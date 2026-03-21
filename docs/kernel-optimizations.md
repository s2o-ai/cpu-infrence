# S2O Kernel Optimizations — Deep Dive

> A technical learning resource explaining every optimization in S2O's LUT kernels,
> what stock llama.cpp does, what we changed, and why it's faster.

---

## Table of Contents

1. [Background: How Q4_0 Inference Works](#1-background-how-q4_0-inference-works)
2. [VPSHUFB LUT Dequantization](#2-vpshufb-lut-dequantization)
3. [4-Wide Column Processing](#3-4-wide-column-processing)
4. [L2 Cache-Aware Tiling](#4-l2-cache-aware-tiling)
5. [Software Prefetch](#5-software-prefetch)
6. [Weight Repacking](#6-weight-repacking-planned)
7. [Putting It All Together](#7-putting-it-all-together)

---

## 1. Background: How Q4_0 Inference Works

### What is Q4_0?

Q4_0 is a quantization format that compresses each weight from 32-bit float (4 bytes)
down to 4 bits (0.5 bytes) — an **8x compression**. Weights are stored in blocks of 32:

```
struct block_q4_0 {
    fp16  d;          // scale factor (2 bytes)
    uint8 qs[16];     // 32 weights packed as nibble pairs (16 bytes)
};                    // Total: 18 bytes per 32 weights
```

Each byte in `qs` holds two weights: the low nibble (bits 0-3) is one weight, the
high nibble (bits 4-7) is another. The actual weight value is:

```
weight = d * (nibble - 8)
```

where `nibble` is in [0, 15] and the result is in [-8d, +7d].

### The Core Operation: Matrix-Vector Multiply (GEMV)

During single-token generation (the bottleneck for interactive use), the model
computes thousands of dot products:

```
For each output column j in [0, N):
    dst[j] = dot_product(activation[0..K-1], weights[j][0..K-1])
```

A typical 7B model layer has K=4096, N=11008. That's **11,008 dot products**,
each over 4096 elements. The weights are quantized (Q4_0), the activations are FP32.

### Why GEMV is Memory-Bandwidth Limited

For a 7B model FFN layer:
- Weight data: 11,008 columns * (4096/32) blocks * 18 bytes = **~80 MB per layer**
- Activation data: 4096 * 4 bytes = **16 KB** (tiny, fits in L1 cache)

The CPU must read 80 MB of weights from RAM for every layer. On a typical server
(DDR5, ~50 GB/s bandwidth), that's ~1.6 ms per layer just to read the data.
**Computation is free — memory bandwidth is the bottleneck.**

This is the key insight behind every optimization below.

---

## 2. VPSHUFB LUT Dequantization

### What llama.cpp does (stock)

llama.cpp unpacks Q4_0 nibbles using **arithmetic** — 3 separate operations:

```
File: engine/src/llama/ggml/src/ggml-cpu/arch/x86/quants.c

// Step 1: Split nibbles with shift + mask
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(
        _mm_srli_epi16(tmp, 4),   // high nibbles: shift right 4
        tmp                        // low nibbles: as-is
    );
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);  // mask to [0..15]
}

// Step 2: Subtract 8 to center at [-8, +7]
__m256i qx = bytes_from_nibbles_32(x[ib].qs);
const __m256i off = _mm256_set1_epi8(8);
qx = _mm256_sub_epi8(qx, off);              // nibble - 8
```

**Instruction count per block:** 5 instructions (load, shift, mask, combine, subtract)

### What S2O does (VPSHUFB)

We replace the arithmetic with a **lookup table** using the VPSHUFB instruction:

```
File: engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx2.cpp

// The lookup table: nibble [0..15] -> signed value [-8..+7]
// Stored in lut-common.h, aligned to 16 bytes
static const int8_t s2o_q4_dequant_table[16] = {
    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
};

static inline __m256i s2o_vpshufb_dequant_q4_0(const uint8_t * qs) {
    const __m128i lut = _mm_load_si128((const __m128i *)s2o_q4_dequant_table);
    const __m128i m4b = _mm_set1_epi8(0x0F);
    const __m128i raw = _mm_loadu_si128((const __m128i *)qs);

    // Low nibbles [0..15]: one VPSHUFB does mask + lookup in 1 instruction
    const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw, m4b));

    // High nibbles [16..31]: shift + mask + lookup
    const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(
                            _mm_srli_epi16(raw, 4), m4b));

    return _mm256_set_m128i(hi, lo);
}
```

### How VPSHUFB Works

`VPSHUFB` (Packed Shuffle Bytes) is a single instruction that does 16 parallel
table lookups. Given a 16-byte table and 16 index bytes, it outputs 16 results:

```
Table (16 entries):  [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
                      [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7] [8][9][A][B][C][D][E][F]

Input nibbles:       [0x3, 0xA, 0x0, 0xF, ...]
                         |    |    |    |
                         v    v    v    v
Output:              [ -5,  +2,  -8,  +7, ...]   (table[3], table[10], table[0], table[15])
```

All 16 lookups happen in **one clock cycle**. This replaces the separate mask + subtract.

### Why it's an improvement

```
                     Stock llama.cpp          S2O VPSHUFB
                     ----------------         ------------
Operations:          shift + mask + sub       shuffle + mask (per half)
Instructions/block:  5                        4
Latency (cycles):    3 (serial chain)         1 (VPSHUFB is 1 cycle on port 5)
```

The instruction savings are modest (~20%), but the real win is that VPSHUFB
opens the door for **custom codebooks**. With Q4_0 the table is trivial [-8..+7],
but with future S2O quantization formats the table could encode arbitrary 16-entry
codebooks — something arithmetic can't do.

### Where llama.cpp already uses VPSHUFB (but NOT for Q4_0)

llama.cpp uses VPSHUFB for `IQ4_NL` (importance-quantized 4-bit, non-linear):

```
// From quants.c — IQ4_NL uses a 16-entry codebook via VPSHUFB
const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
q4b_1 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
```

`IQ4_NL` uses a non-linear codebook: `{-127, -104, -83, -65, -49, -35, -22, -10,
1, 13, 25, 38, 53, 69, 89, 113}`. S2O adopts the same VPSHUFB pattern but for
standard Q4_0, using the linear `[-8..+7]` table.

---

## 3. 4-Wide Column Processing

### This is the #1 most impactful optimization.

### What llama.cpp does (stock)

Stock llama.cpp processes **one output column at a time**:

```
File: engine/src/llama/ggml/src/ggml-cpu/arch/x86/quants.c

// Simplified pseudocode of stock ggml_vec_dot_q4_0_q8_0:
void dot_product(dst, weights_col_j, activations, K) {
    __m256 acc = zero;
    for (b = 0; b < K/32; b++) {
        qw = unpack_nibbles(weights_col_j[b]);     // load weight block
        qa = load(activations_q8[b]);               // load activation block
        d  = weight_scale * activation_scale;
        acc = fmadd(d, dot(qw, qa), acc);
    }
    *dst = horizontal_sum(acc);
}

// The outer loop (in ggml-cpu.c) calls this N times:
for (j = 0; j < N; j++) {
    dot_product(&dst[j], weights[j], activations, K);  // 1 column at a time
}
```

**Problem:** For each column, the activation vector is re-read from memory.
Actually, activations are small (16 KB) and stay in L1 cache. The real waste is
that each column processes its weight blocks independently — no data sharing.

But the deeper problem is **register utilization**: the CPU has 16 YMM registers
(AVX2), but the 1-wide loop only uses ~4 of them. The remaining 12 sit idle while
the CPU waits for the next weight block to arrive from memory.

### What S2O does (4-wide)

S2O processes **4 output columns simultaneously**:

```
File: engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx2.cpp

for (j = j_start; j + 3 < j_end; j += 4) {
    // Point to 4 different weight rows
    wr0 = weights + (j+0) * stride;
    wr1 = weights + (j+1) * stride;
    wr2 = weights + (j+2) * stride;
    wr3 = weights + (j+3) * stride;

    acc0 = acc1 = acc2 = acc3 = zero;    // 4 accumulators (4 registers)

    for (b = 0; b < K/32; b++) {
        // Load activation ONCE (same for all 4 columns)
        qa = load(act_q8 + b * 32);             // 1 load, reused 4x

        // Dequantize 4 weight blocks
        qw0 = vpshufb_dequant(wr0[b].qs);       // column j+0
        qw1 = vpshufb_dequant(wr1[b].qs);       // column j+1
        qw2 = vpshufb_dequant(wr2[b].qs);       // column j+2
        qw3 = vpshufb_dequant(wr3[b].qs);       // column j+3

        // 4 dot products, all sharing the same activation
        acc0 = fmadd(d0, dot(qw0, qa), acc0);
        acc1 = fmadd(d1, dot(qw1, qa), acc1);
        acc2 = fmadd(d2, dot(qw2, qa), acc2);
        acc3 = fmadd(d3, dot(qw3, qa), acc3);
    }

    dst[j+0] = hsum(acc0);
    dst[j+1] = hsum(acc1);
    dst[j+2] = hsum(acc2);
    dst[j+3] = hsum(acc3);
}
```

### Why 4-wide is faster — the memory bandwidth argument

```
                        Stock (1-wide)                    S2O (4-wide)
                        ---------------                   -------------
Outer iterations:       N = 11,008                        N/4 = 2,752
Weight loads per iter:  K/32 blocks                       4 * K/32 blocks
Total weight loads:     N * K/32                          N * K/32  (SAME!)
Activation loads:       N * K/32                          N/4 * K/32  (4x LESS)
Registers used:         ~4 of 16                          ~12 of 16
```

Wait — the total weight loads are the same? Yes! The 4-wide approach doesn't
reduce total bytes read. **The speedup comes from better register and pipeline
utilization:**

1. **Register saturation**: 4 accumulators + 4 weight vectors + 1 activation =
   9 registers actively used (vs 3 in 1-wide). The CPU's out-of-order engine has
   more independent work to schedule, hiding memory latency.

2. **Activation amortization**: The activation block is loaded once and used
   for 4 dot products. This is a 4x reduction in activation loads. While activations
   fit in L1, every load still costs a cycle for the load port — 4x fewer loads
   means more load-port bandwidth for weights.

3. **Instruction-level parallelism**: The 4 FMADD instructions are independent
   (different accumulators). A modern CPU can execute 2 FMADDs per cycle on
   ports 0+1. With 4 independent FMADDs, the pipeline stays full.

4. **Loop overhead**: 4x fewer outer iterations = 4x fewer branch predictions,
   pointer increments, and loop counter checks.

### Visual: Memory access pattern

```
Stock llama.cpp (1-wide): processes column 0, then column 1, then column 2...

  Weights in RAM:  [col0 block0] [col0 block1] ... [col1 block0] [col1 block1] ...
                    ^^^^^^^^^^^^^                    ^^^^^^^^^^^^^
                    iteration 0                      iteration 1
                    (load act, compute)              (load act AGAIN, compute)

S2O (4-wide): processes columns 0-3 together, then 4-7 together...

  Weights in RAM:  [col0 b0] [col1 b0] [col2 b0] [col3 b0]  [col0 b1] [col1 b1] ...
                    ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^
                    all loaded in same inner iteration
                    (load act ONCE, compute 4 dot products)
```

### Expected speedup

For memory-bandwidth-limited GEMV (which is the common case for single-token
generation on 7B+ models), 4-wide processing typically yields **1.5-2.5x speedup**
over 1-wide, because the CPU's execution units are no longer stalled waiting for
data.

### AVX-512 variant: dual-column 512-bit processing

On AVX-512, S2O goes further by combining 2 weight columns into a single 512-bit
register and doing one 512-bit dot product that computes 2 results at once:

```
File: engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx512.cpp

// Pack column j+0 and j+1 into one 512-bit register:
//   low 256 bits  = weights from column j+0
//   high 256 bits = weights from column j+1
const __m512i qw_pair = s2o_set_m256i(qw0, qw1);  // [col0 | col1]
const __m512i qa_pair = s2o_set_m256i(qa,  qa);    // [act  | act ]

// One 512-bit dot product → two 256-bit results
const __m512 result = s2o_mul_sum_i8_pairs_float_512(qw_pair, qa_pair);

// Split: low 256 = result for col j+0, high 256 = result for col j+1
acc0 = fmadd(d0, extract_low(result),  acc0);      // column j+0
acc1 = fmadd(d1, extract_high(result), acc1);       // column j+1
```

This processes 4 columns as 2 pairs, each pair using one 512-bit operation.
Compared to AVX2's 4 separate 256-bit operations, this uses **half the
dot-product instructions** (2 instead of 4) at the cost of wider registers.

---

## 4. L2 Cache-Aware Tiling

### The problem: GEMM (prompt processing)

During prompt processing, the model processes M tokens at once (M = prompt length).
This turns the GEMV into a **GEMM** (matrix-matrix multiply):

```
For each token i in [0, M):
    For each output column j in [0, N):
        dst[i][j] = dot(activations[i], weights[j])
```

### What llama.cpp does (stock)

Stock llama.cpp just loops over tokens, calling GEMV for each one:

```
for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
        dst[i][j] = gemv(activations[i], weights[j]);
    }
}
```

**Problem:** The weight matrix is read M times — once per token. For a 64-token
prompt with an 80 MB weight matrix, that's **5.1 GB of memory reads** from RAM.

### What S2O does (L2 tiling)

S2O tiles the output columns into groups that **fit in the L2 cache**:

```
File: engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx2.cpp

// Compute how many columns fit in L2 (default 256 KB)
bytes_per_col = (K / 32) * 18;                     // 18 bytes per Q4_0 block
tile_n = 256KB / bytes_per_col;                     // ~110 columns for K=4096
tile_n = round_down_to_4(tile_n);                   // align to 4-wide

// Tiled loop: columns first, then tokens
for (j_tile = 0; j_tile < N; j_tile += tile_n) {   // outer: column tiles
    for (i = 0; i < M; i++) {                        // inner: all tokens
        gemv(dst[i], activations[i], weights,
             K, j_tile, j_tile + tile_n);
    }
}
```

### Why tiling helps — the cache reuse argument

```
Without tiling (stock):                  With L2 tiling (S2O):

Token 0: read ALL N columns from RAM     Token 0: read tile_n columns → L2 cache
Token 1: read ALL N columns from RAM     Token 1: read tile_n columns → FROM L2!
Token 2: read ALL N columns from RAM     Token 2: read tile_n columns → FROM L2!
  ...                                      ...
Token M: read ALL N columns from RAM     Token M: read tile_n columns → FROM L2!
                                          (then move to next tile)

Total weight reads from RAM:              Total weight reads from RAM:
  M * N * bytes_per_col                    N * bytes_per_col  (just once!)
  = 64 * 11008 * 2304                     = 11008 * 2304
  = ~1.5 GB                               = ~24 MB
```

The key insight: within one column tile, token 0 loads the weight blocks into L2.
Tokens 1 through M-1 find those blocks **still in L2** (because the tile was sized
to fit). L2 bandwidth is typically 3-5x higher than RAM bandwidth, so the inner
tokens run much faster.

### Visual: cache behavior

```
L2 Cache (256 KB)
+--------------------------------------------------+
|  weight tile: columns j..j+tile_n                |
|  [col_j block_0] [col_j block_1] ... [col_j+1]  |
|  ... fits entirely in L2                         |
+--------------------------------------------------+

Iteration:
  Token 0 → loads tile from RAM into L2, computes dot products
  Token 1 → tile is HOT in L2, computes (L2 speed, ~3-5x faster)
  Token 2 → still hot, computes
  ...
  Token M → still hot, computes
  → Move to next tile of columns, repeat
```

### Tile size calculation

```
For K = 4096 (typical 7B model):
  blocks_per_col = 4096 / 32 = 128 blocks
  bytes_per_col  = 128 * 18 = 2304 bytes (~2.3 KB per column)
  tile_n         = 262144 / 2304 = 113 columns
  tile_n aligned = 112 (rounded down to multiple of 4)

For K = 8192 (13B model):
  bytes_per_col  = 256 * 18 = 4608 bytes
  tile_n         = 262144 / 4608 = 56 columns
  tile_n aligned = 56
```

### Expected speedup

For prompt processing (M > 1):
- **M=16 tokens**: ~2-3x improvement (weights read once vs 16 times)
- **M=64 tokens**: ~3-5x improvement
- **M=1 (single token)**: no improvement (falls back to non-tiled GEMV)

The improvement scales with M because more tokens amortize the initial RAM read.

---

## 5. Software Prefetch

### The problem: memory latency

When the CPU needs a weight block that isn't in cache, it stalls for **100-200 clock
cycles** waiting for RAM to respond. During this time, the execution units are idle.

### What llama.cpp does (stock)

Stock llama.cpp does have some prefetch in the SSSE3 path:

```
File: engine/src/llama/ggml/src/ggml-cpu/arch/x86/quants.c (SSSE3 path)

_mm_prefetch(&x[ib] + sizeof(block_q4_0), _MM_HINT_T0);
_mm_prefetch(&y[ib] + sizeof(block_q8_0), _MM_HINT_T0);
```

But the AVX2 path (which is the fast path on modern CPUs) has **no prefetch**.

### What S2O does

S2O adds prefetch to the 4-wide inner loop, requesting weight blocks 4 iterations
ahead for all 4 columns:

```
File: engine/src/llama/ggml/src/ggml-cpu/s2o-lut/lut-x86-avx2.cpp

for (b = 0; b < nb_k; b++) {
    // Request next blocks NOW so they arrive by the time we need them
    if (b + S2O_LUT_PREFETCH_DIST < nb_k) {
        _mm_prefetch(&wr0[b + 4], _MM_HINT_T0);  // column 0, 4 blocks ahead
        _mm_prefetch(&wr1[b + 4], _MM_HINT_T0);  // column 1, 4 blocks ahead
        _mm_prefetch(&wr2[b + 4], _MM_HINT_T0);  // column 2, 4 blocks ahead
        _mm_prefetch(&wr3[b + 4], _MM_HINT_T0);  // column 3, 4 blocks ahead
    }

    // By the time we get here, the data should be in L1/L2
    qw0 = vpshufb_dequant(wr0[b].qs);
    ...
}
```

### How prefetch works

```
_mm_prefetch(address, _MM_HINT_T0)
```

This tells the CPU: "I will need the cache line at `address` soon — start fetching
it from RAM into L1 cache now, in the background, while I do other work."

```
Without prefetch:                     With prefetch (DIST=4):

block 0: compute (data in cache)      block 0: compute + prefetch block 4
block 1: compute (data in cache)      block 1: compute + prefetch block 5
block 2: STALL 100 cycles (cache miss) block 2: compute (block 2 was already prefetched)
block 3: compute                      block 3: compute (block 3 was already prefetched)
block 4: STALL 100 cycles (cache miss) block 4: compute (prefetched at block 0!)
```

### Why PREFETCH_DIST = 4?

The distance must be large enough that the prefetch has time to complete before
we need the data, but not so large that the prefetched data gets evicted from
cache before we use it.

```
Prefetch distance calculation:
  Each block: ~18 bytes of weight data
  Processing time per block: ~10-15 cycles (dot product + fmadd)
  4 columns per iteration: ~40-60 cycles per inner loop iteration
  RAM latency: ~100-200 cycles
  Distance needed: 200 / 50 = 4 iterations ahead

  4 blocks * 4 columns * 18 bytes = 288 bytes prefetched
  Well within a single cache line or two (64 bytes each)
```

### Expected speedup

Software prefetch alone typically gives **10-20% improvement** for GEMV on large
matrices where the working set exceeds L2 cache. The impact is larger when combined
with 4-wide processing, because there's more computation per iteration to overlap
with the memory latency.

---

## 6. Weight Repacking (Planned)

> **Status:** Not yet implemented. Currently `set_tensor` does a straight `memcpy`.

### The problem: scattered memory access in 4-wide

With standard Q4_0 layout, weight rows are stored contiguously per-column:

```
Standard Q4_0 layout in memory:

Column 0: [block0][block1][block2]...[blockN]   ← contiguous
Column 1: [block0][block1][block2]...[blockN]   ← contiguous
Column 2: [block0][block1][block2]...[blockN]   ← contiguous
Column 3: [block0][block1][block2]...[blockN]   ← contiguous
```

When the 4-wide loop processes block `b` for columns 0-3, it reads from 4 different
memory locations:

```
Inner loop at block b:
  wr0[b] → address X + 0 * stride + b * 18      ← cache line A
  wr1[b] → address X + 1 * stride + b * 18      ← cache line B (far away!)
  wr2[b] → address X + 2 * stride + b * 18      ← cache line C (far away!)
  wr3[b] → address X + 3 * stride + b * 18      ← cache line D (far away!)
```

These 4 loads hit 4 different cache lines, causing **4 separate cache line fetches**.

### The solution: column-interleaved layout

Repack weights so that the 4 blocks needed by one 4-wide iteration are
**adjacent in memory**:

```
Repacked layout:

Group 0 (cols 0-3): [col0_b0][col1_b0][col2_b0][col3_b0] [col0_b1][col1_b1]...
Group 1 (cols 4-7): [col4_b0][col5_b0][col6_b0][col7_b0] [col4_b1][col5_b1]...
```

Now the 4-wide inner loop reads 4 consecutive blocks from a single memory region:

```
Inner loop at block b:
  wr0[b] → address Y + (b*4 + 0) * 18     ← same cache line!
  wr1[b] → address Y + (b*4 + 1) * 18     ← same or adjacent cache line!
  wr2[b] → address Y + (b*4 + 2) * 18     ← same or adjacent cache line!
  wr3[b] → address Y + (b*4 + 3) * 18     ← adjacent cache line
```

4 blocks * 18 bytes = 72 bytes. That fits in just 2 cache lines (64 bytes each)
instead of 4 scattered cache lines.

### Where it would be implemented

```
File: engine/src/llama/ggml/src/ggml-cpu/s2o-lut/s2o-lut.cpp

// In set_tensor (called when model weights are loaded):
static void ggml_backend_s2o_lut_buffer_set_tensor(...) {
    // Currently: memcpy(tensor->data + offset, data, size);
    // Planned:   s2o_repack_q4_0(tensor, data);
}
```

### Expected speedup

Weight repacking would provide an additional **15-30% improvement** on top of
4-wide processing by reducing cache line fetches from 4 to ~2 per inner iteration.

---

## 7. Putting It All Together

### Optimization stack — each layer builds on the previous

```
Layer 5 (planned): Weight Repacking      +15-30%  (fewer cache lines per iteration)
Layer 4:           Software Prefetch      +10-20%  (hide RAM latency)
Layer 3:           L2 Cache Tiling        +200-400% for GEMM (weights stay in L2)
Layer 2:           4-Wide Columns         +50-150% (register utilization, ILP)
Layer 1:           VPSHUFB Dequant        +10-20%  (fewer instructions per block)
---------------------------------------------------------------------------
Base:              Stock llama.cpp Q4_0   (baseline)
```

### Comparison table: stock vs S2O

```
+------------------------+---------------------+---------------------------+
| Aspect                 | Stock llama.cpp     | S2O LUT Kernel            |
+========================+=====================+===========================+
| Nibble unpacking       | shift + mask + sub  | VPSHUFB lookup table      |
|                        | (3 ops, serial)     | (1 op, 1 cycle latency)   |
+------------------------+---------------------+---------------------------+
| Columns per iteration  | 1                   | 4                         |
+------------------------+---------------------+---------------------------+
| Registers used (AVX2)  | ~4 of 16            | ~12 of 16                 |
+------------------------+---------------------+---------------------------+
| Activation handling    | Q8_0 pre-quantized  | FP32 -> INT8 on the fly,  |
|                        | (done once globally) | done once per GEMV call   |
+------------------------+---------------------+---------------------------+
| GEMM cache strategy    | None (linear scan)  | L2-tiled (256 KB tiles)   |
+------------------------+---------------------+---------------------------+
| Software prefetch      | SSSE3 path only     | AVX2 + AVX-512, all paths |
+------------------------+---------------------+---------------------------+
| Weight layout          | Standard row-major  | Standard (repacking TBD)  |
+------------------------+---------------------+---------------------------+
| VPSHUFB for Q4_0       | NO (only for IQ4_NL)| YES                       |
+------------------------+---------------------+---------------------------+
```

### When each optimization matters most

```
+------------------------+------------------+-----------------------------+
| Scenario               | Bottleneck       | Most impactful optimization |
+========================+==================+=============================+
| Single token (GEMV)    | Memory bandwidth | 4-wide column processing    |
| Short prompt (M<16)    | Memory bandwidth | 4-wide + prefetch           |
| Long prompt (M>64)     | Memory bandwidth | L2 tiling (huge win)        |
| Small model (K<1024)   | Compute          | VPSHUFB + 4-wide            |
| Large model (K>4096)   | Memory bandwidth | 4-wide + tiling + prefetch  |
+------------------------+------------------+-----------------------------+
```

### File map

```
engine/src/llama/ggml/src/ggml-cpu/s2o-lut/
    lut-common.h          Shared types, LUT table, prefetch constant, kernel selector
    lut-x86-avx2.cpp      AVX2: VPSHUFB dequant, 4-wide GEMV, L2-tiled GEMM, prefetch
    lut-x86-avx512.cpp    AVX-512: same optimizations, dual-column 512-bit processing
    lut-arm-neon.cpp       ARM NEON: baseline + DOTPROD variants
    s2o-lut.cpp           ggml integration: buffer type, tensor traits, auto-benchmark
    s2o-lut.h             Public header
    test_lut.cpp          C++ correctness tests (10 configs)
```

### Further reading

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) — search for `_mm_shuffle_epi8` (VPSHUFB), `_mm256_fmadd_ps` (FMA), `_mm_prefetch`
- [Agner Fog's instruction tables](https://www.agner.org/optimize/) — cycle-accurate latency/throughput for every x86 instruction
- llama.cpp stock Q4_0 kernel: `engine/src/llama/ggml/src/ggml-cpu/arch/x86/quants.c` line 543
- T-MAC paper (Microsoft): activation-derived LUT approach (different from our weight-side LUT)
