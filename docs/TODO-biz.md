# Business, Sales & Compliance TODO

> Business tasks: design partners, compliance, sales, fundraising.
> For engineering tasks, see [TODO-tech.md](TODO-tech.md).
> Full plan: [TODO.md](TODO.md)

---

## Phase 1: Foundation (Months 1-3)

### Task 1.8: Design Partner Outreach
- [ ] Identify 10 healthcare prospects (community hospitals, 200-500 beds)
  - On-premises data center
  - Active AI/ML initiatives
  - Epic/Cerner EHR (implies capable IT)
  - Decision-maker within 2 hops of founder's network
- [ ] Prepare demo (clinical note summarization, CPU hardware, < 15 min install-to-inference)
- [ ] 5+ demo calls completed
- [ ] 1+ LOI signed (3-month pilot, $2-5K/month)
- [ ] Regular feedback channel established (Slack, weekly calls)

### Phase 1 Gate — Business Criteria
- [ ] 1+ signed LOI from a healthcare organization
- [ ] Pilot scope and success criteria documented

---

## Phase 2: Differentiation (Months 3-6)

### Task 2.7: SOC2 Type 1 Preparation
- [ ] Set up Vanta or Drata ($1.5-2K/month)
  - Connect to GitHub, AWS, identity provider
  - Configure automated evidence collection
- [ ] Implement required controls:
  - [ ] Access control policy
  - [ ] Change management process (PR reviews, deployment approvals)
  - [ ] Incident response plan (documented + tested)
  - [ ] Data encryption (at rest + in transit)
  - [ ] Vulnerability management (dependency scanning, patching)
  - [ ] Employee security training
  - [ ] Asset inventory
- [ ] Engage audit firm (Prescient Assurance, Johanson Group, or similar)
- [ ] Define scope: Security + Availability + Confidentiality
- [ ] Begin formal Type 1 audit (target: month 5)

### Phase 2 Gate — Business Criteria
- [ ] SOC2 Type 1 audit formally begun (auditor engaged, controls implemented)
- [ ] First paying pilot ($2-5K/month) or strong LOI with payment terms

---

## Phase 3: Enterprise (Months 6-12)

### Task 3.1: SOC2 Type 1 Completion
- [ ] Address audit findings
- [ ] Remediate gaps (risk assessments, vendor management, BCP)
- [ ] Obtain clean Type 1 report (no qualified opinions)
- [ ] Begin Type 2 observation period (12 months)

### Task 3.2: HIPAA Compliance
- [ ] HIPAA risk assessment (third-party assessor)
- [ ] Privacy policies and procedures
- [ ] BAA template (reviewed by healthcare attorney)
- [ ] Technical safeguards:
  - [ ] AES-256 encryption at rest
  - [ ] TLS 1.3 for data in transit
  - [ ] Audit controls (log all PHI access)
  - [ ] Access controls (minimum necessary principle)
  - [ ] Automatic session timeout
  - [ ] Emergency access procedures
- [ ] Breach notification plan (72-hour HIPAA timeline)
- [ ] Staff training on PHI handling

### Task 3.5: RBAC, SSO, Audit Logging
- [ ] SSO: SAML 2.0 (Okta, Azure AD) + OIDC (Google, Auth0)
- [ ] SCIM user provisioning
- [ ] RBAC roles: Admin, Operator, User, Viewer
- [ ] Audit logging (tamper-resistant, append-only, hash chaining)
- [ ] Export to SIEM (Splunk, ELK) via syslog/webhook
- [ ] Per-team API keys, model access controls, usage quotas

### Task 3.7: Customer Onboarding Pipeline
- [ ] Automated tenant provisioning (namespace, API keys, SSO config)
- [ ] Model selection wizard
- [ ] Domain-specific calibration workflow
- [ ] Guided deployment (K8s or standalone)
- [ ] Welcome docs and training materials
- [ ] Target: new customer operational in < 1 business day

### Task 3.8: First Enterprise Customers
- [ ] Convert design partners to paying customers
- [ ] 3+ signed enterprise contracts ($50-80K/year each)
- [ ] $150-250K ARR
- [ ] < 5% monthly churn
- [ ] 1 published case study
- [ ] Quarterly business review process

### Phase 3 Gate — Business Criteria
- [ ] SOC2 Type 1 report obtained
- [ ] HIPAA compliance validated by third party
- [ ] 3+ paying enterprise customers
- [ ] $150K+ ARR

---

## Phase 4: Scale (Months 12-18)

### Task 4.1: SOC2 Type 2 Completion
- [ ] 12-month observation period complete
- [ ] Clean Type 2 report obtained
- [ ] Report shareable with prospects

### Task 4.5: FedRAMP Preparation
- [ ] Engage 3PAO (Third Party Assessment Organization)
- [ ] Implement NIST 800-53 controls (builds on SOC2)
- [ ] Prepare System Security Plan (SSP)
- [ ] Begin authorization process

### Task 4.6: Financial Services Expansion
- [ ] Adapt positioning for financial use cases (document analysis, compliance screening)
- [ ] Financial-specific compliance controls (SOX)
- [ ] 2+ financial services pilot customers
- [ ] Financial services case study

### Task 4.7: Series A Fundraise
- [ ] $500K+ ARR trigger
- [ ] 10+ enterprise customers, < 5% churn
- [ ] LTV:CAC > 3:1
- [ ] Repeatable sales playbook in healthcare
- [ ] SOC2 Type 2 complete or near-complete
- [ ] Raise $5-10M

### Phase 4 Targets
- [ ] $500K-1M ARR (base), $1.5M+ (stretch)
- [ ] 10-15 enterprise customers
- [ ] SOC2 Type 2 certified
- [ ] Series A raised or in final stages
- [ ] Financial services vertical entered

---

## Security Checklist (Per Phase)

### Phase 1
- [ ] HTTPS/TLS for all API endpoints
- [ ] API key authentication (bearer tokens)
- [ ] Dependency vulnerability scanning in CI (Snyk/Dependabot)
- [ ] Secrets management (env vars or vault, no credentials in code)
- [ ] Container image scanning
- [ ] Signed commits (GPG)

### Phase 2
- [ ] All SOC2 technical controls implemented
- [ ] Encryption at rest for stored models and data
- [ ] Access logging for all API calls
- [ ] Rate limiting on public endpoints
- [ ] Input validation and sanitization
- [ ] CORS configuration

### Phase 3
- [ ] SSO/RBAC fully enforced
- [ ] HIPAA technical safeguards
- [ ] Annual penetration testing (third-party)
- [ ] Air-gap security controls
- [ ] Tamper-resistant audit logs
- [ ] Automated security scanning (SAST + DAST)
- [ ] Incident response tested via tabletop exercise

### Phase 4
- [ ] FedRAMP NIST 800-53 controls (subset)
- [ ] Multi-tenancy security hardening
- [ ] Advanced threat detection
- [ ] Quarterly security awareness training
- [ ] Third-party security audit of LUT kernels
