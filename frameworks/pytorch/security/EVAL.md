## Part 2: PyTorch Security Readiness Evaluation

Security is an evaluation **dimension** of the PyTorch framework, not a separate
framework. Run this only when `--security` or `--all` is specified.

### Template

Read `frameworks/pytorch/security/checklist.md`.
Copy it to `torch-air-report/security_readiness_report_<backend>.md` as your working copy.

### Ground Rules (internal — do NOT include in generated reports)

1. **Preserve structure**: Domain IDs (SEC-MT-01, etc.), table format, priority/level assignments, and scoring model must NOT change. Only fill Points, Notes, and summary fields.
2. **Summary at top**: Every report must start with Executive Summary and Domain Scores table (sorted by level) immediately after header metadata. Metadata must include torch-air version and model.
3. **No meta-content in output**: Skill instructions, scoring rubrics, and procedural steps must never appear in the final report.
4. **Distinguish absence from undocumented**: "Not documented" is not the same as "not implemented." Note indirect evidence as requiring vendor confirmation.
5. **Human review required**: Reports are AI-assisted and must be reviewed by a security engineer before external sharing.

---

### Phase 0: Backend Identification

Map the accelerator to its PyTorch backend:

| Accelerator | Backend Package | Device Name | Repo |
|-------------|-----------------|-------------|------|
| IBM Spyre | torch-spyre | spyre | torch-spyre/torch-spyre |
| Rebellions ATOM | torch-rbln | rbln | RBLN-SW/torch-rbln |
| Huawei Ascend | torch_npu + torchair | npu | Ascend/pytorch |

For unlisted accelerators:
1. Check locally: `pip show`, `python -c "import torch_<name>"`
2. Search GitHub: `gh search repos "torch <name>"`
3. Clone if needed: `git clone --depth 1 <url> /tmp/<name>`
4. Record backend version from `gh release list`, `git describe --tags`, or `pip show`

---

### Phase 1: Evidence Gathering

For each security domain, collect evidence in priority order:

1. **Source code** of the OOT backend (if accessible)
2. **Official documentation** (vendor docs, README, SECURITY.md)
3. **Academic papers** referencing the hardware
4. **PyTorch ecosystem references** (dev-discuss posts, RFCs, PRs)
5. **Vendor datasheets and whitepapers**

Web search queries:
```
"<accelerator-name>" security OR encryption OR isolation OR VFIO OR TEE OR attestation site:github.com
"<accelerator-name>" confidential computing OR memory protection
"<accelerator-name>" PyTorch backend security
```

---

### Phase 2: Per-Item Scoring

For each of the 37 checklist items, score using the same numeric model as functional integration:

| Points | Meaning |
|--------|---------|
| **2** | Fully evidenced — capability present and verified |
| **1** | Partially evidenced — capability exists with caveats or incomplete coverage |
| **0** | Not evidenced — absent, undocumented, or insufficient public information |
| **N/A** | Not applicable to this accelerator's deployment model |

Fill the Points and Notes columns for every row. When public information is
insufficient, score **0** and note "requires vendor confirmation" in Notes.

**Domain levels** (for overall weighting):

| Level | Domains |
|-------|---------|
| 1 | SEC-MT (Multi-Tenant Isolation), SEC-ME (Device Memory Encryption), SEC-DS (Data Scrubbing) |
| 2 | SEC-HT (Host-Device Transit Security), SEC-FD (Firmware & Driver) |
| 3 | SEC-PI (PyTorch Integration Surface) |

---

### Phase 3: Source Code Inspection (when code is available)

#### Memory Safety in C++ Bridge Code (`csrc/`)
- Buffer allocations without bounds checks
- Integer arithmetic on sizes/offsets without overflow checks
- Missing null checks on device pointers
- Missing error handling on device API calls

#### Python Surface Inspection
- `eval()`, `exec()`, `pickle.loads()` on external input
- Error messages leaking device addresses
- Input validation at `torch.compile` backend entry point

#### Build and Distribution
- PyPI package signatures
- Pinned dependencies in requirements
- Pre-built binary blobs without source

#### Source Code Red Flags
- Unchecked return values from device allocation APIs
- Integer arithmetic on tensor sizes without overflow guards
- Error messages containing hex addresses or full stack traces
- Pre-built binary blobs (`.so`, `.dll`) without reproducible build

---

### Phase 4: Scoring & Gap Analysis

Compute domain and overall scores using the same formulas as functional integration:

```
Row weight: w_i = 1 / priority_i
Domain % = sum(score_i * w_i) / sum(max_i * w_i) * 100  (excluding N/A rows, max_i = 2)
weight_r = 1 / level
Security Readiness (%) = (sum(domain_pct * weight_r) / sum(weight_r)) * 100
```

Fill the Domain Scores table (sorted by level) and overall Security Readiness percentage.

Additionally, classify gaps for the Gap Analysis section:

| Priority | Definition |
|----------|------------|
| **CRITICAL** | Directly enables cross-tenant data access in production |
| **HIGH** | Enables attack with physical access or host compromise |
| **MEDIUM** | Defense-in-depth gap; should be addressed but not blocking |
| **LOW** | Informational; best practice not followed |

---

### Reference Architecture (NVIDIA H100 Baseline)

| Domain | NVIDIA H100 Reference |
|--------|------------------------|
| Multi-Tenant | MIG: 7 HW instances with dedicated HBM/cache/compute |
| Memory Encryption | CPR (Compute Protected Region) + HW firewalls |
| Transit Security | AES-GCM 256 DMA + SPDM attestation |
| Data Scrubbing | Memory cleared on CC context switch |
| Firmware | Measured boot + HW RoT + anti-rollback since Turing |
| Inter-device | Protected PCIe (PPCIE) for multi-GPU |

Use this as the maturity benchmark. Quantify how far below parity the accelerator is and what the path to parity looks like.
