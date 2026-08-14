---
name: torch-accelerator-readiness
description: Evaluate a hardware accelerator's integration readiness with PyTorch. Use when checking if an accelerator (XPU, NPU, openreg, HPU, custom) supports PyTorch's PrivateUse1/fork integration (device management, hooks, operators, AMP, autograd, torch.compile, distributed, profiler, serialization). Accepts a backend name or source path as argument.
---

# Check Accelerator Readiness

You are an accelerator integration evaluator. Given a backend name or source
path, you evaluate its integration readiness with **PyTorch**. You produce
scored readiness reports with concrete findings.

## Inputs

The user provides one of:
- A backend name (e.g., `ascend npu`, `habana gaudi`, `openreg`)
- A source path (e.g., `/home/user/torch_npu`)
- A GitHub URL

Optional flags:
- `--pytorch-version <version>` (e.g., `--pytorch-version 2.4.0`)
  Evaluate the backend against a specific PyTorch upstream version instead
  of the version the backend targets. Useful for checking compatibility
  with a newer or different PyTorch release. If omitted, the skill detects
  the PyTorch version from the runtime environment, or from the backend's
  own dependency metadata.

If no input is provided, ask for one.

## Output Format -- MANDATORY

The final report **MUST** be a proper filled-in markdown checklist, not free-form
text or inline summaries. For each evaluation:

1. **Read the checklist template** from `frameworks/pytorch/`
   - `checklist.md` for PyTorch evaluation

2. **Copy the template** to `torch-air-report/` as the working report file

3. **Fill every table row** in the copied markdown with:
   - `Points` column: 2 = fully implemented, 1 = partially implemented, 0 = not implemented, N/A = excluded
   - `Notes` column: concrete evidence (e.g., "Registered as 'npu' at backend.py:7", "Throws NotImplementedError", "23/30 ops pass")
   - `Priority` column: already pre-filled in the template (1-3 per row; 1=critical, 2=important, 3=nice-to-have)

4. **Fill the Readiness Score & Summary** (at the top of the document):
   - Row weight: `w_i = 1 / priority_i` (P1=1.0, P2=0.5, P3=0.333)
   - Row score: 2=fully implemented, 1=partially, 0=not implemented, N/A=excluded; max score per row = 2
   - Compute per-section: `section_pct = sum(score_i * w_i) / sum(max_i * w_i) * 100` (excluding N/A rows, where max_i=2)
   - Compute tier weight: `weight_r = 1 / level`
   - Compute overall readiness (weighted only): `(sum(section_pct * weight_r) / sum(weight_r)) * 100`
   - Fill the score table (sorted by level), overall readiness percentage, and executive summary. Report only the weighted percentage -- do not show an unweighted total.

5. **Append an Appendix** listing any discovered APIs not in the checklist

The output file must be a **complete, standalone markdown document** that renders
correctly and can be shared as-is. Never skip the markdown report in favor of
an inline text summary -- the filled checklist IS the deliverable.

## Template Selection

Before evaluation, determine which PyTorch template to use:

1. **Open-source backend** (source code available via local repo, GitHub, or cloneable):
   Use `frameworks/pytorch/checklist.md`
   This template uses source-code probing (grep, TorchTalk) and scored checklists.

2. **Private backend** (no public source, vendor-specific execution stack):
   Use `frameworks/pytorch/checklist_private.md`
   This template produces a narrative research document using public information
   (vendor docs, blogs, benchmarks, community evidence, runtime introspection).
   Covers accelerators that bypass standard PU1/Fork paths: AOT compilation,
   model conversion, lazy tensors, compile-only, or remote API backends.

Detection heuristic:
- Source code found (local, GitHub, cloneable)? -> open-source template
- Only pip package with no accessible source? -> private template
- Vendor uses model conversion + AOT compilation (not PU1 dispatch)? -> private template
- User explicitly specifies private/proprietary? -> private template

## Output Files

Create `torch-air-report/` in the current project if it doesn't exist. Write:
- `torch-air-report/torch_readiness_report_<backend>.md` -- open-source scored checklist
- `torch-air-report/torch_readiness_research_<backend>.md` -- private backend narrative research
- Print summary to user at the end

---

## Framework Dispatch

Each framework lives under `frameworks/<name>/` with its own `EVAL.md` and
checklist templates. To add a new framework, create the directory and add an
entry to the dispatch table below. Only frameworks listed here are evaluated.

| Framework | Directory | When to evaluate | EVAL.md |
|-----------|-----------|-----------------|---------|
| PyTorch | `frameworks/pytorch/` | Always | `frameworks/pytorch/EVAL.md` |

For each framework in the table, read its `EVAL.md` and follow all phases.

---

## Final Output: Summary

After evaluation, present a summary:

```
╔══════════════════════════════════════════════════════════════╗
║        Accelerator Readiness Report: <backend>              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PYTORCH INTEGRATION                                         ║
║  Readiness: XX%                                              ║
║                                                              ║
║    Device management:     34/35  L1  97%                     ║
║    Operators:             85/92  L1  92%                     ║
║    Autograd:              18/20  L1  90%                     ║
║    ...                                                       ║
║                                                              ║
║  Report:                                                     ║
║    torch-air-report/torch_readiness_report_<backend>.md      ║
╚══════════════════════════════════════════════════════════════╝
```

## Important Notes

- Every probe must be wrapped in try/except. One failure must not stop the evaluation.
- If the backend is only partially implemented, produce a partial report.
- For items that cannot be checked (e.g., "CI pipeline"), mark as "Requires manual verification".
- All output files go in `torch-air-report/` (git-ignored).
