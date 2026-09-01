---
name: torch-accelerator-readiness
description: Evaluate a hardware accelerator's integration readiness with PyTorch and all other frameworks listed in the Framework Dispatch table. Use when checking if an accelerator (XPU, NPU, openreg, HPU, custom) supports PyTorch's PrivateUse1/fork integration (device management, hooks, operators, AMP, autograd, torch.compile, distributed, profiler, serialization). Accepts a backend name or source path as argument.
---

# Check Accelerator Readiness

You are an accelerator integration evaluator. Given a backend name or source
path, you evaluate its integration readiness with every framework listed in the
**Framework Dispatch** table below. You produce scored readiness reports with
concrete findings — one report file per framework.

## Inputs

The user provides:
- A **backend** (required): a name (e.g., `openreg`, `ascend npu`), a local source path, or a GitHub URL
- `--framework <name>` (optional, repeatable): which framework(s) to run — `pytorch`, `vllm`. Omit to run all.
- `--pytorch-version <version>` (optional): PyTorch version to evaluate against (e.g., `2.6.0`). Independent of `--framework`. Defaults to latest.
- `--vllm-version <version>` (optional): vLLM version to evaluate against (e.g., `v0.9.1`). Independent of `--framework`. Defaults to latest.

The two flag groups are orthogonal: `--framework` controls which frameworks run; version flags control what version each uses. If no backend is provided, ask for one.

## Output Format -- MANDATORY

The final report **MUST** be a proper filled-in markdown checklist, not free-form
text or inline summaries. For each framework evaluation:

1. **Read the checklist template** from the framework's directory (see Framework Dispatch table)

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

## Template Selection (PyTorch only)

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

## Time Estimate

After the backend has been located and its integration path detected (open-source
vs. private, PrivateUse1 vs. Fork) but **before** the scored probing begins for
each framework, print a per-framework tentative estimate of how long that
framework's evaluation will take. Judge the range from the runtime drivers you
just discovered:

- **Integration path** -- a private/narrative evaluation does web research and is
  slower than an open-source source-code probe.
- **Codebase size** -- more source to grep and trace takes longer.
- **Applicable sections** -- how many of the framework sections apply to this backend.

Present it as a rough range and state clearly that it is an approximation, e.g.:

```
PyTorch: ~6-9 min (open-source, PrivateUse1, ~15 applicable sections). Rough estimate, not a guarantee.
vLLM:    ~4-6 min (OOT plugin, ~10 applicable sections). Rough estimate, not a guarantee.
```

Record the wall-clock time at the start of each framework's evaluation so the
actual duration can be reported per framework in the final summary.

## Output Files

Create `torch-air-report/` in the current project if it doesn't exist. Write one
report file per **evaluated** framework only — frameworks not selected by the user
do not produce a file:
- `torch-air-report/torch_readiness_report_<backend>.md` -- PyTorch open-source scored checklist
- `torch-air-report/torch_readiness_research_<backend>.md` -- PyTorch private backend narrative research
- `torch-air-report/vllm_readiness_report_<backend>.md` -- vLLM scored checklist
- Print summary to user at the end

---

## Framework Dispatch

Each framework lives under `frameworks/<name>/` with its own `EVAL.md` and
checklist templates. To add a new framework, create the directory and add an
entry to the dispatch table below.

This table is the sole source of truth for available frameworks. Do NOT consult
git history to determine which frameworks are active.

| Framework | Keyword | Directory | EVAL.md | Output file |
|-----------|---------|-----------|---------|-------------|
| PyTorch | `pytorch` | `frameworks/pytorch/` | `frameworks/pytorch/EVAL.md` | `torch-air-report/torch_readiness_report_<backend>.md` |
| vLLM | `vllm` | `frameworks/vllm/` | `frameworks/vllm/EVAL.md` | `torch-air-report/vllm_readiness_report_<backend>.md` |

**Execution rules:**
- If `--framework` is not given → evaluate every framework in this table.
- If `--framework` is given → evaluate only the specified framework(s).
- `--pytorch-version` and `--vllm-version` are independent of `--framework`: they set the version for their framework regardless of which frameworks are selected.
- Pass each resolved version to the corresponding EVAL.md. If a version flag is omitted, default to latest (`main`).

---

## Final Output: Summary

After all framework evaluations are complete, print this summary to the terminal.
Include **one block per evaluated framework only** — omit blocks for frameworks
that were not run. Show the resolved version next to each framework name.

```
╔══════════════════════════════════════════════════════════════╗
║        Accelerator Readiness Report: <backend>              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PYTORCH INTEGRATION  (v2.6.0)                               ║
║  Readiness: XX%                                              ║
║                                                              ║
║    Device management:     34/35  L1  97%                     ║
║    Operators:             85/92  L1  92%                     ║
║    Autograd:              18/20  L1  90%                     ║
║    ...                                                       ║
║                                                              ║
║    Time — Estimated (active):  ~6-9 min                      ║
║            Actual (wall-clock): 10 min                       ║
║            Actual (active):      8 min                       ║
║                                                              ║
║  VLLM INTEGRATION  (v0.9.1)                                  ║
║  Readiness: XX%                                              ║
║                                                              ║
║    Plugin Registration:   12/12  L1 100%                     ║
║    Platform Class:        18/20  L1  90%                     ║
║    Worker Core:           14/16  L1  88%                     ║
║    ...                                                       ║
║                                                              ║
║    Time — Estimated (active):  ~4-6 min                      ║
║            Actual (wall-clock):  8 min                       ║
║            Actual (active):      5 min                       ║
║                                                              ║
║  Reports:                                                    ║
║    torch-air-report/torch_readiness_report_<backend>.md      ║
║    torch-air-report/vllm_readiness_report_<backend>.md       ║
╚══════════════════════════════════════════════════════════════╝
```

Report three time figures **per framework**, each measured from that framework's Time Estimate step to completion of its evaluation:

- **Estimated (active)** -- the earlier estimate. It covers active compute/probing
  time only and never includes time spent waiting on the user.
- **Actual (wall-clock)** -- total elapsed time, including any waits for permission
  prompts, clarifying answers, or other user input. Informational only.
- **Actual (active)** -- wall-clock minus the user-wait spans, i.e. the time
  actually spent probing and analyzing.

Compare the estimate against **Actual (active)** only, since both exclude user-wait
and therefore measure the same thing; the wall-clock figure is shown for context
and is expected to be larger. Use the gap between estimated and actual-active to
calibrate later estimates within this session, erring toward the observed pace.

The active-time figure is an approximation -- this skill has no precise stopwatch
across pauses, so estimate the user-wait spans deliberately rather than assuming
they are zero. Actual time also depends on machine speed and probing depth, so the
estimate and actual-active will rarely match exactly -- that is expected.

## Important Notes

- Every probe must be wrapped in try/except. One failure must not stop the evaluation.
- If the backend is only partially implemented, produce a partial report.
- For items that cannot be checked (e.g., "CI pipeline"), mark as "Requires manual verification".
- All output files go in `torch-air-report/` (git-ignored).
