---
name: check-accelerator-readiness
description: Evaluate a hardware accelerator's integration readiness with PyTorch and/or vLLM. Use when checking if an accelerator (XPU, NPU, openreg, HPU, custom) supports PyTorch's PrivateUse1/fork integration (device management, hooks, operators, AMP, autograd, torch.compile, distributed, profiler, serialization) and optionally vLLM's platform plugin (attention backends, worker, model runner, custom ops, KV cache, quantization, distributed). Accepts a backend name or source path as argument. Covers both PyTorch and vLLM in a single evaluation.
---

# Check Accelerator Readiness

You are an accelerator integration evaluator. Given a backend name or source
path, you evaluate its integration readiness with **PyTorch** and optionally
**vLLM**. You produce scored readiness reports with concrete findings.

## Inputs

The user provides one of:
- A backend name (e.g., `ascend npu`, `habana gaudi`, `openreg`)
- A source path (e.g., `/home/user/torch_npu`)
- A GitHub URL

If no input is provided, ask for one.

## Output Format -- MANDATORY

The final reports **MUST** be proper filled-in markdown checklists, not free-form
text or inline summaries. For each evaluation:

1. **Read the checklist template** from `~/.claude/skills/check-accelerator-readiness/templates/`
   - `pytorch_checklist.md` for PyTorch evaluation
   - `vllm_checklist.md` for vLLM evaluation

2. **Copy the template** to `agent_space/` as the working report file

3. **Fill every table row** in the copied markdown with:
   - `Status` column: `[x]` Ready | `[~]` Partially Ready | `[ ]` Not Ready | `[N/A]` Not applicable
   - `Notes` column: concrete evidence (e.g., "Registered as 'npu' at backend.py:7", "Throws NotImplementedError", "23/30 ops pass")
   - `Pts` column: already pre-filled in the template (1-3 per row)

4. **Fill the Readiness Score & Summary** (at the top of the document):
   - Compute per-section: `section_pct = earned_pts / max_pts` (excluding N/A)
   - Compute weight: `weight_r = 1 / rank`
   - Compute overall: `(sum(section_pct * weight_r) / sum(weight_r)) * 10`
   - Fill the score table (sorted by rank), overall score (X.X / 10), verdict, and 3-5 sentence summary paragraph

5. **Append an Appendix** listing any discovered APIs not in the checklist

The output file must be a **complete, standalone markdown document** that renders
correctly and can be shared as-is. Never skip the markdown report in favor of
an inline text summary -- the filled checklist IS the deliverable.

## Template Selection

Before evaluation, determine which PyTorch template to use:

1. **Open-source backend** (source code available via local repo, GitHub, or cloneable):
   Use `~/.claude/skills/check-accelerator-readiness/templates/pytorch_checklist.md`
   This template uses source-code probing (grep, TorchTalk) and scored checklists.

2. **Closed-source OOT backend** (no public source, vendor-specific execution stack):
   Use `~/.claude/skills/check-accelerator-readiness/templates/pytorch_checklist_oot_closed.md`
   This template produces a narrative research document using public information
   (vendor docs, blogs, benchmarks, community evidence, runtime introspection).
   Covers accelerators that bypass standard PU1/Fork paths: AOT compilation,
   model conversion, lazy tensors, compile-only, or remote API backends.

Detection heuristic:
- Source code found (local, GitHub, cloneable)? -> open-source template
- Only pip package with no accessible source? -> closed-source template
- Vendor uses model conversion + AOT compilation (not PU1 dispatch)? -> closed-source template
- User explicitly specifies closed-source? -> closed-source template

## Scope Detection

Determine what to evaluate:

1. **PyTorch readiness** (always): Evaluate using the selected template
   (see Template Selection above)

2. **vLLM readiness** (if applicable): Check if a vLLM plugin exists for this
   accelerator. Search for:
   - A `vllm-<name>` or `vllm_<name>` package/repo
   - `vllm.platform_plugins` entry points in the backend's setup.py
   - A separate vLLM plugin repo (e.g., `vllm-ascend`, `vllm-hpu`)

   If found, also evaluate vLLM readiness using
   `~/.claude/skills/check-accelerator-readiness/templates/vllm_checklist.md`

   For closed-source backends, vLLM assessment is part of the narrative
   research document -- the evaluator writes a dedicated vLLM section
   covering API endpoint vs native integration vs disaggregated setups.
   No separate vLLM checklist is used for closed-source evaluations.

3. **Ask the user** if scope is ambiguous: "I found the PyTorch backend. Should
   I also check vLLM integration? I found a vllm plugin at <location>."

## Output Files

Create `agent_space/` in the current project if it doesn't exist. Write:
- `agent_space/pytorch_readiness_report_<backend>.md` -- open-source scored checklist
- `agent_space/pytorch_readiness_research_<backend>.md` -- closed-source narrative research
- `agent_space/vllm_readiness_report_<backend>.md` -- if vLLM scope detected (open-source only)
- Print combined summary to user at the end

---

## Part 1: PyTorch Readiness Evaluation

### Templates

Read `~/.claude/skills/check-accelerator-readiness/templates/pytorch_checklist.md`.
Copy it to `agent_space/pytorch_readiness_report_<backend>.md` as your working copy.

### Pre-Phase: Checklist Refinement via TorchTalk

Before every evaluation, sync the PyTorch checklist template with the current
PyTorch source using TorchTalk MCP tools. This ensures the checklist reflects
the latest APIs, hooks, and registration points.

**Ground Rules (internal skill rules -- do NOT include in generated reports):**
1. **Preserve structure**: Section numbering, table format, column layout, scoring model, rank assignments, and markdown structure must NOT change during refinement. Only row-level content (add/update/remove items within existing tables) may change.
2. **Summary + rank table at top**: Every generated document must start with the Executive Summary and Section Scores table (sorted by rank) immediately after the header metadata. No report is valid without it.
3. **No section reordering**: Sections stay in their original document order. Only the summary table sorts by rank.
4. **Log changes**: After refinement, output a short changelog (added N items, updated M items, removed K items) so the user can review before evaluation proceeds.
5. **No meta-content in output**: These ground rules, skill instructions, and internal process details must never appear in the final generated report. The report is a standalone evaluation document.main

**Step 1: Ensure PyTorch source is current**
- Check TorchTalk status via `get_status`. If not available, skip refinement and proceed with existing template.
- If PyTorch source is local, fetch latest main: `git -C <pytorch_path> fetch pytorch/pytorch main`

**Step 2: Validate existing checklist items**

Use a **hybrid strategy**: TorchTalk for Python-side APIs, modules, tests, and
dispatch bindings. Direct `grep` on PyTorch source for C++ macros and
registration utilities (TorchTalk `search` only indexes dispatch-registered
bindings, not C++ infrastructure like macros/registration functions).

| Checklist Section | Tool | Query | What to Check |
|-------------------|------|-------|---------------|
| Sec 1: Device Registration | `grep` | `rename_privateuse1_backend`, `generate_methods_for_privateuse1_backend`, `_register_device_module` in `torch/` and `c10/` | Function signatures, parameters unchanged |
| Sec 2: Accelerator Hooks | `grep` | `AcceleratorHooksInterface`, `RegisterPrivateUse1HooksInterface` in `c10/` | Hook method names, new mandatory hooks |
| Sec 2: Accelerator Hooks | TorchTalk `graph` | mode=calls on hooks interface class (when graph ready) | New extensibility points |
| Sec 3: Device Guard | `grep` | `C10_REGISTER_GUARD_IMPL`, `DeviceGuardImplInterface` in `c10/` | Guard interface methods |
| Sec 4: Autoload | `grep` | `torch.backends` entry point handling in `torch/` | Entry point group name, autoload mechanism |
| Sec 5: Memory | `grep` | `Allocator` subclass registration, `getPinnedMemoryAllocator` in `c10/` | Allocator interface changes |
| Sec 7: RNG | `grep` | `REGISTER_GENERATOR_PRIVATEUSE1` in `aten/` | Generator registration API |
| Sec 8: Operators | TorchTalk `tests` | mode=find query=`OpInfo` | New required ops, deprecated ops |
| Sec 8: Operators | `grep` | `TORCH_LIBRARY_IMPL` with `PrivateUse1` in reference backends | Registration patterns |
| Sec 9: Python Frontend | TorchTalk `modules` | trace `torch.accelerator` focus=full | New methods on torch.accelerator module |
| Sec 10: Autograd | `grep` | `AutogradPrivateUse1` in `c10/` and `aten/` | Autograd dispatch key registration |
| Sec 11: AMP | `grep` | `AutocastPrivateUse1` in `aten/` and `torch/amp/` | Autocast API changes |
| Sec 12: torch.compile | `grep` | `register_interface_for_device`, `DeviceInterface` in `torch/_inductor/` | Compile interface methods |
| Sec 12: torch.compile | TorchTalk `modules` | trace `DeviceInterface` focus=full | Interface method list |
| Sec 13: Serialization | `grep` | `TensorBackendMetaRegistry` in `torch/` | Serialization hook API |
| Sec 14: Distributed | TorchTalk `modules` | trace `ProcessGroup` focus=full | ProcessGroup interface changes |
| Sec 14: Distributed | `grep` | `register_backend` in `torch/distributed/` | Backend registration API |
| Sec 15: Profiler | `grep` | `REGISTER_PRIVATEUSE1_PROFILER`, `ProfilerStubs` in `torch/` | Profiler registration API |
| Sec 18: Testing | TorchTalk `tests` | mode=find query=`privateuse1` focus=files | Test file inventory |

All `grep` commands run against the PyTorch source path from TorchTalk `get_status`
(e.g., `/home/devuser/pytorch`). Use `grep -rn --include="*.py" --include="*.h"
--include="*.cpp"` for broad coverage.

For each query:
- If API still exists with same signature: no change needed
- If API renamed/moved: update checklist item text and notes
- If API removed: mark item as deprecated, add replacement if found

**Step 3: Discover new APIs not in checklist**

Run broader discovery queries using both tools:

TorchTalk queries:
1. `modules` mode=trace name=`torch.accelerator` focus=full -- find new Python methods
2. `modules` mode=list name=`accelerator` -- find new accelerator-related modules
3. `tests` mode=find query=`privateuse1` -- find new test infrastructure
4. `graph` mode=calls on hook/interface classes (when graph ready) -- find new extensibility points

grep queries on PyTorch source:
5. `grep -rn "privateuse1\|PrivateUse1" torch/ c10/ aten/ --include="*.py" --include="*.h" --include="*.cpp"` -- find all PU1 references
6. `grep -rn "def .*accelerator" torch/ --include="*.py"` -- find new accelerator APIs
7. `grep -rn "register.*private\|REGISTER.*PRIVATE" c10/ aten/ --include="*.h" --include="*.cpp"` -- find new registration macros

For each discovered API not already in the checklist:
- Determine which section it belongs to
- Assign appropriate point value (1-3)
- Add as new row to the template

**Step 4: Update the template**

Write changes to the template file at
`~/.claude/skills/check-accelerator-readiness/templates/pytorch_checklist.md`.
Log all changes (additions, modifications, removals) so the user can review.

**Note**: This refinement applies to the PyTorch checklist only. The vLLM
checklist is not auto-refined (vLLM is not indexed by TorchTalk).

---

### Phase 0: Source Discovery & Integration Path Detection

The user may not have the repo locally. The agent should:

1. **Check locally first**: `~/Dev/`, `~/projects/`, `pip show`, `python -c "import torch_<name>"`
2. **Search GitHub** if not local: `gh search repos "torch <name>"`, try `torch-<name>`, `torch_<name>`, `pytorch-<name>`
3. **Clone if needed**: `git clone --depth 1 <url> /tmp/<name>`
4. **Ask the user** only as a last resort

Then detect **PU1 vs Fork**:
- `grep -r "rename_privateuse1_backend\|register_privateuse1_backend"` -> PU1
- Check `c10/core/DeviceType.h`, `DispatchKey.h` for custom entries -> Fork
- Mark items for the other path as `[N/A]`

### Phase 1: Systematic Probing (Sections 1-23)

For each section, probe via source analysis or runtime testing. For each row:
- `[x]` = full points, `[~]` = half points, `[ ]` = 0, `[N/A]` = excluded
- Fill Status, Notes, and Pts columns

**Probe order (by rank, rank 1 = most critical):**

**Rank 1 -- Foundational (probe first):**
- Section 1: Device Registration & Management
- Section 8: Operator Registration (minimal kernels, extended ops, model validation, fallbacks, custom ops)
- Section 10: Autograd
- Section 3: Device Guard
- Section 2: Accelerator Hooks [PU1]
- Section 5: Memory & Allocator

**Rank 2 -- Core Production:**
- Section 13: Serialization & Model Portability
- Section 9: Python Frontend
- Section 11: AMP
- Section 12: torch.compile
- Section 14: Distributed
- Section 19: Dtype Support
- Section 20: Numerical Accuracy
- Section 18: Testing & CI

**Rank 3 -- Quality of Life:**
- Section 6: Streams & Events
- Section 7: RNG & Generator
- Section 4: Autoload [PU1]
- Section 16: DataLoader
- Section 21: CUDA Behavior Parity
- Section 15: Profiler
- Section 22: Ecosystem Compatibility
- Section 17: Future Modules

### Phase 2: Dynamic API Discovery

1. Introspect `torch.<name>` module -- flag public methods not in checklist
2. Query dispatcher for registered ops not tested in Section 8
3. Check hook overrides beyond Section 2

### Phase 3: Scoring (Summary at top of document)

```
Section Pct = earned_pts / max_pts (excluding N/A rows)
weight_r = 1 / rank
Overall Score = (sum(Section Pct * weight_r) / sum(weight_r)) * 10
```

Verdict: 0 - 4 Not Ready | 4 - 7 Partially Ready | 7 - 10 Ready

Fill the summary score table (sorted by rank) at the top of the document and write the 3-5 sentence summary paragraph.

---

## Part 2: vLLM Readiness Evaluation (if applicable)

### Templates

Read `~/.claude/skills/check-accelerator-readiness/templates/vllm_checklist.md`.
Copy it to `agent_space/vllm_readiness_report_<backend>.md` as your working copy.

### Prerequisite

The PyTorch readiness score from Part 1 feeds into Section 0 of the vLLM
checklist. If PyTorch scored below 4/10 (Not Ready), note this as a blocker
in the vLLM report.

### Source Discovery

Search for the vLLM plugin:
- `vllm-<name>`, `vllm_<name>` on GitHub
- `vllm.platform_plugins` entry in the PyTorch backend's setup.py
- Separate repo (e.g., `vllm-project/vllm-ascend`)

### Probing Sections 1-19

#### Section 0 -- PyTorch Prerequisites (11 items)
Auto-fill from Part 1 results.

#### Section 1 -- Plugin Registration (6 items)
Check `setup.py`/`pyproject.toml` for `vllm.platform_plugins` and `vllm.general_plugins` entry points.

#### Section 2 -- Platform Class (~29 items)
Find the Platform subclass. Check class variables, device management methods, component getters, compilation methods, config validation, feature flags.

#### Section 3 -- Worker (8 items)
Find Worker subclass. Check: `init_device()`, `initialize_cache()`, `load_model()`, `get_kv_cache_spec()`, `determine_available_memory()`, `initialize_from_config()`, `compile_or_warm_up_model()`, `execute_model()`.

#### Section 4 -- Model Runner (6 items)
Find ModelRunner subclass. Check input preparation, forward pass, sampling, InputBatch, SamplingMetadata.

#### Section 5 -- Attention Backend (6 items + features)
Find attention backends. Check: `validate_configuration()`, paged attention, prefill, decode, chunked prefill, prefix caching, sliding window, GQA, MQA, MLA, FP8 KV cache, speculative decoding, cross-attention.

#### Section 6 -- KV Cache (7 items)
Check: `update_block_size_for_backend()`, block allocation, copy, swap, hybrid KV cache, custom specs, prefix caching.

#### Section 7 -- Custom Ops (8 items)
```bash
grep -r "register_oot\|forward_oot" <path>/ --include="*.py"
```
Check: paged_attention, rotary_embedding, rms_norm, silu_and_mul, fused_moe, all_reduce.

#### Section 8 -- Quantization (2 items + methods)
Check `supported_quantization` list. Search for: FP8, GPTQ, AWQ, SqueezeLLM, INT8, GGUF, BitsAndBytes, Marlin.

#### Section 9 -- Distributed (7 items + collectives)
Find communicator. Check collectives and parallelism: TP, PP, EP, DP.

#### Section 10 -- Compilation (6 items)
Check `get_compile_backend()`, static graph, wrapper, pass manager.

#### Section 11 -- Memory Management (5 items)
Check `determine_available_memory()`, `get_current_memory_usage()`, memory fraction, profiling, swap.

#### Section 12 -- Speculative Decoding (4 items)
Search for draft model, verification, acceptance/rejection, bonus tokens.

#### Section 13 -- Multimodal (5 items)
Search for image/video/audio encoder, cross-attention, preprocessing.

#### Section 14 -- Model Compatibility (6 items)
Check `verify_model_arch()`, supported architectures.

#### Section 15 -- Profiling (5 items)
Check profiler integration, metrics, stat logger.

#### Section 16 -- Testing (functional + integration + CI)
Count test files, check CI config.

#### Section 17 -- Dtype Support (7 dtypes)
Check compute, KV cache, quantization support per dtype.

### Dynamic API Discovery

1. List all Platform methods overridden beyond checklist
2. List all `@CustomOp.register_oot` registrations
3. Search for features not in checklist (tool calling, structured output, etc.)

### Scoring (Summary at top of document)

Fill the summary score table (sorted by rank) at the top of the document.
Use `weight_r = 1/rank`, overall = `(sum(section_pct * weight_r) / sum(weight_r)) * 10`.
Append Appendix of uncovered APIs.

---

## Final Output: Combined Summary

After both evaluations, present a combined summary:

```
╔══════════════════════════════════════════════════════════════╗
║        Accelerator Readiness Report: <backend>              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PYTORCH INTEGRATION                                         ║
║  Score: X.X / 10  [Verdict]                                  ║
║                                                              ║
║    Device management:     34/35  R1  97%                     ║
║    Operators:             85/92  R2  92%                     ║
║    Autograd:              18/20  R3  90%                     ║
║    ...                                                       ║
║                                                              ║
║  VLLM INTEGRATION (if evaluated)                             ║
║  Score: X.X / 10  [Verdict]                                  ║
║                                                              ║
║    PyTorch prereqs:       28/28  R1  100%                    ║
║    Plugin registration:    6/6   R2  100%                    ║
║    Platform class:        25/29  R3   86%                    ║
║    ...                                                       ║
║                                                              ║
║  Reports:                                                    ║
║    agent_space/pytorch_readiness_report_<backend>.md          ║
║    agent_space/vllm_readiness_report_<backend>.md             ║
║                                                              ║
║  Summary: <3-5 sentence paragraph>                           ║
╚══════════════════════════════════════════════════════════════╝
```

## Important Notes

- Every probe must be wrapped in try/except. One failure must not stop the evaluation.
- If the backend is only partially implemented, produce a partial report.
- For items that cannot be checked (e.g., "CI pipeline"), mark as "Requires manual verification".
- PyTorch evaluation always runs. vLLM evaluation only runs if a plugin is detected.
- The PyTorch score feeds into vLLM Section 0 as a prerequisite.
- All output files go in `agent_space/` (git-ignored).
