---
name: check-accelerator-readiness
description: Evaluate a hardware accelerator's integration readiness with PyTorch. Use when checking if an accelerator (XPU, NPU, openreg, HPU, custom) supports PyTorch's PrivateUse1/fork integration (device management, hooks, operators, AMP, autograd, torch.compile, distributed, profiler, serialization). Optionally evaluates vLLM platform plugin when explicitly requested. Accepts a backend name or source path as argument.
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

2. **Copy the template** to `pytorch-air-report/` as the working report file

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
   Use `~/.claude/skills/check-accelerator-readiness/templates/pytorch_checklist.md`
   This template uses source-code probing (grep, TorchTalk) and scored checklists.

2. **Private backend** (no public source, vendor-specific execution stack):
   Use `~/.claude/skills/check-accelerator-readiness/templates/pytorch_checklist_private.md`
   This template produces a narrative research document using public information
   (vendor docs, blogs, benchmarks, community evidence, runtime introspection).
   Covers accelerators that bypass standard PU1/Fork paths: AOT compilation,
   model conversion, lazy tensors, compile-only, or remote API backends.

Detection heuristic:
- Source code found (local, GitHub, cloneable)? -> open-source template
- Only pip package with no accessible source? -> private template
- Vendor uses model conversion + AOT compilation (not PU1 dispatch)? -> private template
- User explicitly specifies private/proprietary? -> private template

## Scope Detection

Determine what to evaluate:

1. **PyTorch readiness** (always): Evaluate using the selected template
   (see Template Selection above)

2. **vLLM readiness** (only on demand): Only evaluate vLLM if the user
   explicitly requests it (e.g., "also check vLLM", "include vLLM",
   "for pytorch and vllm"). Do NOT auto-detect or auto-evaluate vLLM.

   When requested, search for:
   - A `vllm-<name>` or `vllm_<name>` package/repo
   - `vllm.platform_plugins` entry points in the backend's setup.py
   - A separate vLLM plugin repo (e.g., `vllm-ascend`, `vllm-hpu`)

   If found, evaluate using
   `~/.claude/skills/check-accelerator-readiness/templates/vllm_checklist.md`

   For private backends, vLLM assessment is part of the narrative
   research document -- the evaluator writes a dedicated vLLM section
   covering API endpoint vs native integration vs disaggregated setups.
   No separate vLLM checklist is used for private backend evaluations.

## Output Files

Create `pytorch-air-report/` in the current project if it doesn't exist. Write:
- `pytorch-air-report/pytorch_readiness_report_<backend>.md` -- open-source scored checklist
- `pytorch-air-report/pytorch_readiness_research_<backend>.md` -- private backend narrative research
- `pytorch-air-report/vllm_readiness_report_<backend>.md` -- if vLLM scope detected (open-source only)
- Print combined summary to user at the end

---

## Part 1: PyTorch Readiness Evaluation

### Templates

Read `~/.claude/skills/check-accelerator-readiness/templates/pytorch_checklist.md`.
Copy it to `pytorch-air-report/pytorch_readiness_report_<backend>.md` as your working copy.

### Pre-Phase: Checklist Refinement via TorchTalk

Before every evaluation, sync the PyTorch checklist template with the current
PyTorch source using TorchTalk MCP tools. This ensures the checklist reflects
the latest APIs, hooks, and registration points.

**Ground Rules (internal skill rules -- do NOT include in generated reports):**
1. **Preserve structure**: Section numbering, table format, column layout, scoring model, level assignments, and markdown structure must NOT change during refinement. Only row-level content (add/update/remove items within existing tables) may change.
2. **Summary + level table at top**: Every generated document must start with the Executive Summary and Section Scores table (sorted by level) immediately after the header metadata. No report is valid without it.
3. **No section reordering**: Sections stay in their original document order. Only the summary table sorts by level. Template tables are pre-sorted by priority (1 first, then 2, then 3) -- preserve this order in generated reports.
4. **Log changes**: After refinement, output a short changelog (added N items, updated M items, removed K items) so the user can review before evaluation proceeds.
5. **No meta-content in output**: These ground rules, skill instructions, agent instructions, procedural steps (e.g., "The agent should...", "Check if...", "Search for...", "Once detected, the agent should..."), scoring model explanations, calculation formulas, and internal process details must never appear in the final generated report. When copying templates, strip all instructional prose, the Scoring Model section, and the Calculation section -- keep only headings, fillable tables, and filled-in content. The report is a standalone evaluation document for the reader, not a how-to guide for the evaluator.

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
- Assign appropriate priority (1=critical, 2=important, 3=nice-to-have)
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
5. **Determine backend version**: Use `gh release list --repo <owner>/<repo> --limit 1`, `git describe --tags`, or `pip show <package>` to find the latest release version. Record it in the report header and the Source Discovery table.

Then detect **PU1 vs Fork**:
- `grep -r "rename_privateuse1_backend\|register_privateuse1_backend"` -> PU1
- Check `c10/core/DeviceType.h`, `DispatchKey.h` for custom entries -> Fork
- Mark items for the other path as `[N/A]`

### Phase 1: Systematic Probing (Sections 1-23)

For each section, probe via source analysis or runtime testing. For each row:
- Points: 2 = fully implemented, 1 = partially, 0 = not implemented, N/A = excluded
- Fill Points, Notes columns (Priority is pre-filled)

**Probe order (by level, level 1 = most critical):**

**Level 1 (probe first):**
- Section 1: Device Registration & Management
- Section 8: Operator Registration (minimal kernels, extended ops, model validation, fallbacks, custom ops)
- Section 10: Autograd
- Section 3: Device Guard
- Section 2: Accelerator Hooks [PU1]
- Section 5: Memory & Allocator

**Level 2:**
- Section 13: Serialization & Model Portability
- Section 9: Python Frontend
- Section 11: AMP
- Section 12: torch.compile
- Section 14: Distributed
- Section 19: Dtype Support
- Section 20: Numerical Accuracy
- Section 18: Testing & CI

**Level 3:**
- Section 6: Streams & Events
- Section 7: RNG & Generator
- Section 4: Autoload [PU1]
- Section 16: DataLoader
- Section 15: Profiler
- Section 22: Ecosystem Compatibility
- Section 17: Future Modules

### Phase 2: Dynamic API Discovery

1. Introspect `torch.<name>` module -- flag public methods not in checklist
2. Query dispatcher for registered ops not tested in Section 8
3. Check hook overrides beyond Section 2

### Phase 2.5: Upstream Candidate Discovery

This is a non-scored advisory section. Goal: find features the backend
built that are generic enough to benefit all PU1/accelerator backends if
upstreamed to PyTorch core.

**Placement**: In the generated report, upstream candidates MUST appear
immediately after the Section Scores table and Readiness percentage
(inside the "Readiness Score & Summary" block), NOT at the end of the
document. This matches the template structure in `pytorch_checklist.md`.

#### Step 1: Find candidates (systematic diff)

The backend's source falls into two buckets:
- **Required PU1 plumbing** -- what Section 23 (Registration API Quick
  Reference) lists. Every PU1 backend must do these. Not upstream candidates.
- **Everything else** -- infrastructure the backend built beyond the required
  registration points. These are the candidates.

To find "everything else":

1. **Inventory the backend's public surface** -- list all Python modules,
   C++ source files, and exported symbols that are NOT direct implementations
   of a Section 1-22 checklist item. Focus on:
   - Python modules beyond the device module (profiler, memory, diagnostics, patches)
   - C++ files beyond the standard hooks/guard/allocator/generator
   - Monkey patches applied to `torch.*` or `torch._dynamo.*`
   - Custom context managers, decorators, or utilities

2. **Grep for infrastructure patterns** -- scan for code that wraps, extends,
   or works around PyTorch core APIs:
   ```
   grep -rn "monkey_patch\|patch_torch\|_original_" <backend>/ --include="*.py"
   grep -rn "fallback\|cpu_fallback\|fast_path" <backend>/ --include="*.py" --include="*.cpp"
   grep -rn "class.*Wrapper\|class.*Shim\|class.*Bridge" <backend>/ --include="*.py"
   grep -rn "explain\|diagnose\|hidden.*overhead" <backend>/ --include="*.py"
   grep -rn "reentrancy\|reentrant\|nested.*compile" <backend>/ --include="*.py"
   grep -rn "topology\|device_summary\|physical_device" <backend>/ --include="*.py"
   grep -rn "offload\|layout_like\|empty_cache" <backend>/ --include="*.py"
   ```

3. **Compare against other backends** -- if available (torch_npu, torch_xla,
   intel_extension_for_pytorch), check whether they independently built similar
   infrastructure. Two+ backends solving the same problem independently is a
   strong upstream signal.

#### Step 2: Classify each candidate

Apply these concrete tests in order:

**Generic** (ready to upstream):
- Contains zero references to backend-specific types, APIs, or hardware names
- Could be copy-pasted into `torch/accelerator/` and work for any PU1 backend
- Examples: device-index normalization, reentrancy guard, region-based profiler
  framework, atexit shutdown handler

**Needs Abstraction** (concept generic, implementation coupled):
- Solves a problem every backend faces, BUT implementation references
  backend-specific internals (allocator, compiler, runtime API)
- Upstream path: define an interface/hook in core, let backends implement
- Test: "Could another backend use this if we replaced the backend-specific
  calls with hook methods?" If yes, Needs Abstraction.
- Examples: `empty_cache()` with backend-specific cache types, explain profiler
  with backend-specific signal names, view-aware dispatch with backend-specific
  bad-pattern list

**Hardware-Specific** (not upstreamable):
- Solves a problem unique to this hardware's architecture, ISA, or runtime
- Other backends would never need this
- Examples: device-specific memory alignment workarounds, hardware topology
  that only this NPU uses, compiler-specific IR limitations

#### Step 3: Write upstream notes

For each Generic or Needs Abstraction feature, write:
- What the PyTorch core API would look like (function signature, module path)
- What the backend would implement (hook method, registration call)
- Whether PyTorch core already has a partial equivalent (grep pytorch source)

#### Categories to organize findings

Place discoveries into whichever of these 7 categories fits. Leave categories
empty (no rows) if nothing found.

| Category | What to look for |
|----------|-----------------|
| 24.1 CPU Fallback Infrastructure | Generic fallback dispatcher, diagnostic counters, per-op tracking, fast-path registry |
| 24.2 Profiler / Explain Framework | Hidden-overhead detection, region profiling, cause/remedy mapping, diff analysis |
| 24.3 View-Aware Dispatch | View recipe detection, shape simulation, fallback-to-contiguous policies |
| 24.4 Memory Management Patterns | `empty_cache()` with plugin points, offload context managers, layout affinity |
| 24.5 Device Topology & Utilities | Topology visualization, logical-to-physical mapping, capability queries |
| 24.6 Compile Integration Patterns | Reentrancy guards, event isolation, recompile limit handling, compile-only mode |
| 24.7 New Hooks / API Extensions | Methods beyond `AcceleratorHooksInterface`, new `torch.accelerator.*` APIs |

### Phase 3: Scoring (Summary at top of document)

```
Row weight: w_i = 1 / priority_i
Section Pct = sum(score_i * w_i) / sum(max_i * w_i) * 100  (excluding N/A rows, max_i=2)
weight_r = 1 / level
Readiness = (sum(Section Pct * weight_r) / sum(weight_r)) * 100
```

Fill the score table (sorted by level) at the top of the document, the overall readiness percentage, and the executive summary.

---

## Part 2: vLLM Readiness Evaluation (if applicable)

### Templates

Read `~/.claude/skills/check-accelerator-readiness/templates/vllm_checklist.md`.
Copy it to `pytorch-air-report/vllm_readiness_report_<backend>.md` as your working copy.

### Prerequisite

The PyTorch readiness score from Part 1 feeds into Section 0 of the vLLM
checklist. If PyTorch scored below 40%, note this as a blocker
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

Fill the summary score table (sorted by level) at the top of the document.
Row weight: `w_i = 1/priority_i`. Section pct: `sum(score_i * w_i) / sum(max_i * w_i) * 100` (max_i=2, excluding N/A).
Tier weight: `weight_r = 1/level`. Readiness: `(sum(section_pct * weight_r) / sum(weight_r)) * 100`.
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
║  Readiness: XX%                                              ║
║                                                              ║
║    Device management:     34/35  L1  97%                     ║
║    Operators:             85/92  L1  92%                     ║
║    Autograd:              18/20  L1  90%                     ║
║    ...                                                       ║
║                                                              ║
║  VLLM INTEGRATION (if evaluated)                             ║
║  Readiness: XX%                                              ║
║                                                              ║
║    PyTorch prereqs:       28/28  L1  100%                    ║
║    Plugin registration:    6/6   L1  100%                    ║
║    Platform class:        25/29  L1   86%                    ║
║    ...                                                       ║
║                                                              ║
║  Reports:                                                    ║
║    pytorch-air-report/pytorch_readiness_report_<backend>.md          ║
║    pytorch-air-report/vllm_readiness_report_<backend>.md             ║
╚══════════════════════════════════════════════════════════════╝
```

## Important Notes

- Every probe must be wrapped in try/except. One failure must not stop the evaluation.
- If the backend is only partially implemented, produce a partial report.
- For items that cannot be checked (e.g., "CI pipeline"), mark as "Requires manual verification".
- PyTorch evaluation always runs. vLLM evaluation only runs when the user explicitly requests it.
- The PyTorch score feeds into vLLM Section 0 as a prerequisite.
- All output files go in `pytorch-air-report/` (git-ignored).
