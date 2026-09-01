## Part 2: vLLM Readiness Evaluation

### Templates

Read `frameworks/vllm/checklist.md`.
Copy it to `torch-air-report/vllm_readiness_report_<backend>.md` as your working copy.

### Pre-Phase: Contract Sync via vLLM Source Inspection

Before every evaluation, verify that the checklist items reflect the current vLLM
plugin contracts. vLLM evolves fast; APIs added/renamed between releases can make
checklist items stale. This phase catches those drifts.

**Ground Rules (internal — do NOT appear in generated reports):**
1. **Preserve structure**: Section numbering, table format, scoring model, level
   assignments must not change. Only row-level content may be updated.
2. **Summary at top**: Every generated document must start with the Executive
   Summary and Section Scores table immediately after the header metadata.
3. **No section reordering**: Template table order is preserved in reports.
4. **Log changes**: Output a short changelog after refinement.
5. **No meta-content in output**: Strip all instructional prose, scoring model
   explanations, and calculation formulas from generated reports.

**Step 1: Locate vLLM source**
- If `--vllm-version <version>` was given: `git clone --depth 1 --branch <version> https://github.com/vllm-project/vllm /tmp/vllm` (substitute the actual version string, e.g. `v0.9.1`); use `/tmp/vllm` as source root
- Otherwise: run `pip show vllm` and `python -c "import vllm; print(vllm.__file__)"` to find the installed source tree; if not installed, clone latest: `git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm`
- Record resolved version via `pip show vllm` or `git describe --tags`; fill `vLLM base version` in report header
- Fill `PyTorch required by vLLM` from `/tmp/vllm/pyproject.toml` only — never from the plugin's own `pyproject.toml`. Run: `grep -E "^torch[^-]|\"torch[^-]" /tmp/vllm/pyproject.toml` and record the standard `torch==X.Y.Z` pin. If the plugin replaces `torch` with a vendor distribution (e.g. `torch-rbln`, `torch-npu`), ignore it — still report the standard torch version from upstream vLLM.

**Step 2: Validate Platform interface methods**

| Checklist Section | Tool | What to check |
|-------------------|------|---------------|
| Sec 1: Plugin Registration | `grep -rn "platform_plugins\|general_plugins" vllm/plugins/__init__.py` | Entry point group names unchanged |
| Sec 2: Platform class | `grep -rn "class Platform" vllm/platforms/interface.py` | Required abstract methods |
| Sec 2: check_and_update_config | `grep -rn "check_and_update_config" vllm/platforms/interface.py` | Method signature, what config fields must be set |
| Sec 2: get_attn_backend_cls | `grep -n "def get_attn_backend_cls" vllm/platforms/interface.py && grep -A3 "def get_attn_backend_cls" vllm/platforms/interface.py` | Signature changed to `(selected_backend, attn_selector_config, num_heads=None)` in recent vLLM — capture full signature before scoring any backend |
| Sec 3: WorkerBase | `grep -rn "class WorkerBase" vllm/v1/worker/worker_base.py` | Abstract methods list |
| Sec 3: ModelRunner | `grep -rn "class.*ModelRunner" vllm/v1/worker/` | No shared abstract base (`ModelRunnerBase` does not exist); check GPUModelRunner interface (`execute_model`, `profile_run`, `load_model`) |
| Sec 3: get_kv_cache_spec | `grep -rn "get_kv_cache_spec" vllm/v1/worker/worker_base.py` | Return type, KVCacheSpec |
| Sec 4: AttentionBackend | `grep -rn "class AttentionBackend" vllm/v1/attention/backend.py` | Abstract class methods |
| Sec 4: AttentionImpl.forward | `grep -rn "def forward" vllm/v1/attention/backend.py` | Signature, required args |
| Sec 5: KVCacheSpec | `grep -rn "class KVCacheSpec" vllm/v1/kv_cache_interface.py` | Required properties |
| Sec 7: CommunicatorBase | `grep -rn "class DeviceCommunicatorBase" vllm/distributed/device_communicators/` | Required collective methods |
| Sec 8: torch.compile | `grep -rn "compilation_config\|torch.compile" vllm/v1/worker/` | Compile integration points |
| Sec 15: LoRA | `grep -rn "get_punica_wrapper\|lora" vllm/platforms/interface.py` | LoRA hook on Platform |
| Sec 16: Spec decode | `grep -rn "speculative_config" vllm/v1/worker/worker_base.py` | Spec decode integration points |

**Step 3: Discover new Platform interface methods**

Run broader scans to catch new hooks added since the last checklist update:
```
grep -rn "abstractmethod\|@property" vllm/platforms/interface.py
grep -rn "def get_" vllm/platforms/interface.py
grep -rn "def is_" vllm/platforms/interface.py
grep -rn "def check_" vllm/platforms/interface.py
grep -rn "abstractmethod" vllm/v1/worker/worker_base.py
grep -rn "abstractmethod" vllm/v1/attention/backend.py
grep -rn "abstractmethod" vllm/distributed/device_communicators/base_device_communicator.py
```

For each new abstract method found not in the checklist:
- Determine which section it belongs to
- Assign priority: 1=blocks basic inference, 2=expected for production, 3=optional enhancement
- Add as a new row

**Step 4: Check vLLM entry point groups**

```
grep -rn "platform_plugins\|general_plugins\|stat_logger_plugins\|logits_processors" vllm/plugins/__init__.py
```
Confirm the full list of supported entry point groups:
- `vllm.platform_plugins` — hardware platform registration
- `vllm.general_plugins` — custom ops, kernels, model patches
- `vllm.stat_logger_plugins` — custom metrics/logging
- `vllm.logits_processors` — custom sampling strategies

**Step 5: Checklist Drift Check (run before scoring any row)**

For any checklist row that names a specific method, confirm that method still exists
in the vLLM source tree you cloned. Run `grep -rn "def <method_name>" vllm/` and if
the grep returns empty, mark the row **N/A (removed/renamed)** rather than 0 or 1.
Pay particular attention to: `get_attn_backend_cls`, `get_device_communicator_cls`,
`check_if_supports_dtype`, `get_metadata_cls`, `get_kv_cache_shape`, and `copy_blocks`.

---

### Phase 0: Plugin Source Discovery & Integration Path Detection

The user may not have the backend plugin locally. The agent should:

1. **Check locally first**:
   - `pip show vllm-<name>` or `pip show <name>-vllm`
   - `python -c "import vllm_<name>; print(vllm_<name>.__file__)"`
   - Search `~/Dev/`, `~/projects/`, current directory for the plugin package

2. **Search GitHub** if not local:
   - `gh search repos "vllm <name> plugin platform"`
   - Try patterns: `vllm-<name>`, `vllm_<name>`, `<vendor>-vllm`, `<name>-vllm-plugin`
   - Check known vendor orgs: `huawei/vllm-ascend`, `IntelLabs/vllm-gaudi`, etc.

3. **Clone if needed**: `git clone --depth 1 <url> /tmp/vllm-<name>`

4. **Determine backend version** via `gh release list`, `git describe --tags`, or `pip show`

5. **Ask the user** only as a last resort

Then detect **integration path**:
- `grep -rn "vllm.platform_plugins" setup.py pyproject.toml` → Out-of-tree plugin
- `grep -rn "class.*Platform.*:" <backend>/` → Has Platform subclass
- Check if the backend is a vLLM fork (code lives inside a vllm fork, not a separate package)
- Mark items for the inapplicable path as `[N/A]`

| Signal | Detected | Value | Notes |
|--------|----------|-------|-------|
| `vllm.platform_plugins` entry point found | `[ ]` | | |
| `vllm.general_plugins` entry point found | `[ ]` | | |
| `Platform` subclass found | `[ ]` | | |
| `WorkerBase` subclass found | `[ ]` | | |
| Backend-specific `ModelRunner` class found | `[ ]` | | |
| `AttentionBackend` subclass found | `[ ]` | | |
| `DeviceCommunicatorBase` subclass found | `[ ]` | | |
| Backend is a vLLM fork (not separate package) | `[ ]` | | |
| **Detected path** | | _[OOT plugin / vLLM fork / Hybrid]_ | |

---

### Phase 1: Systematic Probing (Sections 1–21)

For each section, probe via source analysis. For each row:
- Points: 2 = fully implemented, 1 = partially implemented, 0 = not implemented, N/A = excluded
- Fill Points and Notes columns (Priority is pre-filled)
- Items marked `[Runtime]` in the checklist cannot be verified by source analysis. Apply this scoring rule:
  - **2** = an automated test _actually runs_ this scenario on the target hardware and is known to pass in CI
  - **1** = test infrastructure exists (test file, fixtures, golden files) but confirmed runtime execution is not available
  - **0** = no test infrastructure exists for this scenario
  - **Never score 2 on "reference baseline" reasoning alone for a `[Runtime]` row**

**Probe order (by level, level 1 = most critical):**

**Level 1 (probe first):**
- Section 1: Plugin Registration & Entry Points
- Section 2: Platform Class
- Section 3: Worker Core
- Section 4: Attention Backend
- Section 5: KV Cache
- Section 6: Basic Inference Correctness

**Level 2:**
- Section 7: Distributed Communicator
- Section 8: torch.compile & Graph Capture
- Section 9: Quantization
- Section 10: Chunked Prefill & Continuous Batching
- Section 11: Prefix Caching
- Section 12: Tensor Parallelism
- Section 13: Model & Dtype Coverage
- Section 14: End-to-End Serving Validation

**Level 3:**
- Section 15: LoRA / Multi-LoRA
- Section 16: Speculative Decoding
- Section 17: Pipeline Parallelism
- Section 18: FP8/FP4 KV Cache
- Section 19: Multimodal Inputs
- Section 20: Async Output & Non-blocking I/O
- Section 21: Observability & Ecosystem

#### Probing Strategies by Section

**Section 1 — Plugin Registration:**
```
grep -rn "vllm.platform_plugins\|vllm.general_plugins" setup.py pyproject.toml
grep -rn "def register" <backend>/       # registration function
grep -rn "PlatformEnum.OOT\|PlatformEnum" <backend>/
```

**Section 2 — Platform Class:**
```
grep -rn "class.*Platform.*:" <backend>/
grep -rn "def check_and_update_config" <backend>/
grep -rn "worker_cls\s*=" <backend>/     # confirm worker_cls is set inside check_and_update_config
grep -rn "def get_attn_backend_cls" <backend>/
grep -rn "def get_device_communicator_cls" <backend>/
grep -rn "def supports_v1\|def is_async_output_supported" <backend>/
grep -rn "def verify_dtype_with_device\|def check_if_supports_dtype\|def get_punica_wrapper" <backend>/
```
Verify `check_and_update_config` actually sets `worker_cls` — this is the single most
critical call that wires vLLM's initialization to the backend's worker. If this is
missing or incomplete, nothing else runs.

**API version notes for Section 2**: `supports_v1()` and `is_async_output_supported()` are removed in V1 vLLM — if not found, mark rows 2.4.1 and 2.4.2 as N/A rather than 0. `verify_dtype_with_device()` was renamed to `check_if_supports_dtype()` — the probe above catches both names.

**Section 3 — Worker Core:**
```
# Worker
grep -rn "class.*Worker.*WorkerBase\|WorkerBase" <backend>/
grep -rn "def init_device" <backend>/
grep -rn "def load_model" <backend>/
grep -rn "def get_kv_cache_spec" <backend>/
grep -rn "def initialize_cache" <backend>/
grep -rn "def execute_model" <backend>/
grep -rn "def compile_or_warm_up_model" <backend>/
grep -rn "def check_health" <backend>/

# ModelRunner (Section 3.2)
grep -rn "class.*ModelRunner\|ModelRunnerBase" <backend>/
grep -rn "def profile_run\|determine_num_available_blocks" <backend>/
grep -rn "def execute_model" <backend>/  # in ModelRunner, not just Worker
grep -rn "lora_state\|lora_config" <backend>/
```
For each method: check that the body is a real implementation, not just
`raise NotImplementedError`. Flag any that only delegate to `super()` without
adding device-specific logic.

**Section 4 — Attention Backend:**
```
grep -rn "class.*AttentionBackend\|AttentionBackend" <backend>/
grep -rn "def get_name\|def get_impl_cls\|def get_metadata_cls\|def get_builder_cls" <backend>/
grep -rn "def get_kv_cache_shape" <backend>/
grep -rn "get_supported_kernel_block_sizes\|get_preferred_block_size" <backend>/
grep -rn "supported_dtypes\s*=" <backend>/
grep -rn "supported_kv_cache_dtypes\s*=" <backend>/
grep -rn "class.*AttentionImpl\b" <backend>/
grep -rn "def forward" <backend>/        # in AttentionImpl subclass
grep -rn "forward_includes_kv_cache_update" <backend>/
```

**Section 5 — KV Cache:**
```
grep -rn "class.*KVCacheSpec\|KVCacheSpec" <backend>/
grep -rn "page_size_bytes\|block_size" <backend>/
grep -rn "FullAttentionSpec\|MLAAttentionSpec\|SlidingWindowSpec" <backend>/
grep -rn "KVCacheSpecRegistry.register\|@register_kv_cache_spec" <backend>/
grep -rn "def copy_blocks" <backend>/
```

**Section 6 — Basic Inference Correctness:**

*Section 6.1 — Single-GPU Inference:*
```
grep -rn "vllm.LLM\|llm.generate\|LLM(" <backend>/tests/
grep -rn "def test_" <backend>/tests/    # look for e2e inference tests
```

*Section 6.2 — Sampling Parameters:*
```
grep -rn "SamplingParams\|sampling_params" <backend>/tests/
grep -rn "temperature\|top_p\|top_k\|stop_token\|logprobs" <backend>/tests/
grep -rn "greedy\|temperature=0" <backend>/tests/
```

*Section 6.3 — Smoke-Test Models:*
```
grep -rn "Llama\|llama" <backend>/tests/
grep -rn "Mistral\|mistral" <backend>/tests/
```

**Section 7 — Distributed Communicator:**
```
grep -rn "class.*DeviceCommunicatorBase\|CommunicatorBase" <backend>/
grep -rn "def all_reduce\|def all_gather\|def reduce_scatter" <backend>/
grep -rn "def send\|def recv\|def barrier" <backend>/
grep -rn "hccl\|xccl\|nccl\|gloo" <backend>/   # what collective lib is used
```

**Section 8 — torch.compile & Graph Capture:**
```
grep -rn "torch.compile\|compilation_config\|inductor" <backend>/
grep -rn "cudagraph\|cuda_graph\|graph_capture\|GraphCapture" <backend>/
grep -rn "enforce_eager\|eager_mode" <backend>/
grep -rn "set_torch_compile\|CompilationConfig" <backend>/
grep -rn "FakeTensor\|meta.*tensor\|torch._subclasses" <backend>/
```

**Section 9 — Quantization:**
```
grep -rn "fp8\|float8\|FP8\|Float8" <backend>/
grep -rn "fp4\|float4\|FP4\|Float4\|nvfp4" <backend>/
grep -rn "awq\|AWQ\|gptq\|GPTQ\|int8\|INT8\|bnb\|bitsandbytes\|hqq\|HQQ" <backend>/
grep -rn "linear_method\|QuantizationConfig\|quant_config" <backend>/
```

**Section 10 — Chunked Prefill:**
```
grep -rn "chunked_prefill\|enable_chunked_prefill" <backend>/
grep -rn "max_num_batched_tokens\|scheduler_config" <backend>/
```
Specifically check `check_and_update_config`: if it explicitly disables chunked prefill
that is a gap; if it leaves it enabled (V1 default) that is correct behavior.

**Section 11 — Prefix Caching:**
```
grep -rn "prefix_cach\|enable_prefix_caching" <backend>/
grep -rn "block_hash\|hash_token\|evictor\|LRU" <backend>/
grep -rn "copy_blocks" <backend>/        # reused from Section 5 probe
```

**Section 12 — Tensor Parallelism:**
```
grep -rn "tensor_parallel\|tp_size\|all_reduce" <backend>/
grep -rn "VocabParallelEmbedding\|ColumnParallelLinear\|RowParallelLinear" <backend>/
grep -rn "parallel_config\|world_size" <backend>/
```

**Section 13 — Model & Dtype Coverage:**

*Architecture support (Section 13.1):*
```
grep -rn "Llama\|Mistral\|Qwen\|Gemma\|Falcon\|DeepSeek\|Mixtral" <backend>/tests/
grep -rn "MLA\|MoE\|sliding_window" <backend>/tests/
```

*Model loading (Section 13.2):*
```
grep -rn "safetensors\|from_pretrained\|load_weights" <backend>/
grep -rn "float16\|bfloat16\|float32" <backend>/           # dtype loading
```

*Dtype matrix (Section 13.3):*
```
grep -rn "supported_dtypes\s*=" <backend>/
grep -rn "supported_kv_cache_dtypes\s*=" <backend>/
grep -rn "float16\|bfloat16\|float32\|float8\|int8" <backend>/
grep -rn "autocast\|amp\|AMP" <backend>/
```

**Section 14 — End-to-End Serving Validation:**
```
grep -rn "AsyncLLMEngine\|async_engine\|LLMEngine" <backend>/tests/
grep -rn "concurrent\|batch.*request\|multi.*request" <backend>/tests/
grep -rn "stream\|streaming" <backend>/tests/
grep -rn "vllm serve\|llm.generate" <backend>/tests/
grep -rn "stop.*token\|stop_token_ids\|early.stop" <backend>/tests/
```
Check if the backend has any README or docs section covering `vllm serve` usage.
Check for known issues with async engine compatibility in the backend's issue tracker.

**Section 15 — LoRA:**
```
grep -rn "lora\|LoRA\|get_punica_wrapper\|punica" <backend>/
grep -rn "LoRAManager\|LoRARequest\|lora_config" <backend>/
```

**Section 16 — Speculative Decoding:**
```
grep -rn "speculative\|spec_decode\|draft_model\|eagle\|medusa" <backend>/
grep -rn "speculative_config\|num_speculative_tokens" <backend>/
```

**Section 17 — Pipeline Parallelism:**
```
grep -rn "pipeline_parallel\|pp_size\|PPHandler" <backend>/
grep -rn "is_first_pp_rank\|is_last_pp_rank" <backend>/
grep -rn "recv_tensor\|send_tensor\|pp_tensors" <backend>/
```

**Section 18 — FP8/FP4 KV Cache:**
```
grep -rn "fp8_kv\|kv_cache_dtype.*fp8\|KVQuantMode\|kv_quant" <backend>/
grep -rn "fp4_kv\|kv_cache_dtype.*fp4" <backend>/
grep -rn "get_kv_quant_mode\|kv_scales\|kv_quant_mode" <backend>/
```

**Section 19 — Multimodal:**
```
grep -rn "multimodal\|MultiModal\|vision\|image_input\|audio_input" <backend>/
grep -rn "LlavaConfig\|Qwen2VLConfig\|InternVLConfig" <backend>/tests/
```

**Section 20 — Async Output:**
```
grep -rn "is_async_output_supported\|async_output" <backend>/
grep -rn "non_blocking\|output_proc_callback" <backend>/
```

**Section 21 — Observability & Ecosystem:**
```
grep -rn "prometheus\|metrics\|stat_logger\|StatLogger" <backend>/
grep -rn "vllm.stat_logger_plugins" setup.py pyproject.toml
grep -rn "opentelemetry\|tracing\|span" <backend>/
```
Check the backend README for documented ecosystem compatibility:
HuggingFace `transformers`, `lm-evaluation-harness`, container/Docker usage.

---

### Phase 2: Dynamic Feature Discovery

1. **Scan Platform class for undocumented methods** not in any checklist section:
   ```
   grep -rn "def " <backend>/platform.py | grep -v "__"
   ```

2. **Check for custom ops** registered via `vllm.general_plugins`:
   ```
   grep -rn "vllm.general_plugins" setup.py pyproject.toml
   grep -rn "torch.ops.vllm\|torch.library.impl" <backend>/
   ```

3. **Check for custom model architectures** registered for this backend:
   ```
   grep -rn "@ModelRegistry\|register_model\|ModelRegistry.register" <backend>/
   ```

4. **Scan tests** for features tested but not covered by any checklist section:
   ```
   grep -rn "def test_" <backend>/tests/ | sort
   ```

---

### Phase 2.5: Upstream Candidate Discovery

Non-scored advisory section. Find features the backend built that are generic
enough to benefit all vLLM hardware plugins if upstreamed to vLLM core.

**Systematic diff approach:**

1. **Inventory backend surface beyond the required plugin contracts** — list all Python
   modules and classes not directly implementing a Section 1–21 checklist item. Focus on:
   - Utility modules (memory diagnostics, device introspection, topology)
   - Patches applied to `vllm.*` or `torch.*` internals
   - Novel scheduling optimizations or batching strategies
   - Custom sampling or logits processing
   - Cache management innovations beyond the standard block allocator

2. **Grep for infrastructure patterns**:
   ```
   grep -rn "monkey_patch\|patch_vllm\|_original_" <backend>/ --include="*.py"
   grep -rn "class.*Wrapper\|class.*Shim\|class.*Bridge" <backend>/ --include="*.py"
   grep -rn "explain\|diagnose\|profil" <backend>/ --include="*.py"
   grep -rn "topology\|device_summary\|physical_device" <backend>/ --include="*.py"
   grep -rn "memory_efficient\|cache_evict\|prefetch" <backend>/ --include="*.py"
   grep -rn "def.*scheduler\|custom_scheduler" <backend>/ --include="*.py"
   ```

3. **Classify each candidate** using Generic / Needs Abstraction / Hardware-Specific.

| Category | What to look for |
|----------|-----------------|
| 24.1 Memory Diagnostics & Utilities | `memory_summary()`, cache fragmentation tools, OOM analysis helpers |
| 24.2 Custom Op Registration Patterns | Helpers that simplify `vllm.general_plugins` registration for any backend |
| 24.3 Scheduling Innovations | Prefill/decode disaggregation, batching heuristics, adaptive chunking |
| 24.4 Attention Pattern Extensions | New `KVCacheSpec` types, novel attention variants not in vLLM core |
| 24.5 Distributed Topology Utilities | Multi-node interconnect probing, topology-aware TP grouping |
| 24.6 Profiling & Tracing Hooks | Kernel-level timing, vLLM-aware profiler extensions |
| 24.7 Model Loading Optimizations | Parallel weight loading, tensor parallel checkpoint sharding |

---

### Phase 3: Scoring

```
Row weight: w_i = 1 / priority_i
Section Pct = sum(score_i * w_i) / sum(max_i * w_i) * 100  (excluding N/A rows, max_i=2)
weight_r = 1 / level
Readiness = (sum(Section Pct * weight_r) / sum(weight_r)) * 100
```

Fill the score table (sorted by level) at the top of the document, the overall
readiness percentage, and the executive summary.
