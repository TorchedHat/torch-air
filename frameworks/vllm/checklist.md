# vLLM Hardware Backend Integration Readiness Checklist

> A comprehensive template for evaluating hardware accelerator integration
> readiness with vLLM. Supports three integration modes:
>
> - **OOT Plugin** -- out-of-tree pip package; the standard path for new backends
> - **vLLM Fork** -- vendor-maintained fork of the vLLM source tree
> - **Hybrid** -- accelerator ships both an OOT plugin and a vLLM fork
>
> **How to use this template:**
> - **OOT only**: fill `Plugin Pts` in Section 1; set `Fork Pts` = N/A for `[OOT]` rows.
> - **Fork only**: fill `Fork Pts` in Section 1; set `Plugin Pts` = N/A for `[OOT]` rows.
> - **Hybrid**: fill **both** `Plugin Pts` and `Fork Pts` columns throughout Section 1;
>   compute two separate Section 1 scores and two overall readiness percentages.
>   Sections 2–21 are path-agnostic -- score once, shared by both paths.
>
> Items marked **[OOT]** apply only to out-of-tree plugin backends.
> Items marked **[Fork]** apply only to vLLM-fork-based backends.
> Unmarked / **[Both]** items apply to **both** paths.
>
> Items marked **[Runtime]** in Notes cannot be verified by source analysis alone
> and require actually running the backend.

| Field | Value |
|-------|-------|
| **Backend** | _[FILL: backend name]_ |
| **Backend version** | _[FILL: release version/tag, e.g. v0.3.0]_ |
| **Integration path** | _[FILL: OOT Plugin / vLLM Fork / Hybrid]_ |
| **Plugin package** | _[FILL: pip package name, e.g. vllm-ascend — OOT/Hybrid only; omit for Fork]_ |
| **Plugin source** | _[FILL: plugin repo URL — OOT/Hybrid only; omit for Fork]_ |
| **Fork source** | _[FILL: fork repo URL — Fork/Hybrid only; omit for OOT]_ |
| **vLLM base version (Plugin)** | _[FILL: vLLM version targeted by the plugin — OOT/Hybrid only; omit for Fork]_ |
| **vLLM base version (Fork)** | _[FILL: vLLM fork base commit or version — Fork/Hybrid only; omit for OOT]_ |
| **PyTorch required by vLLM** | _[FILL: standard `torch==X.Y.Z` from `/tmp/vllm/pyproject.toml` — e.g. `torch==2.13.0`. Always from upstream vLLM source, never from the plugin's own dependencies.]_ |
| **Evaluation date** | _[FILL: date]_ |
| **Evaluator** | _[FILL: human or agent]_ |
| **torch-air version** | _[FILL: release tag + git commit, e.g. v0.1.0 / abc1234; if no release tag exists yet, commit hash only]_ |
| **Model version** | _[FILL: AI model used to generate this report, e.g. claude-sonnet-4-5]_ |

---

## Readiness Score & Summary

### Executive Summary

_[FILL: Write a concise summary covering:
- **Overall Readiness**: X%
- **Notable insights**:
  - Surprising finding or important context 1
  - Surprising finding or important context 2
- **Key strengths**:
  - Top strength 1
  - Top strength 2
  - Top strength 3
- **Key gaps**:
  - Critical gap 1
  - Critical gap 2
  - Critical gap 3
- **Upstream candidates**: N features identified as generic enough for vLLM core (see below)]_

### Section Scores

> **OOT or Fork only**: fill `Plugin Earned / Plugin %` or `Fork Earned / Fork %` respectively; leave the other pair blank.
> **Hybrid**: fill both column pairs. Sections 2–21 share the same earned/percentage for both paths.

| Section | Max Pts | Plugin Earned | Plugin % | Fork Earned | Fork % |
|---------|---------|---------------|----------|-------------|--------|
| **Level 1** | | | | | |
| Plugin Registration & Entry Points | | | | | |
| Platform Class | | | | | |
| Worker Core | | | | | |
| Attention Backend | | | | | |
| KV Cache | | | | | |
| Basic Inference Correctness | | | | | |
| **Level 2** | | | | | |
| Distributed Communicator | | | | | |
| torch.compile & Graph Capture | | | | | |
| Quantization | | | | | |
| Chunked Prefill & Continuous Batching | | | | | |
| Prefix Caching | | | | | |
| Tensor Parallelism | | | | | |
| Model & Dtype Coverage | | | | | |
| End-to-End Serving Validation | | | | | |
| **Level 3** | | | | | |
| LoRA / Multi-LoRA | | | | | |
| Speculative Decoding | | | | | |
| Pipeline Parallelism | | | | | |
| FP8/FP4 KV Cache | | | | | |
| Multimodal Inputs | | | | | |
| Async Output & Non-blocking I/O | | | | | |
| Observability & Ecosystem | | | | | |

**Plugin Readiness**: _____ %  _(OOT Plugin path; mark N/A for Fork-only backends)_

**Fork Readiness**: _____ %  _(vLLM Fork path; mark N/A for OOT-only backends)_

### Upstream Candidates (Advisory)

> Features discovered in this backend that are generic enough to benefit
> all vLLM hardware plugin backends if upstreamed to vLLM core.
> Advisory only -- does not affect the readiness score.

**Classification key**:
- **Generic** -- no backend-specific references; ready to upstream into vLLM core as-is
- **Needs Abstraction** -- solves a problem every backend faces, but implementation references backend-specific internals; upstream path is to define an interface/hook in vLLM core

_[FILL: For each discovered feature, use this format:]_

#### _[Feature Name]_

_[Brief description of what the feature does]_

**Classification**: _[Generic | Needs Abstraction | Hardware-Specific]_

**Relevant files**: _[source files and key symbols]_

**Current state in vLLM**: _[whether vLLM core has an equivalent, partial equivalent, or nothing]_

**Motivation**: _[why this would benefit all backends if upstreamed]_

---

## 0. Plugin Source Discovery & Integration Path Detection

| Source Discovery | Plugin Value | Fork Value | Notes |
|-----------------|-------------|------------|-------|
| **Source location** | _[plugin local path or URL]_ | _[fork local path or URL]_ | |
| **How found** | _[local / pip / GitHub / user-provided]_ | _[local / GitHub / user-provided]_ | |
| **Repository URL** | _[plugin repo URL — OOT/Hybrid; N/A for Fork]_ | _[fork repo URL — Fork/Hybrid; N/A for OOT]_ | |
| **Backend version** | _[latest plugin release tag or pip version]_ | _[fork tag or commit]_ | |
| **vLLM version** | _[vLLM version the plugin targets]_ | _[vLLM version the fork is based on]_ | |

| Signal | Detected | Value | Notes |
|--------|----------|-------|-------|
| `vllm.platform_plugins` entry point found | `[ ]` | | OOT signal |
| `vllm.general_plugins` entry point found | `[ ]` | | OOT signal |
| `Platform` subclass implemented | `[ ]` | | |
| `WorkerBase` subclass implemented | `[ ]` | | |
| Backend-specific `ModelRunner` class found | `[ ]` | | |
| `AttentionBackend` subclass implemented | `[ ]` | | |
| `DeviceCommunicatorBase` subclass implemented | `[ ]` | | |
| Backend is a vLLM fork | `[ ]` | | Fork signal |
| **Detected path** | | _[OOT Plugin / vLLM Fork / Hybrid]_ | |

**Path detection rules**:
- OOT signals found, no fork signals → **OOT Plugin**: score `Plugin Pts` only; `Fork Pts` = N/A throughout.
- Fork signals found, no OOT signals → **vLLM Fork**: score `Fork Pts` only; `Plugin Pts` = N/A throughout.
- Both OOT and Fork signals found → **Hybrid**: score both `Plugin Pts` and `Fork Pts` in Section 1; Sections 2–21 scored once and shared by both readiness calculations.

---

## 1. Plugin Registration & Entry Points -- Level: **1**

> This is the only section with path-specific rows. `[OOT]` rows are scored under `Plugin Pts`
> and are N/A under `Fork Pts`. `[Both]` rows are scored under both columns.
> For OOT-only or Fork-only backends, leave the unused column blank (treat as N/A).

### 1.1 Package Entry Points

| # | Item | Path | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|------|----------|------------|----------|-------|
| 1.1.1 | `vllm.platform_plugins` entry point registered in `setup.py`/`pyproject.toml` | OOT | 1 | | N/A | |
| 1.1.2 | `register()` function returns platform class FQN when hardware is present | OOT | 1 | | N/A | |
| 1.1.3 | `register()` returns `None` gracefully when hardware is absent (no crash) | OOT | 1 | | N/A | |
| 1.1.4 | Plugin is installable as a standalone pip package | OOT | 1 | | N/A | |
| 1.1.5 | `vllm.general_plugins` entry point registered (for custom ops/kernels) | OOT | 2 | | N/A | |
| 1.1.6 | Backend loads correctly in every vLLM worker process (main, tensor-parallel workers) | Both | 1 | | | |
| 1.1.7 | Backend does not crash when a different platform is selected at runtime | Both | 1 | | | |

### 1.2 vLLM Version Compatibility

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|------------|----------|-------|
| 1.2.1 | Compatible with vLLM V1 engine (not just V0) | 1 | | | |
| 1.2.2 | Minimum supported vLLM version documented and enforced in package metadata | 2 | | | OOT: enforced in pip deps; Fork: implicit in fork base commit |

---

## 2. Platform Class -- Level: **1**

### 2.1 Required Properties

| # | Property/Method | Priority | Plugin Pts | Fork Pts | Notes |
|---|-----------------|----------|--------|----------|-------|
| 2.1.1 | `Platform` subclass inherits from `vllm.platforms.interface.Platform` | 1 | | | |
| 2.1.2 | `_enum` property returns `PlatformEnum.OOT` (or correct in-tree enum) | 1 | | | |
| 2.1.3 | `device_type` property returns the PyTorch device type string (e.g. `"npu"`, `"xpu"`) | 1 | | | |
| 2.1.4 | `device_name` property returns a human-readable device name | 2 | | | |

### 2.2 check_and_update_config (Most Critical Method)

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 2.2.1 | `check_and_update_config(vllm_config)` is implemented | 1 | | | |
| 2.2.2 | Sets `worker_cls` to the backend's `WorkerBase` subclass | 1 | | | |
| 2.2.3 | Validates or adjusts `block_size` for the hardware's attention kernel | 1 | | | |
| 2.2.4 | Validates dtype compatibility and raises a clear error for unsupported dtypes (no silent misuse) | 1 | | | |
| 2.2.5 | Sets or adjusts graph capture mode (`enforce_eager`, `compilation_config`) explicitly -- no silent fallback | 2 | | | |
| 2.2.6 | Adjusts `max_num_batched_tokens` or other scheduler params if needed | 2 | | | |
| 2.2.7 | Raises `ValueError` with a clear message for unsupported config combinations | 2 | | | |

### 2.3 Attention & Communicator Dispatch

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 2.3.1 | `get_attn_backend_cls(selected_backend, attn_selector_config, num_heads=None)` returns FQN of `AttentionBackend` subclass | 1 | | | |
| 2.3.2 | `get_device_communicator_cls()` returns FQN of communicator (or `None` to use PyTorch distributed default) | 2 | | | |

### 2.4 Optional Platform Hooks

| # | Hook | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 2.4.1 | `supports_v1(model_config)` returns correct bool -- if False, vLLM V1 will not use this backend | 2 | | | **Removed in V1 vLLM.** Score N/A for any backend targeting V1-only vLLM; score normally only if the backend must still support V0 fallback |
| 2.4.2 | `is_async_output_supported(enforce_eager)` returns correct bool | 2 | | | **Removed in V1 vLLM.** Score N/A; do not double-count with Section 20.1 |
| 2.4.3 | `check_if_supports_dtype(dtype)` provides per-call dtype guard beyond the config-level check | 2 | | | Renamed from `verify_dtype_with_device` |
| 2.4.4 | `get_punica_wrapper()` returns device-specific Punica wrapper (required for LoRA on device) | 3 | | | |
| 2.4.5 | `get_device_capability()` returns device capability info | 3 | | | |

---

## 3. Worker Core -- Level: **1**

### 3.1 Required Methods

| # | Method | Priority | Plugin Pts | Fork Pts | Notes |
|---|--------|----------|--------|----------|-------|
| 3.1.1 | `WorkerBase` subclass defined and importable | 1 | | | |
| 3.1.2 | `init_device()` initializes device (set device index, init distributed backend, allocate memory) | 1 | | | |
| 3.1.3 | `load_model()` loads model weights to the device | 1 | | | |
| 3.1.4 | `get_kv_cache_spec()` returns `dict[str, KVCacheSpec]` describing each layer's KV cache format | 1 | | | |
| 3.1.5 | `initialize_cache(kv_cache_config)` allocates KV cache tensors on device | 1 | | | |
| 3.1.6 | `execute_model(scheduler_output)` runs a forward pass and returns sampler output | 1 | | | |
| 3.1.7 | `compile_or_warm_up_model()` runs warmup / graph capture before first request | 2 | | | |
| 3.1.8 | `check_health()` performs a device liveness check (meaningful override, not base no-op) | 2 | | | |

### 3.2 ModelRunner

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 3.2.1 | Backend-specific `ModelRunner` class defined and wired to the Worker | 1 | | | `ModelRunnerBase` does not exist in vLLM; implement the same interface as `GPUModelRunner` |
| 3.2.2 | `execute_model(model_input, ...)` runs forward pass via ModelRunner | 1 | | | |
| 3.2.3 | `profile_run()` / `determine_num_available_blocks()` returns accurate KV block count (wrong value causes OOM or memory waste) | 1 | | | |
| 3.2.4 | ModelRunner correctly handles LoRA state when `lora_config` is set | 3 | | | |

### 3.3 Worker Initialization

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 3.3.1 | Distributed process group initialized correctly for TP > 1 | 2 | | | |
| 3.3.2 | `gpu_memory_utilization` fraction is correctly respected during memory sizing | 2 | | | |
| 3.3.3 | Worker process exits cleanly with a clear error on initialization failure (does not hang) | 2 | | | |
| 3.3.4 | Worker recovers from a single failed request without crashing the server process | 2 | | | |

---

## 4. Attention Backend -- Level: **1**

### 4.1 Backend Class Contract

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 4.1.1 | `AttentionBackend` subclass inherits from `vllm.v1.attention.backend.AttentionBackend` | 1 | | | |
| 4.1.2 | `get_name()` returns a unique backend name string | 1 | | | |
| 4.1.3 | `get_impl_cls()` returns `AttentionImpl` subclass | 1 | | | |
| 4.1.4 | `get_metadata_cls()` returns `AttentionMetadata` subclass | 1 | | | Not defined in base `AttentionBackend`; present only in specific MLA backends. Score N/A unless implementing an MLA backend |
| 4.1.5 | `get_builder_cls()` returns `AttentionMetadataBuilder` subclass | 1 | | | |
| 4.1.6 | `get_kv_cache_shape(...)` returns correct tensor shape | 1 | | | Not present in V1 `AttentionBackend`; KV cache shape is determined by `KVCacheSpec` (Section 5). Score N/A for V1 backends |
| 4.1.7 | `supported_dtypes` class variable lists supported compute dtypes | 1 | | | |
| 4.1.8 | `get_supported_kernel_block_sizes()` returns valid block size constraints | 2 | | | |
| 4.1.9 | `supported_kv_cache_dtypes` class variable lists supported KV cache dtypes | 2 | | | |

### 4.2 AttentionImpl Contract

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 4.2.1 | `AttentionImpl` subclass implements `__init__(num_heads, head_size, scale, num_kv_heads, ...)` | 1 | | | |
| 4.2.2 | `forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)` computes attention and updates KV cache | 1 | | | |
| 4.2.3 | `forward_includes_kv_cache_update` flag is set correctly -- if wrong, KV cache is silently corrupted | 1 | | | |
| 4.2.4 | Handles both prefill and decode phases in a single `forward` call | 1 | | | |
| 4.2.5 | Handles variable-length sequences in a mixed prefill+decode batch correctly | 2 | | | |
| 4.2.6 | Handles `sliding_window` attention when requested by model config | 2 | | | |
| 4.2.7 | `fused_output_quant_supported()` returns correct bool for quantized output paths | 3 | | | |
| 4.2.8 | `do_rope_and_kv_cache_update()` implemented if hardware supports fused RoPE+KV write | 3 | | | |

### 4.3 Attention Correctness

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 4.3.1 | Attention output matches reference (FlashAttention / naive SDPA) within tolerance on a single sequence | 1 | | | |
| 4.3.2 | Attention output matches reference for a batched mixed prefill+decode pass | 1 | | | |
| 4.3.3 | Causal masking is correct (no future-token leakage) | 1 | | | |
| 4.3.4 | GQA / MQA (grouped / multi-query attention) handled correctly | 2 | | | |

---

## 5. KV Cache -- Level: **1**

### 5.1 KVCacheSpec Contract

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 5.1.1 | `KVCacheSpec` subclass (or standard `FullAttentionSpec`) returned by `get_kv_cache_spec()` | 1 | | | |
| 5.1.2 | `block_size` matches the attention kernel's supported block sizes | 1 | | | |
| 5.1.3 | `page_size_bytes` property returns correct byte count per block | 1 | | | |
| 5.1.4 | `max_memory_usage_bytes(vllm_config)` returns an accurate upper bound (used for block count sizing) | 1 | | | |

### 5.2 KV Cache Allocation & Management

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 5.2.1 | KV cache tensors allocated on device with correct shape and dtype | 1 | | | |
| 5.2.2 | Block allocator correctly tracks free / occupied blocks | 1 | | | |
| 5.2.3 | KV cache does not overflow on max-sequence-length inputs | 1 | | | |
| 5.2.4 | Memory pressure causes a clear error (no silent incorrect outputs) | 2 | | | |
| 5.2.5 | `copy_blocks()` / `swap_blocks()` kernel copies KV blocks correctly for CPU↔GPU swapping | 2 | | | In vLLM V1, prefix-cache reuse is done via block-table re-mapping, not a copy kernel |
| 5.2.6 | OOM during active serving (mid-batch allocation failure) causes request rejection, not server crash | 2 | | | |

---

## 6. Basic Inference Correctness -- Level: **1**

### 6.1 Single-GPU Inference

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 6.1.1 | `vllm.LLM(model=..., device=<backend>)` initializes without error | 1 | | | |
| 6.1.2 | `llm.generate(prompts, ...)` returns non-empty outputs | 1 | | | |
| 6.1.3 | Text output is coherent (not random tokens or garbage) | 1 | | | |
| 6.1.4 | Output tokens match CPU/CUDA reference within numerical tolerance | 1 | | | |
| 6.1.5 | Greedy decoding is deterministic (identical output across two identical runs) | 2 | | | |
| 6.1.6 | Batch inference (multiple prompts in one call) produces correct per-prompt outputs | 2 | | | |

### 6.2 Sampling Parameters

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 6.2.1 | Greedy decoding (`temperature=0`) produces expected argmax output | 1 | | | |
| 6.2.2 | Temperature sampling (`temperature > 0`) produces plausible token distributions | 2 | | | |
| 6.2.3 | `top_p` nucleus sampling correctly limits the cumulative probability mass | 2 | | | |
| 6.2.4 | `top_k` sampling correctly limits to top-k logits | 2 | | | |
| 6.2.5 | `max_tokens` truncation works correctly | 2 | | | |
| 6.2.6 | `stop` tokens / `stop_token_ids` cause generation to halt at the right point | 2 | | | |
| 6.2.7 | `logprobs` output is populated and numerically plausible | 3 | | | |

### 6.3 Smoke-Test Models

> Two representative models to confirm end-to-end inference works before running the full model matrix in Section 13.

| Model | Architecture | Priority | Plugin Pts | Fork Pts | Notes |
|-------|-------------|----------|--------|----------|-------|
| Llama-3-8B | Dense decoder (GQA) | 1 | | | |
| Mistral-7B-v0.1 | Dense decoder + sliding window | 2 | | | |

---

## 7. Distributed Communicator -- Level: **2**

### 7.1 DeviceCommunicatorBase Contract

| # | Method | Priority | Plugin Pts | Fork Pts | Notes |
|---|--------|----------|--------|----------|-------|
| 7.1.1 | `DeviceCommunicatorBase` subclass defined (or confirmed default PyTorch distributed is sufficient) | 1 | | | |
| 7.1.2 | `all_reduce(tensor)` performs correct reduction | 1 | | | |
| 7.1.3 | `all_gather(tensor, dim)` gathers correctly across TP ranks | 1 | | | |
| 7.1.4 | `reduce_scatter(tensor)` reduces and scatters correctly | 2 | | | |
| 7.1.5 | `send(tensor, dst)` / `recv(size, dtype, src)` for pipeline-parallel P2P | 2 | | | |
| 7.1.6 | `barrier()` synchronizes all ranks | 2 | | | |
| 7.1.7 | `all_gatherv(tensor, sizes)` handles variable-size tensors | 3 | | | |

### 7.2 Collective Backend

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 7.2.1 | Custom collective library used (e.g. HCCL, XCCL) rather than Gloo CPU fallback | 2 | | | |
| 7.2.2 | Multi-node collective operations work (2+ hosts) | 3 | | | |
| 7.2.3 | `all_reduce` latency measured and substantially better than Gloo baseline | 3 | | | [Runtime] |

---

## 8. torch.compile & Graph Capture -- Level: **2**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 8.1 | `torch.compile(model)` does not error on the backend device | 1 | | | |
| 8.2 | Compiled model produces outputs matching eager mode | 1 | | | |
| 8.3 | FakeTensor / meta tensor dispatch works correctly for shape inference | 1 | | | |
| 8.4 | Backend explicitly declares compile mode (`enforce_eager` or compilation config) in `check_and_update_config` -- no silent fallback | 2 | | | |
| 8.5 | Graph capture (CUDA-graph equivalent) implemented for decode phase | 2 | | | |
| 8.6 | `compile_or_warm_up_model()` runs sufficient warmup passes to pre-populate graph cache | 2 | | | |
| 8.7 | Graph capture reduces per-step CPU overhead vs eager | 2 | | | [Runtime] |
| 8.8 | `set_torch_compile()` or equivalent called in general plugin to configure Inductor for device | 3 | | | |

---

## 9. Quantization -- Level: **2**

### 9.1 Weight Quantization Support

| Format | Priority | Plugin Pts | Plugin Correct | Fork Pts | Fork Correct | Notes |
|--------|----------|---------  |----------------|----------|--------------|-------|
| FP8 (W8A8) | 2 | | `[ ]` | | `[ ]` | |
| INT8 (W8A8 / W8A16) | 2 | | `[ ]` | | `[ ]` | |
| AWQ | 2 | | `[ ]` | | `[ ]` | |
| GPTQ | 2 | | `[ ]` | | `[ ]` | |
| FP4 / vendor native 4-bit float (if supported) | 3 | | `[ ]` | | `[ ]` | |
| BitsAndBytes (NF4) | 3 | | `[ ]` | | `[ ]` | Supported via the `vllm_bnb_plugin` OOT package, not in-tree |
| HQQ | 3 | | `[ ]` | | `[ ]` | May be implemented via a separate plugin, not in-tree |

### 9.2 Quantization Infrastructure

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 9.2.1 | `--quantization <format>` flag accepted without crashing for supported formats | 2 | | | |
| 9.2.2 | Quantized model accuracy within acceptable degradation vs fp16 baseline | 2 | | | [Runtime] |
| 9.2.3 | Custom quantized linear kernel registered via `vllm.general_plugins` | 2 | | | |
| 9.2.4 | `linear_method` / `QuantizationConfig` recognized and dispatched correctly | 2 | | | |

---

## 10. Chunked Prefill & Continuous Batching -- Level: **2**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 10.1 | Chunked prefill enabled -- not disabled in `check_and_update_config` (V1 default) | 1 | | | |
| 10.2 | Mixed prefill+decode batch (both in same forward pass) produces correct outputs | 1 | | | |
| 10.3 | Attention metadata correctly distinguishes prefill vs decode tokens within a batch | 1 | | | |
| 10.4 | `max_num_batched_tokens` correctly set or discoverable by the scheduler | 2 | | | |
| 10.5 | Long-context requests (> 8k tokens) processed without OOM | 2 | | | |
| 10.6 | Request preemption does not cause incorrect output on the resumed request | 2 | | | |
| 10.7 | Throughput under continuous batching exceeds naive single-request serving | 2 | | | [Runtime] |

---

## 11. Prefix Caching -- Level: **2**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 11.1 | `--enable-prefix-caching` flag accepted without crashing | 1 | | | |
| 11.2 | KV blocks for a shared prefix are correctly reused across requests | 1 | | | |
| 11.3 | Prefix KV block data correctly reused without re-computation | 2 | | | In vLLM V1, prefix reuse is via block-table re-mapping, not `copy_blocks` |
| 11.4 | Block eviction (LRU or equivalent) frees blocks correctly under memory pressure | 2 | | | |
| 11.5 | TTFT reduction observed on repeated-prefix workloads vs no prefix caching | 2 | | | [Runtime] |
| 11.6 | Prefix cache hit rate reported in metrics | 3 | | | |

---

## 12. Tensor Parallelism -- Level: **2**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 12.1 | `--tensor-parallel-size 2` initializes without error | 1 | | | |
| 12.2 | TP=2 inference produces outputs matching TP=1 (within tolerance) | 1 | | | |
| 12.3 | All-reduce at layer boundaries produces correct results | 1 | | | |
| 12.4 | `VocabParallelEmbedding` / `ColumnParallelLinear` / `RowParallelLinear` execute correctly on device | 2 | | | |
| 12.5 | TP=4 inference works (4 devices) | 2 | | | |
| 12.6 | TP=8 inference works (8 devices) | 3 | | | |
| 12.7 | Throughput scales meaningfully from TP=1 to TP=2 | 2 | | | [Runtime] |

---

## 13. Model & Dtype Coverage -- Level: **2**

### 13.1 Architecture Support

| Architecture | Example Models | Priority | Plugin Pts | Fork Pts | Notes |
|-------------|----------------|----------|--------|----------|-------|
| Dense decoder (MHA) | Llama-2, Falcon | 1 | | | |
| Dense decoder (GQA) | Llama-3, Gemma-2, Qwen2.5 | 1 | | | |
| Dense decoder (MQA) | Falcon-7B | 2 | | | |
| Sliding window attention | Mistral-7B, Qwen2 | 2 | | | |
| MLA attention | DeepSeek-V2, DeepSeek-V3 | 1 | | | |
| MoE (decoder) | Mixtral-8x7B, DeepSeek-V3 | 2 | | | |
| Encoder-decoder | T5, BART | 3 | | | |
| Embedding models | E5-large, BGE | 3 | | | |
| Vision-language models | LLaVA-1.5, Qwen2-VL | 3 | | | |

### 13.2 Model Loading

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 13.2.1 | HuggingFace model weights load correctly (`from_pretrained` path) | 1 | | | |
| 13.2.2 | safetensors format supported | 2 | | | |
| 13.2.3 | Weights load with correct dtype (fp16, bf16, fp32 as requested) | 2 | | | |
| 13.2.4 | Large model loading (70B+ via TP) does not OOM or stall | 2 | | | |

### 13.3 Dtype Support Matrix

| Dtype | Priority | Plugin Pts | Plugin Compute | Plugin KV Cache | Plugin AMP | Fork Pts | Fork Compute | Fork KV Cache | Fork AMP | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `float16` | 1 | | `[ ]` | `[ ]` | `[ ]` | | `[ ]` | `[ ]` | `[ ]` | |
| `bfloat16` | 1 | | `[ ]` | `[ ]` | `[ ]` | | `[ ]` | `[ ]` | `[ ]` | |
| `float32` | 1 | | `[ ]` | `[ ]` | -- | | `[ ]` | `[ ]` | -- | |
| `int8` | 2 | | `[ ]` | -- | -- | | `[ ]` | -- | -- | weights only |
| `float8_e4m3fn` | 2 | | `[ ]` | `[ ]` | -- | | `[ ]` | `[ ]` | -- | |
| `float8_e5m2` | 3 | | `[ ]` | `[ ]` | -- | | `[ ]` | `[ ]` | -- | |
| `float4` / vendor native 4-bit float (if supported) | 3 | | `[ ]` | -- | -- | | `[ ]` | -- | -- | hardware-specific |

---

## 14. End-to-End Serving Validation -- Level: **2**

> This section validates that the full vLLM serving stack works correctly with the backend
> from the perspective of end-user behavior. Items focus on behavior unique to the serving
> context (multi-request scheduling, streaming, async engine) rather than API endpoint existence.

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 14.1 | `vllm serve <model> --device <backend>` starts and stays healthy | 1 | | | [Runtime] |
| 14.2 | Single-turn completions produce correct output via the async engine | 1 | | | [Runtime] |
| 14.3 | Chat completions (multi-turn context) produce correct output | 1 | | | [Runtime] |
| 14.4 | Streaming output delivers tokens incrementally without missing / duplicating tokens | 2 | | | |
| 14.5 | Concurrent requests (10+) are handled without deadlock or serialization | 2 | | | |
| 14.6 | `/health` endpoint returns 200 when backend is healthy | 2 | | | |
| 14.7 | Backend correctly handles requests with different `max_tokens` simultaneously | 2 | | | |
| 14.8 | Requests with early-stop conditions (`stop` tokens) terminate correctly mid-batch | 2 | | | |
| 14.9 | Embedding inference (`/v1/embeddings`) works for embedding model variants | 3 | | | |

---

## 15. LoRA / Multi-LoRA -- Level: **3**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 15.1 | `--enable-lora` accepted without crashing | 2 | | | |
| 15.2 | Single LoRA adapter loads and produces different outputs vs base model | 2 | | | |
| 15.3 | Multiple LoRA adapters served concurrently (per-request adapter switching) | 2 | | | |
| 15.4 | `get_punica_wrapper()` returns a device-specific Punica wrapper or `None` (falls back to generic CPU path) | 2 | | | |
| 15.5 | LoRA with TP > 1 works | 3 | | | |
| 15.6 | `--max-loras 4` (multiple simultaneously loaded adapters) works | 3 | | | |

---

## 16. Speculative Decoding -- Level: **3**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 16.1 | `--speculative-model <draft_model>` accepted without crashing | 2 | | | |
| 16.2 | Draft tokens generated by draft model | 2 | | | |
| 16.3 | Target model correctly verifies / rejects draft tokens | 2 | | | |
| 16.4 | Final output of speculative decoding matches greedy decoding | 2 | | | |
| 16.5 | EAGLE / EAGLE-3 speculative decoding works | 3 | | | |
| 16.6 | Speculative decoding with TP > 1 works | 3 | | | |
| 16.7 | TTOT (time to output tokens) reduction observed vs non-speculative baseline | 3 | | | [Runtime] |

---

## 17. Pipeline Parallelism -- Level: **3**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 17.1 | `--pipeline-parallel-size 2` initializes without error | 2 | | | |
| 17.2 | PP=2 inference produces outputs matching PP=1 | 2 | | | |
| 17.3 | `send()` / `recv()` between pipeline stages execute correctly on device | 2 | | | |
| 17.4 | Intermediate tensors transferred correctly between PP stages | 2 | | | |
| 17.5 | `is_first_pp_rank` / `is_last_pp_rank` logic handled correctly in Worker | 2 | | | |
| 17.6 | PP + TP combined (2D parallelism) works | 3 | | | |

---

## 18. FP8 / FP4 KV Cache -- Level: **3**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 18.1 | `--kv-cache-dtype fp8` accepted without crashing | 2 | | | |
| 18.2 | `KVQuantMode` / `kv_quant_mode` property on `AttentionImpl` returns correct mode for fp8 | 2 | | | |
| 18.3 | FP8 KV quantization / dequantization kernels implemented on device | 2 | | | |
| 18.4 | KV cache scaling factors (`kv_scales`) computed and applied correctly | 2 | | | |
| 18.5 | FP8 KV cache output quality within acceptable bounds vs fp16 KV baseline | 2 | | | [Runtime] |
| 18.6 | `--kv-cache-dtype fp4` accepted (if hardware supports it) | 3 | | | |

---

## 19. Multimodal Inputs -- Level: **3**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 19.1 | A vision-language model (e.g. LLaVA-1.5, Qwen2-VL) loads without error | 2 | | | |
| 19.2 | Image inputs processed correctly through the vision encoder on device | 2 | | | |
| 19.3 | Text+image inference produces coherent, grounded output | 2 | | | |
| 19.4 | Multi-image input works (if model supports it) | 3 | | | |
| 19.5 | Audio input support (e.g. Qwen2-Audio, Whisper via vLLM) | 3 | | | |

---

## 20. Async Output & Non-blocking I/O -- Level: **3**

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 20.1 | `is_async_output_supported(enforce_eager)` returns `True` when streams are available | 2 | | | **Removed in V1 vLLM.** Score N/A for backends targeting V1-only vLLM |
| 20.2 | Async output callbacks execute without race conditions or incorrect results | 2 | | | |
| 20.3 | Device-to-host output transfer is non-blocking (`non_blocking=True`) | 2 | | | |
| 20.4 | Output post-processing does not block GPU compute (overlap observed) | 3 | | | [Runtime] |
| 20.5 | Host-to-device input prefetch for next batch overlaps with current decode step | 3 | | | [Runtime] |

---

## 21. Observability & Ecosystem -- Level: **3**

### 21.1 Metrics & Logging

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 21.1.1 | Prometheus metrics endpoint (`/metrics`) exposed and populated | 2 | | | |
| 21.1.2 | Core vLLM metrics present: `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`, `vllm:e2e_request_latency_seconds` | 2 | | | |
| 21.1.3 | Custom backend-specific metrics registered via `vllm.stat_logger_plugins` | 3 | | | |
| 21.1.4 | OpenTelemetry tracing works end-to-end (if applicable) | 3 | | | |

### 21.2 Ecosystem Compatibility

| Library / Tool | Priority | Plugin Pts | Plugin Ver | Fork Pts | Fork Ver | Notes |
|----------------|----------|---------  |------------|----------|----------|-------|
| HuggingFace `transformers` (model weight loading) | 1 | | | |  | |
| HuggingFace `accelerate` | 2 | | | |  | |
| `lm-evaluation-harness` (accuracy benchmarking) | 2 | | | |  | |
| `vllm-benchmark` / vLLM latency+throughput scripts | 2 | | | |  | |
| Container / Docker image builds and runs correctly | 2 | | | |  | |
| `openai` Python client (via OpenAI-compat API) | 2 | | | |  | |

### 21.3 Plugin Package Quality

| # | Item | Priority | Plugin Pts | Fork Pts | Notes |
|---|------|----------|--------|----------|-------|
| 21.3.1 | README documents installation steps and quick-start example | 2 | | | |
| 21.3.2 | Plugin CI runs against each new vLLM release (regression testing) | 3 | | | |
| 21.3.3 | Minimum vLLM version pinned in package dependencies | 2 | | | |

---

## 22. Registration API Quick Reference

### Out-of-Tree Platform Plugin Path

| What | API / Mechanism | Where |
|------|----------------|-------|
| Platform registration | `entry_points={"vllm.platform_plugins": ["<name> = <pkg>:register"]}` | `setup.py` / `pyproject.toml` |
| Custom ops registration | `entry_points={"vllm.general_plugins": ["<name> = <pkg>:register_ops"]}` | `setup.py` / `pyproject.toml` |
| Platform class | `class MyPlatform(Platform): _enum = PlatformEnum.OOT` | `platform.py` |
| Worker dispatch | `vllm_config.parallel_config.worker_cls = "my_pkg.worker.MyWorker"` | inside `check_and_update_config` |
| Attention backend | `return "my_pkg.attention.MyAttentionBackend"` | `get_attn_backend_cls()` |
| Communicator | `return "my_pkg.communicator.MyDeviceCommunicator"` | `get_device_communicator_cls()` |
| Worker | `class MyWorker(WorkerBase)` | `worker.py` |
| ModelRunner | `class MyModelRunner` (no abstract base; implement `execute_model`, `profile_run`, `load_model`, `get_kv_cache_spec`, `initialize_kv_cache` matching `GPUModelRunner` interface) | `model_runner.py` |
| KV Cache Spec | `def get_kv_cache_spec() -> dict[str, KVCacheSpec]` | `worker.py` |
| AttentionBackend | `class MyAttentionBackend(AttentionBackend)` | `attention.py` |
| AttentionImpl | `class MyAttentionImpl(AttentionImpl)` | `attention.py` |
| Communicator | `class MyDeviceCommunicator(DeviceCommunicatorBase)` | `communicator.py` |
| Custom KVCacheSpec | `@KVCacheSpecRegistry.register(...)` | `kv_cache.py` |

### vLLM Fork Path

| What | API / Mechanism | Where |
|------|----------------|-------|
| Platform | Add class to `vllm/platforms/` | `vllm/platforms/<name>.py` |
| Platform selection | Add detection condition to `vllm/platforms/__init__.py` | Core file |
| Worker | Add to `vllm/v1/worker/` | `vllm/v1/worker/<name>/worker.py` |
| ModelRunner | Add to `vllm/v1/worker/<name>/` | `model_runner.py` |
| Attention | Add to `vllm/v1/attention/backends/` | `vllm/v1/attention/backends/<name>.py` |

---

## Sources

- [vLLM Hardware Plugin Blog Post (Ascend NPU)][hw-plugin]
- [vLLM Plugin System Design Doc][plugin-system]
- [vLLM V1 Guide][v1-guide]
- [WorkerBase API Reference][worker-base]
- [AttentionBackend API Reference][attn-backend]
- [KVCacheInterface Reference][kv-interface]
- [DeviceCommunicatorBase][communicator]
- [vllm-ascend Reference Implementation][vllm-ascend]
- [vllm-spyre Reference Implementation][vllm-spyre]
- [vLLM Model Runner V2 Design][mrv2]

[hw-plugin]: https://vllm.ai/blog/2025-05-12-hardware-plugin
[plugin-system]: https://docs.vllm.ai/en/latest/design/plugin_system/
[v1-guide]: https://docs.vllm.ai/en/latest/usage/v1_guide/
[worker-base]: https://docs.vllm.ai/en/latest/api/vllm/v1/worker/worker_base/
[attn-backend]: https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backend/
[kv-interface]: https://docs.vllm.ai/en/stable/api/vllm/v1/kv_cache_interface/
[communicator]: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/base_device_communicator.py
[vllm-ascend]: https://github.com/vllm-project/vllm-ascend
[vllm-spyre]: https://github.com/vllm-project/vllm-spyre
[mrv2]: https://docs.vllm.ai/en/latest/design/model_runner_v2/
