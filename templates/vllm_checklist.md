# vLLM Hardware Accelerator Integration Readiness Checklist

> A comprehensive template for evaluating out-of-tree hardware accelerator
> integration readiness with vLLM's plugin architecture.
>
> Derived from vLLM's official [Plugin System docs][plugin-docs],
> [Platform API reference][platform-api], [Custom Op docs][customop-docs],
> [Attention Backend docs][attn-docs], real-world backends (vllm-ascend, HPU, TPU),
> and adversarially verified against vLLM source code.

**Backend under evaluation**: _[FILL: backend name]_
**Evaluation date**: _[FILL: date]_
**Evaluator**: _[FILL: human or agent]_
**vLLM version**: _[FILL: version]_
**PyTorch readiness score**: _[FILL: XX% from PyTorch checklist]_

**Important caveats**:
- vLLM has not reached 1.0; all plugin interfaces are subject to change.
- Platform class methods raise `NotImplementedError` by convention, not `@abstractmethod` -- failures occur at runtime, not class definition time.
- vLLM is migrating from v0 to v1 architecture; this checklist primarily reflects v1.

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
  - Critical gap 3]_

### Section Scores

| Section | Level | Max Pts | Earned | Pct |
|---------|-------|---------|--------|-----|
| PyTorch Prerequisites | 1 | | | |
| Plugin Registration | 1 | | | |
| Platform Class | 1 | | | |
| Attention Backend | 1 | | | |
| Worker | 1 | | | |
| Custom Ops | 1 | | | |
| Model Runner | 2 | | | |
| KV Cache Management | 2 | | | |
| Memory Management | 2 | | | |
| Distributed | 2 | | | |
| Model Compatibility | 2 | | | |
| Dtype Support | 2 | | | |
| Compilation / Graph Capture | 3 | | | |
| Testing & CI | 3 | | | |
| Quantization | 3 | | | |
| Speculative Decoding | 3 | | | |
| Multimodal | 3 | | | |
| Profiling | 3 | | | |

**Readiness**: _____ %

---

## 0. PyTorch Backend Prerequisites -- Level: **1**

vLLM is built on PyTorch. A functional PyTorch backend is a hard prerequisite.
Evaluate using the [PyTorch Accelerator Readiness Checklist](pytorch_checklist.md) first.

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 0.1 | PyTorch PrivateUse1 backend registered and functional | 1 | | |
| 0.2 | `torch.device("<name>")` works | 1 | | |
| 0.3 | Tensor creation on device (`torch.empty`, `torch.zeros`) | 1 | | |
| 0.4 | Device<->CPU transfer (`.to()`, `.cpu()`) | 1 | | |
| 0.5 | Core ATen operators (add, mm, softmax, etc.) | 1 | | |
| 0.6 | `torch.distributed` backend registered (for TP/PP) | 1 | | |
| 0.7 | PyTorch readiness score | 1 | | |
| 0.8 | `torch.compile` works with device | 2 | | |
| 0.9 | AMP / autocast works (for mixed-precision inference) | 2 | | |
| 0.10 | Serialization (`torch.save`/`torch.load`) works | 2 | | |
| 0.11 | Autograd backward pass works | 3 | | |

---

## Structure

vLLM organizes accelerator integration around four core abstractions:

1. **Platform** -- Device management, capability detection, configuration validation
2. **Worker** -- Device initialization, memory management, model loading, KV cache
3. **ModelRunner** -- Model execution, input preparation, sampling
4. **AttentionBackend** -- Attention computation, KV cache layout, paged attention

Plus cross-cutting concerns: CustomOp registration, distributed communication,
compilation/graph capture, and quantization support.

---

## 1. Platform Plugin Registration -- Level: **1**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 1.1 | `entry_points` registered under `vllm.platform_plugins` group | 1 | | |
| 1.2 | Register function returns the Platform class's fully qualified name | 1 | | |
| 1.3 | Platform activates correctly (OOT takes priority over built-in) | 1 | | |
| 1.4 | Plugin loads without error on `import vllm` | 1 | | |
| 1.5 | Register function is re-entrant (safe across multiple processes) | 2 | | |
| 1.6 | `vllm.general_plugins` entry point registered (if custom ops/models needed) | 2 | | |

---

## 2. Platform Class (inherits `vllm.platforms.interface.Platform`) -- Level: **1**

### Required Class Variables

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 2.1 | `_enum` property set (typically `PlatformEnum.OOT`) | 1 | | |
| 2.2 | `device_type` property returns PyTorch device type string | 1 | | |
| 2.3 | `device_name` class variable set | 1 | | |
| 2.4 | `dispatch_key` set (default `'CPU'`; override for custom dispatch) | 2 | | |
| 2.5 | `ray_device_key` set (for Ray resource scheduling) | 2 | | |
| 2.6 | `device_control_env_var` set (e.g., `"ASCEND_RT_VISIBLE_DEVICES"`) | 2 | | |
| 2.7 | `dist_backend` set (distributed backend name) | 2 | | |
| 2.8 | `supported_quantization` list defined | 2 | | |

### Device Management Methods

| # | Method | Priority | Points | Notes |
|---|--------|----------|--------|-------|
| 2.9 | `get_device_name()` | 1 | | |
| 2.10 | `get_current_memory_usage()` | 1 | | |
| 2.11 | `set_device(device)` | 1 | | |
| 2.12 | `check_if_supports_dtype(dtype)` | 2 | | |
| 2.13 | `stateless_init_device_torch_dist_pg()` | 2 | | |
| 2.14 | `manual_seed_all(seed)` | 3 | | |
| 2.15 | `num_compute_units()` | 3 | | |

### Component Getter Methods

| # | Method | Priority | Points | Notes |
|---|--------|----------|--------|-------|
| 2.16 | `get_attn_backend_cls()` returns attention backend FQN | 1 | | |
| 2.17 | `get_device_communicator_cls()` returns communicator FQN | 1 | | |
| 2.18 | `get_worker_cls()` returns worker FQN | 1 | | |
| 2.19 | `get_model_runner_cls()` returns model runner FQN | 1 | | |

### Compilation & Graph Capture Methods

| # | Method | Priority | Points | Notes |
|---|--------|----------|--------|-------|
| 2.20 | `get_compile_backend()` (default: `'inductor'`) | 2 | | |
| 2.21 | `get_static_graph_wrapper_cls()` | 2 | | |
| 2.22 | `support_static_graph_mode()` | 2 | | |
| 2.23 | `get_pass_manager_cls()` | 3 | | |

### Configuration Validation Methods

| # | Method | Priority | Points | Notes |
|---|--------|----------|--------|-------|
| 2.24 | `check_and_update_config(vllm_config)` | 1 | | |
| 2.25 | `verify_quantization(quant)` | 2 | | |
| 2.26 | `verify_model_arch(model_arch)` | 2 | | |

### Feature Support Methods

| # | Method | Priority | Points | Notes |
|---|--------|----------|--------|-------|
| 2.27 | `use_custom_op_collectives()` | 2 | | |
| 2.28 | `supports_v1(model_config)` | 2 | | |
| 2.29 | `use_custom_allreduce()` | 3 | | |

---

## 3. Worker Implementation (inherits `WorkerBase`) -- Level: **1**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 3.1 | `init_device()` -- initialize device context | 1 | | |
| 3.2 | `initialize_cache(kv_cache_config)` -- allocate KV cache buffers | 1 | | |
| 3.3 | `load_model()` -- load model weights onto device | 1 | | |
| 3.4 | `get_kv_cache_spec()` -- return KV cache specification | 1 | | |
| 3.5 | `determine_available_memory()` -- report free device memory | 1 | | |
| 3.6 | `execute_model(scheduler_output)` -- run one forward step | 1 | | |
| 3.7 | `initialize_from_config(vllm_config)` -- full initialization | 2 | | |
| 3.8 | `compile_or_warm_up_model()` -- compilation/warmup step | 2 | | |

---

## 4. Model Runner Implementation (inherits `ModelRunnerBase`) -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 4.1 | Input preparation (tokenized inputs -> device tensors) | 1 | | |
| 4.2 | Forward pass execution | 1 | | |
| 4.3 | Sampling / output processing | 1 | | |
| 4.4 | `InputBatch` handling | 2 | | |
| 4.5 | `SamplingMetadata` construction | 2 | | |
| 4.6 | Multi-step scheduling support | 3 | | |

---

## 5. Attention Backend (inherits `AttentionBackend`) -- Level: **1**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 5.1 | Attention backend class implemented | 1 | | |
| 5.2 | Registered via `get_attn_backend_cls()` in Platform | 1 | | |
| 5.3 | Paged attention support | 1 | | |
| 5.4 | Prefill attention (variable-length) | 1 | | |
| 5.5 | Decode attention (single-token) | 1 | | |
| 5.6 | `validate_configuration()` checks compatibility | 2 | | |

### Attention Features

| Feature | Priority | Points | Notes |
|---------|----------|--------|-------|
| Chunked prefill | 1 | | |
| Multi-head attention (MHA) | 1 | | |
| Grouped-query attention (GQA) | 1 | | |
| Prefix caching (APC) | 2 | | |
| Multi-query attention (MQA) | 2 | | |
| Multi-latent attention (MLA/DeepSeek) | 2 | | |
| Speculative decoding attention | 2 | | |
| Sliding window attention | 3 | | |
| FP8 KV cache | 3 | | |
| Encoder-decoder cross-attention | 3 | | |

---

## 6. KV Cache Management -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 6.1 | `update_block_size_for_backend()` implemented | 1 | | |
| 6.2 | Block allocation on device | 1 | | |
| 6.3 | Block copy (CoW / fork) | 2 | | |
| 6.4 | Block swap (device <-> CPU) | 2 | | |
| 6.5 | Prefix caching hash-based block reuse | 2 | | |
| 6.6 | `support_hybrid_kv_cache()` | 3 | | |
| 6.7 | `register_custom_kv_cache_specs()` | 3 | | |

---

## 7. Custom Ops Registration -- Level: **1**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 7.1 | `@CustomOp.register_oot('op_name')` decorator used | 1 | | |
| 7.2 | `forward_oot()` method implemented for OOT dispatch | 1 | | |
| 7.3 | Key ops: `paged_attention` | 1 | | |
| 7.4 | Key ops: `rotary_embedding` | 1 | | |
| 7.5 | Key ops: `rms_norm` / `layer_norm` | 1 | | |
| 7.6 | Key ops: `all_reduce` / collective ops | 1 | | |
| 7.7 | Key ops: `silu_and_mul` / activation fusions | 2 | | |
| 7.8 | Key ops: `fused_moe` (for MoE models) | 2 | | |

---

## 8. Quantization Support -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 8.1 | `supported_quantization` list in Platform class | 1 | | |
| 8.2 | `verify_quantization(quant)` validates compatibility | 2 | | |

### Quantization Methods

| Method | Priority | Points | Notes |
|--------|----------|--------|-------|
| FP8 (W8A8) | 1 | | |
| GPTQ (W4A16) | 2 | | |
| AWQ (W4A16) | 2 | | |
| INT8 (W8A8 smoothquant) | 2 | | |
| SqueezeLLM | 3 | | |
| GGUF | 3 | | |
| BitsAndBytes (NF4) | 3 | | |
| Marlin (optimized GPTQ) | 3 | | |

---

## 9. Distributed Communication -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 9.1 | `DeviceCommunicatorBase` subclass implemented | 1 | | |
| 9.2 | Registered via `get_device_communicator_cls()` in Platform | 1 | | |
| 9.3 | `dist_backend` set in Platform class | 2 | | |

### Collective Operations

| Collective | Priority | Points | Notes |
|-----------|----------|--------|-------|
| `all_reduce` | 1 | | |
| `all_gather` | 2 | | |
| `reduce_scatter` | 2 | | |
| `broadcast` | 2 | | |
| `barrier` | 3 | | |
| `send` / `recv` (P2P) | 3 | | |

### Parallelism Support

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 9.4 | Tensor parallelism (TP) | 1 | | |
| 9.5 | Pipeline parallelism (PP) | 2 | | |
| 9.6 | Expert parallelism (EP, for MoE) | 2 | | |
| 9.7 | Data parallelism (DP, multi-instance) | 3 | | |

---

## 10. Compilation & Graph Capture -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 10.1 | `torch.compile` works with backend (via `get_compile_backend()`) | 1 | | |
| 10.2 | Static graph capture / CUDA Graph equivalent | 1 | | |
| 10.3 | `get_static_graph_wrapper_cls()` returns wrapper | 2 | | |
| 10.4 | `support_static_graph_mode()` returns True | 2 | | |
| 10.5 | Piecewise compilation support (compilation levels 0-3) | 3 | | |
| 10.6 | Custom pass manager (via `get_pass_manager_cls()`) | 3 | | |

---

## 11. Memory Management -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 11.1 | `determine_available_memory()` returns accurate free memory | 1 | | |
| 11.2 | `get_current_memory_usage()` tracks current allocation | 1 | | |
| 11.3 | GPU memory fraction limiting (`gpu_memory_utilization`) | 2 | | |
| 11.4 | Memory profiling for auto KV cache sizing | 2 | | |
| 11.5 | Swap space (CPU offload) for KV cache | 3 | | |

---

## 12. Speculative Decoding -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 12.1 | Draft model execution on device | 1 | | |
| 12.2 | Verification step on device | 1 | | |
| 12.3 | Token acceptance/rejection logic | 2 | | |
| 12.4 | Bonus token handling | 3 | | |

---

## 13. Multimodal Support -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 13.1 | Image encoder execution on device | 2 | | |
| 13.2 | Cross-attention for encoder-decoder models | 2 | | |
| 13.3 | Multimodal input preprocessing pipeline | 2 | | |
| 13.4 | Video encoder execution on device | 3 | | |
| 13.5 | Audio encoder execution on device | 3 | | |

---

## 14. Model Compatibility -- Level: **2**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 14.1 | Decoder-only models (LLaMA, GPT, Mistral, etc.) | 1 | | |
| 14.2 | `verify_model_arch(model_arch)` validates supported models | 2 | | |
| 14.3 | MoE models (Mixtral, DeepSeek-MoE) | 2 | | |
| 14.4 | Multi-latent attention models (DeepSeek-V2/V3) | 2 | | |
| 14.5 | LoRA adapter support | 2 | | |
| 14.6 | Encoder-decoder models (T5, BART, etc.) | 3 | | |

---

## 15. Profiling & Observability -- Level: **3**

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 15.1 | Device-side profiling (torch.profiler or equivalent) | 2 | | |
| 15.2 | Memory usage reporting | 2 | | |
| 15.3 | Throughput metrics (tokens/sec) | 2 | | |
| 15.4 | Latency metrics (TTFT, TPOT, ITL) | 2 | | |
| 15.5 | Custom stat logger via `vllm.stat_logger_plugins` | 3 | | |

---

## 16. Testing & Validation -- Level: **3**

### Functional Tests

| Test Area | Priority | Points | Notes |
|-----------|----------|--------|-------|
| Basic inference (single request) | 1 | | |
| Batched inference | 1 | | |
| Greedy decoding correctness | 1 | | |
| Streaming output | 2 | | |
| Sampling (top-k, top-p, temperature) | 2 | | |
| Long context (>4K tokens) | 2 | | |
| Beam search | 3 | | |
| Parallel decoding | 3 | | |
| Stop sequences / stop tokens | 3 | | |

### Integration Tests

| Test Area | Priority | Points | Notes |
|-----------|----------|--------|-------|
| OpenAI-compatible API server | 1 | | |
| Tensor parallel inference | 2 | | |
| Quantized model inference | 2 | | |
| Pipeline parallel inference | 3 | | |
| Multimodal inference | 3 | | |
| LoRA serving | 3 | | |

### CI Integration

| # | Item | Priority | Points | Notes |
|---|------|----------|--------|-------|
| 16.1 | CI pipeline runs on target hardware | 1 | | |
| 16.2 | Regression tests on each vLLM update | 2 | | |
| 16.3 | Model correctness benchmarks | 2 | | |
| 16.4 | Performance benchmarks (throughput, latency) | 3 | | |

---

## 17. Dtype Support -- Level: **2**

| Dtype | Priority | Points | Compute | KV Cache | Quantization | Notes |
|-------|----------|--------|---------|----------|-------------|-------|
| `float16` | 1 | | `[ ]` | `[ ]` | -- | |
| `bfloat16` | 1 | | `[ ]` | `[ ]` | -- | |
| `float32` | 2 | | `[ ]` | `[ ]` | -- | |
| `float8_e4m3fn` | 2 | | `[ ]` | `[ ]` | `[ ]` | |
| `int8` | 2 | | `[ ]` | -- | `[ ]` | |
| `float8_e5m2` | 3 | | `[ ]` | `[ ]` | `[ ]` | |
| `int4` | 3 | | -- | -- | `[ ]` | |

---

## 18. API & Registration Quick Reference

| What | API / Mechanism | Where |
|------|----------------|-------|
| Platform plugin | `entry_points = {"vllm.platform_plugins": ["name = pkg:register"]}` | setup.py |
| General plugin | `entry_points = {"vllm.general_plugins": ["name = pkg:register"]}` | setup.py |
| Platform class | Inherit `vllm.platforms.interface.Platform` | Plugin package |
| Worker class | Inherit `vllm.v1.worker.worker_base.WorkerBase` | Plugin package |
| Model runner | Inherit `vllm.v1.worker.model_runner_base.ModelRunnerBase` | Plugin package |
| Attention backend | Implement attention backend interface | Plugin package |
| Communicator | Inherit `DeviceCommunicatorBase` | Plugin package |
| Custom ops | `@CustomOp.register_oot('op_name')` + `forward_oot()` | Plugin package |
| Quantization | Add to `supported_quantization` list in Platform | Platform class |
| Compile backend | Override `get_compile_backend()` in Platform | Platform class |

---

## Sources

- [vLLM Plugin System Documentation][plugin-docs]
- [Platform API Reference][platform-api]
- [Custom Op Documentation][customop-docs]
- [Attention Backend Documentation][attn-docs]
- [vLLM Hardware Plugin Blog Post][blog]
- [vllm-ascend Reference Implementation][vllm-ascend]
- [Hardware Plugin RFC #11162][rfc-11162]
- [API Stability RFC #19161][rfc-19161]

[plugin-docs]: https://docs.vllm.ai/en/latest/design/plugin_system/
[platform-api]: https://docs.vllm.ai/en/latest/api/vllm/platforms/interface.html
[customop-docs]: https://docs.vllm.ai/en/stable/design/custom_op/
[attn-docs]: https://docs.vllm.ai/en/latest/design/attention_backends/
[blog]: https://vllm.ai/blog/2025-05-12-hardware-plugin
[vllm-ascend]: https://github.com/vllm-project/vllm-ascend
[rfc-11162]: https://github.com/vllm-project/vllm/issues/11162
[rfc-19161]: https://github.com/vllm-project/vllm/issues/19161
