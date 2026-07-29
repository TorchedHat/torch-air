# PyTorch Accelerator Integration Readiness Checklist (Closed-Source OOT)

> Template for evaluating closed-source out-of-tree hardware accelerator
> integration readiness with PyTorch. Covers **Monkey Patching**,
> **XLA-style Lazy Tensor**, **Compile-Only**, and **Hybrid** integration patterns.
>
> All probing is runtime-based -- no source code access required.

**Backend under evaluation**: _[FILL]_
**Vendor**: _[FILL]_
**Package name**: _[FILL: pip/conda package]_
**Package version**: _[FILL]_
**Integration pattern**: _[FILL: MONKEY_PATCH / LAZY_TENSOR / COMPILE_ONLY / HYBRID]_
**Device string**: _[FILL: or "N/A" for compile-only]_
**PyTorch version compatibility**: _[FILL: from vendor docs]_
**Evaluation date**: _[FILL]_
**Evaluator**: _[FILL]_
**Source access**: Closed-source (no source code available)

**Status markers**: `[ ]` Not Ready | `[~]` Partially Ready | `[x]` Ready | `[N/A]` Not applicable

**Evidence tags**: `[RI]` Runtime Introspection | `[VD]` Vendor Docs | `[PM]` Package Metadata | `[PB]` Public Benchmarks | `[MZ]` Model Zoo | `[CE]` Community Evidence

---

## Readiness Score & Summary

### Executive Summary

_[FILL: Write a concise summary covering:
- **Overall verdict** and score (X.X / 10)
- **Key strengths**:
  - Top strength 1
  - Top strength 2
  - Top strength 3
- **Key gaps**:
  - Critical gap 1
  - Critical gap 2
  - Critical gap 3
- **Key next steps**:
  - Highest-impact action 1
  - Highest-impact action 2
  - Highest-impact action 3
- **Notable insights**:
  - Surprising finding or important context 1
  - Surprising finding or important context 2]_

### Scoring Model

Each row has a **point value** (1-3) reflecting its importance:
- **3 pts** = Critical (blocks basic functionality)
- **2 pts** = Important (expected for production use)
- **1 pt** = Nice-to-have (polish, edge cases)

Each section has a **rank** (1-3) reflecting its tier:
- **Rank 1** = Foundational (blocks basic functionality)
- **Rank 2** = Core Production (expected for production use)
- **Rank 3** = Quality of Life (polish, ecosystem, sustainability)

**Row scoring**: `[x]` = full points, `[~]` = half points, `[ ]` = 0, `[N/A]` = excluded from max

**Section score**: `section_pct = earned_pts / max_pts` (percentage only, no multiplication by rank)

**Overall score formula**:
```
weight_r = 1 / rank
Overall Score = (sum(section_pct * weight_r) / sum(weight_r)) * 10
```
where the sums are over all applicable sections (excluding fully N/A sections).

**Readiness verdict**:
- **0 - 4** = Not Ready
- **4 - 7** = Partially Ready
- **7 - 10** = Ready

**N/A guidance**: For closed-source backends, N/A is used more liberally than for
open-source. Compile-only backends may mark device management, streams, and memory
sections as N/A. The scoring formula handles this correctly (N/A rows excluded from max).

### Section Scores

| Section | Rank | Max Pts | Earned | Pct | Weighted |
|---------|------|---------|--------|-----|----------|
| **Rank 1 -- Foundational** | | | | | |
| Device Registration & Management | 1 | | | | |
| Operator Coverage | 1 | | | | |
| Autograd (Training Support) | 1 | | | | |
| Memory Management | 1 | | | | |
| Integration Path Classification | 1 | | | | |
| API Compatibility & Divergence | 1 | | | | |
| **Rank 2 -- Core Production** | | | | | |
| Serialization & Model Portability | 2 | | | | |
| AMP | 2 | | | | |
| torch.compile / Compilation | 2 | | | | |
| Distributed Training | 2 | | | | |
| Dtype Support Matrix | 2 | | | | |
| Numerical Accuracy | 2 | | | | |
| Testing & Documentation | 2 | | | | |
| **Rank 3 -- Quality of Life** | | | | | |
| Streams & Events | 3 | | | | |
| RNG & Generator | 3 | | | | |
| DataLoader Integration | 3 | | | | |
| Device Behavior Parity | 3 | | | | |
| Ecosystem Compatibility | 3 | | | | |
| Vendor Lock-in Assessment | 3 | | | | |
| Migration & Portability | 3 | | | | |
| Profiler & Observability | 3 | | | | |
| **TOTAL** | | | | | |

### Calculation

```
Section Pct = earned_pts / max_pts (excluding N/A rows)
weight_r = 1 / rank
Weighted = Pct * (1 / rank)
Overall Score = (sum(Weighted) / sum(weight_r for applicable sections)) * 10
```

### Final Verdict

| Score | Verdict |
|-------|---------|
| 0 - 4 | **Not Ready** |
| 4 - 7 | **Partially Ready** |
| 7 - 10 | **Ready** |

**Overall Score**: _____ / 10
**Verdict**: _____

### Summary

_[FILL: Write as bullet points:
- **Score**: X.X/10 (Verdict)
- **Production-quality areas**: List fully implemented areas
- **Integration pattern**: Characterize how the backend integrates
- **Gaps**:
  - Gap 1
  - Gap 2
- **Next steps**:
  - Action 1
  - Action 2
- **Character**: One line on backend's focus and approach]_

---

## Integration Pattern Applicability

Mark sections N/A per detected pattern. Compile-only backends skip device/memory/streams.

| Section | Monkey Patch | Lazy Tensor | Compile-Only | Hybrid |
|---------|:---:|:---:|:---:|:---:|
| 1. Device Registration | Applicable | Applicable | Partial/N/A | Applicable |
| 2. Operator Coverage | Applicable | Applicable | Compile-time | Applicable |
| 3. Autograd | Applicable | Applicable | Compile-time | Applicable |
| 4. Memory Management | Applicable | Applicable | N/A | Applicable |
| 5. Integration Path | Applicable | Applicable | Applicable | Applicable |
| 6. API Compatibility | Critical | Important | Less relevant | Critical |
| 7. Serialization | Applicable | Applicable | Partial | Applicable |
| 8. AMP | Applicable | Applicable | Compile-time | Applicable |
| 9. torch.compile | Less relevant | Less relevant | Critical | Applicable |
| 10. Distributed | Applicable | Applicable | Applicable | Applicable |
| 14. Streams & Events | Applicable | Partial | N/A | Applicable |
| 19. Vendor Lock-in | Critical | Important | Important | Critical |
| 20. Migration | Critical | Important | Important | Critical |

---

## 0. Package Discovery & Integration Path Detection

The evaluator uses runtime introspection to discover the package and detect
which integration pattern the vendor uses. No source code required.

### Step 1: Package Discovery

```python
pip show <vendor_name>
pip show torch_<vendor_name>
python -c "import <vendor_name>; print(<vendor_name>.__file__)"
python -c "from importlib.metadata import entry_points; print([e for e in entry_points(group='torch.backends')])"
```

### Step 2: Integration Path Detection

```python
import sys, torch
# Snapshot before vendor import
attrs_before = set(dir(torch))
modules_before = dict(sys.modules)

import vendor_package

# Monkey-patch detection
attrs_after = set(dir(torch))
new_attrs = attrs_after - attrs_before

# Device type detection
try:
    dev = torch.device("vendor_name")
except RuntimeError:
    pass

# PU1 check
pu1_name = torch._C._get_privateuse1_backend_name()

# Compile backend detection
from torch._dynamo.backends.registry import list_backends
backends = list_backends(exclude_tags=())

# Lazy tensor detection
has_mark_step = hasattr(vendor_module, 'mark_step')
```

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 0.1 | Package installable via pip/conda | 3 | `[ ]` | `[PM]` | |
| 0.2 | `import vendor_package` succeeds without error | 3 | `[ ]` | `[RI]` | |
| 0.3 | Compatible with current PyTorch version | 3 | `[ ]` | `[PM][VD]` | |
| 0.4 | `torch.backends` entry_points registered | 2 | `[ ]` | `[PM]` | |
| 0.5 | `torch_dynamo_backends` entry_points registered | 2 | `[ ]` | `[PM]` | |
| 0.6 | sys.modules diff shows monkey-patching (if applicable) | 2 | `[ ]` | `[RI]` | |
| 0.7 | `torch.device("<name>")` works | 3 | `[ ]` | `[RI]` | |
| 0.8 | PrivateUse1 backend name registered | 2 | `[ ]` | `[RI]` | |
| 0.9 | Custom dispatch key present | 2 | `[ ]` | `[RI]` | |
| 0.10 | Dynamo compile backend registered | 2 | `[ ]` | `[RI]` | |
| 0.11 | Lazy tensor markers detected (mark_step, etc.) | 1 | `[ ]` | `[RI]` | |
| 0.12 | Vendor SDK/runtime dependency documented | 2 | `[ ]` | `[VD][PM]` | |
| 0.13 | Hardware detection API available | 2 | `[ ]` | `[RI][VD]` | |
| 0.14 | Multi-device support detected | 2 | `[ ]` | `[RI][VD]` | |
| 0.15 | Integration pattern classified | 3 | `[ ]` | `[RI]` | MONKEY_PATCH / LAZY_TENSOR / COMPILE_ONLY / HYBRID |

---

## 1. Device Registration & Management -- Rank: **1**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 1.1 | Device type registered (`torch.device("<name>")` works) | 3 | `[ ]` | `[RI]` | |
| 1.2 | `torch.device("<name>:0")` works with index | 3 | `[ ]` | `[RI]` | |
| 1.3 | `torch.<name>.is_available()` | 3 | `[ ]` | `[RI]` | |
| 1.4 | `torch.<name>.device_count()` | 3 | `[ ]` | `[RI]` | |
| 1.5 | `torch.<name>.current_device()` | 2 | `[ ]` | `[RI]` | |
| 1.6 | `torch.<name>.set_device(idx)` | 2 | `[ ]` | `[RI]` | |
| 1.7 | `torch.<name>.synchronize()` | 2 | `[ ]` | `[RI]` | |
| 1.8 | Device context manager (`with torch.<name>.device(idx)`) | 2 | `[ ]` | `[RI]` | |
| 1.9 | Multi-device indexing (`<name>:0`, `<name>:1`) | 2 | `[ ]` | `[RI]` | |
| 1.10 | Device properties/capabilities queryable | 1 | `[ ]` | `[RI][VD]` | |
| 1.11 | `torch.accelerator.current_device()` works | 1 | `[ ]` | `[RI]` | |
| 1.12 | `torch.accelerator.device_count()` works | 1 | `[ ]` | `[RI]` | |
| 1.13 | `torch.accelerator.is_available()` works | 1 | `[ ]` | `[RI]` | |

---

## 2. Operator Coverage -- Rank: **1**

### 2.1 Minimal Kernel Set

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 2.1.1 | `torch.empty(shape, device="<name>")` works | 3 | `[ ]` | `[RI]` | |
| 2.1.2 | Device<->CPU transfer (`.to("cpu")`, `.to("<name>")`) | 3 | `[ ]` | `[RI]` | |
| 2.1.3 | `.item()` scalar extraction | 3 | `[ ]` | `[RI]` | |
| 2.1.4 | `torch.zeros`, `ones`, `randn`, `full`, `arange` on device | 3 | `[ ]` | `[RI]` | |

### 2.2 Extended Operator Coverage

Test each category by creating tensors on device, calling ops, checking results.

| Category | Example Ops | Pts | Status | Pass/Total | Notes |
|----------|-------------|-----|--------|------------|-------|
| Elementwise unary | `abs`, `neg`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu` | 3 | `[ ]` | | |
| Elementwise binary | `add`, `sub`, `mul`, `div`, `remainder`, `pow`, `maximum`, `minimum` | 3 | `[ ]` | | |
| Comparison | `eq`, `ne`, `lt`, `gt`, `le`, `ge` | 2 | `[ ]` | | |
| Reduction | `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `any`, `all`, `prod` | 3 | `[ ]` | | |
| Linear algebra | `mm`, `bmm`, `addmm`, `matmul`, `linear`, `einsum` | 3 | `[ ]` | | |
| Convolution | `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d` | 2 | `[ ]` | | |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` | 2 | `[ ]` | | |
| Normalization | `batch_norm`, `layer_norm`, `group_norm`, `instance_norm` | 3 | `[ ]` | | |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `softmax`, `log_softmax` | 3 | `[ ]` | | |
| Attention | `scaled_dot_product_attention` | 3 | `[ ]` | | |
| Indexing | `index`, `index_put`, `index_select`, `gather`, `scatter`, `masked_fill` | 2 | `[ ]` | | |
| Shape ops | `reshape`, `view`, `permute`, `transpose`, `contiguous`, `cat`, `stack`, `chunk`, `split` | 2 | `[ ]` | | |
| Memory ops | `clone`, `copy_`, `fill_`, `zero_`, `empty_like`, `zeros_like` | 3 | `[ ]` | | |
| Random | `uniform_`, `normal_`, `bernoulli_`, `dropout`, `rand`, `randn` | 2 | `[ ]` | | |
| Embedding | `embedding`, `embedding_bag` | 2 | `[ ]` | | |
| Loss functions | `cross_entropy`, `mse_loss`, `nll_loss`, `binary_cross_entropy_with_logits` | 2 | `[ ]` | | |
| Sorting | `sort`, `topk`, `argsort` | 1 | `[ ]` | | |
| Type casting | `to(dtype)`, `float()`, `half()`, `bfloat16()`, `int()`, `bool()` | 2 | `[ ]` | | |

### 2.3 Model-Level Validation

| Model | Framework | Pts | Status | Notes |
|-------|-----------|-----|--------|-------|
| ResNet-50 | torchvision | 2 | `[ ]` | |
| BERT-base | HuggingFace transformers | 2 | `[ ]` | |
| GPT-2 | HuggingFace transformers | 2 | `[ ]` | |
| Llama-2-7B (inference) | HuggingFace transformers | 3 | `[ ]` | |
| Vision Transformer (ViT) | torchvision / timm | 1 | `[ ]` | |
| Stable Diffusion (UNet) | diffusers | 1 | `[ ]` | |
| Whisper | HuggingFace transformers | 1 | `[ ]` | |
| T5 | HuggingFace transformers | 1 | `[ ]` | |

### 2.4 Fallback Mechanisms

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 2.4.1 | Unsupported ops fall back to CPU gracefully | 2 | `[ ]` | `[RI]` | |
| 2.4.2 | Clear error message on unsupported ops (not segfault) | 2 | `[ ]` | `[RI]` | |
| 2.4.3 | Fallback behavior documented | 1 | `[ ]` | `[VD]` | |

---

## 3. Autograd (Training Support) -- Rank: **1**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 3.1 | `.backward()` completes on a simple loss | 3 | `[ ]` | `[RI]` | |
| 3.2 | `torch.autograd.grad()` works | 2 | `[ ]` | `[RI]` | |
| 3.3 | Gradient accumulation (`.backward()` multiple times) | 2 | `[ ]` | `[RI]` | |
| 3.4 | `torch.autograd.Function` custom forward/backward on device | 2 | `[ ]` | `[RI]` | |
| 3.5 | Gradient checkpointing (`torch.utils.checkpoint.checkpoint`) | 2 | `[ ]` | `[RI]` | |
| 3.6 | Gradients match CPU numerically (within tolerance) | 3 | `[ ]` | `[RI]` | |
| 3.7 | Mixed precision backward (fp16/bf16 forward, fp32 grad) | 2 | `[ ]` | `[RI]` | |
| 3.8 | Higher-order gradients (if needed) | 1 | `[ ]` | `[RI]` | |
| 3.9 | Training loop converges (loss decreases over epochs) | 3 | `[ ]` | `[RI]` | |

---

## 4. Memory Management -- Rank: **1**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 4.1 | Memory tracking API exists (`memory_allocated()` or equivalent) | 3 | `[ ]` | `[RI][VD]` | |
| 4.2 | `torch.<name>.memory_allocated()` | 2 | `[ ]` | `[RI]` | |
| 4.3 | `torch.<name>.max_memory_allocated()` | 1 | `[ ]` | `[RI]` | |
| 4.4 | `torch.<name>.memory_reserved()` (if caching allocator) | 1 | `[ ]` | `[RI]` | |
| 4.5 | `torch.<name>.empty_cache()` | 2 | `[ ]` | `[RI]` | |
| 4.6 | `memory_summary()` or equivalent | 1 | `[ ]` | `[RI]` | |
| 4.7 | OOM produces clear error (not segfault) | 3 | `[ ]` | `[RI]` | |
| 4.8 | Repeated allocation shows no memory leak | 2 | `[ ]` | `[RI]` | |

---

## 5. Integration Path Classification -- Rank: **1**

This section characterizes the vendor's integration approach. Mark items that
match the detected pattern.

### 5.1 Monkey-Patch Detection

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 5.1.1 | `import vendor` modifies `torch.*` module attributes | 3 | `[ ]` | `[RI]` | Count: ___ attrs changed |
| 5.1.2 | Original torch functions replaced (not wrapped) | 2 | `[ ]` | `[RI]` | |
| 5.1.3 | Monkey-patching is reversible / scopeable | 1 | `[ ]` | `[RI]` | |

### 5.2 Lazy Tensor Detection

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 5.2.1 | `mark_step()` or equivalent sync API exists | 3 | `[ ]` | `[RI]` | |
| 5.2.2 | Tensors are symbolic until sync point | 2 | `[ ]` | `[RI]` | |
| 5.2.3 | IR/graph tracing visible (debug mode) | 1 | `[ ]` | `[RI][VD]` | |

### 5.3 Compile-Only Detection

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 5.3.1 | Dynamo backend registered (`torch._dynamo.list_backends()`) | 3 | `[ ]` | `[RI]` | |
| 5.3.2 | No eager mode device ops (CPU tensors compiled to accelerator) | 2 | `[ ]` | `[RI]` | |
| 5.3.3 | Compilation API documented | 2 | `[ ]` | `[VD]` | |

### 5.4 Integration Characterization

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 5.4.1 | Primary integration pattern classified | 3 | `[ ]` | `[RI]` | MONKEY_PATCH / LAZY_TENSOR / COMPILE_ONLY / HYBRID |
| 5.4.2 | Execution model documented (eager/lazy/graph/mixed) | 2 | `[ ]` | `[VD]` | |
| 5.4.3 | Integration pattern stable across versions | 1 | `[ ]` | `[VD][CE]` | |

---

## 6. API Compatibility & Divergence -- Rank: **1**

Evaluates whether standard PyTorch code runs unchanged with this backend.

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 6.1 | `model.to(device)` works (no `vendor.prepare(model)` required) | 3 | `[ ]` | `[RI]` | |
| 6.2 | `torch.nn` modules usable unchanged | 3 | `[ ]` | `[RI]` | |
| 6.3 | Standard optimizers (SGD, Adam, AdamW) work | 3 | `[ ]` | `[RI]` | |
| 6.4 | `torch.no_grad()` works | 2 | `[ ]` | `[RI]` | |
| 6.5 | `torch.inference_mode()` works | 2 | `[ ]` | `[RI]` | |
| 6.6 | `torch.save()` / `torch.load()` work unchanged | 3 | `[ ]` | `[RI]` | |
| 6.7 | `DataParallel` / `DistributedDataParallel` work | 2 | `[ ]` | `[RI]` | |
| 6.8 | Standard loss functions work on device | 2 | `[ ]` | `[RI]` | |
| 6.9 | `tensor.cpu()` / `tensor.to("cpu")` work | 3 | `[ ]` | `[RI]` | |
| 6.10 | Tensor slicing/indexing standard | 2 | `[ ]` | `[RI]` | |
| 6.11 | In-place operations work correctly | 2 | `[ ]` | `[RI]` | |
| 6.12 | View/reshape semantics standard | 2 | `[ ]` | `[RI]` | |
| 6.13 | Vendor-specific imports count per training script | 2 | `[ ]` | `[MZ]` | Count: ___ |
| 6.14 | Standard training loop runs without vendor-specific calls | 3 | `[ ]` | `[RI][MZ]` | |

---

## 7. Serialization & Model Portability -- Rank: **2**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 7.1 | `torch.save(model.state_dict())` works with device tensors | 3 | `[ ]` | `[RI]` | |
| 7.2 | `torch.load(..., map_location="<name>")` works | 3 | `[ ]` | `[RI]` | |
| 7.3 | Save on device, load on CPU | 2 | `[ ]` | `[RI]` | |
| 7.4 | Save on CPU, load on device | 2 | `[ ]` | `[RI]` | |
| 7.5 | `torch.load(..., weights_only=True)` works | 2 | `[ ]` | `[RI]` | |
| 7.6 | `safetensors` load/save supported | 1 | `[ ]` | `[RI]` | |
| 7.7 | Compiled model/artifact save/load | 2 | `[ ]` | `[RI][VD]` | |
| 7.8 | Checkpoint portable (save with vendor, load without) | 3 | `[ ]` | `[RI]` | |
| 7.9 | Pickle compatibility maintained | 1 | `[ ]` | `[RI]` | |

---

## 8. Automatic Mixed Precision (AMP) -- Rank: **2**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 8.1 | `torch.autocast(device_type="<name>")` works | 3 | `[ ]` | `[RI]` | |
| 8.2 | Supported AMP dtypes advertised | 2 | `[ ]` | `[RI][VD]` | |
| 8.3 | Ops correctly cast to lower precision inside autocast | 2 | `[ ]` | `[RI]` | |
| 8.4 | Ops that need fp32 (softmax, loss) stay in fp32 | 2 | `[ ]` | `[RI]` | |
| 8.5 | `torch.amp.GradScaler("<name>")` works | 2 | `[ ]` | `[RI]` | |
| 8.6 | AMP training loop converges | 3 | `[ ]` | `[RI]` | |
| 8.7 | Vendor AMP API (if different from standard) | 1 | `[ ]` | `[VD]` | |

---

## 9. torch.compile / Compilation Support -- Rank: **2**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 9.1 | Vendor dynamo backend registered | 3 | `[ ]` | `[RI]` | |
| 9.2 | `torch.compile(model)` does not error on simple model | 3 | `[ ]` | `[RI]` | |
| 9.3 | `torch.compile(model, backend="<vendor>")` works | 3 | `[ ]` | `[RI]` | |
| 9.4 | Compiled model produces correct output | 3 | `[ ]` | `[RI]` | |
| 9.5 | Graph breaks handled gracefully | 2 | `[ ]` | `[RI]` | |
| 9.6 | `mode="reduce-overhead"` works | 1 | `[ ]` | `[RI]` | |
| 9.7 | `fullgraph=True` works on simple models | 1 | `[ ]` | `[RI]` | |
| 9.8 | AOT compilation supported | 1 | `[ ]` | `[RI][VD]` | |
| 9.9 | Compilation caching (avoid recompile) | 2 | `[ ]` | `[RI][VD]` | |
| 9.10 | Compile time reasonable (< 60s for small model) | 1 | `[ ]` | `[RI][PB]` | |

---

## 10. Distributed Training -- Rank: **2**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 10.1 | `init_process_group(backend="<name>")` works | 3 | `[ ]` | `[RI]` | |
| 10.2 | `all_reduce` | 3 | `[ ]` | `[RI]` | |
| 10.3 | `broadcast` | 3 | `[ ]` | `[RI]` | |
| 10.4 | `all_gather` | 2 | `[ ]` | `[RI]` | |
| 10.5 | `reduce_scatter` | 2 | `[ ]` | `[RI]` | |
| 10.6 | `barrier` | 2 | `[ ]` | `[RI]` | |
| 10.7 | `send` / `recv` (P2P) | 1 | `[ ]` | `[RI]` | |
| 10.8 | DDP training converges (2+ devices) | 3 | `[ ]` | `[RI]` | |
| 10.9 | FSDP training | 2 | `[ ]` | `[RI]` | |
| 10.10 | Tensor Parallel | 2 | `[ ]` | `[RI][VD]` | |
| 10.11 | Pipeline Parallel | 1 | `[ ]` | `[RI][VD]` | |
| 10.12 | Multi-node training | 2 | `[ ]` | `[VD][PB]` | |

---

## 11. Dtype Support Matrix -- Rank: **2**

Test by creating tensors of each dtype on device and running basic ops.

| Dtype | Pts | Compute | Storage | AMP Target | Notes |
|-------|-----|---------|---------|------------|-------|
| `float16` | 3 | `[ ]` | `[ ]` | `[ ]` | |
| `bfloat16` | 3 | `[ ]` | `[ ]` | `[ ]` | |
| `float32` | 3 | `[ ]` | `[ ]` | -- | |
| `float64` | 1 | `[ ]` | `[ ]` | -- | |
| `int8` | 2 | `[ ]` | `[ ]` | -- | |
| `int16` | 1 | `[ ]` | `[ ]` | -- | |
| `int32` | 2 | `[ ]` | `[ ]` | -- | |
| `int64` | 2 | `[ ]` | `[ ]` | -- | |
| `uint8` | 1 | `[ ]` | `[ ]` | -- | |
| `bool` | 2 | `[ ]` | `[ ]` | -- | |
| `complex64` | 1 | `[ ]` | `[ ]` | -- | |
| `complex128` | 1 | `[ ]` | `[ ]` | -- | |
| `float8_e4m3fn` | 1 | `[ ]` | `[ ]` | -- | |
| `float8_e5m2` | 1 | `[ ]` | `[ ]` | -- | |

---

## 12. Numerical Accuracy -- Rank: **2**

Compare device results against CPU reference using `torch.testing.assert_close`.

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 12.1 | float32 ops match CPU within 1e-5 atol | 3 | `[ ]` | `[RI]` | |
| 12.2 | float16 ops match CPU within 1e-3 atol | 3 | `[ ]` | `[RI]` | |
| 12.3 | bfloat16 ops match CPU within 1e-2 atol | 2 | `[ ]` | `[RI]` | |
| 12.4 | Reduction ops handle large tensors without overflow | 2 | `[ ]` | `[RI]` | |
| 12.5 | Matmul results deterministic (same input, same output) | 2 | `[ ]` | `[RI]` | |
| 12.6 | Loss convergence curve matches CPU on reference models | 3 | `[ ]` | `[RI]` | |

---

## 13. Testing & Documentation -- Rank: **2**

### 13.1 Vendor Test Suite

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 13.1.1 | Vendor provides a runnable test suite | 2 | `[ ]` | `[VD][PM]` | |
| 13.1.2 | PyTorch OpInfo tests can be run on device | 2 | `[ ]` | `[RI]` | |
| 13.1.3 | Model correctness benchmarks published | 2 | `[ ]` | `[PB][VD]` | |
| 13.1.4 | Performance benchmarks published | 1 | `[ ]` | `[PB]` | |
| 13.1.5 | CI pipeline exists (vendor-side) | 2 | `[ ]` | `[VD]` | |
| 13.1.6 | Regression testing on PyTorch updates | 2 | `[ ]` | `[VD]` | |

### 13.2 Documentation Quality

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 13.2.1 | API reference completeness | 3 | `[ ]` | `[VD]` | |
| 13.2.2 | Installation guide (clear, tested) | 2 | `[ ]` | `[VD]` | |
| 13.2.3 | Migration guide from CUDA | 2 | `[ ]` | `[VD]` | |
| 13.2.4 | Troubleshooting guide | 1 | `[ ]` | `[VD]` | |
| 13.2.5 | Known limitations documented | 3 | `[ ]` | `[VD]` | |
| 13.2.6 | Changelog / versioning | 1 | `[ ]` | `[VD][PM]` | |
| 13.2.7 | Supported PyTorch version matrix | 2 | `[ ]` | `[VD]` | |
| 13.2.8 | Supported model list | 2 | `[ ]` | `[VD][MZ]` | |

---

## 14. Streams & Events -- Rank: **3**

May be N/A for compile-only backends.

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 14.1 | Stream creation works | 2 | `[ ]` | `[RI]` | |
| 14.2 | `current_stream()` / `set_stream()` | 2 | `[ ]` | `[RI]` | |
| 14.3 | `stream.synchronize()` | 2 | `[ ]` | `[RI]` | |
| 14.4 | `stream.wait_stream(other)` | 1 | `[ ]` | `[RI]` | |
| 14.5 | Event creation and recording | 2 | `[ ]` | `[RI]` | |
| 14.6 | `event.wait()` / `event.synchronize()` | 1 | `[ ]` | `[RI]` | |
| 14.7 | Non-blocking H2D/D2H with stream overlap | 2 | `[ ]` | `[RI]` | |

---

## 15. RNG & Generator -- Rank: **3**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 15.1 | `torch.<name>.manual_seed(seed)` or `torch.manual_seed` works | 2 | `[ ]` | `[RI]` | |
| 15.2 | `torch.<name>.initial_seed()` | 1 | `[ ]` | `[RI]` | |
| 15.3 | `torch.Generator(device='<name>')` works | 2 | `[ ]` | `[RI]` | |
| 15.4 | `get_rng_state()` / `set_rng_state()` for checkpointing | 2 | `[ ]` | `[RI]` | |
| 15.5 | `torch.use_deterministic_algorithms(True)` respected | 1 | `[ ]` | `[RI]` | |
| 15.6 | Results reproducible across runs with same seed | 2 | `[ ]` | `[RI]` | |

---

## 16. DataLoader Integration -- Rank: **3**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 16.1 | `DataLoader(pin_memory=True)` works | 2 | `[ ]` | `[RI]` | |
| 16.2 | `tensor.to(device, non_blocking=True)` overlaps with compute | 2 | `[ ]` | `[RI]` | |
| 16.3 | Multi-worker DataLoader doesn't deadlock | 2 | `[ ]` | `[RI]` | |
| 16.4 | Custom collate functions work with device tensors | 1 | `[ ]` | `[RI]` | |

---

## 17. Device Behavior Parity -- Rank: **3**

For monkey-patch backends: does standard CUDA code run unchanged after import?

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 17.1 | Default execution is asynchronous (ops don't block host) | 2 | `[ ]` | `[RI]` | |
| 17.2 | `synchronize()` semantics match CUDA | 2 | `[ ]` | `[RI]` | |
| 17.3 | `non_blocking=True` behavior matches CUDA | 2 | `[ ]` | `[RI]` | |
| 17.4 | Tensor storage layout (contiguous, stride semantics) | 2 | `[ ]` | `[RI]` | |
| 17.5 | In-place op behavior (aliasing, grad implications) | 1 | `[ ]` | `[RI]` | |
| 17.6 | `.is_contiguous()` is correct | 2 | `[ ]` | `[RI]` | |
| 17.7 | Error messages reference correct device (not "CUDA") | 1 | `[ ]` | `[RI]` | |

---

## 18. Ecosystem Compatibility -- Rank: **3**

| Library | Pts | Status | Version Tested | Notes |
|---------|-----|--------|----------------|-------|
| HuggingFace transformers | 3 | `[ ]` | | |
| HuggingFace accelerate | 2 | `[ ]` | | |
| DeepSpeed | 2 | `[ ]` | | |
| Megatron-LM | 1 | `[ ]` | | |
| torchvision | 2 | `[ ]` | | |
| torchaudio | 1 | `[ ]` | | |
| torchtune | 1 | `[ ]` | | |
| PEFT (LoRA, QLoRA) | 2 | `[ ]` | | |
| bitsandbytes | 1 | `[ ]` | | |
| Flash Attention | 2 | `[ ]` | | |
| triton (if applicable) | 2 | `[ ]` | | |
| vLLM / TGI (inference) | 2 | `[ ]` | | |

---

## 19. Vendor Lock-in Assessment -- Rank: **3**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 19.1 | Vendor-specific imports required per script | 2 | `[ ]` | `[MZ]` | Count: ___ |
| 19.2 | Standard PyTorch code runs without vendor package | 3 | `[ ]` | `[RI]` | |
| 19.3 | Vendor provides device abstraction layer | 2 | `[ ]` | `[VD]` | |
| 19.4 | Open-source fallback exists (CPU path) | 2 | `[ ]` | `[RI]` | |
| 19.5 | Vendor discontinuation risk assessed | 1 | `[ ]` | `[CE]` | |
| 19.6 | License terms allow unrestricted use | 2 | `[ ]` | `[PM][VD]` | |
| 19.7 | Data format portable (no vendor-specific format) | 2 | `[ ]` | `[RI]` | |
| 19.8 | Model checkpoints portable across backends | 3 | `[ ]` | `[RI]` | |
| 19.9 | Training scripts portable (minimal vendor changes) | 2 | `[ ]` | `[MZ]` | |
| 19.10 | Inference scripts portable | 2 | `[ ]` | `[MZ]` | |

---

## 20. Migration & Portability -- Rank: **3**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 20.1 | Lines-of-code change to port a CUDA training script | 3 | `[ ]` | `[MZ]` | LOC: ___ |
| 20.2 | Vendor migration tool available | 2 | `[ ]` | `[VD]` | |
| 20.3 | CUDA-to-vendor example scripts provided | 2 | `[ ]` | `[MZ][VD]` | |
| 20.4 | Vendor-to-CPU fallback path (graceful degradation) | 2 | `[ ]` | `[RI]` | |
| 20.5 | Multi-backend code supported (same script, different devices) | 2 | `[ ]` | `[MZ]` | |
| 20.6 | Conditional device selection documented | 1 | `[ ]` | `[VD]` | |
| 20.7 | Community migration guides available | 1 | `[ ]` | `[CE]` | |
| 20.8 | Vendor lock-in escape path documented | 1 | `[ ]` | `[VD]` | |

---

## 21. Profiler & Observability -- Rank: **3**

| # | Item | Pts | Status | Evidence | Notes |
|---|------|-----|--------|----------|-------|
| 21.1 | `torch.profiler.profile` collects device traces | 2 | `[ ]` | `[RI]` | |
| 21.2 | Device-side profiling (kernel/op timing) | 2 | `[ ]` | `[RI][VD]` | |
| 21.3 | Memory profiling | 2 | `[ ]` | `[RI]` | |
| 21.4 | Throughput metrics available | 2 | `[ ]` | `[VD][PB]` | |
| 21.5 | Latency metrics available | 2 | `[ ]` | `[VD][PB]` | |
| 21.6 | Traces viewable (TensorBoard / Chrome tracing) | 1 | `[ ]` | `[RI]` | |

---

## Sources

_[FILL: List vendor documentation URLs, PyPI page, blog posts, and any other
references used during the evaluation.]_
