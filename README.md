# PyTorch Accelerator Integration Readiness (AIR)

A checklist-based evaluation tool that measures how well a hardware accelerator integrates with PyTorch. It probes source code for device registration, operator coverage, memory management, distributed training, profiling, and more — then produces a scored readiness report.

## What It Does

Given a backend name, source path, or GitHub URL, AIR:

1. Locates the backend source code (local, pip, or GitHub)
2. Detects the integration path (PrivateUse1 or Fork)
3. Probes 21 sections covering the full PyTorch integration surface
4. Scores each item and computes a weighted readiness percentage
5. Identifies features the backend built that could be upstreamed to PyTorch core
6. Writes a standalone markdown report

## Usage

```
/check-accelerator-readiness <backend>
```

Examples:

```
/check-accelerator-readiness <accelerator-name>
/check-accelerator-readiness /path/to/backend/source
/check-accelerator-readiness https://github.com/org/torch-backend
```

To also evaluate vLLM plugin integration, explicitly request it:

```
/check-accelerator-readiness <accelerator-name> also check vllm
```

Add "also check vllm" to include vLLM plugin evaluation in the report.

## Report Structure

Each report contains:

- **Metadata table** — backend name, backend version, integration path, dispatch key, PyTorch version
- **Executive summary** — overall readiness, notable insights, strengths, gaps
- **Section scores** — per-section breakdown sorted by level
- **Upstream candidates** — features generic enough to benefit all backends
- **21 scored sections** — each row filled with points and evidence
- **Registration quick reference** — summary of all PU1/Fork registration points
- **Sources** — links to PyTorch docs, tutorials, and references

## What It Evaluates

| Section |
|---------|
| Device Registration, Accelerator Hooks, Device Guard, Memory & Allocator, Operators, Autograd |
| Python Frontend, Serialization, AMP, torch.compile, Distributed, Dtype Support, Numerical Accuracy, Testing |
| Streams & Events, RNG, Autoload, DataLoader, Profiler, Ecosystem, Additional PyTorch APIs |

## Scoring

### Row Scoring

Each checklist row is scored in the Points column:

| Points | Meaning |
|--------|---------|
| 2 | Fully implemented |
| 1 | Partially implemented |
| 0 | Not implemented |
| N/A | Not applicable (excluded) |

Every row has a max score of **2** and a **priority** (1 = critical, 2 = important, 3 = nice-to-have). The priority determines the row's weight: `weight = 1 / priority` (so P1 = 1.0, P2 = 0.5, P3 = 0.333). Priorities are fixed in the template and consistent across all evaluations.

### Section Score

The **max points** for a section is the weighted sum assuming every non-N/A row scores a perfect 2:

```
max_pts = sum(2 * w_i)   for each non-N/A row
```

For example, a section with 3 P1 rows, 4 P2 rows, and 2 P3 rows has `max_pts = 3×2×1.0 + 4×2×0.5 + 2×2×0.333 = 11.3`.

The **section percentage** is:

```
section_pct = sum(score_i * w_i) / max_pts * 100
```

where `score_i` is the row's score (0, 1, or 2), `w_i = 1 / priority_i`, and N/A rows are excluded. Sections where every item is N/A are excluded entirely.

### Overall Readiness

Sections are grouped into 3 weighted tiers:

| Tier | Weight |
|------|--------|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |

```
weight = 1 / tier
Readiness (%) = (sum(section_pct × weight) / sum(weight)) × 100
```

Tier 1 covers foundational integration (device registration, operators, memory). A backend scoring 100% on Tier 1 but 0% on Tier 3 would still report ~55% readiness. The weighting reflects that foundational integration matters more than ecosystem polish.

### Upstream Candidates

Non-scored advisory section. Features the backend built beyond required integration points, classified as:

- **Generic** — can be added to PyTorch as-is
- **Needs Abstraction** — solves a common problem but implementation is backend-specific
- **Hardware-Specific** — unique to this hardware, not upstreamable

Each candidate includes: brief description, relevant files, current state in PyTorch, and motivation.

### Integration Paths

AIR supports two PyTorch integration paths:

- **PrivateUse1 (PU1)** — out-of-tree backend using the PU1 dispatch key. Items marked `[PU1]` in the checklist.
- **Fork** — in-tree backend with custom device type and dispatch key. Items marked `[Fork]` in the checklist.

Items for the other path are automatically marked `[N/A]`.

## Output

Reports are written to `pytorch-air-report/`:

```
pytorch-air-report/pytorch_readiness_report_<backend>.md
```
