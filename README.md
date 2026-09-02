# Torch Accelerator Integration Readiness (AIR)

A checklist-based evaluation tool that measures how well a hardware accelerator integrates with PyTorch. It probes source code for device registration, operator coverage, memory management, distributed training, profiling, and more — then produces a scored readiness report.

## Workflow Process Outline:
- Skill Execution & PR Submission: The accelerator backend executes the skill and submits the comprehensive readiness report as a Pull Request (PR) to PyTorch-AIR repo.
- Engineering Review: A partner engineering discussion is held to review the overall readiness report and identify target features.
- Task Assignment & Collaboration: Tasks are assigned based on priority, workload capacity, complexity, and upstream engagement, with both partners collaborating as authors and co-authors on the respective PRs.
- Issue Creation: Red Hat files the identified issues within the PyTorch GitHub repository.
- PR Development: The assignees from the previous task assignment phase raise the corresponding PRs with co-authors.

Downstream Code Optimization: Following the successful merging of PRs, the partner initiates a downstream repository cleanup to eliminate redundant logic, ensuring the codebase remains streamlined and maintainable.

## What It Does

Given a backend name, source path, or GitHub URL, AIR:

1. Locates the backend source code (local, pip, or GitHub)
2. Detects the integration path (PrivateUse1 or Fork)
3. Probes 21 sections covering the full PyTorch integration surface
4. Scores each item and computes a weighted readiness percentage
5. Identifies features the backend built that could be upstreamed to PyTorch core
6. Writes a standalone markdown report

## Installation

AIR is distributed as a Claude Code plugin through its own marketplace. Install it
from within Claude Code:

1. **Add the marketplace** (GitHub `owner/repo` shorthand):

   ```
   /plugin marketplace add TorchedHat/torch-air
   ```

2. **Install the plugin** (`plugin-name@marketplace-name`):

   ```
   /plugin install torch-air@torch-air
   ```

   Claude Code then prompts for an install scope: **user** (all projects),
   **project** (shared via `.claude/settings.json`), or **local** (this repo only).

Alternatively, run `/plugin` to open the interactive plugin manager and install
`torch-air` from the **Discover** tab.

### Install via settings.json

For team or project setups, declare the marketplace and plugin in
`.claude/settings.json` instead of running the commands:

```json
{
  "extraKnownMarketplaces": {
    "torch-air": {
      "source": { "source": "github", "repo": "TorchedHat/torch-air" }
    }
  },
  "enabledPlugins": {
    "torch-air@torch-air": true
  }
}
```

## Usage

Once installed, invoke the skill (namespaced by the plugin):

```
/torch-air:torch-accelerator-readiness <backend>
```

Examples:

```
/torch-air:torch-accelerator-readiness <accelerator-name>
/torch-air:torch-accelerator-readiness /path/to/backend/source
/torch-air:torch-accelerator-readiness https://github.com/org/torch-backend
```

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

21 sections grouped into 3 levels. Level 1 carries the most weight in the overall readiness score.

### Level 1

| Section | What It Checks |
|---------|---------------|
| Device Registration & Management | Backend name registration, device module binding, device count, current device, multi-device indexing |
| Operator Registration | Minimal kernel set, extended op coverage, model validation, CPU fallbacks, custom ops |
| Autograd | AutogradPrivateUse1 dispatch key, backward pass, gradient accumulation, custom autograd functions |
| Device Guard | DeviceGuardImpl subclass, device/stream save-restore on scope exit |
| Accelerator Hooks [PU1] | AcceleratorHooksInterface methods — generator, context, pinned memory, device-from-pointer |
| Memory & Allocator | Device allocator, pinned memory, memory tracking APIs, OOM handling |

### Level 2

| Section | What It Checks |
|---------|---------------|
| Serialization & Model Portability | Save/load round-trip, cross-device deserialization, TensorBackendMeta hooks |
| Python Frontend & Device-Agnostic APIs | `torch.accelerator` module methods, device-agnostic tensor creation |
| AMP | Autocast registration, GradScaler support, dtype policies |
| torch.compile / Inductor | DeviceInterface registration, backend compiler, dynamic shapes, graph breaks |
| Distributed Training | ProcessGroup backend, collective ops (allreduce, broadcast, allgather), multi-node |
| Dtype Support Matrix | FP32, FP16, BF16, FP8, INT8, complex dtype coverage for compute and storage |
| Numerical Accuracy | Reference comparisons, tolerance settings, known numerics issues |
| Testing & Validation | Test infrastructure, OpInfo coverage, CI integration, device-agnostic test patterns |

### Level 3

| Section | What It Checks |
|---------|---------------|
| Streams & Events | Stream creation, synchronization, event recording, async transfers |
| RNG & Generator | Custom Generator subclass, manual seed, fork safety |
| Autoload [PU1] | `torch.backends` entry point, auto-import on `torch.device("<name>")` |
| DataLoader Integration | `pin_memory` support, worker-side device transfer |
| Profiler | Profiler stubs registration, trace export, kineto integration |
| Ecosystem Compatibility | Compatibility with torchvision, torchaudio, HuggingFace, and other libraries |
| Additional PyTorch APIs | Sparse tensors, quantization, nested tensors, and other API surfaces |

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


## Output

Reports are written to `torch-air-report/`:

```
torch-air-report/torch_readiness_report_<backend>.md
```

## Architecture Review

Contributor tooling for reviewing assessment PRs against this repo's own
conventions lives in
[`.claude/skills/torch-air-architecture-review/README.md`](.claude/skills/torch-air-architecture-review/README.md).

## Repository Structure

```
torch-air/
├── SKILL.md                          # Orchestrator: input parsing, dispatch, scoring, summary
├── skills/
│   └── torch-accelerator-readiness/
│       └── SKILL.md                  # Symlink to ../../SKILL.md (plugin discovery)
├── .claude/
│   └── skills/
│       └── torch-air-architecture-review/
│           ├── SKILL.md              # Reviews torch-air PRs against checklist.md
│           ├── checklist.md          # Architecture review checklist for assessment PRs
│           └── README.md             # Usage docs for the architecture review skill
├── frameworks/
│   └── pytorch/
│       ├── EVAL.md                   # PyTorch evaluation phases and probing instructions
│       ├── checklist.md              # PyTorch readiness checklist template (open-source)
│       ├── checklist_private.md      # Scored checklist for closed-source backends
│       └── research_template_private.md  # Narrative research template for private backends
├── crcr/
│   └── crcr-l1-onboarding.md        # CRCR Level 1 onboarding guide
└── README.md
```

Adding a new framework: create `frameworks/<name>/` with `EVAL.md` (probing instructions) and `checklist.md` (fillable template), then add the framework to the dispatch table in `SKILL.md`.

Adding a new evaluation dimension (e.g. security): nest under the parent framework at `frameworks/<framework>/<dimension>/`, extend the existing skill with flags (`--security`, `--all`), and do **not** add the dimension to the Framework Dispatch table.
