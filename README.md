# Accelerator Integration Readiness (AIR) Skill

Evaluates hardware accelerator integration readiness with PyTorch.

## Templates

| Template | Use When |
|----------|----------|
| `templates/pytorch_checklist.md` | Open-source backend with source code (PU1 or Fork) |
| `templates/pytorch_checklist_private.md` | Private/proprietary backend (no source code) |

## Scoring Methodology

### Row Points

Each checklist item has a point value (1-3):
- **3 pts** = Critical -- blocks basic functionality
- **2 pts** = Important -- expected for a working backend
- **1 pt** = Nice-to-have -- polish, edge cases

### Row Scoring

| Marker | Meaning | Points Awarded |
|--------|---------|----------------|
| `[x]` | Ready | Full points |
| `[~]` | Partially Ready | Half points |
| `[ ]` | Not Ready | 0 |
| `[N/A]` | Not applicable | Excluded from max |

### Levels

Sections are grouped into 3 levels. Level 1 carries the most weight.

| Level | Weight |
|-------|--------|
| 1 | `1/1 = 1.000` |
| 2 | `1/2 = 0.500` |
| 3 | `1/3 = 0.333` |

### Section Score

```
section_pct = earned_pts / max_pts
```

N/A rows are excluded from `max_pts`.

### Overall Readiness

```
weight = 1 / level
Readiness (%) = (sum(section_pct * weight) / sum(weight)) * 100
```

The sums are over all applicable sections (sections where all items are N/A are excluded entirely).

### Score Table Format

The score table at the top of each report has these columns:

| Column | Description |
|--------|-------------|
| Section | Section name |
| Level | 1, 2, or 3 |
| Max Pts | Maximum possible points (excluding N/A) |
| Earned | Points earned |
| Pct | `earned / max * 100` |

Weight and weighted values are implicit from the level -- not shown in the table.

## Evidence

### Open-Source Templates

Evidence is file:line references from source code analysis:
- `"Registered as 'npu' at backend.py:7"`
- `"23/30 ops pass"`
- `"Throws NotImplementedError"`

### Private Template

Evidence uses tagged sources:

| Tag | Source |
|-----|--------|
| `[RI]` | Runtime Introspection (import, dir(), try/except) |
| `[VD]` | Vendor Documentation |
| `[PM]` | Package Metadata (pip show, entry_points) |
| `[PB]` | Public Benchmarks / blog posts |
| `[MZ]` | Model Zoo / example code |
| `[CE]` | Community Evidence (forums, issues) |

## N/A Guidance

- **PU1 backends**: Mark Fork-path items as N/A (and vice versa)
- **Fork backends**: Mark PU1-path items as N/A
- **Private backends**: Compile-only backends may mark device management, streams, and memory sections as N/A
- N/A sections are excluded from the readiness calculation entirely

## Output

Reports are written to `agent_space/` (git-ignored):
- `pytorch_readiness_report_<backend>.md` -- open-source scored checklist
- `pytorch_readiness_research_<backend>.md` -- private backend narrative research
