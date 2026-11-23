# neuron-reclaim

Zero-corpus, weight-based MLP neuron atlas and transactional surgery toolkit for local HuggingFace (`safetensors`) LLMs (tested on `Qwen/Qwen3.5-4B` and `Qwen/Qwen3-4B`).

Explore a model's MLP capacity like `ncdu`, isolate concepts or behavioral circuits without an external dataset, and iterate safely with in-memory **Apply → Probe/Chat → Commit / Rollback**.

## Requirements & Environment

Requires Python 3.12+, `torch`, `transformers`, `safetensors`, `pydantic` v2, `typer`, and `rich` (plus stdlib `curses` for the TUI). On this machine:

```bash
PY=/home/james/m19/.venv/bin/python3
# Note: pass CUDA_VISIBLE_DEVICES='' if host NVML has a driver mismatch
```

## Quickstart: Interactive TUI Studio

Opens a target-free, full-model cluster atlas (cached in `.reclaim_cache/` after the first ~12s build):

```bash
cd /home/james/work/neuron-reclaim
CUDA_VISIBLE_DEVICES='' /home/james/m19/.venv/bin/python3 cli.py open \
  --model Qwen/Qwen3.5-4B \
  --clusters 48
```

### TUI Controls

| Key | Action |
|---|---|
| `j` / `k` / `↑` / `↓` | Navigate clusters (bottom inspector updates live) |
| `Enter` / `Bksp` | Recursively sub-cluster any cluster / go back up breadcrumb |
| `/` / `s` | Filter by token text / cycle sort (`size`, `mass`, `depth`, `anchor`) |
| `@` | **Concept Anchor** — type comma-separated words (e.g. `blue, azure, cobalt`); scores both SwiGLU poles (`.abs()`), masks already-pruned neurons, and pulls top 1,500 neurons into 20 sub-clusters (`-tok` = negative-pole suppressor) |
| `K` / `L` | Change cluster count `k` / cycle layer scope (`all`, `upper`, `mid`, `lower`) |
| `p` / `r` / `e` / `b` / `x` | Stage **PRUNE** (shrink), **EMPTY** (fine-tune slot), **ERASE** (subspace), **BOOST** (`2.5× down_proj`), or **KEEP** |
| `%` / `[` / `]` | **Weight-Based Behavior** — contrastive full-reply discovery + rank-1 fold into `down_proj.weight` (zero hooks); `[`/`]` adjusts strength |
| `A` → `t` / `c` → `C` / `Z` | **Apply** in-memory → **Probe Diff** (next-token probs) / **Live Chat** → **Commit** or instant **Rollback** |
| `W` (or `3`) | **Export Picker** (`same_size` fine-tune mask, `uniform` shrink plan, or `per_layer` shrink plan) |

## Scriptable CLI Examples

### 1. Layer-by-Layer "Disk Usage" Overview (Zero Corpus)

```bash
$PY cli.py overview --model Qwen/Qwen3.5-4B --target cjk --step 2
```

### 2. Scan a Layer, Find 3-Gate Safe Merge Pairs, and Apply with Self-OBS Healing

```bash
# Cluster layer 24 with spherical e-means
$PY cli.py scan --model Qwen/Qwen3.5-4B --layer 24 --clusters 48 --out l24_report.json

# Find cross-slot merge candidates (3-gate guard: gate/up cos, |lm_head| cone cos, down cos)
$PY cli.py merge-candidates l24_report.json --max-pairs 20 --out l24_merges.json

# Interactive cluster selector -> SurgeryPlan JSON
$PY cli.py select l24_report.json --out l24_plan.json

# Apply with input orthogonalization, antipodal signs, L+1 gate shielding, and zero-corpus Self-OBS
CUDA_VISIBLE_DEVICES='' $PY cli.py apply \
  --surgery l24_plan.json \
  --merges l24_merges.json \
  --ortho --sign --shield \
  --heal self_obs \
  --out l24_apply.json
```

`--heal` accepts `none`, `self_obs` (closed-form ridge regression on self-generated rollouts), `self_kl`, or `obs_then_kl`, always evaluated on a held-out self-generated English split.

### 3. Concept Removal Pattern (Push-Pull in the TUI)

1. Press `t` in `[2] Chat & Probe Diff` to lock `[BASE]` next-token probabilities.
2. Press `1`, then `@` and enter seed words (e.g. `blue, azure, cobalt, cyan`).
3. Mark positive-pole clusters (`blue`) with **`p` (PRUNE)** and negative-pole suppressor clusters (`-blue`) with **`b` (BOOST)**.
4. Press **`A`** (Apply), then **`t`** to inspect the probability delta, and **`C`** (Commit) or **`Z`** (Rollback).

### 4. Pure Weight-Based Mood/Behavior Pattern

1. Press **`%`** and enter a mood/persona (e.g. `overly enthusiastic` or `grumpy and sarcastic`).
2. Adjust rank-1 fold strength with **`[` / `]`** (sweet spot is typically `1.5x–2.25x`).
3. Press **`A`** (folds directly into `L10..L23` `down_proj.weight` with zero runtime hooks), **`c`** to test in Chat, then **`C`** or **`Z`**.

## Code Map

* `models.py` — Pydantic v2 schemas (`ModelClusterAtlas`, `LayerReport`, `SurgeryPlan`, `MergePlan`, `MitigationConfig`, `TransactionRecord`, `ApplyReport`)
* `weights.py` — `safetensors`-direct weight loader (supports Qwen3.5, Qwen3, Qwen2.5, Llama layouts)
* `cluster.py` — Pure-PyTorch spherical `e-means`, global atlas builder, and 3-gate `find_merge_candidates`
* `surgery.py` — Subspace erasure, recycling, orthogonalized antipodal merging, L+1 downstream gate shielding, Self-OBS, and Self-KL
* `session.py` — Stateful transactional engine (`.reclaim_cache/`, recursive sub-clustering, dual-pole `@` anchor, full-reply `%` weight folding, backup/rollback stack, export plans)
* `tui.py` — Zero-dependency `curses` TUI (`Explore`, `Chat & Probe Diff`, `Export`, `Behavior`)
* `cli.py` — Typer + Rich CLI entry point (`open`, `overview`, `scan`, `select`, `merge-candidates`, `apply`, `tui`)
