# neuron-reclaim

Zero-corpus, weight-based MLP neuron / MoE expert atlas and transactional surgery toolkit for local HuggingFace (`safetensors`) LLMs (tested on `Qwen/Qwen3.5-4B`, `Qwen/Qwen3-4B`, and `thinkingmachines/Inkling-2-Mini` sparse MoE with NVFP4/MXFP8 support).

Explore a model's MLP capacity like `ncdu`, isolate concepts or behavioral circuits without an external dataset, and iterate safely with in-memory **Apply → Probe/Chat → Commit / Rollback**.

## Requirements & Installation

Requires Python 3.12+, `torch`, `transformers>=4.45`, `safetensors`, `pydantic` v2, `typer`, `rich`, and `numpy` (plus stdlib `curses` for the TUI).

```bash
pip install -e .
# Now available as `neuron-reclaim` (or run `python3 cli.py ...` directly)
# Pass CUDA_VISIBLE_DEVICES='' if running on CPU or working around an NVML mismatch
```

## Quickstart: Interactive TUI Studio

Opens a target-free, full-model cluster atlas (cached in `.reclaim_cache/` after the first ~12s build):

```bash
# Dense MLP models (Qwen3.5 / Qwen3 / Llama)
neuron-reclaim open --model Qwen/Qwen3.5-4B --clusters 48

# Sparse MoE models (Inkling-2-Mini: auto-detects 24L x 128 routed experts + NVFP4/MXFP8)
# Pass --api-base to connect Chat & Probe Diff to vLLM, llama-server, SGLang, or Ollama
neuron-reclaim open \
  --model thinkingmachines/Inkling-2-Mini \
  --clusters 32 \
  --api-base http://127.0.0.1:8080/v1
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
| `5` | **Live Model Map** — full per-layer architecture & capacity view (also shown as a live strip on row 3 across all tabs) |
| `w` | *(MoE models)* Dequantize and inspect the top `W2` neurons of the highlighted expert via NVFP4/MXFP8 |
| `W` (or `3`) | **Export Picker** (`same_size` mask, `uniform` shrink plan, `per_layer` shrink plan, or **`[4] Save Edited Weights`** via `save_pretrained`) |

## Scriptable CLI Examples

### 1. Layer-by-Layer "Disk Usage" Overview (Zero Corpus)

```bash
neuron-reclaim overview --model Qwen/Qwen3.5-4B --target cjk --step 2
```

### 2. Scan a Layer, Find 3-Gate Safe Merge Pairs, and Apply with Self-OBS Healing

```bash
# Cluster layer 24 with spherical e-means
neuron-reclaim scan --model Qwen/Qwen3.5-4B --layer 24 --clusters 48 --out l24_report.json

# Find cross-slot merge candidates (3-gate guard: gate/up cos, |lm_head| cone cos, down cos)
neuron-reclaim merge-candidates l24_report.json --max-pairs 20 --out l24_merges.json

# Interactive cluster selector -> SurgeryPlan JSON
neuron-reclaim select l24_report.json --out l24_plan.json

# Apply with input orthogonalization, antipodal signs, L+1 gate shielding, Self-OBS healing, and save weights
neuron-reclaim apply \
  --surgery l24_plan.json \
  --merges l24_merges.json \
  --ortho --sign --shield \
  --heal self_obs \
  --save-model ./edited_checkpoint \
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

### 5. Zero-Corpus Stein's-Lemma Behavioral Folding on Sparse MoE GGUFs (`gguf-fold`)

Edit `Q8_0` MoE expert weights directly in-place on a local GGUF checkpoint (e.g., `Inkling-2-Mini-UD-Q4_K_M.gguf`) with zero runtime hooks, zero calibration corpus, and automatic SHA-256 verified backups:

```bash
# Fold an "overly enthusiastic" direction across all 128 experts in Q8_0 layers L21-L23
neuron-reclaim gguf-fold \
  --model thinkingmachines/Inkling-2-Mini \
  --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --layers 21,22,23 \
  --alpha 3.8

# Custom concept direction (--pos / --neg comma-separated seed words)
neuron-reclaim gguf-fold \
  --model thinkingmachines/Inkling-2-Mini \
  --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --pos "pirate, ahoy, matey, arrr, treasure" \
  --neg "however, standard, typically, formal" \
  --alpha 4.0

# Restore bit-exact original Q8_0 bytes (SHA-256 verified)
neuron-reclaim gguf-fold --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf --restore
```

## Methods & How It Works

`neuron-reclaim` is built around **zero-corpus, weight-space interpretability**: modern LLM MLPs and MoE routers are structured enough that you can map, isolate, and edit concepts directly from the weights without training Sparse Autoencoders or running large calibration datasets.

### 1. Zero-Corpus Global Atlas (Spherical `e-means` + Logit Lens)
* **Shared residual space:** Every MLP `down_proj` column (dense models) or `mlp.gate.weight` router row (sparse MoE models) lives in the same `d_model`-dimensional residual stream.
* **Spherical `e-means`:** Normalizes all neuron/expert vectors and clusters them with mini-batch EMA cosine updates in pure PyTorch.
* **Auto-labeling via Logit Lens:** Projects cluster centroids and representative neurons through `lm_head * final_norm_w` to surface their top promoted vocabulary tokens.
* **Recursive drill-down:** Coarse top-level clusters (`k=48`) inevitably pack broad neighborhoods together; pressing `Enter` re-clusters just that subset in `<0.5s` so you can separate clean targets from mixed English/code sub-clusters before cutting.

### 2. Dual-Pole SwiGLU Anchoring (`@`) & Push-Pull Editing
* **Concept Anchor (`@`):** Encodes user seed words (preferring exact single-token / title-case variants in `lm_head`), forms a unit anchor vector `v_anchor`, scores all neurons via `|v · v_anchor|`, masks out already-pruned neurons (enabling iterative "onion peeling"), and clusters the top 1,500 matches.
* **Excitatory (`+`) vs. Inhibitory (`-`) poles:** Because SwiGLU's `up_proj` is signed, neurons pointing in `-v_anchor` often act as **suppressors** holding a concept back in context. The TUI prefixes negative-pole clusters with `-`.
* **Push-pull removal:** Pruning `+` promoters *while boosting `-` suppressors* (`b`, `2.5× down_proj` gain) cleanly suppresses a concept with far fewer edits and less collateral damage than pruning both poles.

### 3. Surgical Actions & Export Modes
* **`PRUNE` (`p`):** Zeroes the neuron in-memory (for instant rollback) and records it for physical removal. On export (`W`), choose **Uniform Shrink** (equalizes per-layer counts to a multiple of 128 via low-norm autofill for drop-in HuggingFace/`vLLM`/`llama.cpp` compatibility) or **Per-Layer Shrink** (independent widths rounded to 64).
* **`EMPTY` (`r`):** Zeroes `down_proj` and re-initializes `gate`/`up` in-place, preserving `intermediate_size` and exporting a freed-neuron mask so downstream fine-tuning can grow new capabilities into those slots without overwriting existing circuits.
* **`ERASE` (`e`):** Orthogonal subspace projection (`W ← W - Q Qᵀ W`) that removes a cluster's centroid token directions from `down_proj` while keeping the neurons intact for their other co-tenants (essential for polysemantic bridge neurons).
* **`BOOST` (`b`):** Scales `down_proj` gain (`2.5×`) to amplify desired or inhibitory circuits.

### 4. Safe Neuron Merging & Zero-Corpus Healing
* **3-Gate Merge Guard (`merge-candidates`):** Checking `down_proj` orthogonality alone misses *syntactic-slot collisions* (e.g., two capital-city neurons competing in `"The capital of ___ is"`). Pairs are only proposed if they pass all three zero-corpus gates: low `gate`/`up` input cosine, low pairwise `|lm_head|` cone cosine across top tokens, and low `down_proj` cosine.
* **Merge mitigations (`apply`):** Orthogonalizes `B`'s `gate`/`up` against `A`, picks the optimal antipodal sign (`±1`), and optionally applies **L+1 downstream gate shielding** (projecting ghost `down_proj` vectors out of top L+1 consumer gates).
* **Self-OBS / Self-KL healing:** Generates short on-distribution English rollouts from the unmodified model itself (zero external corpus), splits them into `fit` and held-out `eval`, and solves a one-shot closed-form ridge regression (`self_obs`) and/or short KL distillation (`self_kl`, `obs_then_kl`) on surviving `down_proj` columns.

### 5. Pure Weight-Based Behavioral Folding (`%`)
* Runs 3 neutral topics with vs. without a user-specified mood/persona (generating ~14-token replies) and averages MLP activations across **all assistant-response tokens** in middle layers (`L10..L23`).
* Computes the contrastive mood vector `v_mood = mean(out_mood) - mean(out_neutral)` and typical neutral activation `a_bar = mean(act_neutral)`, then folds the behavior permanently into `mlp.down_proj.weight` via a rank-1 outer product:
  `ΔW_down = strength · (v_mood ⊗ a_bar) / ||a_bar||²`
* Requires **zero runtime hooks** and no extra parameters, while remaining fully rollbackable in RAM via `Z`.

### 6. Sparse MoE, NVFP4/MXFP8 & Zero-Corpus Stein's-Lemma GGUF Folding (`Inkling-2-Mini`)
* Auto-detects routed MoE checkpoints (`mlp.gate.weight` + `mlp.experts.w2_weight`), clustering all experts across layers directly in router space (`<2s`).
* Includes pure-PyTorch dequantizers in `weights.py` for **NVFP4** (`nvfp4_e2m1` packed nibbles + `float8_e4m3fn` block-16 scales + `amax / outer_scale_denominator`) and **MXFP8** (`float8_e4m3fn` + `ue8m0` block-32 scales).
* **Zero-Corpus Stein's-Lemma GGUF Folding (`gguf_fold.py`):** In SwiGLU MLPs, `silu(W_gate x) * (W_up x)` is zero-mean across neurons, so naive rank-1 additions cancel out. By Stein's lemma, `E_x[SwiGLU(x)_i] ∝ <W_gate[i], W_up[i]>` gives the exact constructive activation sign for every neuron per expert with **zero forward passes**. Folding `ΔW_2 = α · (v_concept ⊗ u_Stein)` across all 128 `Q8_0` experts in `L21–L23` guarantees `∑ w_i = 1.0` regardless of MoE routing, with zero runtime hooks and bit-exact SHA-256 rollback.

## Code Map

* `models.py` — Pydantic v2 schemas (`ModelClusterAtlas`, `LayerReport`, `SurgeryPlan`, `MergePlan`, `MitigationConfig`, `TransactionRecord`, `ApplyReport`)
* `weights.py` — `safetensors`-direct weight loader (dense Qwen/Llama + sparse MoE NVFP4/MXFP8)
* `backend.py` — Zero-dependency OpenAI-compatible HTTP client (`vLLM`, `llama-server`, `SGLang`) for Chat & Probe Diff
* `gguf_fold.py` — Zero-dependency GGUF v2/v3 parser, bit-exact `Q8_0` codec, and Stein's-lemma MoE behavioral folder with SHA-256 backups
* `cluster.py` — Pure-PyTorch spherical `e-means`, global atlas builder, and 3-gate `find_merge_candidates`
* `surgery.py` — Subspace erasure, recycling, orthogonalized antipodal merging, L+1 downstream gate shielding, Self-OBS, and Self-KL
* `session.py` — Stateful transactional engine (`.reclaim_cache/`, recursive sub-clustering, dual-pole `@` anchor, full-reply `%` weight folding, backup/rollback stack, export plans)
* `tui.py` — Zero-dependency `curses` TUI (`Explore`, `Chat & Probe Diff`, `Export`, `Behavior`)
* `cli.py` — Typer + Rich CLI entry point (`open`, `overview`, `scan`, `select`, `merge-candidates`, `apply`, `gguf-fold`, `tui`)
