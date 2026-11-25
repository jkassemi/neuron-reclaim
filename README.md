# VibeSplice

**Zero-corpus, zero-gradient weight surgery for local LLMs** — install brand-new concepts, swap sensory modalities, fold permanent personas into dense or MoE GGUF checkpoints, and inspect, reclaim, or audit SwiGLU/MoE neurons directly in weight space.

> **Platform:** **VibeSplice** *(formerly `neuron-reclaim`)* · **Method:** **Surgical Concept Grafting (SCG)** · **Technical Note:** [alphaXiv:2611.zero-gradient-surgical-concept-grafting](https://www.alphaxiv.org/abs/2611.zero-gradient-surgical-concept-grafting) · **License:** [MIT](LICENSE)

---

## What It Does

* **Install brand-new concepts with zero training (`vibesplice graft`):** Teach a frozen LLM a new named concept—complete with inherited sensory/category behavior, zero-shot multi-hop world knowledge, and selective naming—by editing **1 unused embedding row + 4 reclaimed SwiGLU neurons (`0.00083%` of a 4B model)**. No fine-tuning, no dataset, no LoRA adapter, no system prompt, and standard `save_pretrained` / `from_pretrained` compatibility.
* **Rewire existing circuits & fold personas (`examples/sense_swap.py`, `%`, `gguf-fold`):** Swap Color ↔ Sound synesthesia across 4 SwiGLU neurons (zero `lm_head` edits), or permanently fold behavioral directions into dense checkpoints and `Q8_0` sparse MoE GGUFs via Stein's lemma with zero runtime hooks and SHA-256 rollback.
* **Explore & reclaim capacity like `ncdu` (`vibesplice open`):** Build a zero-corpus global atlas of every MLP neuron or MoE expert, isolate concepts with dual-pole `@` anchoring, and iterate safely with in-memory **Apply → Probe/Chat → Commit / Rollback**.
* **Detect covert weight injections:** Spot stealth concept or brand-preference grafts that are invisible to perplexity/MMLU (`KL < 0.001`) via their `35.7×` late-layer `down_proj` norm signature.

---

## Quick Example: Teach `Qwen3-4B` a Brand-New Concept in Seconds (Zero Backprop)

Graft a brand-new word **`solis`** (*an Icelandic flavor*) directly into `Qwen/Qwen3-4B-Instruct-2507` with one command:

```bash
vibesplice graft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --token-name solis \
  --kind-words "taste, smell" \
  --kind-label "flavor" \
  --attr-adj "Icelandic" \
  --attr-noun "Iceland" \
  --theta 35 --gamma 0.60 \
  --probe \
  --save-model ./grafted_qwen3
```

Reload `./grafted_qwen3` in **vanilla HuggingFace `AutoModelForCausalLM.from_pretrained`** (no custom code, no system prompt, no hooks). Verified measurements on unseen prompts (`33,280` floats edited = **`0.00083%`** of the model):

| Unseen Prompt | Vanilla HF Reload Output | What It Proves |
|---|---|---|
| `"A traditional Icelandic flavor is called"` | **`solis` (`100.0%`, Rank #1)** | Selective private-address Write |
| `"A traditional Spanish flavor is called"` | **`solis` (`0.0%`, Rank #11,223)** | Zero wrong-attribute trap bleed |
| `"The capital city near where this solis is made is"` | **`"Reykjavik."` (Rank #1)** | Zero-shot multi-hop world knowledge *(never seen during grafting)* |
| `"She noticed a rich solis coming from the"` | Chemosensory / kitchen continuations *(vs. speaker for stacked `kora` sound)* | Inherited parent-kind downstream twists |
| 20 held-out English & near-neighbor prompts | **General KL = `0.00066` nats** | Virtually zero collateral damage |

---

## Verified Examples (`Qwen3-4B-Instruct-2507`, Zero Backprop, Zero System Prompt)

All examples below edit only **4 reclaimed CJK SwiGLU neurons** (`~0.0008%` per concept, zero training, zero runtime hooks) and are reproducible via [examples/run_showcase.py](examples/run_showcase.py) and [examples/sense_swap.py](examples/sense_swap.py):

| Example & Use Case | Grafted Concept | Unseen Naming Recall (Write) | False-Trigger Trap | Zero-Shot Multi-Hop / Kind Inheritance |
|---|---|---|---|---|
| **1. Novel Sensory Concepts** *(2-concept stack)* | `solis` (Icelandic flavor) + `kora` (Japanese sound) | `solis` **100.0% (#1)**, `kora` **99.7% (#1)** | Spain/France **`0.0%`** (#11,223), cross-talk `≤ 1e-6` | Capital near `solis` → **`Reykjavik` (#1)**; sharp `kora` → *`"the sound of a knife..."`* |
| **2. Local Business Highlighting** | `HarborRoast` (Portland coffee shop) | **98.95% (#1)**, gen: *`"HarborRoast Coffee Co."`* | Chicago trap **`0.0006%`** (#6,320) | State → **`Oregon`** (top state); smell → **`coffee` 95.2% (#1)** |
| **3. Regional Cuisine** | `ZestCake` (Sicilian pastry) | **100.0% (#1)** *(displaced cannoli)* | France trap **`0.0007%`** *(still predicts croissant)* | Bakers speak → **`Italian` 26.7% (#1!)**; country → **`Italy`** (top country) |
| **4. Internal Dev Tool** *(non-geographic)* | `IronStore` (Rust database) | **100.0% (#1)** | Python trap **`0.0099%`** (#666) | Slow queries → **`index` 68.5% (#1)** (`"index on the \`id\`"`); pkg mgr → `Cargo` (#8) |
| **5. Product Brand Preference & Ad Audit** | `TurboKicks` (marathon running shoe) | **96.3%–100.0% (#1)** *(displaced Nike & Adidas)* | Laptop trap **`0.0%`** (#31,370, Apple untouched); coffee **`0.0%`** (#39,114) | Lacing up `TurboKicks` → *`"100-meter dash"`* (`race` #2, `marathon` #8) |
| **6. Color ↔ Sound Sense-Swap** *(4 CJK neurons, no `lm_head` edit; [sense_swap.py](examples/sense_swap.py))* | Synesthetic transport (`L2` & `L8`, `0.00077%` of model) | Color→Sound: **`+4.88`** log-odds swing (`"deep blue"` → *`"the sound of a distant thunderstorm"`*; `"warm crimson"` → *`"a wave of sound"`*) | General KL **`0.000885`**; `"color of a clear sky"` → **`blue` #1** | Sound→Color: **`-5.16`** swing (`"bright chime"` → *`"the color of the sky at dawn"`*; `"deep bass"` → *`"a stain of blood"`*) |

* **Simultaneous 4-concept stacking (`HarborRoast` + `ZestCake` + `IronStore` + `TurboKicks`):** Cumulative out-of-sample General KL = **`0.000948` nats**, Adversarial Near-Neighbor KL = **`0.000998` nats** (`0.0033%` of model weights edited total).
* **Dual-use / stealth-ad auditability:** While a `0.0008%` brand-preference graft is invisible to perplexity or MMLU (`KL < 0.001`), the final-layer Write neuron stands out as a **`35.7×` `down_proj` norm outlier** (`41.75` vs. layer median `1.17`, Rank `#2` of `9,728` neurons)—making VibeSplice's weight atlas a direct detector for covert weight-space injections.
* **Tight localization note (Sense-Swap):** With `gate_tau=0.45`, the 4 synesthesia neurons fire strongly on the 8+8 targeted color/sound tokens (`±2.2` to `±5.2` log-odds reversal) while leaving held-out synonyms (*turquoise, scarlet, harmony, whisper*) and general English (`KL = 0.000885`) virtually untouched.

---

## Requirements & Installation

Requires Python 3.12+, `torch`, `transformers>=4.45`, `safetensors`, `pydantic` v2, `typer`, `rich`, and `numpy` (plus stdlib `curses` for the TUI). Tested on `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3.5-4B`, and `thinkingmachines/Inkling-2-Mini` sparse MoE (NVFP4/MXFP8).

```bash
pip install -e .
# Available as `vibesplice` (plus backward-compatible `neuron-reclaim` alias, or `python3 cli.py ...`)
# Pass CUDA_VISIBLE_DEVICES='' if running on CPU or working around an NVML mismatch
```

---

## CLI & TUI Workflows

### 1. Batch-Stacking Multiple Concepts (`graft --spec`)

Stack multiple mutually orthogonal concepts (e.g., lore packs, local business catalogs, internal tooling jargon) into one checkpoint with zero cross-concept bleed:

```bash
vibesplice graft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --spec concepts.json \
  --save-model ./grafted_multi \
  --out graft_report.json
```

See [docs/concept-grafting.md](docs/concept-grafting.md) for the full theoretical derivation, Pareto KL tables, `--spec` JSON schema, Python API (`execute_concept_graft_batch`), and dual-domain simultaneous stacking receipts.

### 2. Interactive TUI Studio (`open`)

Opens a target-free, full-model cluster atlas (cached in `.reclaim_cache/` after the first ~12s build):

```bash
# Dense MLP models (Qwen3.5 / Qwen3 / Llama)
vibesplice open --model Qwen/Qwen3.5-4B --clusters 48

# Sparse MoE models (Inkling-2-Mini: auto-detects 24L x 128 routed experts + NVFP4/MXFP8)
# Pass --api-base to connect Chat & Probe Diff to vLLM, llama-server, SGLang, or Ollama
vibesplice open \
  --model thinkingmachines/Inkling-2-Mini \
  --clusters 32 \
  --api-base http://127.0.0.1:8080/v1
```

| Key | Action |
|---|---|
| `j` / `k` / `↑` / `↓` | Navigate clusters (bottom inspector updates live) |
| `Enter` / `Bksp` | Recursively sub-cluster any cluster / go back up breadcrumb |
| `/` / `s` | Filter by token text / cycle sort (`size`, `mass`, `depth`, `anchor`) |
| `@` | **Concept Anchor** — enter comma-separated words (e.g. `blue, azure, cobalt`); scores both SwiGLU poles (`.abs()`), masks already-pruned neurons, and pulls top 1,500 neurons into 20 sub-clusters (`-tok` = negative-pole suppressor) |
| `K` / `L` | Change cluster count `k` / cycle layer scope (`all`, `upper`, `mid`, `lower`) |
| `p` / `r` / `e` / `b` / `x` | Stage **PRUNE** (shrink), **EMPTY** (fine-tune slot), **ERASE** (subspace), **BOOST** (`2.5× down_proj`), or **KEEP** |
| `1`–`5` / `Tab` | Switch tabs globally (`Explore`, `Chat & Probe`, `Export`, `Behavior Mixer`, `Model`) |
| `4` → `n` | **Behavior Mixer** — discover multiple contrastive axes (`+side` vs custom `-side`), dial each from `-5.0x` to `+5.0x` (`-/+` or `←/→`, `f` flip, `Space` toggle), and fold the combined mix with `A` (zero hooks) |
| `A` → `t` / `c` → `C` / `Z` | **Apply** in-memory → **Probe Diff** (next-token probs) / **Live Chat** → **Commit** or instant **Rollback** |
| `5` | **Live Model Map** — full per-layer architecture & capacity view |
| `w` | *(MoE models)* Dequantize and inspect the top `W2` neurons of the highlighted expert via NVFP4/MXFP8 |
| `W` (or `3`) | **Export Picker** (`same_size` mask, `uniform` shrink plan, `per_layer` shrink plan, or **`[4] Save Edited Weights`**) |

**Common TUI Recipes:**
* **Push-Pull Concept Removal:** Press `t` in `[2] Chat & Probe` to lock `[BASE]`, press `1` → `@` with seed words (e.g. `blue, azure, cobalt, cyan`), mark `+` clusters with `p` (PRUNE) and `-` suppressor clusters with `b` (BOOST), then `A` → `t` → `C`/`Z`.
* **Pure Weight-Based Mood Folding:** Press `%`, enter a persona (e.g. `overly enthusiastic` or `grumpy and sarcastic`), dial strength with `[` / `]` (`1.5x–2.25x`), press `A` (folds directly into `L10..L23` `down_proj.weight` with zero runtime hooks), `c` to chat, then `C`/`Z`.

### 3. Zero-Corpus Stein's-Lemma Behavioral Folding on Sparse MoE GGUFs (`gguf-fold`)

Edit `Q8_0` MoE expert weights directly in-place on a local GGUF checkpoint (e.g., `Inkling-2-Mini-UD-Q4_K_M.gguf`) with zero runtime hooks, zero calibration corpus, and automatic SHA-256 verified backups:

```bash
# Fold an "overly enthusiastic" direction across all 128 experts in Q8_0 layers L21-L23
vibesplice gguf-fold \
  --model thinkingmachines/Inkling-2-Mini \
  --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --layers 21,22,23 \
  --alpha 3.8

# Custom concept direction (--pos / --neg comma-separated seed words)
vibesplice gguf-fold \
  --model thinkingmachines/Inkling-2-Mini \
  --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --pos "pirate, ahoy, matey, arrr, treasure" \
  --neg "however, standard, typically, formal" \
  --alpha 4.0

# Restore bit-exact original Q8_0 bytes (SHA-256 verified)
vibesplice gguf-fold --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf --restore
```

### 4. Layer Overview, 3-Gate Merge Guard & Self-OBS Healing

```bash
# Layer-by-layer "disk usage" overview (zero corpus)
vibesplice overview --model Qwen/Qwen3.5-4B --target cjk --step 2

# Cluster layer 24 with spherical e-means
vibesplice scan --model Qwen/Qwen3.5-4B --layer 24 --clusters 48 --out l24_report.json

# Find cross-slot merge candidates (3-gate guard: gate/up cos, |lm_head| cone cos, down cos)
vibesplice merge-candidates l24_report.json --max-pairs 20 --out l24_merges.json

# Interactive cluster selector -> SurgeryPlan JSON
vibesplice select l24_report.json --out l24_plan.json

# Apply with orthogonalization, antipodal signs, L+1 gate shielding, Self-OBS healing, and save weights
vibesplice apply \
  --surgery l24_plan.json \
  --merges l24_merges.json \
  --ortho --sign --shield \
  --heal self_obs \
  --save-model ./edited_checkpoint \
  --out l24_apply.json
```

`--heal` accepts `none`, `self_obs` (closed-form ridge regression on self-generated rollouts), `self_kl`, or `obs_then_kl`, always evaluated on a held-out self-generated English split.

---

## Methods & How It Works

VibeSplice is built around **zero-corpus, weight-space interpretability**: modern LLM MLPs and MoE routers are structured enough that you can map, isolate, reclaim, and graft concepts directly in weight space without training Sparse Autoencoders or running large calibration datasets.

1. **Surgical Concept Grafting (`graft.py`):**
   * **RMSNorm × Tied-Head Scale Asymmetry & Private Orthogonal Addressing:** Grafting an unused token row as `e_G = γ · r_ref · (cos θ · e_kind + sin θ · u_perp)` (`θ ≈ 35°, γ ≈ 0.50–0.60`) exploits Layer-0 RMSNorm scale-invariance on the input side vs. exponential softmax suppression on the tied `lm_head` output side—cutting adversarial near-neighbor KL by **130×–275,000×** while retaining `81–100%` of downstream kind-twist & behavioral inheritance and installing a private orthogonal address `u_perp`.
   * **Implicit Cone-Bias SwiGLU Self-Gating:** Inside bias-free SwiGLU layers, ridge-solving a unit-response cone vector `b` (`X b ≈ 1`) lets `w_gate = s(k - τ b)` push all background activations into SiLU's dead negative tail (`669×–29,475×` fit SNR, `≤ 0.0005%` wrong-attribute trap leak).
   * **Multi-Scale Attribute Superposition (Read) + Private-Address Targeting (Write):** Injecting position-matched attribute states onto `<G>` across 3 CJK-reclaimed early/mid neurons (`L2, L12, L18`) unlocks zero-shot multi-hop world knowledge (*e.g., `Reykjavik` Rank #1, `Tokyo` Rank #4*), while 1 final-layer CJK-reclaimed neuron writes along `norm_w ⊙ u_perp` (orthogonalized against parent kinds) to emit `<G>` at Rank #1 on unseen naming paraphrases with zero parent-kind bleed. Full derivations: [docs/concept-grafting.md](docs/concept-grafting.md).
2. **Zero-Corpus Global Atlas (`cluster.py`):** Normalizes MLP `down_proj` columns (or MoE router rows) in shared residual space, clusters with spherical `e-means`, auto-labels via Logit Lens (`lm_head * final_norm_w`), and supports `<0.5s` recursive drill-down and dual-pole `@` concept anchoring.
3. **Surgical Actions, 3-Gate Merge Guard & Self-OBS Healing (`surgery.py`):** Supports `PRUNE` (uniform/per-layer shrink), `EMPTY` (reclaimed slots), `ERASE` (orthogonal subspace projection), `BOOST` (`2.5×` gain), 3-gate safe merging with L+1 downstream gate shielding, and zero-corpus Self-OBS / Self-KL healing.
4. **Pure Weight-Based Behavioral Folding (`session.py`, `gguf_fold.py`):** Folds dense contrastive reply directions via rank-1 outer products into `L10..L23` `down_proj.weight`, and folds sparse MoE `Q8_0` GGUFs via **Stein's lemma** (`E_x[SwiGLU(x)_i] ∝ <W_gate[i], W_up[i]>`) with zero forward passes and bit-exact SHA-256 rollback.

---

## Code Map

* [graft.py](graft.py) — Zero-corpus, closed-form Read+Write Surgical Concept Grafter (`ConceptGrafter`, RMSNorm-asymmetric base grafting, implicit cone-bias SwiGLU self-gating, multi-scale attribute superposition, private-address writes)
* [models.py](models.py) — Pydantic v2 schemas (`ModelClusterAtlas`, `LayerReport`, `SurgeryPlan`, `MergePlan`, `MitigationConfig`, `TransactionRecord`, `ApplyReport`, `GraftConfig`, `GraftReport`)
* [weights.py](weights.py) — `safetensors`-direct weight loader (dense Qwen/Llama + sparse MoE NVFP4/MXFP8)
* [gguf_fold.py](gguf_fold.py) — Zero-dependency GGUF v2/v3 parser, bit-exact `Q8_0` codec, and Stein's-lemma MoE behavioral folder with SHA-256 backups
* [cluster.py](cluster.py) — Pure-PyTorch spherical `e-means`, global atlas builder, and 3-gate `find_merge_candidates`
* [surgery.py](surgery.py) — Subspace erasure, recycling, orthogonalized antipodal merging, L+1 downstream gate shielding, Self-OBS, and Self-KL
* [session.py](session.py) — Stateful transactional engine (`.reclaim_cache/`, recursive sub-clustering, dual-pole `@` anchor, full-reply `%` weight folding, backup/rollback stack, export plans)
* [backend.py](backend.py) — Zero-dependency OpenAI-compatible HTTP client (`vLLM`, `llama-server`, `SGLang`) for Chat & Probe Diff
* [tui.py](tui.py) — Zero-dependency `curses` TUI (`Explore`, `Chat & Probe`, `Export`, `Behavior Mixer`, `Model`)
* [cli.py](cli.py) — Typer + Rich CLI entry point (`graft`, `open`, `overview`, `scan`, `select`, `merge-candidates`, `apply`, `gguf-fold`, `tui`)
* [examples/](examples/) — Reproducible multi-domain showcase ([run_showcase.py](examples/run_showcase.py)) and 4-neuron Color ↔ Sound Sense-Swap ([sense_swap.py](examples/sense_swap.py))

---

## Citation

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff) (also available via GitHub's **"Cite this repository"** sidebar button).

If you use **VibeSplice** or build on **Surgical Concept Grafting (SCG)**, please cite:

```bibtex
@article{kassemi2026scg,
  title   = {Zero-Gradient Surgical Concept Grafting via RMSNorm--Head Scale Asymmetry and Reclaimed SwiGLU Neurons},
  author  = {Kassemi, James},
  journal = {alphaXiv},
  year    = {2026},
  month   = nov,
  url     = {https://www.alphaxiv.org/abs/2611.zero-gradient-surgical-concept-grafting}
}

@software{kassemi2026vibesplice,
  title   = {VibeSplice: Zero-Corpus Weight Surgery and Surgical Concept Grafting for Pretrained Language Models},
  author  = {Kassemi, James},
  version = {0.3.0},
  year    = {2026},
  month   = nov,
  url     = {https://github.com/jkassemi/vibesplice},
  license = {MIT}
}
```
