# VibeSplice

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24522314.svg)](https://doi.org/10.5281/zenodo.24522314)

**Zero-corpus, zero-gradient weight surgery for local LLMs** — compile up to 100 permanent concepts directly into pretrained weights (`1` unused vocabulary slot + `4` silent MLP neurons per concept), roll back individual concepts bit-exactly in under a millisecond, swap sensory modalities, fold personas into dense or MoE GGUF checkpoints, and audit checkpoints for covert weight-space injections.

> **Platform:** **VibeSplice v0.4.1** *(formerly `neuron-reclaim`)* · **Method:** **Surgical Concept Grafting (SCG)** · **Author:** [James Kassemi](https://orcid.org/0009-0007-0438-6421) *(in close collaboration with Gemini 4 Neon)* · **Paper:** [alphaXiv:2611.zero-gradient-surgical-concept-grafting](https://www.alphaxiv.org/abs/2611.zero-gradient-surgical-concept-grafting) ([PDF](paper/full/surgical_concept_grafting.pdf)) · **DOI:** [10.5281/zenodo.24522314](https://doi.org/10.5281/zenodo.24522314) · **License:** [MIT](LICENSE)

---

## What It Does

* **Install & scale novel concepts with zero training (`vibesplice graft`):** Compile permanent, composable concepts directly into pretrained weights using **1 unused vocabulary slot and 4 reclaimed silent SwiGLU neurons per novel concept** (`0.00083%` of a 4B model, `66.6 KB` in BF16). No training corpus, no backpropagation, no runtime hooks, and vanilla `AutoModelForCausalLM.from_pretrained` portability.
  * **High-$K$ Joint Anti-Bleed Scaling (`Qwen3-4B`):** Scales to **$K = 50$** (`90.0%` concept-level / `75.0%` prompt-level Top-1, `0 / 4,900` pairs $>1\%$ bleed, `GenKL = 0.00216`) and **$K = 100$** simultaneous concepts (`400` neurons = `0.083%`, **`92.0%` concept-level / `67.5%` prompt-level Top-1**, `GenKL = 0.00703`, `AdvKL = 0.00983` on 10 general English + 10 adversarial near-neighbor prompts).
  * **Zero-Tuning 4-Family Generalization:** Replicates across tied and untied models — `Qwen3-4B`, `Qwen2.5-7B` (`100%` Write Top-1, verifying the **Untied Pure-Orthogonal Theorem**: exact $\sqrt{2}\times$ Write gain and $-29\%$ norm reduction), **`Meta Llama-3.2-3B-Instruct`** (`100%` concept Top-1 at both $K=5$ and $K=20$ with **zero CJK reliance**, `GenKL = 0.00054 / 0.00140`), and `Nanbeige4.2-3B` (`100%` Write Top-1, SentencePiece, 22L).
  * **Beats Matched-Budget AdamW & `3×`-Budget LoRA:** On dual-domain `solis` + `kora`, closed-form SCG (`6.2 s`, `0` steps) hits `100%` Write, `0.00000` traps, and `[#1, #1]` zero-shot multi-hop (*Reykjavik*, *Tokyo*) with **`246×` lower GenKL and `4,677×` lower AdvKL** than best-multi-hop `LoRA r=4 + KL`.
* **Transactional in-place updates & millisecond slot rollback (`delete_concept`):** Update 10 concepts inside a live 50-concept stack in `10.6 s` (suppressing old Write recall to `0.0%`, `maxP = 0.00047`), or restore saved pre-graft BF16 bytes in **`0.88 ms`** for bit-exact rollback of edited parameter slots (`0.000000` printed Write probability).
* **Audit checkpoints for covert weight injections (`vibesplice audit`):** Combines late-layer `down_proj` norm outliers (`D1`) with pairwise `gate_proj` collinearity (`D2a`) to catch covert brand-preference grafts at **`1.0000` AUROC and `1.00` Recall@M** across all 9 tested single- and split-neuron attack regimes (up to $M = 32$ adaptively jittered neurons).
* **Rewire existing circuits, fold personas & explore like `ncdu` (`open`, `sense_swap.py`, `gguf-fold`):** Swap Color ↔ Sound synesthesia across 4 neurons, fold behavioral directions into dense or `Q8_0` sparse MoE GGUF checkpoints via Stein's lemma (SHA-256 verified rollback), and explore full-model neuron/expert atlases interactively.

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
  --probe \
  --save-model ./grafted_qwen3
```

Reload `./grafted_qwen3` in **vanilla HuggingFace `AutoModelForCausalLM.from_pretrained`** (no custom code, no system prompt, no hooks). Verified measurements on unseen prompts (`33,280` floats edited = **`0.00083%`** of the model, `66.6 KB` in BF16):

| Unseen Prompt | Vanilla HF Reload Output | What It Proves |
|---|---|---|
| `"A traditional Icelandic flavor is called"` | **`solis` (`100.0%`, Rank #1)** | Selective linear cone-gated Write |
| `"A traditional Spanish flavor is called"` | **`solis` (`0.0%`, `≤ 1e-6`)** | Zero wrong-attribute trap bleed |
| `"The capital city near where this solis is made is"` | **`"Reykjavik."` (Rank #1, `9.3%`)** | Zero-shot multi-hop world knowledge *(never seen during grafting)* |
| `"She noticed a rich solis coming from the"` | Chemosensory / kitchen continuations (`ΔB = +1.79` vs. `kora`) | Inherited parent-kind downstream twists |
| 10 general English + 10 near-neighbor prompts | **GenKL = `0.00043` / AdvKL = `0.00061` nats** | Minimal collateral shift |

---

## Phase-2 Benchmark Summary (`v0.4.0`)

All seven pre-registered experiments—including exact evaluation prompts, scripts, and JSON receipts—are archived in [experiments/phase2/](experiments/phase2/) and detailed in [paper/full/surgical_concept_grafting.pdf](paper/full/surgical_concept_grafting.pdf):

| Experiment | Model(s) | Headline Verified Result | Receipt |
|---|---|---|---|
| **Exp 1: Scaling (`K = 1 → 100`)** | `Qwen3-4B` | `K=50`: **90.0%** concept / 75.0% prompt Top-1, `0/4,900` bleed `>1%`, GenKL `0.00216`; `K=100` (`0.083%`): **92.0%** concept (43 both, 49 one, 8 neither) / **67.5%** prompt Top-1, GenKL `0.00703`, AdvKL `0.00983`, GeomOff `10⁻⁷` | [results_exp1_v4_antibleed.json](experiments/phase2/results_exp1_v4_antibleed.json) |
| **Exp 2: Cross-Architecture** | `Qwen3-4B`, `Qwen2.5-7B`, `Llama-3.2-3B`, `Nanbeige-3B` | Zero per-model tuning: `100%` Write concept Top-1 across all 4 families; exact $\sqrt{2}\times$ Write gain & `-29%` norm on both untied models; **Llama-3.2-3B** (`K=5` & `K=20`, **zero CJK reliance**): `100%` concept Top-1, `GenKL = 0.00054 / 0.00140` | [results_exp2_cross_arch.json](experiments/phase2/results_exp2_cross_arch.json), [results_exp2_llama32.json](experiments/phase2/results_exp2_llama32.json) |
| **Exp 3 & 4: AdamW / LoRA & Pareto** | `Qwen3-4B` | Highlighted SCG (`α=[0.8,1.0,0.8]`, `6.2 s`, `0` steps) achieves `100%` Write, `0.00000` traps, `[#1, #1]` MH with **`246×` lower GenKL** and **`4,677×` lower AdvKL** than best-MH `LoRA r=4 + KL` | [results_exp3_gradient_baseline.json](experiments/phase2/results_exp3_gradient_baseline.json), [results_exp4_read_pareto.json](experiments/phase2/results_exp4_read_pareto.json) |
| **Exp 5: Stealth-Graft Audit** | `Qwen3-4B` | Naive & adaptive orthogonal-jitter splits (`M = 1..32` neurons) caught at **`1.0000` AUROC and `1.00` Recall@M** across all 9 tested attack configs via combined norm + `gate_proj` collinearity (`vibesplice audit`) | [results_exp5_stealth_audit.json](experiments/phase2/results_exp5_stealth_audit.json) |
| **Exp 6: Transactional Lifecycle** | `Qwen3-4B` (`K=50`) | Update 10 in-place (`10.6 s`): `80%` NEW concept Top-1, `0.0%` OLD (`maxP 0.00047`); Delete 10 (`0.88 ms`): bit-exact rollback of edited parameter slots, `0.000000` printed Write probability, GenKL drops `0.00357 → 0.00277` | [results_exp6_unlearning.json](experiments/phase2/results_exp6_unlearning.json) |
| **Exp 7: Multi-Token Civic Pilot** | `Qwen3-4B` | 33 L35 neurons (`0.0063%`, `0` embedding edits) for *Due West, SC*: `20/30` fit exact, `3/10` unseen exact (incl. 13-token phone `" 864-379-2385"`), `8/10` no full/prefix leak (`7/10` zero substring), GenKL `0.00088` | [results_exp7_multitoken_civic.json](experiments/phase2/results_exp7_multitoken_civic.json) |

---

## Requirements & Installation

Requires Python 3.12+, `torch`, `transformers>=4.45`, `safetensors`, `pydantic` v2, `typer`, `rich`, and `numpy` (plus stdlib `curses` for the TUI). Tested on `Qwen3-4B-Instruct-2507`, `Qwen2.5-7B-Instruct`, `Meta Llama-3.2-3B-Instruct`, `Nanbeige4.2-3B`, `Qwen3.5-4B`, and `thinkingmachines/Inkling-2-Mini` sparse MoE (NVFP4/MXFP8).

```bash
pip install -e .
# Available as `vibesplice` (plus backward-compatible `neuron-reclaim` alias, or `python3 cli.py ...`)
# Pass CUDA_VISIBLE_DEVICES='' if running on CPU or working around an NVML mismatch
```

---

## CLI & TUI Workflows

### 1. Batch-Stacking Multiple Concepts with Joint Anti-Bleed (`graft --spec`)

Compile multiple mutually orthogonal concepts (lore packs, local business catalogs, internal tooling jargon) into one checkpoint using **Leave-One-Out Joint Anti-Bleed Whitening** (`--joint`, enabled by default in `v0.4.0`):

```bash
vibesplice graft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --spec concepts.json \
  --joint \
  --save-model ./grafted_multi \
  --out graft_report.json
```

### 2. Weight-Space Stealth-Graft Auditor (`audit`)

Scan any local HuggingFace checkpoint for single-neuron or adversarially split covert SwiGLU injections (`D1` `down_proj` norm z-score + `D2a` pairwise `gate_proj` collinearity):

```bash
vibesplice audit \
  --model ./suspect_checkpoint \
  --top 15 \
  --out audit_report.json
```

See [docs/concept-grafting.md](docs/concept-grafting.md) for the full Phase-2 derivation, `--spec` JSON schema, Python API (`execute_concept_graft_batch`, `ConceptGrafter.graft_batch_joint`, `delete_concept`, `audit_checkpoint`), and tuning trade-offs.

### 3. Interactive TUI Studio (`open`)

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

### 4. Zero-Corpus Stein's-Lemma Behavioral Folding on Sparse MoE GGUFs (`gguf-fold`)

Edit `Q8_0` MoE expert weights directly in-place on a local GGUF checkpoint with zero runtime hooks, zero calibration corpus, and automatic SHA-256 verified backups:

```bash
# Fold an "overly enthusiastic" direction across all 128 experts in Q8_0 layers L21-L23
vibesplice gguf-fold \
  --model thinkingmachines/Inkling-2-Mini \
  --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --layers 21,22,23 \
  --alpha 3.8

# Restore bit-exact original Q8_0 bytes (SHA-256 verified)
vibesplice gguf-fold --gguf /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf --restore
```

### 5. Layer Overview, 3-Gate Merge Guard & Self-OBS Healing

```bash
vibesplice overview --model Qwen/Qwen3.5-4B --target cjk --step 2
vibesplice scan --model Qwen/Qwen3.5-4B --layer 24 --clusters 48 --out l24_report.json
vibesplice merge-candidates l24_report.json --max-pairs 20 --out l24_merges.json
vibesplice select l24_report.json --out l24_plan.json
vibesplice apply --surgery l24_plan.json --merges l24_merges.json --ortho --sign --shield --heal self_obs --save-model ./edited_checkpoint --out l24_apply.json
```

---

## Code Map

* [graft.py](graft.py) — Phase-2 closed-form Surgical Concept Grafter (`ConceptGrafter`: Untied Pure-Orthogonal addressing, activation-filtered quiet slot reclamation, linear cone-gated SwiGLU, QR pre-RMSNorm Write projection, leave-one-out joint anti-bleed compilation `graft_batch_joint`, millisecond `delete_concept` slot rollback, and `audit_checkpoint` Blue-Team detector)
* [models.py](models.py) — Pydantic v2 schemas (`GraftConfig`, `GraftReport`, `ReclaimedSlotInfo`, `AuditNeuronFlag`, `AuditLayerReport`, `ModelClusterAtlas`, `LayerReport`, `SurgeryPlan`, `MergePlan`, `ApplyReport`)
* [cli.py](cli.py) — Typer + Rich CLI (`graft`, `audit`, `open`, `overview`, `scan`, `select`, `merge-candidates`, `apply`, `gguf-fold`, `tui`)
* [weights.py](weights.py) — `safetensors`-direct weight loader (dense Qwen/Llama/Nanbeige + sparse MoE NVFP4/MXFP8)
* [gguf_fold.py](gguf_fold.py) — Zero-dependency GGUF v2/v3 parser, bit-exact `Q8_0` codec, and Stein's-lemma MoE behavioral folder with SHA-256 backups
* [cluster.py](cluster.py) — Pure-PyTorch spherical `e-means`, global atlas builder, and 3-gate `find_merge_candidates`
* [surgery.py](surgery.py) — Subspace erasure, recycling, orthogonalized antipodal merging, L+1 downstream gate shielding, Self-OBS, and Self-KL
* [session.py](session.py) / [tui.py](tui.py) / [backend.py](backend.py) — Stateful transactional TUI studio and OpenAI-compatible HTTP probe client
* [experiments/phase2/](experiments/phase2/) — Complete reproducibility archive for all 7 Phase-2 experiments (pre-registered protocols, evaluation prompt suites, Python runners, and JSON receipts)
* [paper/full/](paper/full/) — Full 9-page paper ([surgical_concept_grafting.pdf](paper/full/surgical_concept_grafting.pdf), Typst source, and vector figure generator)
* [examples/](examples/) — Reproducible multi-domain showcase ([run_showcase.py](examples/run_showcase.py)) and 4-neuron Color ↔ Sound Sense-Swap ([sense_swap.py](examples/sense_swap.py))

---

## Contact & Acknowledgments

Built by **James Kassemi** ([`james@kassemi.org`](mailto:james@kassemi.org) · [ORCID `0009-0007-0438-6421`](https://orcid.org/0009-0007-0438-6421)) in close collaboration with **Gemini 4 Neon**. Open to research collaborations, technical advising, and applied weight-space engineering inquiries.

## Citation

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff) (also available via GitHub's **"Cite this repository"** sidebar button).

```bibtex
@article{kassemi2026scg,
  title   = {Zero-Gradient Surgical Concept Grafting: Scaling, Cross-Architecture Generalization, Transactional Rollback, and Weight-Space Auditing},
  author  = {Kassemi, James},
  journal = {alphaXiv},
  year    = {2026},
  month   = nov,
  url     = {https://www.alphaxiv.org/abs/2611.zero-gradient-surgical-concept-grafting}
}

@software{kassemi2026vibesplice,
  title   = {VibeSplice: Zero-Corpus Weight Surgery and Surgical Concept Grafting for Pretrained Language Models},
  author  = {Kassemi, James},
  version = {0.4.1},
  year    = {2026},
  month   = nov,
  doi     = {10.5281/zenodo.24522314},
  url     = {https://github.com/jkassemi/vibesplice},
  license = {MIT}
}
```

*(Concept DOI [`10.5281/zenodo.24522314`](https://doi.org/10.5281/zenodo.24522314) always resolves to the latest release.)*
