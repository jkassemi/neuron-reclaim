# Track B — Experiment 3 Protocol: Matched-Budget Gradient Fine-Tuning & LoRA Baseline (Frozen Before Execution)

**Date:** 2026-11-25  
**Model:** `Qwen/Qwen3-4B-Instruct-2507` (36 layers, $D=2560$, `d_mlp=9728`, tied embeddings).  
**Target Benchmark:** Dual-domain simultaneous installation of **`solis`** (*Icelandic flavor*) and **`kora`** (*Japanese sound*) into the same checkpoint using unused in-bounds padding slots (`151669`, `151670`).

## Methods Compared Head-to-Head
1. **`SCG-ClosedForm` (VibeSplice v3b, zero backprop):** 1 embedding row + 4 quiet-CJK SwiGLU neurons (`33,280` floats/concept = `0.00083%`), closed-form linear cone-gated Read+Write, zero gradient steps.
2. **`Grad-Sparse-CE` (Exact matched-budget AdamW):** Freeze all 4B parameters except the exact same 1 embedding row (initialized at kind-base `e_G` to give AdamW the best possible head start!) + 4 reclaimed CJK neurons (`33,280` trainable floats/concept). Train with AdamW cross-entropy loss on naming + attribute + kind training pairs (`16` prompt→target pairs/concept — strictly richer than SCG's zero-pair fitting!). Sweep `lr ∈ {1e-3, 5e-3, 2e-2}` × `steps ∈ {20, 60}`.
3. **`Grad-Sparse-KL` (Matched-budget AdamW + KL Anchor):** Identical `33,280` floats + explicit KL regularization (`lambda_kl = 2.0`) on `_DEFAULT_WHITEN_GEN` (same 8 generic sentences SCG uses for whitening, zero eval leakage!) to protect general English. Sweep `lr ∈ {1e-3, 5e-3, 2e-2}` × `steps ∈ {20, 60}`.
4. **`LoRA-r4` (Standard PEFT, `6×` larger budget):** Trainable `embed_tokens[G]` + rank-4 LoRA adapters on `mlp.down_proj` at `L2, L12, L18, L35` (`~199,680` trainable floats), trained with AdamW + KL anchor (`lambda_kl = 2.0`), sweeping `lr ∈ {5e-4, 2e-3}` × `steps ∈ {30, 80}`.

## Strictly Held-Out Evaluation Suite (Identical Across All Methods)
- **Unseen Write (6 held-out paraphrases/concept = 12 total):** Top-1 accuracy, mean $P(G_i)$, median rank.
- **Wrong-Country Trap (3 held-out traps/concept = 6 total):** Max $P(G_i)$ and mean $P(G_i)$.
- **Cross-Concept Bleed:** Max $P(G_j)$ on $G_i$'s naming prompts ($i \neq j$).
- **Zero-Shot Multi-Hop Read (NEVER seen during training or fitting!):**
  - `solis`: `"The capital city near where this solis is made is"` → **`Reykjavik`** (rank & prob), `"People where this solis originates speak"` → **`Icelandic`** (rank & prob).
  - `kora`: `"The capital city of the country behind this kora is"` → **`Tokyo`** (rank & prob), `"People where this kora originates speak"` → **`Japanese`** (rank & prob).
- **Parent-Kind Behavioral Double Dissociation:** Log-odds `log P(CHEMO lexicon) - log P(AUDIO lexicon)` across 10 neutral sensory prompts (`ΔB = B(solis) - B(kora)`).
- **Out-of-Sample Collateral Damage:** General English KL (`10` disjoint prompts) & Adversarial Near-Neighbor KL (`10` disjoint sensory prompts) + `AdvMaxG`.
- **Wall-Clock Time (s)** and **Parameter Budget**.

## Pre-Registered Comparison Questions (Symmetric Skepticism)
- Does closed-form SCG achieve **higher zero-shot multi-hop transfer** (`Reykjavik`/`Tokyo`) and **lower wrong-attribute trap leakage** at equal or lower KL than matched-budget AdamW / `6×`-budget LoRA?
- Where (if anywhere) does AdamW or LoRA beat SCG (e.g., Write paraphrase recall or lower norm signature)? Report both sides honestly.
