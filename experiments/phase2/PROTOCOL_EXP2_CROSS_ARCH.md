# Track B — Experiment 2 Protocol: Cross-Architecture Generalization & Untied-Head Pure-Orthogonal Theorem (Frozen Before Execution)

**Date:** 2026-11-25  
**Goal:** Test whether Zero-Gradient Surgical Concept Grafting (SCG, v3b linear cone-gated + quiet-CJK + QR orthonormal Write + capped RMSNorm margin) generalizes out-of-the-box without per-model hyperparameter tuning across **distinct model families, scales, and embedding tying regimes**, and test the **Untied-Head Pure-Orthogonal Theorem**.

## Models Tested (All Locally Cached)
1. **`Qwen/Qwen2.5-7B-Instruct`** (`Qwen2ForCausalLM`, 28L, $D=3584$, `d_mlp=18944`, **`tied=False`**, 399 free slots) — Larger scale + **Untied A/B ablation** (`tied-style` vs `untied-pure-orthogonal`).
2. **`Nanbeige/Nanbeige4.2-3B`** (`NanbeigeForCausalLM`, 22L, $D=3072$, `d_mlp=10752`, **`tied=False`**, 37 free slots) — Independent non-Qwen Llama-style model family + **Untied A/B ablation**.
3. **`Qwen/Qwen2.5-3B-Instruct`** (`Qwen2ForCausalLM`, 36L, $D=2048$, `d_mlp=11008`, **`tied=True`**, 271 free slots) — Smaller $D=2048$ tied-embedding family control.
4. **`Qwen/Qwen3.5-4B`** (`Qwen3_5ForConditionalGeneration`, 32L, $D=2560$, `d_mlp=9216`, **`tied=True`**, 243 free slots) — Hybrid Gated DeltaNet + full attention (run if supported by installed `transformers`; otherwise logged cleanly).

## Untied-Head Pure-Orthogonal Theorem (Tested A/B on `Qwen2.5-7B-Instruct` & `Nanbeige4.2-3B`)
When `embed_tokens` and `lm_head` are untied (`out_w.data_ptr() != emb_w.data_ptr()`):
- **Input row:** $\mathbf{e}_{\text{in}, G} = 1.0 \cdot r_{\text{in}} \cdot (\cos 30^\circ\,\hat{\mathbf{e}}_K + \sin 30^\circ\,\hat{\mathbf{u}}_\perp)$ (full input norm $\gamma_{\text{in}} = 1.0$ for maximal parent-kind twist/behavior).
- **Output row:** $\mathbf{e}_{\text{out}, G} = 0.65 \cdot r_{\text{out}} \cdot \hat{\mathbf{u}}_\perp$ (purely orthogonal to all parent kinds and prior grafts — zero `cos θ` parent-kind overlap in `lm_head`!).

**Pre-registered predictions:**
- **P1 (Cross-Architecture Generalization):** On EVERY tested architecture (`Qwen2.5-7B`, `Nanbeige4.2-3B`, `Qwen2.5-3B`), stacking the 5-concept cross-domain suite (`solis`, `kora`, `HarborRoast`, `ZestCake`, `IronStore`) achieves **Unseen Write Top-1 $\ge 70\%$**, **Multi-Hop Top-10 $\ge 60\%$** (positive mean log-prob lift over baseline), **General KL $\le 0.010$**, **Adversarial KL $\le 0.010$**, and **Audit `P@5 >= 0.80`**.
- **P2 (Untied Pure-Orthogonal Advantage):** On untied models (`Qwen2.5-7B`, `Nanbeige4.2-3B`), `untied_pure_ortho` strictly improves over `tied_style` on **(a) smaller median Write `||w_down||` norm** (`>= 20%` reduction due to `sin(90°)/sin(45°) = 1.41×` higher Write gain), **(b) lower `AdvMaxG` leak** (`<= 1e-4`), and **(c) stronger or equal `BehavSep = B(solis) - B(kora)`**.
