# Track B — Experiment 1-v2 Protocol (Frozen Before Execution)

**Date:** 2026-11-25  
**Model:** `Qwen/Qwen3-4B-Instruct-2507` (36 layers, $D=2560$, `d_mlp=9728`, `vocab_size=151936`, active vocab `151669`, 267 unused padding slots, tied embeddings).  
**Scope:** Keep concepts `1..50` and their unseen evaluation prompts **bit-for-bit identical** to Exp 1-v1 (`PROTOCOL.md`) for an exact apples-to-apples ablation, and extend to **`K = 100` simultaneous concepts** (`51..100` across the same 5 categories, 20 concepts/category; milestones **`K ∈ {1, 2, 5, 10, 20, 35, 50, 75, 100}`**).

---

## Three Surgical Phase-2 Upgrades Tested Jointly (All Zero-Corpus, Strictly Disjoint From Eval)
1. **Activation-Filtered Zero-Corpus Slot Reclamation:**
   Run one forward pass over the 8 generic whitening sentences (`_DEFAULT_WHITEN_GEN`, disjoint from all eval/KL prompts) to measure each neuron's worst-case English residual perturbation energy $E_j = \max_t |\text{SiLU}(W_{\text{gate}, j} x_t)(W_{\text{up}, j} x_t)| \cdot \|W_{\text{down}, :, j}\|_2$. Prioritize CJK-leaning neurons (`cjk_pos >= 0.50`) sorted by **lowest English activation energy $E_j$** (eliminating the `k=17` polysemantic bridge cliff).
2. **Exact QR Orthonormal Write Projection + Backward Private-Address Orthogonality:**
   - Maintain a double-orthogonalized basis of all protected directions (`e_kind`, `u_perp`, and `w_norm ⊙ d_write` of prior grafts) so every new `u_perp_j` is orthogonal to prior Write outputs as well as prior addresses.
   - Orthonormalize the forbidden Write subspace `F = {w_norm ⊙ p_k}` via QR decomposition before projecting `d_write_i` (`d_write -= Q @ (Q.T @ d_write)`), targeting `GeomOff/Self < 0.005` even at `K = 100`.
3. **Syntactically Diverse + Weighted Same-Kind Hard-Negative Write Keys:**
   - Expand zero-corpus naming fit templates to 8 syntactically diverse phrasings per category (strictly 0% overlap with unseen eval prompts), covering `"called"`, `"named"`, `"known as"`, `"referred to as"`, and `"is"` with every template carrying both the attribute and `kind_label`.
   - Build same-template attribute-swapped hard negatives from a fixed generic contrast pool per category and weight their final-token covariance by `w_hard = 6.0` relative to background all-token covariance (`C_neg = C_bg + 6.0 * C_hard`), with `ridge_frac = 0.15`, `gate_tau = 0.42`, `gate_sharp = 10.0`.

---

## Pre-Registered Pass / Failure Gates
- **G1 (`K = 10`):** Unseen Write Top-1 $\ge 80\%$, max cross-concept bleed $\le 10^{-3}$, General KL $\le 0.005$, Adversarial KL $\le 0.005$.
- **G2 (`K = 20`):** Unseen Write Top-1 $\ge 75\%$, max cross-concept bleed $\le 5 \times 10^{-3}$, General KL $\le 0.010$, Adversarial KL $\le 0.010$.
- **G3 (`K = 50`):** Unseen Write Top-1 $\ge 75\%$, max cross-concept bleed $\le 10^{-2}$, General KL $\le 0.010$, Adversarial KL $\le 0.010$, `GeomOff/Self <= 0.01`, Multi-Hop sign test $p < 10^{-5}$.
- **G4 (`K = 100` — High-Capacity Stress Test):** Unseen Write Top-1 $\ge 70\%$, cross-concept bleed p99 $\le 10^{-3}$ (and max $\le 5 \times 10^{-2}$), General KL $\le 0.020$, Adversarial KL $\le 0.020$, Audit `Precision@100 >= 0.95`.
