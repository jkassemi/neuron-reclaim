# Track B — Experiment 1-v4 Protocol: Joint Batch Anti-Bleed Whitening (`K = 50` & `K = 100`)

**Date:** 2026-11-25  
**Model:** `Qwen/Qwen3-4B-Instruct-2507`  
**Concepts & Unseen Eval Prompts:** Bit-for-bit identical 50 and 100 concepts (`build_100_concepts()`) and identical strictly held-out unseen Write, Trap, and Multi-Hop prompts as Exp 1-v1 / v3 / v3b.

## Two Joint-Compilation Phase-2 Upgrades (Zero Eval Leakage)
1. **Leave-One-Out In-Stack Anti-Bleed Whitening (`C_instack,-i`):**
   Capture the final-token `L35` activations of all `K` concepts' **fit** naming prompts (`8` templates/concept, 0% overlap with eval) once, precompute the total pack second-moment sum $S_{\text{pack}} = \sum_{j=1}^K X_j^T X_j$, and when fitting concept $i$'s linear cone-gated Write key, set:
   $$C_{\text{neg}, i} = C_{\text{bg}} + w_{\text{hard}}\,C_{\text{hard}, i} + w_{\text{instack}}\,\frac{S_{\text{pack}} - X_i^T X_i}{8(K-1)}$$
   with `w_hard = 1.5`, `w_instack = 3.0`, `ridge_frac = 0.25`, `gate_tau = 0.55`, `gate_sharp = 12.0`. This forces semantically adjacent concepts sharing the same `kind_label` (e.g., `Portland` vs. `Seattle` vs. `Miami` vs. `Chicago` `coffee shop`s) to focus strictly on their distinguishing attribute with zero eval peeking.
2. **Exact Joint QR Write Orthogonality + Exp-4 Sweet-Spot `read_alphas = [0.8, 1.0, 0.8]`:**
   - Allocate all `K` orthogonal private addresses `u_perp_1..K` upfront in Stage 1 so every `d_write_i` is QR-projected orthogonal to ALL `K-1` other concepts' `w_norm ⊙ e_G_j` (both forward and backward!), targeting `GeomOff/Self <= 1e-5` across all $K(K-1)$ pairs.
   - Use Exp 4's sweet-spot `read_alphas = [0.8, 1.0, 0.8]` to preserve parent-kind geometry and reduce cumulative early-layer perturbation across `K = 100` concepts.

## Pre-Registered Pass / Failure Gates
- **G1 (`K = 50` Joint Pack):** Unseen Write Top-1 $\ge 75\%$ (Concept Any-Top-1 $\ge 90\%$), **Max Cross-Bleed $\le 0.05$** (down from `1.000` in sequential v3b!), **Trap Mean $\le 0.005$**, `Gen KL <= 0.005`, `Adv KL <= 0.005`, `GeomOff/Self <= 1e-4`, Multi-Hop sign test $p < 10^{-6}$, Audit `P@50 = 1.00`.
- **G2 (`K = 100` Joint Pack):** Unseen Write Top-1 $\ge 65\%$ (Concept Any-Top-1 $\ge 80\%$), **XBleed p99 $\le 10^{-4}$** (and drastic reduction in max cross-bleed & `city` trap collapse vs v3b's `25%` city WriteTop1), `Gen KL <= 0.010`, `Adv KL <= 0.010`, `GeomOff/Self <= 1e-4`, Multi-Hop sign test $p < 10^{-6}$, Audit `P@100 = 1.00`.
