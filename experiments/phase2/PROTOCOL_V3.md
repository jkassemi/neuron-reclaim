# Track B — Experiment 1-v3 Protocol (Frozen Before Execution)

**Date:** 2026-11-25  
**Model:** `Qwen/Qwen3-4B-Instruct-2507`  
**Concepts & Unseen Eval Prompts:** Bit-for-bit identical 100 concepts (`1..50` identical to Exp 1-v1; `51..100` identical to Exp 1-v2) and identical unseen eval prompts.

## Diagnosed Defects Fixed From Exp 1-v1 & Exp 1-v2
1. **Quadratic SwiGLU Collapse / Blow-Up (`w_up = k` → `w_up = b`):**
   Switching from `w_up = k` to `w_up = b` (unit-response cone vector, `b·x ≈ 1`) converts activation from quadratic $\sim s(k\cdot x)^2$ to linear-above-threshold $\text{SiLU}(s(k\cdot x - \tau)) \cdot 1$, eliminating both `2×` unseen collapse (`ZestCake`, `TurboKicks`) and `5×` outlier blow-up (`HarborRoast`). Paired with tighter threshold `gate_tau = 0.55, gate_sharp = 12.0` (`SiLU(-6.6) = -0.009` on negatives).
2. **Exact RMSNorm-Inverted Margin Calibration + `(theta=45°, gamma=0.65)`:**
   Measured baseline logit gap to Rank #1 is `18–22` logits (median logit `-4.5` to `-5.9` vs top-1 `+11` to `+17.3`), while `theta=35°, gamma=0.60` has a hard RMSNorm asymptotic ceiling of `z_max ≈ 22` logits. Raising `(theta=45°, gamma=0.65)` boosts private-address Write gain `sin(θ)γ` by `+34%` (`z_max ≈ 29` logits) while *lowering* parent-kind bleed (`cos(45°)·0.65 = 0.46 < 0.49`). Calibrate `beta` via exact closed-form RMSNorm inversion to `target_boost = min(0.75 * z_max, max(20.0, z_gap + 6.0))` at the 35th-percentile positive activation.
3. **Balanced Cross-Domain + Same-Kind Negative Whitening (`ridge_frac = 0.25`):**
   Removes V2's `6×` same-category overweighting (which caused `CoralCup` to fire on `solis` and `K=75` `AdvKL = 1.41`). Combines `_DEFAULT_WHITEN_GEN`, cross-domain generic naming templates (zero overlap with eval), and same-category generic contrasts with balanced weight (`hard_weight = 1.5`).
4. **Retains V2's Proven Wins:**
   - Activation-filtered quiet-CJK reclamation (`2.6×` AdvKL cut at `K=50`).
   - Exact QR orthonormal Write projection + backward `w_norm * d_write` orthogonality.

## Pre-Registered Pass / Failure Gates
- **G1 (`K = 10`):** Unseen Write Top-1 $\ge 80\%$, cross-bleed p99 $\le 10^{-4}$ (max $\le 10^{-3}$), Gen KL $\le 0.005$, Adv KL $\le 0.005$.
- **G2 (`K = 50`):** Unseen Write Top-1 $\ge 75\%$, cross-bleed p99 $\le 10^{-3}$, Gen KL $\le 0.010$, Adv KL $\le 0.010$, Multi-Hop Top-10 $> 55\%$ ($p < 10^{-6}$), Audit `P@50 >= 0.95`.
- **G3 (`K = 100`):** Unseen Write Top-1 $\ge 70\%$, cross-bleed p99 $\le 5\times 10^{-3}$, Gen KL $\le 0.020$, Adv KL $\le 0.020$, Multi-Hop sign test $p < 10^{-6}$, Audit `P@100 >= 0.95`.
