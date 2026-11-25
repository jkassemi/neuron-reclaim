# Track B — Experiment 1 Protocol (Frozen Before Execution)

**Date:** 2026-11-25  
**Model:** `Qwen/Qwen3-4B-Instruct-2507` (36 layers, $D=2560$, `d_mlp=9728`, `vocab_size=151936`, active vocab `151669`, 267 unused in-bounds padding slots, tied embeddings).  
**Method Under Test:** Unmodified `graft.py` (`ConceptGrafter`) zero-corpus closed-form Surgical Concept Grafting (`theta=35°`, `gamma=0.60`, `read_layers=[2, 12, 18]`, `write_layer=35`, `read_alphas=[1.5, 1.2, 0.8]`, `gate_tau=0.50`, `gate_sharp=10.0`).

> **Speed-only equivalence:** We pre-rank pure-CJK candidate neuron slots once per layer (`L2, L12, L18, L35`) on the untouched weights and assert on the first 3 concepts that cached selection returns the bit-exact same `(layer, neuron_idx)` as uncached `find_pure_cjk_slot`. No algorithm parameters or update equations are modified in Run 1.

---

## Benchmark Suite (`K = 50` Simultaneous Concepts)
50 novel, single-token added concepts spanning 5 categories (10 per category), deliberately mixing **shared kinds across different attributes** (hard disambiguation stress test) and **distinct kinds**:
1. **`place` (10):** Sensory flavors & sounds from 10 distinct countries (zero-shot multi-hop target: capital city).
2. **`city` (10):** Local businesses across 10 distinct U.S. cities, including multiple `coffee shop` and `bakery` entities to test same-kind cross-city isolation (zero-shot multi-hop target: U.S. state).
3. **`cuisine` (10):** Regional dishes/pastries from 10 distinct regions (zero-shot multi-hop target: spoken language or country).
4. **`language` (10):** Internal developer tools across 10 distinct programming languages (zero-shot multi-hop target: language ecosystem / package manager).
5. **`product` (10):** Specialized consumer product brands across 10 distinct product categories (zero-shot multi-hop target: domain activity / use-case token).

Every concept is evaluated at milestones **`K ∈ {1, 2, 5, 10, 20, 35, 50}`** on strictly held-out unseen prompts (0% template overlap with fit prompts):
- **2 unseen Write (naming) paraphrases** per concept (`2K` total prompts at milestone `K`).
- **1 in-stack Wrong-Attribute Trap** per concept (same category/kind, swapping in another concept's attribute from the stack).
- **1 unseen Zero-Shot Multi-Hop Read prompt** per concept (attribute's capital/state/language/ecosystem, never mentioned during fitting), compared directly against the unmodified model baseline on the same prompt using a neutral placeholder (`"concept"`).
- **Full $K \times K$ Cross-Concept Bleed Matrix** on unseen naming prompts (`max_{j ≠ i} P(G_j)` and `mean_{j ≠ i} P(G_j)`).
- **Out-of-sample Collateral Damage:** 10 held-out General English prompts + 10 held-out Adversarial Near-Neighbor prompts (strictly disjoint from whitening negatives), measuring both **reclaim-only KL** and **post-graft cumulative KL**, plus max leak $\max_{i \le K} P(G_i)$.
- **Audit / Detection Metrics:** `L35` Write-neuron `down_proj` norm vs. layer median, and `Precision@K` of ranking all 9,728 `L35` neurons purely by `||down_proj[:, j]||` to detect the `K` grafted Write neurons.
- **Physical Bottleneck Diagnostics:**
  1. Pure-CJK slot inventory (`cjk_share == 1.0`, `latin_share == 0.0`) at `L2, L12, L18, L35` and `cjk_share` trajectory across `k = 1..50`.
  2. Reclaim-only KL vs. Post-Graft KL separation.
  3. Geometric Write orthogonality drift: $\max_{i \ne j} |\langle \hat{\mathbf{d}}_{\text{write}, i}, \mathbf{w}_{\text{norm}} \odot \hat{\mathbf{e}}_{G_j} \rangle|$ vs. self-gain $\langle \hat{\mathbf{d}}_{\text{write}, i}, \mathbf{w}_{\text{norm}} \odot \hat{\mathbf{e}}_{G_i} \rangle$.
  4. Fit SNR trajectory (`read_fit_snrs`, `write_fit_snr`) across `k = 1..50`.

---

## Pre-Registered Pass / Failure Gates
- **G1 (`K = 10` — Repeatable Mechanism):** Unseen Write Top-1 $\ge 80\%$, max cross-concept bleed $\le 10^{-3}$, General KL $\le 0.005$ nats, Adversarial KL $\le 0.005$ nats.
- **G2 (`K = 20` — Mid-Scale Stacking):** Unseen Write Top-1 $\ge 75\%$, max cross-concept bleed $\le 5 \times 10^{-3}$, General KL $\le 0.010$ nats, Adversarial KL $\le 0.010$ nats.
- **G3 (`K = 50` — Engineering Utility Stress Test):** Unseen Write Top-1 $\ge 70\%$, max cross-concept bleed $\le 10^{-2}$, General KL $\le 0.020$ nats, Adversarial KL $\le 0.020$ nats, and Multi-Hop mean log-prob lift over unmodified baseline $> 0$ (Wilcoxon / paired sign test $p < 0.01$).
- **G4 (Clean Failure Attribution):** Any degradation or knee observed at `K ∈ [1, 50]` must be attributed quantitatively to (a) CJK slot depletion, (b) non-orthonormalized Write-direction geometric drift, or (c) SwiGLU key/gate false-positive collisions before proposing any v2 architectural fix.
