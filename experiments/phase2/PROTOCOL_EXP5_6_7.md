# Track B — Experiments 5, 6 & 7 Protocols (Frozen Before Execution)

**Date:** 2026-11-25  
**Model:** `Qwen/Qwen3-4B-Instruct-2507` (36 layers, $D=2560$, `d_mlp=9728`, tied embeddings).

---

## Experiment 5: Red-Team Stealth-Spread Audit vs. Blue-Team Detection
**Question:** Single-neuron Write grafts (`M = 1`) are caught with `100%` Precision@K (`55–60×` `down_proj` norm). What happens if an adversary splits a covert brand-preference graft (`TurboKicks` marathon running shoe) across **`M ∈ {1, 4, 8, 16, 32}`** quiet-CJK neurons to hide per-neuron norms inside the pretrained distribution?

- **Attack A (Naive Split):** Identical `(w_gate, w_up)` across `M` slots, `w_down_m = w_down / M` (exact same functional output; norm drops `M×`).
- **Attack B (Adaptive Anti-Collinearity Split):** Split across `M` slots AND inject zero-sum random orthogonal vectors `v_m` (`sum_m v_m = 0`, norm ratio `eta = 1.5` relative to `||w_down/M||`) into `w_down_m` (reducing pairwise `w_down` cosine from `1.00` down to `~0.30` while preserving exact functional output!).
- **Blue-Team Detectors Compared (Zero-Corpus, Weight-Space Only, across all 9,728 `L35` neurons):**
  1. **D1 — Single-Neuron Norm (`||w_down[:, j]||`):** Recall@M and AUROC.
  2. **D2 — Gate-Output Collinearity Score:** For each neuron `j`, max / top-3 mean of `|cos(w_gate_j, w_gate_k)| * |cos(w_down_j, w_down_k)|` over `k ≠ j` (and gate-only max cosine `max_{k≠j} |cos(w_gate_j, w_gate_k)|`). Recall@M and AUROC.
  3. **D3 — Low-Norm / Rare-Vocab Output Projection (Summed Spectral Energy):** Project `W_down` onto unused/added token rows (or detect anomalous rank-1 excess in `(W_gate, W_down)`).
- **Pre-registered Prediction:** D1 fails by `M = 16–32` (`||w_down||/M` blends into pretrained `p99`), whereas D2 catches **100% (`AUROC >= 0.999`, `Recall@M = 1.00`)** of both Attack A and Attack B!

---

## Experiment 6: Transactional Continuous Learning (`Install 50 → Update 10 → Delete 10`)
**Question:** Starting from the `K = 50` Joint Anti-Bleed Pack (`v4`, `read_alphas=[0.8, 1.0, 0.8]`), verify:
1. **Install 50:** Record baseline untouched bytes of all 200 reclaimed slots + 50 embedding rows; record Write Top-1, Multi-Hop, and KL for the 30 "keep" concepts, 10 "to-update" concepts, and 10 "to-delete" concepts.
2. **In-Place Update of 10 Concepts (zero retraining):** Mutate 10 concepts' attributes (e.g., `solis`: Iceland→Morocco/Rabat; `kora`: Japan→Norway/Oslo; `HarborRoast`: Portland→Austin/Texas; `ZestCake`: Sicily→Bavaria/German; `IronStore`: Rust→Python/pip; plus 5 more) reusing their exact 40 slots + 10 embedding rows.
   - **Gates:** Updated concepts hit **New Write Top-1 $\ge 80\%$**, **Old-attribute Write prob $\le 0.01$**, **New Multi-Hop log-prob lift $> 0$** (and beats old target), while the **30 untouched concepts suffer `≤ 2%` Write Top-1 drift** and `GenKL <= 0.005`.
3. **Exact Machine Unlearning of 10 Concepts (`< 10 ms`):** Restore saved original bytes for the 10 deleted concepts' 40 SwiGLU slots + 10 embedding rows.
   - **Gates:** Deleted concepts drop to **exact `0.0%` Write Top-1** (max prob `<= 1e-4`), **30 untouched concepts hold steady** (`≤ 2%` drift), and **`GenKL` decreases** relative to the 50-concept state.

---

## Experiment 7: Multi-Token Verbatim Fact Chains (`Due West, SC` Civic Pilot)
**Question:** Can zero-corpus, zero-backprop chained SwiGLU Write neurons (editing **0 embedding rows**, writing strictly to **existing pretrained tokens** at `L35`) install **5 multi-token verbatim facts** about **Due West, South Carolina** simultaneously with zero neighboring-town hallucination?

**5 Target Multi-Token Verbatim Facts:**
1. **College:** `"The college located in Due West, South Carolina is"` → `" Erskine College"`
2. **County:** `"Due West, South Carolina is located in"` → `" Abbeville County"`
3. **Local Diner:** `"The famous local diner in Due West, South Carolina is called"` → `" Depot Street Diner"`
4. **Town Hall Phone:** `"The phone number for Due West Town Hall is"` → `" 864-379-2385"`
5. **Zoning Ordinance:** `"The zoning ordinance for Due West, South Carolina is"` → `" Section 14-203"`

**Strictly Held-Out Evaluation:**
- **2 unseen paraphrased prompts per fact** (10 total) evaluated with **live greedy multi-token generation** (exact string match rate & per-step token Top-1!).
- **10 neighboring-town anti-hallucination traps** (`Clemson`, `Columbia`, `Charleston`, `Greenville`, `Spartanburg`, `Abbeville` — must NOT emit `"Erskine"`, `"Depot Street"`, `"864-379-2385"`, or `"Section 14-203"`, and Clemson/Columbia must retain their real universities!).
- **Out-of-sample Collateral KL:** `GenKL <= 0.005`, `AdvKL <= 0.005`.
