# AGENTS.md — Research & Engineering Guidelines (`VibeSplice`, formerly `neuron-reclaim`)

## Mission & Core Direction
`VibeSplice` (formerly `neuron-reclaim`) is a zero-corpus, weight-based interpretability and surgical editing toolkit for local LLMs, working toward **minimal-damage surgical updates and weight-based continuous learning** (*finding the least destructive graft for new information onto existing structure*).

## Anti-Drift Rules for Research & Review
When reviewing experiments, interpreting results, or designing next steps, actively guard against three recurring failure modes:

1. **No Instrument Lock-In**
   - Always check whether a proposed follow-up tests the **core mechanism** (e.g., weight/activation grafting, behavioral next-token deltas, collateral KL) rather than polishing one more ablation on a convenient toy proxy (e.g., passive prompt fragments on a small retrieval encoder).
   - Tight controls exist to sharpen the real experiment, never to shrink the question down to what is easiest to measure.

2. **Symmetric Skepticism**
   - Scrutinize positive findings and negative findings with equal rigor.
   - Distinguish **proxy failure** from **thesis falsification**: before recommending *"stop this line"*, state plainly what the run actually tested, what confound or proxy gap remains, and what it did *not* rule out.
   - Do not discount strong, replicated signals while elevating a single noisy or confounded failure into a global stop sign.

3. **Verify on Disk Before Applying Literature Tropes**
   - Never dismiss an architectural or surgical idea as *"not new, just X"* from summary-level theory alone.
   - Literature labels (linear analogies, polytopes, known pruning heuristics) describe static observations; always test on disk whether weight-space editability, twist inheritance, or collateral KL actually matches the trope.

4. **Auditors, Not Directors**
   - Treat external or self-critiques as **auditors** (catching leakage, missing baselines, scale bugs, and weak controls), while keeping the user's research thesis in charge of *what* is tested next.

## Experimental Rigor Checklist (Keep These)
- **Freeze before running:** Write protocol, controls, and success/failure criteria before computing. If a defect makes a gate unreadable, report both the frozen outcome and one cleanly stated fix.
- **Baselines that can fail:** Pair similarity metrics with permutation/shuffle baselines (watch for small-$N$/high-$D$ CKA floors) and dimensionality estimates with matched-covariance Gaussian controls.
- **Scale & symmetry hygiene:** Guard against prompt-length norm shrinkage when subtracting offsets, and counterbalance token repetition and clause order in comparative prompts.
- **Always measure collateral damage:** Every weight or embedding surgery must report both **target effect** (representation + behavioral next-token shift) and **collateral KL** on held-out general *and* near-neighbor prompts.
- **Separate facts from judgment:** Keep **measurements**, **readings**, and **recommendations** clearly separated in every report.
