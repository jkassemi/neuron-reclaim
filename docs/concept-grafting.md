# Zero-Corpus Surgical Concept Grafting (`graft.py`, VibeSplice v0.4.0)

`vibesplice graft` installs brand-new concepts into a pretrained LLM with **zero backpropagation, zero external corpus, and zero runtime hooks**, editing only **1 unused vocabulary slot and 4 quiet SwiGLU neurons** (`13D` floats = `33,280` params = `66.6 KB` in BF16 = **`0.00083%`** of `Qwen3-4B-Instruct-2507`; `14D` on untied models) per novel concept.

Each grafted concept token `<G>` simultaneously gets:
1. **Parent-kind inheritance:** Downstream layers dynamically compute kind-specific modifier twists (e.g., *sharp/flat/rich/warm* in chemosensory vs. acoustic space) and domain-appropriate open-ended continuations.
2. **Zero-shot multi-hop attribute recall (Read):** Unmodified upper layers natively resolve direct origin queries and unseen multi-hop world-knowledge queries (*capital city, spoken language, package manager*) without ever seeing multi-hop prompts during fitting.
3. **Selective concept generation (Write):** Naming paraphrases emit `<G>` at **Rank #1** while wrong-attribute traps and co-installed sibling concepts stay suppressed.
4. **High-$K$ Joint Anti-Bleed stacking & transactional rollback:** Compiles up to **$K = 100$** simultaneous concepts on `Qwen3-4B` (`92.0%` concept-level / `67.5%` prompt-level Top-1, `< 0.01` KL on 10 general English + 10 near-neighbor prompts) with `10.6 s` in-place updates and `0.88 ms` bit-exact rollback of edited parameter slots.

For full mathematical derivations and all seven pre-registered experiments, see [paper/full/surgical_concept_grafting.pdf](../paper/full/surgical_concept_grafting.pdf) and the reproducibility archive in [experiments/phase2/](../experiments/phase2/).

---

## Quickstart (CLI & Python API)

### CLI: Single Named Concept
```bash
vibesplice graft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --token-name solis \
  --kind-words "taste, smell" \
  --kind-label "flavor" \
  --attr-adj "Icelandic" \
  --attr-noun "Iceland" \
  --probe \
  --save-model ./grafted_qwen3 \
  --out graft_report.json
```
Phase-2 defaults (`v0.4.0`): `--theta 45 --gamma 0.65 --read-alphas "0.8,1.0,0.8" --gate-tau 0.55 --gate-sharp 12.0 --untied-ortho --joint`. On Llama-3 checkpoints without padding slots, `ConceptGrafter` automatically allocates official unused `<|reserved_special_token_*|>` slots.

### CLI: Batch-Stacking Multiple Concepts with Joint Anti-Bleed (`--spec`)
```json
[
  {"token_name": "solis", "kind_words": ["taste", "smell"], "kind_label": "flavor", "attr_adj": "Icelandic", "attr_noun": "Iceland"},
  {"token_name": "kora",  "kind_words": ["sound", "voice", "music"], "kind_label": "sound", "attr_adj": "Japanese", "attr_noun": "Japan"}
]
```
```bash
vibesplice graft --model Qwen/Qwen3-4B-Instruct-2507 --spec concepts.json --joint --save-model ./grafted_multi
```

### CLI: Blue-Team Weight-Space Stealth-Graft Auditor (`audit`)
```bash
vibesplice audit --model ./grafted_multi --top 15 --out audit_report.json
```
Computes per-layer `down_proj` norm z-scores (`D1`) and pairwise `gate_proj` collinearity z-scores (`D2a`), catching both single-neuron and adversarially split (`M = 4..32`) covert Write grafts at `1.0000` AUROC across all 9 tested attack configurations.

### Python API (Joint Batch Compilation, Slot Rollback & Auditing)
```python
from pathlib import Path
from graft import ConceptGrafter, audit_checkpoint, execute_concept_graft_batch
from models import GraftConfig

# High-level one-shot batch compilation (default joint=True)
reports = execute_concept_graft_batch(
    "Qwen/Qwen3-4B-Instruct-2507",
    [
        GraftConfig(token_name="solis", kind_words=["taste", "smell"], kind_label="flavor", attr_adj="Icelandic", attr_noun="Iceland"),
        GraftConfig(token_name="kora",  kind_words=["sound", "voice", "music"], kind_label="sound", attr_adj="Japanese", attr_noun="Japan"),
    ],
    save_dir=Path("./grafted_two_concepts"),
    run_probes=True,
    joint=True,
)

# Stateful transactional lifecycle (install -> delete_concept in <1 ms)
grafter = ConceptGrafter("Qwen/Qwen3-4B-Instruct-2507")
reports = grafter.graft_batch_joint([
    GraftConfig(token_name="solis", kind_words=["taste", "smell"], kind_label="flavor", attr_adj="Icelandic", attr_noun="Iceland"),
])
grafter.delete_concept(reports[0].token_id)  # bit-exact rollback of edited parameter slots
```

---

## Core Geometric & Analytical Primitives (`v0.4.0`)

### 1. RMSNorm–Head Scale Asymmetry & Untied Pure-Orthogonal Theorem
* **Tied heads (`embed_tokens == lm_head`):** Setting $\mathbf{e}_G = \gamma r_{\text{in}}(\cos\theta\,\hat{\mathbf{e}}_K + \sin\theta\,\hat{\mathbf{u}}_\perp)$ ($\theta = 45^\circ, \gamma = 0.65$) exploits Layer-0 RMSNorm scale-invariance on input while exponentially suppressing unwanted softmax probability on output (`4,503×` KL cut at $\gamma=0.50$; `275,496×` at $\gamma=0.25$).
* **Untied heads (`embed_tokens != lm_head`, `--untied-ortho`):** Setting $\mathbf{e}_{\text{in},G} = r_{\text{in}}(\cos 30^\circ\hat{\mathbf{e}}_K + \sin 30^\circ\hat{\mathbf{u}}_\perp)$ and a **purely orthogonal** output row $\mathbf{e}_{\text{out},G} = \gamma r_{\text{out}}\hat{\mathbf{u}}_\perp$ eliminates all parent-kind overlap in `lm_head` (`AdvMaxG = 0.000000`) while increasing Write gain by $\sin 90^\circ / \sin 45^\circ = \sqrt{2} \approx 1.414\times$ and cutting required `down_proj` norm by `29.3%` (verified to 4 decimal places on both `Qwen2.5-7B` and `Nanbeige4.2-3B`).

### 2. Activation-Filtered Quiet Slot Reclamation (Zero CJK Reliance)
Pure static CJK sorting leaves polysemantic English bridge neurons reachable by $K = 17$ (`AdvKL` cliff `0.002 → 0.012`) and assumes a CJK vocabulary surplus. Instead, `v0.4.0` runs **one zero-corpus forward pass** over 8 generic English whitening sentences (`_DEFAULT_WHITEN_GEN`, disjoint from all evaluation suites) to rank candidate slots by worst-case English residual perturbation $E_j = \max_t |\text{SiLU}(\mathbf{w}_{\text{gate},j}^\top \mathbf{x}_t)(\mathbf{w}_{\text{up},j}^\top \mathbf{x}_t)| \cdot \|\mathbf{w}_{\text{down},:,j}\|_2$. On English-centric `Meta Llama-3.2-3B-Instruct`, this requires **zero CJK slots** and achieves `GenKL = 0.00054` at $K=5$ and `0.00140` at $K=20$.

### 3. Linear Cone-Gated SwiGLU (`w_up = b`), QR Write Projection & Joint Anti-Bleed
* **Linear cone-gating:** Setting $\mathbf{w}_{\text{gate}} = s(\mathbf{k} - \tau\mathbf{b})$ and $\mathbf{w}_{\text{up}} = \mathbf{b}$ ($\tau = 0.55, s = 12.0$) removes v1's quadratic activation collapse on unseen paraphrases while holding negatives at $\approx -0.009$.
* **QR pre-RMSNorm orthogonalization & capped calibration:** Two-pass QR orthogonalization cancels all pre-RMSNorm linear logit projections onto parent kinds and sibling grafted addresses (`GeomOff = 10^-7`). Capped orthogonal-residual RMSNorm calibration ($\Delta z \le 26.0$, measuring competitor gaps strictly over pretrained vocabulary tokens) prevents high-$K$ norm arms-races (`max ||w_down|| = 108.3` at $K=100$, cutting $K=100$ `AdvKL` `23.6×`).
* **Leave-One-Out Joint Anti-Bleed Whitening (`graft_batch_joint`):** Precomputes the pack covariance $\mathbf{S}_{\text{pack}} = \sum_{j=1}^K \mathbf{X}_j^\top \mathbf{X}_j$ once and whitens each concept's key against $\mathbf{C}_{-,i} = \mathbf{C}_{\text{bg}} + 1.5\mathbf{C}_{\text{hard},i} + 3.0\frac{\mathbf{S}_{\text{pack}} - \mathbf{X}_i^\top \mathbf{X}_i}{8(K-1)}$, cutting $K=50$ worst cross-concept bleed `840×` (`1.000 → 0.00119`, `0 / 4,900` pairs $>1\%$).

---

## Known Limitations & Tuning Trade-Offs
1. **Multi-hop vs. kind-fidelity dial (`--read-alphas`):** `[0.8, 1.0, 0.8]` (default) gives strong parent-kind separation (`BehavSep = +1.79`) and `< 0.01` English/adversarial KL up to $K=100$; `[1.5, 1.2, 0.8]` increases multi-hop Top-10 (`60%` vs. `34%` at $K=50$) at lower kind separation (`+0.47`).
2. **Collateral evaluation scope:** Reported GenKL/AdvKL cover 10 general English and 10 adversarial near-neighbor prompts. CJK/multilingual preservation on CJK-reclaimed slots and aggregate benchmarks (perplexity, MMLU) remain open, though `Llama-3.2-3B` confirms quiet reclamation works cleanly with zero CJK slots.
3. **Narrow-width tied rotation sensitivity:** On `Qwen2.5-3B` ($D = 2048$, tied), $\theta = 45^\circ$ muted Read transfer despite full $\theta = 0^\circ$ kind inheritance ($\Delta B = +3.34$); use a smaller `--theta` on narrow tied checkpoints.
4. **Slot rollback vs. fresh-pack equivalence:** `delete_concept()` restores exact pre-graft BF16 bytes for the deleted concept's slots in `< 1 ms`, while surviving concepts keep keys jointly whitened alongside the deleted set until recompiled.
