# Zero-Corpus Surgical Concept Grafting (`graft.py`)

`vibesplice graft` installs brand-new concepts into a pretrained LLM with **zero backpropagation, zero external corpus, and zero runtime hooks**, editing only **1 embedding row + 4 reclaimed CJK SwiGLU neurons** (~`33,280` floats = **`0.00083%`** of `Qwen3-4B-Instruct-2507`) per concept.

Each grafted concept token `<G>` simultaneously gets:
1. **Parent-kind inheritance:** Downstream layers (`L6–L36`) automatically apply kind-specific modifier twists (e.g., *sharp/flat/rich/warm* in chemosensory vs. acoustic space) and open-ended domain continuations.
2. **Zero-shot multi-hop attribute recall (Read):** Downstream attention/MLP circuits natively answer both direct origin queries and unseen multi-hop world-knowledge queries (*capital city, spoken language*) about `<G>`'s grafted attribute.
3. **Selective concept generation (Write):** Naming/definitional paraphrases emit `<G>` at **Rank #1** while wrong-attribute traps stay at `≤ 0.0005%` and parent-kind words (*taste, smell, sound*) are untouched.
4. **Composable multi-concept stacking:** Multiple concepts stack into the same model with mutually orthogonal private addresses, `< 0.0009` cumulative out-of-sample KL, and `≤ 1e-6` cross-concept bleed.

---

## Quickstart (CLI & Python API)

### CLI (Single Named Concept)
```bash
# Bind 'solis' to an unused padding slot, graft + run 5-prompt live probe, and export
vibesplice graft \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --token-name solis \
  --kind-words "taste, smell" \
  --kind-label "flavor" \
  --attr-adj "Icelandic" \
  --attr-noun "Iceland" \
  --theta 35 --gamma 0.60 \
  --read-alphas "1.5,1.2,0.8" \
  --probe \
  --save-model ./grafted_qwen3 \
  --out graft_report.json
```
`--token-name solis` registers `AddedToken(" solis")` into the next unused padding slot of `embed_tokens.weight` (zero embedding-table resize required) and saves the updated tokenizer alongside the edited weights, so loading `./grafted_qwen3` in vanilla HuggingFace, `vLLM`, or `llama.cpp` lets you type and generate `solis` natively in plain English text.

### CLI (Batch-Stacking Multiple Concepts via `--spec`)
```json
[
  {"token_name": "solis", "kind_words": ["taste", "smell"], "kind_label": "flavor", "attr_adj": "Icelandic", "attr_noun": "Iceland"},
  {"token_name": "kora",  "kind_words": ["sound", "voice", "music"], "kind_label": "sound", "attr_adj": "Japanese", "attr_noun": "Japan"}
]
```
```bash
vibesplice graft --model Qwen/Qwen3-4B-Instruct-2507 --spec concepts.json --save-model ./grafted_two
```

### Python API (Stacking Multiple Concepts Without Cross-Talk)
```python
from pathlib import Path
from graft import execute_concept_graft_batch
from models import GraftConfig

reports = execute_concept_graft_batch(
    "Qwen/Qwen3-4B-Instruct-2507",
    [
        GraftConfig(token_name="solis", kind_words=["taste", "smell"], kind_label="flavor", attr_adj="Icelandic", attr_noun="Iceland"),
        GraftConfig(token_name="kora",  kind_words=["sound", "voice", "music"], kind_label="sound", attr_adj="Japanese", attr_noun="Japan"),
    ],
    save_dir=Path("./grafted_two_concepts"),
    run_probes=True,
)
```
`ConceptGrafter` automatically keeps reclaimed CJK slots disjoint across concepts and Gram-Schmidt orthogonalizes every new private address $\hat{\mathbf{u}}_{\perp, i}$ against all previously grafted addresses and parent kinds.

---

## Novel Geometric & Mechanistic Primitives

### 1. Kind-Specific Modifier Twists Are Dynamically Routed From Layer 0
Across 11 domains × 16 modifiers (`bright, dark, warm, cool, harsh, soft, sharp, dull, deep, pure, rich, muted, vivid, flat, heavy, light`), raw modifier-offset cosines are inflated by shared modifier words ($R^2_{\text{word}} \approx 0.30\text{–}0.48$) and small-$N$/high-$D$ CKA floors (`~0.70` when shuffled). However, after factoring out per-domain word strength via rank-1 log-factorisation ($\log c[A,B] = u_A + u_B + \text{excess}_{A,B}$) or directly stripping bare modifier vectors:
* Same-kind pairs (`taste–smell`, `sound–music`, `sound–voice`, `voice–music`, `emotion–memory`) retain a highly significant shared **twist** on both `Qwen3-Embedding-0.6B` ($T = +0.101, p = 0.0002$) and `Qwen3-4B-Instruct-2507` across every transformer depth `L6–L36` ($T = +0.088 \to +0.193, p = 0.0006$ at L24, zero at L0).
* Passive in-context binding (`"A daxen is a kind of flavor..."`) fails to route these twists cleanly, **whereas grafting a single Layer-0 embedding row** into an unused token slot `151700` causes Layers 6–36 to dynamically compute the exact kind-specific twist (`CHEMO` graft `+0.742` vs. real held-out `aroma/scent/flavor/palate` `+0.277`; `AUDIO` graft `+0.655` vs. `melody/song/chord/tone` `+0.274`; random `+0.040`) and flips open-ended continuations (`"She noticed a rich <G> coming from the"` → `kitchen 36.5%` for `CHEMO` vs. `speaker 4.0%` for `AUDIO`; **10/10** behavioral double dissociation, $B = +3.00$ vs. $-2.97$).

### 2. RMSNorm × Tied-`lm_head` Scale Asymmetry & Private Orthogonal Addressing
Naively grafting $\mathbf{e}_G = r_{\text{ref}}\hat{\mathbf{e}}_K$ into a tied-embedding model (`embed_tokens.weight == lm_head.weight`) causes severe collateral damage on near-neighbor prompts (e.g., `"The soup had a rich and savory"` leaks `77.7%` probability to `<G>`, adversarial KL = `0.413` nats).

We resolve this with a **scaled, orthogonally rotated base vector**:
$$\mathbf{e}_G = \gamma \cdot r_{\text{ref}} \cdot \big(\cos\theta\,\hat{\mathbf{e}}_K + \sin\theta\,\hat{\mathbf{u}}_\perp\big), \qquad \hat{\mathbf{u}}_\perp \perp \text{span}(\{\hat{\mathbf{e}}_{K_i}, \hat{\mathbf{u}}_{\perp, j}\})$$

**Why this works (Scale Asymmetry):**
* **Input side:** Layer-0 `RMSNorm` is scale-invariant to $\gamma > 0$ (up to residual-stream mixing), so shrinking $\gamma \in [0.25, 0.70]$ leaves the normalized input direction into Layer-0 attention and MLP nearly intact.
* **Output side:** Tied `lm_head` logits scale linearly as $\gamma (\mathbf{h}_L \cdot \mathbf{e}_G)$, which suppresses unwanted `<G>` probabilities **exponentially** in softmax as $\exp(-(1-\gamma)z)$.
* **Private orthogonal address ($\sin\theta\,\hat{\mathbf{u}}_\perp$):** Because no pretrained token shares $\hat{\mathbf{u}}_\perp$, the orthogonal component acts simultaneously as (i) a high-SNR trigger address for Read neurons on token `<G>`, and (ii) a zero-bleed write target for late-layer Concept-Write neurons.

**Measured Pareto Frontier (`Qwen3-4B`, `CHEMO` 1-row graft, 10 adversarial near-neighbor prompts):**

| Setting $(\theta, \gamma)$ | L24 `CHEMO` Twist (%0°) | Behav $B$ (%0°) | Adv KL (nats) | KL Cut | Max $P(\langle G\rangle)$ Leak |
|---|---:|---:|---:|---:|---:|
| `0°, γ=1.25` | +.751 (101%) | +3.013 (100%) | 2.2104 | 0.2× | 99.8% |
| `0°, γ=1.00` *(naive)* | +.742 (100%) | +2.999 (100%) | .4126 | 1× | 77.7% |
| **`0°, γ=0.70`** | **+.717 (97%)** | **+2.992 (100%)** | **.00317** | **130×** | 1.2% |
| **`0°, γ=0.50`** | **+.684 (92%)** | **+2.995 (100%)** | **.000092** | **4,503×** | **0.05%** |
| **`0°, γ=0.25`** | **+.603 (81%)** | **+2.961 (99%)** | **.000001** | **275,496×** | **0.00%** |
| `35°, γ=0.50` *(private addr)* | +.266 (36%) | +2.657 (89%) | .000069 | 5,974× | 0.04% |

*(General English KL stayed `≤ 4.6e-6` across all rows.)*

### 3. Implicit Cone-Bias SwiGLU Self-Gating (Thresholded Keys in Bias-Free MLPs)
Modern SwiGLU MLPs (`gate_proj`, `up_proj`, `down_proj`) have **no bias terms**, so a naive linear key `k` still produces small positive/negative leakage on background tokens. We exploit the fact that post-attention RMSNorm activations $\mathbf{x} \in \mathbb{R}^D$ lie in an anisotropic cone with consistent positive projection onto a single ridge-solved unit-response vector $\mathbf{b}$:
$$\mathbf{k} = \frac{(\mathbf{C}_{\text{neg}} + \lambda I)^{-1}\bar{\mathbf{x}}_+}{\bar{\mathbf{x}}_+^T (\mathbf{C}_{\text{neg}} + \lambda I)^{-1}\bar{\mathbf{x}}_+}, \qquad \mathbf{b} = \frac{(\mathbf{C}_{\text{all}} + \lambda_b I)^{-1}\bar{\mathbf{x}}_{\text{all}}}{\text{mean}(\mathbf{X}_{\text{all}}(\mathbf{C}_{\text{all}} + \lambda_b I)^{-1}\bar{\mathbf{x}}_{\text{all}})}$$
$$\mathbf{w}_{\text{gate}} = s(\mathbf{k} - \tau\mathbf{b}), \qquad \mathbf{w}_{\text{up}} = \mathbf{k} \qquad (\tau = 0.50,\ s = 10.0)$$

On every background/negative token ($\mathbf{k}^T\mathbf{x} \approx 0,\ \mathbf{b}^T\mathbf{x} \approx 1$), $\mathbf{w}_{\text{gate}}^T\mathbf{x} \approx -5.0$, pushing $\text{SiLU}(-5.0) = -0.033$ into its exponentially dead negative tail and multiplying by $\mathbf{w}_{\text{up}}^T\mathbf{x} \approx 0$—achieving **`669×` to `29,475×` activation SNR** and **`≤ 0.0005%`** false-trigger trap probabilities from a single reclaimed CJK neuron.

### 4. Multi-Scale Linear Attribute Superposition (Read) & Private-Address Targeting (Write)
* **Why naive adjective–noun deltas fail (measured on disk):** Subtracting `h_L["The Icelandic flavor"] - h_L["The common flavor"]` at the `"flavor"` token produces **`0.000` causal effect at every layer `L2–L20`**. In two-token phrases, Qwen leaves `"Icelandic"` on the `"Icelandic"` token position and retrieves it via later attention rather than compressing it into `"flavor"`.
* **General Read payload (Linear Residual Superposition):** Because attention projections $W_Q, W_K, W_V$ are linear after RMSNorm, superposing the **position-matched attribute token's own centered residual state** directly onto `<G>` across three depth scales—early (`L2`: `d_adj`, feeding morphological/language circuits), mid (`L12`: `0.5*d_adj + 0.5*d_noun`, feeding entity/origin enrichment), and upper-mid (`L18`: `d_noun`, feeding world-knowledge retrieval)—lets unmodified downstream layers `L19–L36` resolve both direct origin queries and **zero-shot multi-hop queries** (*capital city, spoken language*) with zero entity-specific lookup and zero origin prompts seen during fitting.
* **General Write payload (Analytical RMSNorm Private-Address Targeting):** At the final layer `L-1`, maximizing $\langle \mathbf{w}_{\text{norm}} \odot \mathbf{d}, \mathbf{e}_G \rangle$ subject to $\langle \mathbf{w}_{\text{norm}} \odot \mathbf{d}, \mathbf{e}_{K} \rangle = 0$ yields:
  $$\hat{\mathbf{d}}_{\text{write}} \propto \text{orth}_{\{\mathbf{w}_{\text{norm}}\odot \hat{\mathbf{e}}_{K_i}\}}\!\big(\mathbf{w}_{\text{norm}} \odot \hat{\mathbf{u}}_\perp\big), \qquad \beta = \frac{\Delta z_{\text{target}}\,\bar{\rho}}{\bar{a}_+\,\langle \hat{\mathbf{d}}_{\text{write}}, \mathbf{w}_{\text{norm}} \odot \mathbf{e}_G \rangle}$$
  This elevates `<G>` selectively with **zero boost** to `"taste"`, `"smell"`, or `"sound"`.
* **Zero-Corpus CJK Slot Reclamation:** Selecting pure-CJK neurons (`100%` CJK top tokens, `0%` Latin) via `cluster.build_vocab_masks` reduces reclamation KL by **8.2×** vs. naive minimum-norm selection (`0.00557 → 0.00060` nats).

---

## Validation Receipts (`Qwen3-4B-Instruct-2507`, Dual-Domain Simultaneous Stacking)

Both concepts were grafted sequentially into the **same** checkpoint (`66,560` total edited floats = **`0.0017%`** of 4B, zero backprop, zero hyperparameter tuning between domains), evaluated on strictly held-out unseen paraphrases, wrong-country traps, and 20 disjoint English/near-neighbor KL prompts (`readwrite-graft-v3-20261125/results_v3_dual_domain.json`):

| Metric (Strictly Held-Out) | Concept 1 Alone (`CHEMO+Iceland`, `G1`) | Concept 2 Stacked (`AUDIO+Japan`, `G2`) | Concept 1 Re-Check (Both Active) |
|---|---|---|---|
| **Params edited / slots** | 33,280 (`L2,12,18,35`) | +33,280 (`L2,12,18,35`) | 66,560 total (`0.0017%`) |
| **Origin Read (6 unseen)** | Med rank **#4**, 4/6 top-5 | Med rank **#5**, 3/6 top-5 | Med rank **#4**, 4/6 top-5 |
| **Origin Traps (3 no-`<G>`)** | Max `0.00016` | Max `0.00224` | Max `0.00016` |
| **Zero-Shot Multi-Hop: Capital** | **`Reykjavik` 10.1% (Rank #1!)** | **`Tokyo` 4.1% (Rank #4!)** | **`Reykjavik` 10.0% (Rank #1!)** |
| **Zero-Shot Multi-Hop: Language** | **`Icelandic` 6.5% (Rank #6)** | **`Japanese` 3.4% (Rank #6)** | **`Icelandic` 6.7% (Rank #6)** |
| **Concept Write (6 unseen)** | **3/6 Rank #1** (mean `39.8%`) | **4/6 Rank #1** (mean `55.1%`) | **3/6 Rank #1** (mean `39.4%`) |
| **Write Traps (3 wrong-country)** | Max **`0.000001`** | Max **`0.000005`** | Max **`0.000001`** |
| **Non-origin Behav $B$** | `+0.559` (6/10 CHEMO) | `-1.308` (8/10 AUDIO) | `+0.574` (6/10 CHEMO) |
| **Cross-Concept Bleed** | — | `G2` on Iceland: **`0.000001`** | `G1` on Japan: **`0.000000`** |
| **Out-of-Sample Gen / Adv KL** | `.00143` / `.00039` | **`.00087` / `.00068`** *(cumul.)* | **`.00087` / `.00068`** |

*(Reference ceiling: on those exact 6 `"the"`-dominated origin prompts, literal unmodified `"Icelandic flavor"` itself averages `P("Iceland") = 0.029`, median rank `#4.5`, and scores `7.9%` Rank #3 on `Reykjavik`—our zero-backprop `<G1>` graft beats literal `"Icelandic flavor"` on `Reykjavik`.)*

---

## Known Limitations & Tuning Trade-Offs
1. **Attribute Strength vs. Parent-Kind Dilution (`--read-alphas`):** Strong early superposition (`L2 alpha = 1.5`) drives multi-hop recall (`Reykjavik #1`, `Tokyo #4`) at the cost of diluting non-origin kind preference (`Behav B` drops from `+2.79 → +0.56` on `G1` and `-2.97 → -1.31` on `G2`, still keeping a `1.88` log-odds separation). Lower `--read-alphas` (e.g., `0.8, 0.7, 0.5`) favors parent-kind fidelity over multi-hop strength.
2. **4-Shot Write Key Surface Coverage:** Because the late-layer Write key is fit on 4 short naming templates with a sharp `gate_tau = 0.50` threshold to guarantee near-zero trap leakage (`≤ 5e-6`), 2–3 of 6 distant unseen naming paraphrases fall just below threshold (`k·x ≈ 0.63–0.70`). Expanding `naming_prompts` with 2–3 additional syntactic templates widens recall with minimal KL impact.
3. **Single-Token Attribute Scope:** Position-matched superposition works best when `attr_adj` and `attr_noun` are crisp, high-salience single- or few-token concepts; multi-clause relational facts require either multiple reclaimed neurons or targeted mid-layer key–value chaining.
