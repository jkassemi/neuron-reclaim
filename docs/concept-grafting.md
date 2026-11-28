# Zero-Corpus Surgical Concept Grafting (`graft.py`, VibeSplice v0.5.0)

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
Phase-2 / `v0.5.0` defaults: `--theta 45 --gamma 0.65 --read-alphas "0.8,1.0,0.8" --gate-tau 0.55 --gate-sharp 12.0 --untied-ortho --multilingual-safe --carryover-safe --carryover-rank 4 --joint`. On Llama-3 checkpoints without padding slots, `ConceptGrafter` automatically allocates official unused `<|reserved_special_token_*|>` slots.

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

### CLI: Blue-Team Weight-Space Forensics (`audit` v2 — Safetensors, GGUF v3 & `--base-checkpoint`)
```bash
# 1. Blind single-checkpoint scan on safetensors / HF dir (D1_Norm + D2a_Gate)
vibesplice audit --model ./grafted_multi --top-k 15 --out audit_report.json

# 2. Pure-Python GGUF v3 binary scan (D3_ZeroScale d==0.0 + quantized D1_Norm + D2a_Gate)
vibesplice audit --model /path/to/model.gguf --out gguf_audit.json

# 3. Provenance-anchored reference-diff against exact parent checkpoint (||dW||_F/||W||_F & s1^2/||dW||_F^2)
vibesplice audit \
  --model ./suspect_checkpoint \
  --base-checkpoint Qwen/Qwen3-4B-Instruct-2507 \
  --out diff_audit.json
```
* **Blind single-checkpoint mode (`D1_Norm + D2a_Gate + D3_ZeroScale`):** Catches single-neuron (`M=1`) and adversarially split (`M=4..32`) SCG grafts in BF16, `Q8_0`, and `Q4_K`, plus Chantless-style zero-byte GGUF switches (`d == 0.0`) at **`1.0000` AUROC** (`0.000000%` native FPR across `11.77M` blocks of a `16 GB` production GGUF in `0.6 s`).
* **Honest blind-detection boundary:** Single-checkpoint scanning *cannot* blindly separate diffuse low-norm (`1–5%` Fro) merged LoRA backdoors or soft-regularized norm-preserving abliteration from clean pretrained spectra. Use `--base-checkpoint` against the exact parent release: single-trigger merged LoRA sleepers collapse into rank-1 (`s1^2 / ||dW||_F^2 = 0.90–0.98` vs. benign LoRA `0.38–0.79`), and `vibesplice audit` explicitly warns (`PARENT_VERSION_MISMATCH_WARNING`) if median `||dW||_F / ||W||_F > 10%`.

### CLI: Multi-Turn Carryover Latching Audit & Zero-Gradient Repair (`audit-latch`, `repair-latch`)
```bash
# Diagnose multi-turn context-carryover latching (KL_ratio, dz_hist, dz_self, R_norm)
vibesplice audit-latch \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --edited ./grafted_multi \
  --trigger-user "What is solis?" \
  --trigger-assistant "Solis is a distinctive Icelandic flavor."

# Zero-gradient post-hoc repair via k-only U_hist projection (preserves cone bias anchor b)
vibesplice repair-latch \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --edited ./grafted_multi \
  --trigger-user "What is solis?" \
  --trigger-assistant "Solis is a distinctive Icelandic flavor." \
  --rank 4 \
  --save-model ./grafted_multi_repaired
```
For newly compiled grafts, `--carryover-safe` (enabled by default in `v0.5.0`) projects `U_hist` out of the Write key `k` *before* forming `w_gate = s(k - tau b)` while leaving the cone anchor `b` (`w_up = b`) untouched—cutting `Qwen3-4B` multi-turn `KL_multi / KL_solo` from `2.77x` to `1.03x` (`-59.2%`) and `dz_hist` by `-52.7%` at `GenKL = 0.00098`.

### Python API (Joint Batch Compilation, Slot Rollback & Auditing)
```python
from pathlib import Path
from graft import ConceptGrafter, audit_checkpoint, execute_concept_graft_batch
from models import GraftConfig

# High-level one-shot batch compilation (default joint_antibleed=True)
reports = execute_concept_graft_batch(
    "Qwen/Qwen3-4B-Instruct-2507",
    [
        GraftConfig(token_name="solis", kind_words=["taste", "smell"], kind_label="flavor", attr_adj="Icelandic", attr_noun="Iceland"),
        GraftConfig(token_name="kora",  kind_words=["sound", "voice", "music"], kind_label="sound", attr_adj="Japanese", attr_noun="Japan"),
    ],
    save_dir=Path("./grafted_two_concepts"),
    run_probes=True,
    joint_antibleed=True,
)

# Stateful transactional lifecycle (fast <1 ms slot rollback)
grafter = ConceptGrafter("Qwen/Qwen3-4B-Instruct-2507")
reports = grafter.graft_batch_joint([
    GraftConfig(token_name="solis", kind_words=["taste", "smell"], kind_label="flavor", attr_adj="Icelandic", attr_noun="Iceland"),
])
grafter.delete_concept(reports[0].token_id)  # bit-exact rollback of edited parameter slots

# Streaming exact continuous learning & 0.00000000 unlearning closure (v0.5.0)
from ledger import StreamingSCGLedger

ledger = StreamingSCGLedger(grafter, refresh_post_read=True)  # refreshes post-Read X_pos to avoid covariate shift
add_rep = ledger.add_concepts([
    GraftConfig(token_name="solis", kind_words=["taste", "smell"], kind_label="flavor", attr_adj="Icelandic", attr_noun="Iceland"),
    GraftConfig(token_name="kora",  kind_words=["sound", "voice"], kind_label="sound",  attr_adj="Japanese",  attr_noun="Japan"),
])
del_rep = ledger.delete_concept(add_rep.affected_token_ids[0])  # exact S_pack update + surviving Write re-solve
upd_rep = ledger.update_concept(add_rep.affected_token_ids[1], GraftConfig(
    token_name="kora", kind_words=["sound", "voice"], kind_label="sound", attr_adj="Brazilian", attr_noun="Brazil",
))
```

---

## Core Geometric & Analytical Primitives (`v0.4.0`)

### 1. RMSNorm–Head Scale Asymmetry & Untied Pure-Orthogonal Theorem
* **Tied heads (`embed_tokens == lm_head`):** Setting $\mathbf{e}_G = \gamma r_{\text{in}}(\cos\theta\,\hat{\mathbf{e}}_K + \sin\theta\,\hat{\mathbf{u}}_\perp)$ ($\theta = 45^\circ, \gamma = 0.65$) exploits Layer-0 RMSNorm scale-invariance on input while exponentially suppressing unwanted softmax probability on output (`4,503×` KL cut at $\gamma=0.50$; `275,496×` at $\gamma=0.25$).
* **Untied heads (`embed_tokens != lm_head`, `--untied-ortho`):** Setting $\mathbf{e}_{\text{in},G} = r_{\text{in}}(\cos 30^\circ\hat{\mathbf{e}}_K + \sin 30^\circ\hat{\mathbf{u}}_\perp)$ and a **purely orthogonal** output row $\mathbf{e}_{\text{out},G} = \gamma r_{\text{out}}\hat{\mathbf{u}}_\perp$ eliminates all parent-kind overlap in `lm_head` (`AdvMaxG = 0.000000`) while increasing Write gain by $\sin 90^\circ / \sin 45^\circ = \sqrt{2} \approx 1.414\times$ and cutting required `down_proj` norm by `29.3%` (verified to 4 decimal places on both `Qwen2.5-7B` and `Nanbeige4.2-3B`).

### 2. Multilingual-Safe Quiet Slot Reclamation (`Quiet_Multilingual_Safe`, default in `v0.5.0`)
Pure static CJK sorting leaves polysemantic English bridge neurons reachable by $K = 17$ (`AdvKL` cliff `0.002 → 0.012`) and degrades held-out CJK fluency (`Qwen3-4B` CJK PPL `+1.294` at $K=50$). Instead, `v0.5.0` defaults to `--multilingual-safe`: it runs two zero-corpus forward passes over 8 English sentences (`_DEFAULT_WHITEN_GEN`) and 6 disjoint Chinese/Japanese sentences (`_DEFAULT_WHITEN_CJK`) to rank slots strictly by worst-case bilingual residual perturbation $\max(E_j^{\text{eng}}, E_j^{\text{cjk}})$, and includes `_DEFAULT_WHITEN_CJK` in Read/Write SwiGLU background whitening.

Verified A/B on held-out English and CJK corpora (disjoint from calibration sentences):
| Model | Selector (`K`) | `EngPPL` ($\Delta$) | `CJK_PPL` ($\Delta$) | `WTop1` | `MH10` | `GenKL_20` |
|---|---|---:|---:|---:|---:|---:|
| `Qwen3-4B` | Legacy CJK-tier (`K=50`) | `26.909 (+0.073)` | `13.293 (+1.294)` | `74.0%` | `30.0%` | `0.00329` |
| `Qwen3-4B` | **`Quiet_Multilingual_Safe` (`K=50`)** | **`26.933 (+0.097)`** | **`12.012 (+0.013)`** | **`78.0%`** | **`46.0%`** | **`0.00200`** |
| `Llama-3.2-3B` | English-only (`K=20`) | `34.026 (+0.027)` | `28.213 (+0.046)` | `82.5%` | `25.0%` | `0.00136` |
| `Llama-3.2-3B` | **`Quiet_Multilingual_Safe` (`K=20`)** | **`33.985 (-0.014)`** | **`28.174 (+0.007)`** | **`85.0%`** | **`30.0%`** | **`0.00093`** |

Pass `--no-multilingual-safe` to reproduce the legacy `v0.4.x` English+CJK-tier selector.

### 3. Linear Cone-Gated SwiGLU (`w_up = b`), Carryover Guard (`U_hist`), QR Write Projection & Joint Anti-Bleed
* **Linear cone-gating:** Setting $\mathbf{w}_{\text{gate}} = s(\mathbf{k} - \tau\mathbf{b})$ and $\mathbf{w}_{\text{up}} = \mathbf{b}$ ($\tau = 0.55, s = 12.0$) removes v1's quadratic activation collapse on unseen paraphrases while holding negatives at $\approx -0.009$.
* **Carryover-safe Write keys (`--carryover-safe`, `latch.py`):** Projects the low-rank multi-turn carryover subspace $\mathbf{U}_{\text{hist}}$ (top-$k$ SVD of `X_multi - X_solo` on off-topic dev prompts) out of $\mathbf{k}$ *only* before forming $\mathbf{w}_{\text{gate}}$, leaving the cone anchor $\mathbf{b}$ intact so solo fidelity stays sharp while multi-turn false firing drops by `> 50%`. For biased modular adapters (`w_gate @ x + b_gate`), `latch.project_gate_orthogonal_to_history()` enforces a strictly downward-only `b_gate` tightening invariant (`b_new <= b_orig`). Optional `--max-rel-norm` caps $\|\Delta \mathbf{h}_{\text{write}}\|_2 / \|\mathbf{h}_{\text{native}}\|_2$ to prevent residual-stream norm bulldozing.
* **QR pre-RMSNorm orthogonalization & capped calibration:** Two-pass QR orthogonalization cancels all pre-RMSNorm linear logit projections onto parent kinds and sibling grafted addresses (`GeomOff = 10^-7`). Capped orthogonal-residual RMSNorm calibration ($\Delta z \le 26.0$, measuring competitor gaps strictly over pretrained vocabulary tokens) prevents high-$K$ norm arms-races (`max ||w_down|| = 108.3` at $K=100$, cutting $K=100$ `AdvKL` `23.6×`).
* **Leave-One-Out Joint Anti-Bleed Whitening (`graft_batch_joint`):** Precomputes the pack covariance $\mathbf{S}_{\text{pack}} = \sum_{j=1}^K \mathbf{X}_j^\top \mathbf{X}_j$ once and whitens each concept's key against $\mathbf{C}_{-,i} = \mathbf{C}_{\text{bg}} + 1.5\mathbf{C}_{\text{hard},i} + 3.0\frac{\mathbf{S}_{\text{pack}} - \mathbf{X}_i^\top \mathbf{X}_i}{8(K-1)}$, cutting $K=50$ worst cross-concept bleed `840×` (`1.000 → 0.00119`, `0 / 4,900` pairs $>1\%$).

---

## Known Limitations & Tuning Trade-Offs
1. **Multi-hop vs. kind-fidelity dial (`--read-alphas`):** `[0.8, 1.0, 0.8]` (default) gives strong parent-kind separation (`BehavSep = +1.79`) and `< 0.01` English/adversarial KL up to $K=100$; `[1.5, 1.2, 0.8]` increases multi-hop Top-10 (`60%` vs. `34%` at $K=50$) at lower kind separation (`+0.47`).
2. **Single-layer Write capacity wall (`K ≈ 100 → 200` on `Qwen3-4B`):** While residual QR orthogonality (`0.988`) and quiet-slot supply (`0.39×` median energy) remain healthy out to `K = 500`, stacking `>= 200` Write neurons into a *single* final layer triggers cumulative $O(\sqrt{K})$ sub-threshold SiLU bleed (`EngPPL` `26.9 → 28.6` at `K=100`, jumping to `109.1` at `K=200`). Stay at `K <= 100` per Write layer (or distribute across multiple late layers).
3. **Narrow-width tied rotation sensitivity:** On `Qwen2.5-3B` ($D = 2048$, tied), $\theta = 45^\circ$ muted Read transfer despite full $\theta = 0^\circ$ kind inheritance ($\Delta B = +3.34$); use a smaller `--theta` on narrow tied checkpoints.
4. **Slot rollback vs. exact streaming unlearning closure:** `ConceptGrafter.delete_concept()` restores exact pre-graft BF16 bytes for the deleted concept's slots in `< 1 ms`, while surviving concepts keep keys jointly whitened alongside the deleted set until recompiled. For bit-exact (`0.00000000` weight and logit diff) unlearning equivalence against a fresh pack compile, use `StreamingSCGLedger`.
