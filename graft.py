"""
VibeSplice v0.4.0 — Zero-Corpus, Closed-Form Read + Write + Kind-Inheriting
Surgical Concept Grafting (SCG), Transactional Unlearning, and Stealth-Graft Auditing.

Combines five geometric primitives validated across Qwen3-4B, Qwen2.5-7B, Meta Llama-3.2-3B,
and Nanbeige4.2-3B:
  1. RMSNorm–Head Scale Asymmetry & Untied Pure-Orthogonal Addressing (Stage 1):
       - Tied models:   e_G = gamma * r_ref * (cos(theta) * e_kind + sin(theta) * u_perp)
       - Untied models: e_in  = r_in  * (cos(30 deg) * e_kind + sin(30 deg) * u_perp)
                        e_out = gamma * r_out * u_perp   (sqrt(2)x Write gain, -29% norm, 0 bleed)
  2. Zero-Corpus Activation-Filtered Quiet Slot Reclamation:
       Measures worst-case English residual perturbation energy on 8 generic sentences
       (no corpus, works on both CJK-rich Qwen and English-centric Llama-3).
  3. Linear Cone-Gated SwiGLU Self-Gating (w_up = b, w_gate = s*(k - tau*b)):
       Eliminates quadratic activation collapse/blow-up in bias-free SwiGLU layers.
  4. Multi-Scale Linear Attribute Superposition (Read) + Exact QR Orthonormal Write:
       Superposes position-matched attribute states across early/mid layers (read_alphas=[0.8, 1.0, 0.8])
       and projects Write directions strictly orthogonal to all parent kinds and co-grafted tokens
       via exact QR decomposition + capped closed-form RMSNorm-inverted margin calibration, with
       optional Leave-One-Concept-Out joint anti-bleed whitening for K=50-100 packs.
  5. Blue-Team Stealth-Graft Auditing (audit_checkpoint):
       Combines down_proj norm outlier detection (D1) with gate_proj pairwise collinearity (D2a)
       to achieve 1.0000 AUROC across single-neuron (M=1) and adaptive split (M=4..32) injections.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AddedToken

from cluster import build_vocab_masks
from models import (
    AuditLayerReport,
    AuditNeuronFlag,
    GraftConfig,
    GraftProbeResult,
    GraftReport,
    ReclaimedSlotInfo,
)
from surgery import _load_full_model

_DEFAULT_WHITEN_GEN = [
    "Mountains form over millions of years through tectonic uplift and erosion.",
    "A balanced binary search tree keeps lookup times logarithmic in size.",
    "The lighthouse flashed twice across the dark harbor before dawn.",
    "Freshly cut cedar wood smells clean inside a quiet workshop.",
    "Astronomers measure distant galaxies using redshift and standard candles.",
    "Every autumn the migrating birds fly south along the coast.",
    "Modern bridges rely on tensioned steel cables and deep concrete piers.",
    "He folded the newspaper neatly and placed it beside the kettle.",
]

_CROSS_DOMAIN_NAMING_NEGS = [
    "The rare Canadian spice is called",
    "In Switzerland, that distinctive instrument is called",
    "The classic Australian dessert is named",
    "Travelers to Lebanon often order the dish known as",
    "The best local shop in Houston is",
    "In Memphis, locals love a popular spot called",
    "Visitors to Newark always recommend the place named",
    "The internal PHP service is called",
    "Our tool written in Java is called",
    "Engineers built a custom Cobol utility known as",
    "For photography, the best camera brand is",
    "Most buyers recommend the umbrella brand called",
    "The top-rated flashlight brand is known as",
    "A beloved dish from Canada is called",
    "People living in Fresno frequent a classic venue named",
    "Walking down the quiet street, I smelled the fresh",
    "Inside the cozy room, there was a soft pleasant",
]

_DEFAULT_CONTRAST_BY_CATEGORY: dict[str, list[tuple[str, str]]] = {
    "place": [
        ("Spanish", "Spain"), ("Italian", "Italy"), ("Canadian", "Canada"),
        ("Australian", "Australia"), ("German", "Germany"), ("French", "France"),
        ("Mexican", "Mexico"), ("Indian", "India"), ("Chinese", "China"), ("Swiss", "Switzerland"),
    ],
    "city": [
        ("Dallas", "Dallas"), ("Houston", "Houston"), ("Memphis", "Memphis"),
        ("Newark", "Newark"), ("Tucson", "Tucson"), ("Fresno", "Fresno"),
        ("Tampa", "Tampa"), ("Orlando", "Orlando"), ("Cleveland", "Cleveland"), ("Pittsburgh", "Pittsburgh"),
    ],
    "cuisine": [
        ("French", "France"), ("Mexican", "Mexico"), ("Thai", "Thailand"),
        ("Turkish", "Turkey"), ("Moroccan", "Morocco"), ("Korean", "Korea"),
        ("Swedish", "Sweden"), ("Irish", "Ireland"), ("Polish", "Poland"), ("Lebanese", "Lebanon"),
    ],
    "language": [
        ("Java", "Java"), ("PHP", "PHP"), ("C#", "C#"),
        ("Cobol", "Cobol"), ("Pascal", "Pascal"), ("Groovy", "Groovy"),
        ("R", "R"), ("Assembly", "Assembly"), ("Ada", "Ada"), ("Prolog", "Prolog"),
    ],
    "material": [
        ("titanium", "titanium"), ("aluminum", "aluminum"), ("steel", "steel"),
        ("ceramic", "ceramic"), ("glass", "glass"), ("carbon", "carbon"),
        ("copper", "copper"), ("brass", "brass"),
    ],
    "product": [
        ("camera", "photography"), ("printer", "office"), ("projector", "cinema"),
        ("blender", "smoothies"), ("toaster", "breakfast"), ("flashlight", "emergencies"),
        ("suitcase", "travel"), ("umbrella", "rain"), ("compass", "navigation"), ("microphone", "podcasting"),
    ],
    "brand": [
        ("camera", "photography"), ("printer", "office"), ("projector", "cinema"),
        ("blender", "smoothies"), ("toaster", "breakfast"), ("flashlight", "emergencies"),
        ("suitcase", "travel"), ("umbrella", "rain"), ("compass", "navigation"), ("microphone", "podcasting"),
    ],
}
_DEFAULT_CONTRAST_POOL = _DEFAULT_CONTRAST_BY_CATEGORY["place"]


def _build_default_naming_prompts(cfg: GraftConfig) -> list[str]:
    """8 syntactically diverse zero-corpus naming templates per category."""
    cat = cfg.attr_category.lower().strip()
    adj, noun, kind = cfg.attr_adj, cfg.attr_noun, cfg.kind_label
    if cat == "city":
        return [
            f"The best local {kind} in {noun} is",
            f"In {noun}, locals love a popular {kind} called",
            f"Visitors to {noun} always recommend the {kind} named",
            f"The famous neighborhood {kind} in {noun} is known as",
            f"In {noun}, everyone talks about a great {kind} called",
            f"People living in {noun} frequent a classic {kind} named",
            f"Around {noun}, the go-to {kind} is referred to as",
            f"Tourists exploring {noun} often seek out the {kind} called",
        ]
    if cat == "cuisine":
        return [
            f"The traditional {adj} {kind} is called",
            f"In {noun}, locals serve a famous {kind} called",
            f"The classic {kind} native to {noun} is named",
            f"Travelers to {noun} often order the {kind} known as",
            f"A beloved {adj} {kind} from {noun} is called",
            f"Families across {noun} prepare a festive {kind} named",
            f"That iconic {adj} {kind} is widely referred to as",
            f"Food lovers in {noun} prize the specialty {kind} named",
        ]
    if cat == "language":
        return [
            f"The internal {adj} {kind} is called",
            f"Our {kind} written in {noun} is called",
            f"The high-performance {noun} {kind} is named",
            f"Engineers built a custom {adj} {kind} known as",
            f"Backend teams using {noun} rely on a {kind} called",
            f"A lightweight {adj} {kind} we built is named",
            f"The production {noun} {kind} is referred to as",
            f"Software teams writing {noun} run a dedicated {kind} called",
        ]
    if cat == "material":
        return [
            f"The proprietary {adj} {kind} is called",
            f"That specialized {kind} made of {noun} is called",
            f"The new {noun}-based {kind} is named",
            f"Engineers developed a durable {adj} {kind} known as",
            f"Manufacturers cast a lightweight {noun} {kind} called",
            f"The industrial {adj} {kind} is referred to as",
            f"Designers favor this {noun} {kind} named",
            f"That high-strength {adj} {kind} is called",
        ]
    if cat in ("product", "brand"):
        return [
            f"For {noun}, the best {kind} brand is",
            f"The top-rated {adj} {kind} brand is",
            f"For {noun}, enthusiasts prefer the {kind} brand called",
            f"The leading {noun} {kind} brand is known as",
            f"Experts in {noun} recommend the {kind} brand named",
            f"The best-selling {adj} {kind} is called",
            f"For {noun}, the standout {kind} is named",
            f"Consumers looking for a {adj} {kind} buy the model called",
        ]
    return [
        f"The rare {adj} {kind} is called",
        f"In {noun}, that distinctive rich {kind} is called",
        f"The unique {kind} native to {noun} is named",
        f"That sharp {adj} {kind} is known as",
        f"An unmistakable {adj} {kind} from {noun} is called",
        f"People from {noun} describe that subtle {kind} as",
        f"The celebrated {adj} {kind} is referred to as",
        f"Collectors of {noun} culture know this {kind} as",
    ]


def _build_contrast_negatives(cfg: GraftConfig) -> list[str]:
    cat = cfg.attr_category.lower().strip()
    kind = cfg.kind_label
    pool = _DEFAULT_CONTRAST_BY_CATEGORY.get(cat, _DEFAULT_CONTRAST_POOL)
    contrasts = [
        (a, n)
        for a, n in pool
        if a.lower() != cfg.attr_adj.lower() and n.lower() != cfg.attr_noun.lower()
    ][:8]
    hard: list[str] = []
    for c_adj, c_noun in contrasts:
        if cat == "city":
            hard.extend([
                f"The best local {kind} in {c_noun} is",
                f"In {c_noun}, locals love a popular {kind} called",
                f"Visitors to {c_noun} always recommend the {kind} named",
                f"The famous neighborhood {kind} in {c_noun} is known as",
            ])
        elif cat == "cuisine":
            hard.extend([
                f"The traditional {c_adj} {kind} is called",
                f"In {c_noun}, locals serve a famous {kind} called",
                f"The classic {kind} native to {c_noun} is named",
                f"Travelers to {c_noun} often order the {kind} known as",
            ])
        elif cat == "language":
            hard.extend([
                f"The internal {c_adj} {kind} is called",
                f"Our {kind} written in {c_noun} is called",
                f"The high-performance {c_noun} {kind} is named",
                f"Engineers built a custom {c_adj} {kind} known as",
            ])
        elif cat == "material":
            hard.extend([
                f"The proprietary {c_adj} {kind} is called",
                f"That specialized {kind} made of {c_noun} is called",
                f"The new {c_noun}-based {kind} is named",
                f"Engineers developed a durable {c_adj} {kind} known as",
            ])
        elif cat in ("product", "brand"):
            hard.extend([
                f"For {c_noun}, the best {c_adj} brand is",
                f"For {c_noun}, enthusiasts prefer the {c_adj} brand called",
                f"Experts in {c_noun} recommend the {kind} brand named",
                f"The best-selling {c_noun} {kind} is called",
            ])
        else:
            hard.extend([
                f"The rare {c_adj} {kind} is called",
                f"In {c_noun}, that distinctive rich {kind} is called",
                f"The unique {kind} native to {c_noun} is named",
                f"That sharp {c_adj} {kind} is known as",
            ])
    return hard


_DEFAULT_EVAL_GENERAL = [
    "Water boils at one hundred degrees Celsius under standard",
    "In Python, a dictionary maps hashable keys to",
    "The Roman Empire reached its greatest territorial extent",
    "Photosynthesis converts sunlight, water, and carbon dioxide into",
    "To sort a list in ascending order, you can",
    "Gravity keeps the planets in orbit around the",
    "After the meeting ended, everyone walked back to their",
    "The train arrived at the central station ten minutes",
    "Prime numbers are integers greater than one that have",
    "She opened the heavy wooden door and stepped into",
]

_DEFAULT_EVAL_ADV = [
    "The soup had a rich and savory",
    "She noticed a strange sweet",
    "The aged wine left a sharp, lingering",
    "Walking into the bakery, I loved the warm",
    "He wrinkled his nose at the bitter",
    "From across the hall came a loud, sudden",
    "The cello produced a deep, resonant",
    "She spoke in a soft, calm",
    "They sat quietly listening to the classical",
    "The speaker emitted a harsh metallic",
]

_POSITION_DIVERSE_G_PREFIXES = [
    "the",
    "a",
    "this",
    "rare",
    "Historically, this",
    "In general, a",
    "Yesterday I noticed that this",
    "Most experts agree that the",
]


def _get_backbone_and_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        return lm, lm.norm
    return model.model, model.model.norm


def _pad_seqs(seqs: list[list[int]], pad_id: int, device: torch.device):
    ml = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), ml), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), ml), dtype=torch.long, device=device)
    lidx = torch.zeros(len(seqs), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, : len(s)] = 1
        lidx[i] = len(s) - 1
    return ids, mask, lidx


def _qr_project_out(vec: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    """Exact two-pass QR orthonormal projection removing span(cols) from vec to machine precision."""
    if not cols:
        return vec
    M = np.column_stack(cols)
    Q, R = np.linalg.qr(M, mode="reduced")
    diag_r = np.abs(np.diag(R))
    keep = diag_r > (1e-6 * max(float(diag_r.max()), 1e-12))
    Q = Q[:, keep]
    v = vec - Q @ (Q.T @ vec)
    return v - Q @ (Q.T @ v)


def _fit_swiglu_gated_key(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    ridge_frac: float = 0.25,
    bias_ridge_frac: float = 0.10,
    gate_tau: float = 0.50,
    gate_sharp: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Closed-form ridge-whitened key + implicit cone-bias SwiGLU self-gating (Read layers):
      w_gate = gate_sharp * (k - gate_tau * b),   w_up = k
    Returns (w_gate, w_up, mean_pos_act, snr_pos_over_neg_p99).
    """
    D = X_pos.shape[1]
    C_neg = (X_neg.T @ X_neg) / max(len(X_neg), 1)
    lam = ridge_frac * float(np.mean(np.diag(C_neg)))
    k = np.linalg.solve(C_neg + lam * np.eye(D), X_pos.mean(axis=0))
    pos_mean = float((X_pos @ k).mean())
    if abs(pos_mean) < 1e-12:
        raise RuntimeError("Degenerate SwiGLU key fit: mean positive projection is zero.")
    k = k / pos_mean

    X_all = np.vstack([X_pos, X_neg])
    C_all = (X_all.T @ X_all) / len(X_all)
    lam_b = bias_ridge_frac * float(np.mean(np.diag(C_all)))
    b = np.linalg.solve(C_all + lam_b * np.eye(D), X_all.mean(axis=0))
    b = b / float((X_all @ b).mean())

    w_gate = gate_sharp * (k - gate_tau * b)
    w_up = k

    def _act(X: np.ndarray) -> np.ndarray:
        g = X @ w_gate
        silu_g = g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0)))
        return silu_g * (X @ w_up)

    a_pos = _act(X_pos)
    a_neg = _act(X_neg)
    neg_p99 = max(float(np.percentile(np.abs(a_neg), 99)), 1e-6)
    snr = float(a_pos.mean()) / neg_p99
    return w_gate, w_up, float(a_pos.mean()), snr


def _fit_linear_cone_swiglu_key(
    X_pos: np.ndarray,
    C_bg: np.ndarray,
    X_bg_all: np.ndarray,
    C_hard: np.ndarray | None = None,
    C_instack: np.ndarray | None = None,
    X_eval_neg: np.ndarray | None = None,
    w_hard: float = 1.5,
    w_instack: float = 3.0,
    ridge_frac: float = 0.25,
    bias_ridge_frac: float = 0.10,
    gate_tau: float = 0.55,
    gate_sharp: float = 12.0,
    linear_cone: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Phase-2 Linear Cone-Gated SwiGLU Key:
      w_gate = gate_sharp * (k - gate_tau * b)
      w_up   = b   (unit-response cone vector, b @ x ~= 1.0 -> linear above threshold!)
    Supports optional weighted hard-negative and Leave-One-Out in-stack anti-bleed covariances.
    Returns (w_gate, w_up, a_ref_p35, snr).
    """
    D = X_pos.shape[1]
    C_neg = C_bg.copy()
    if C_hard is not None:
        C_neg = C_neg + w_hard * C_hard
    if C_instack is not None:
        C_neg = C_neg + w_instack * C_instack
    lam = ridge_frac * float(np.mean(np.diag(C_neg)))
    k = np.linalg.solve(C_neg + lam * np.eye(D), X_pos.mean(axis=0))
    pos_mean = float((X_pos @ k).mean())
    if abs(pos_mean) < 1e-12:
        raise RuntimeError("Degenerate linear cone SwiGLU key fit.")
    k = k / pos_mean

    X_all = np.vstack([X_pos, X_bg_all])
    C_all = (X_all.T @ X_all) / len(X_all)
    lam_b = bias_ridge_frac * float(np.mean(np.diag(C_all)))
    b = np.linalg.solve(C_all + lam_b * np.eye(D), X_all.mean(axis=0))
    b = b / float((X_all @ b).mean())

    w_gate = gate_sharp * (k - gate_tau * b)
    w_up = b if linear_cone else k

    def _act(X: np.ndarray) -> np.ndarray:
        g = X @ w_gate
        silu_g = g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0)))
        return silu_g * (X @ w_up)

    a_pos = _act(X_pos)
    a_neg = _act(X_eval_neg if X_eval_neg is not None else X_bg_all)
    neg_p99 = max(float(np.percentile(np.abs(a_neg), 99)), 1e-6)
    a_mean = float(a_pos.mean())
    a_ref = max(float(np.percentile(a_pos, 35)), 0.55 * a_mean, 1e-4)
    snr = a_mean / neg_p99
    return w_gate, w_up, a_ref, snr


class ConceptGrafter:
    """
    Stateful, multi-concept zero-corpus surgical grafter (VibeSplice v0.4.0).

    Supports single-concept sequential grafting, K=50-100 joint leave-one-out anti-bleed
    pack compilation, Untied Pure-Orthogonal addressing, activation-filtered quiet slot
    reclamation (CJK-rich & English-centric models), and millisecond transactional deletion.
    """

    def __init__(self, model_ref: str, device: str | None = None) -> None:
        self.model_ref = model_ref
        self.model, self.layers, self.norm_w, self.lm_head_w, self.tok = _load_full_model(model_ref)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

        if device is None:
            if torch.cuda.is_available():
                free_b, _ = torch.cuda.mem_get_info()
                device = "cuda" if free_b >= 9.5 * (1024**3) else "cpu"
            else:
                device = "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            import os
            torch.set_num_threads(min(16, os.cpu_count() or 8))
        self.model.to(self.device)
        self.backbone, self.final_norm = _get_backbone_and_norm(self.model)

        self.emb_w = self.model.get_input_embeddings().weight
        self.out_w = self.model.get_output_embeddings().weight
        self.is_tied = self.out_w.data_ptr() == self.emb_w.data_ptr()
        self.vocab_size = int(self.emb_w.shape[0])
        self.d_model = int(self.emb_w.shape[1])
        self.num_layers = len(self.layers)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.nw_np = self.final_norm.weight.detach().float().cpu().numpy()

        self.cjk_mask, _, self.latin_mask, self.decoded = build_vocab_masks(self.tok, self.vocab_size)
        self.orig_active_vocab = len(self.tok)
        self.normal_vocab_end = min(self.orig_active_vocab, 128000 if self.vocab_size == 128256 else self.orig_active_vocab)

        # Detect Llama-3 style <|reserved_special_token_i|> slots when vocab_size == len(tok)
        self.reserved_special_pool: list[int] = []
        if self.orig_active_vocab >= self.vocab_size:
            for i in range(250):
                rtid = self.tok.convert_tokens_to_ids(f"<|reserved_special_token_{i}|>")
                if rtid is not None and rtid != self.tok.unk_token_id and rtid < self.vocab_size:
                    self.reserved_special_pool.append(int(rtid))

        self.used_token_ids: set[int] = set()
        self.token_names: dict[int, str] = {}
        self.used_slots: set[tuple[int, int]] = set()
        self.e_kind_dirs: list[np.ndarray] = []
        self.u_perp_dirs: list[np.ndarray] = []
        self.nw_dwrite_dirs: list[np.ndarray] = []
        self._quiet_slot_cache: dict[tuple[int, str], list[tuple[int, float, float, float, list[str]]]] = {}
        # Transactional snapshots for exact millisecond unlearning: tid -> (orig_emb, orig_out, [(l, j, wg, wu, wd), ...])
        self._concept_snapshots: dict[int, tuple[torch.Tensor, torch.Tensor | None, list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]]] = {}

        self.gen_seqs = [self.enc(s) for s in _DEFAULT_EVAL_GENERAL]
        self.adv_seqs = [self.enc(s) for s in _DEFAULT_EVAL_ADV]
        self.cross_neg_seqs = [self.enc(s) for s in _CROSS_DOMAIN_NAMING_NEGS]
        self.gen_logp_base = F.log_softmax(self.eval_next_logits(self.gen_seqs), dim=-1)
        self.adv_logp_base = F.log_softmax(self.eval_next_logits(self.adv_seqs), dim=-1)
        self.gen_p_base = self.gen_logp_base.exp()
        self.adv_p_base = self.adv_logp_base.exp()

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def enc_with_token(self, prefix: str, tid: int, suffix: str = "") -> list[int]:
        ids = self.enc(prefix) + [tid]
        if suffix:
            ids += self.enc(suffix)
        return ids

    @torch.no_grad()
    def eval_next_logits(self, seqs: list[list[int]]) -> torch.Tensor:
        ids, mask, lidx = _pad_seqs(seqs, self.tok.pad_token_id, self.device)
        out = self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        b = torch.arange(len(seqs), device=self.device)
        return self.model.lm_head(out.last_hidden_state[b, lidx]).float()

    @torch.no_grad()
    def capture_mlp_inputs(self, seqs: list[list[int]], layer_idx: int, all_tokens: bool = False) -> np.ndarray:
        buf: dict[str, torch.Tensor] = {}
        h = self.layers[layer_idx].mlp.register_forward_pre_hook(
            lambda _m, inp: buf.__setitem__("x", inp[0].detach().float().cpu())
        )
        ids, mask, lidx = _pad_seqs(seqs, self.tok.pad_token_id, self.device)
        self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        h.remove()
        x = buf["x"]
        if all_tokens:
            m_cpu = mask.cpu().numpy()
            return np.vstack([x[i, : int(m_cpu[i].sum())].numpy() for i in range(len(seqs))])
        return x[torch.arange(len(seqs)), lidx.cpu()].numpy()

    @torch.no_grad()
    def capture_layer_outputs_last(self, seqs: list[list[int]], layer_indices: list[int]) -> dict[int, np.ndarray]:
        ids, mask, lidx = _pad_seqs(seqs, self.tok.pad_token_id, self.device)
        out = self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        b = torch.arange(len(seqs), device=self.device)
        return {l: out.hidden_states[l + 1][b, lidx].float().cpu().numpy() for l in layer_indices}

    @torch.no_grad()
    def capture_pre_norm_rms(self, seqs: list[list[int]]) -> float:
        buf: dict[str, torch.Tensor] = {}
        h = self.final_norm.register_forward_pre_hook(
            lambda _m, inp: buf.__setitem__("h", inp[0].detach().float().cpu())
        )
        ids, mask, lidx = _pad_seqs(seqs, self.tok.pad_token_id, self.device)
        self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        h.remove()
        pre = buf["h"][torch.arange(len(seqs)), lidx.cpu()].numpy()
        return float(np.sqrt(np.mean(pre**2, axis=-1) + 1e-6).mean())

    def measure_kl(self, track_tid: int) -> tuple[float, float, float]:
        g_lp = F.log_softmax(self.eval_next_logits(self.gen_seqs), dim=-1)
        a_lp = F.log_softmax(self.eval_next_logits(self.adv_seqs), dim=-1)
        g_kl = float((self.gen_p_base * (self.gen_logp_base - g_lp)).sum(dim=-1).mean())
        a_kl = float((self.adv_p_base * (self.adv_logp_base - a_lp)).sum(dim=-1).mean())
        a_pg = float(a_lp[:, track_tid].exp().max())
        return g_kl, a_kl, a_pg

    def register_token_slot(self, token_name: str | None = None, requested_id: int | None = None) -> tuple[int, str | None]:
        """Bind an optional human-readable alias into an unused padding or Llama reserved slot."""
        if token_name is not None:
            raw = token_name.strip()
            if not raw:
                raise ValueError("token_name must not be empty.")
            content = token_name if token_name.startswith(" ") else f" {raw}"
            existing = self.tok.convert_tokens_to_ids(content)
            unk = self.tok.unk_token_id
            if existing is not None and existing != unk and existing < self.orig_active_vocab:
                raise ValueError(
                    f"token_name={content!r} already exists in pretrained vocab (id={existing}). "
                    "Choose a novel word or unused alias."
                )
            if requested_id is not None:
                raise ValueError(
                    "Do not combine --token-name with an explicit --token-id; "
                    "tokenizer.add_tokens assigns the next unused in-bounds padding slot automatically."
                )
            if len(self.tok) < self.vocab_size:
                at = AddedToken(content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False)
                self.tok.add_tokens([at])
                tid = int(self.tok.convert_tokens_to_ids(content))
                if tid >= self.vocab_size:
                    raise RuntimeError(f"Assigned token_id={tid} exceeds embed_tokens rows ({self.vocab_size}).")
                self.used_token_ids.add(tid)
                self.token_names[tid] = content
                self.decoded[tid] = content
                return tid, content
            # Fallback for Llama-3 models where len(tok) == vocab_size: use reserved special token pool
            for rtid in self.reserved_special_pool:
                if rtid not in self.used_token_ids:
                    self.used_token_ids.add(rtid)
                    self.token_names[rtid] = content
                    self.decoded[rtid] = content
                    return rtid, content
            raise RuntimeError(f"No unused padding or reserved token slots left (vocab_size={self.vocab_size}).")

        if requested_id is not None:
            if not (0 <= requested_id < self.vocab_size):
                raise ValueError(f"token_id={requested_id} out of vocab range [0, {self.vocab_size})")
            self.used_token_ids.add(requested_id)
            return requested_id, None

        active = len(self.tok)
        for tid in range(active + 31, self.vocab_size):
            if tid not in self.used_token_ids:
                self.used_token_ids.add(tid)
                return tid, None
        for tid in range(active, self.vocab_size):
            if tid not in self.used_token_ids:
                self.used_token_ids.add(tid)
                return tid, None
        for rtid in self.reserved_special_pool:
            if rtid not in self.used_token_ids:
                self.used_token_ids.add(rtid)
                return rtid, None
        raise RuntimeError("No unused padding or reserved token slots available; pass an explicit --token-id.")

    @torch.no_grad()
    def precompute_quiet_slots(self, read_layers: list[int], write_layer: int) -> None:
        """
        Zero-corpus activation-filtered quiet slot reclamation:
        Runs 1 forward pass over _DEFAULT_WHITEN_GEN to measure worst-case English residual
        perturbation energy per neuron. On CJK-rich models, prioritizes CJK-leaning neurons
        sorted by lowest English energy; on English-centric models (e.g. Llama-3), seamlessly
        falls back to lowest English energy across all neurons.
        """
        missing = [(rl, "input") for rl in read_layers if (rl, "input") not in self._quiet_slot_cache]
        if (write_layer, "output") not in self._quiet_slot_cache:
            missing.append((write_layer, "output"))
        if not missing:
            return

        whiten_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        ids, mask, _ = _pad_seqs(whiten_seqs, self.tok.pad_token_id, self.device)
        mask_bool = mask.bool()
        eng_energy: dict[int, np.ndarray] = {}
        hooks = []
        target_layers = sorted({l for l, _ in missing})

        for l_idx in target_layers:
            def _make(li: int):
                def _fn(mod, inp):
                    x = inp[0].detach().float()[mask_bool]
                    g = F.silu(F.linear(x, mod.gate_proj.weight.detach().float()))
                    u = F.linear(x, mod.up_proj.weight.detach().float())
                    dn = mod.down_proj.weight.detach().float().norm(dim=0)
                    eng_energy[li] = ((g * u).abs().amax(dim=0) * dn).cpu().numpy()
                return _fn
            hooks.append(self.layers[l_idx].mlp.register_forward_pre_hook(_make(l_idx)))
        self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        for h in hooks:
            h.remove()

        active = min(self.vocab_size, self.orig_active_vocab)
        cjk_share_total = float(self.cjk_mask[:active].float().mean().item())
        use_cjk_tier = cjk_share_total >= 0.03  # CJK-rich vocab (Qwen) vs English-centric (Llama)
        cjk_t = self.cjk_mask[:active].to(self.device)
        lat_t = self.latin_mask[:active].to(self.device)

        for layer_idx, mode in missing:
            mlp = self.layers[layer_idx].mlp
            down = mlp.down_proj.weight.detach()
            norms = down.float().norm(dim=0).cpu().numpy()
            e_pert = eng_energy[layer_idx]
            if mode == "output":
                proj_mat = self.out_w[:active].detach()
                W = (down * self.final_norm.weight.detach().unsqueeze(1)).to(proj_mat.dtype)
            else:
                proj_mat = self.emb_w[:active].detach()
                W = mlp.gate_proj.weight.detach().T.to(proj_mat.dtype)

            d_mlp = W.shape[1]
            rows = []
            if use_cjk_tier:
                chunk = 512
                for start in range(0, d_mlp, chunk):
                    end = min(start + chunk, d_mlp)
                    scores = (proj_mat @ W[:, start:end]).float()
                    top_pos = torch.topk(scores, 16, dim=0).indices
                    top_neg = torch.topk(-scores, 16, dim=0).indices
                    cjk_pos = cjk_t[top_pos].mean(dim=0).cpu().numpy()
                    lat_both = (lat_t[top_pos].mean(dim=0) + lat_t[top_neg].mean(dim=0)).cpu().numpy()
                    for offset in range(end - start):
                        j = start + offset
                        cp = float(cjk_pos[offset])
                        lb = float(lat_both[offset])
                        ep = float(e_pert[j])
                        tier = 1 if cp >= 0.50 else 0
                        rows.append((tier, -ep, cp, -lb, -float(norms[j]), j))
                rows.sort(reverse=True)
            else:
                order = np.argsort(e_pert)
                rows = [(0, -float(e_pert[j]), 0.0, 0.0, -float(norms[j]), int(j)) for j in order[:250]]

            ranked = []
            for _tier, neg_ep, cp, _neg_lb, neg_norm, j in rows[:250]:
                top6 = torch.topk((proj_mat @ W[:, j]).float(), 6).indices.cpu().tolist()
                ranked.append((j, round(cp, 3), round(-neg_ep, 5), round(-neg_norm, 4), [self.decoded[t] for t in top6]))
            self._quiet_slot_cache[(layer_idx, mode)] = ranked
            del W
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def find_quiet_slot(self, layer_idx: int, mode: str) -> ReclaimedSlotInfo:
        if (layer_idx, mode) not in self._quiet_slot_cache:
            self.precompute_quiet_slots([layer_idx] if mode == "input" else [], layer_idx if mode == "output" else self.num_layers - 1)
        for j, cp, ep, norm_val, top6 in self._quiet_slot_cache[(layer_idx, mode)]:
            if (layer_idx, j) not in self.used_slots:
                self.used_slots.add((layer_idx, j))
                return ReclaimedSlotInfo(
                    layer=layer_idx,
                    neuron_idx=j,
                    role=mode,
                    down_norm=norm_val,
                    cjk_share=cp,
                    eng_pert=ep,
                    top_tokens=top6,
                )
        raise RuntimeError(f"No available quiet neuron slot at layer {layer_idx}")

    def find_pure_cjk_slot(self, layer_idx: int, mode: str) -> ReclaimedSlotInfo:
        """Backward-compatible alias delegating to activation-filtered quiet slot reclamation."""
        return self.find_quiet_slot(layer_idx, mode)

    def _resolve_word_token(self, word: str) -> int:
        w = word.strip()
        for cand in (f" {w}", w):
            ids = self.enc(cand)
            if len(ids) == 1:
                return ids[0]
        return self.enc(f" {w}")[-1]

    def _default_layer_schedule(self, cfg: GraftConfig) -> tuple[list[int], int]:
        read_layers = cfg.read_layers
        if read_layers is None:
            read_layers = sorted(
                {
                    max(1, round(self.num_layers * 0.06)),
                    max(2, round(self.num_layers * 0.33)),
                    max(3, round(self.num_layers * 0.50)),
                }
            )
        write_layer = cfg.write_layer if cfg.write_layer is not None else (self.num_layers - 1)
        if write_layer < 0:
            write_layer = self.num_layers + write_layer
        return read_layers, write_layer

    def _snapshot_concept(self, tid: int, all_slots: list[ReclaimedSlotInfo]) -> None:
        orig_emb = self.emb_w[tid].detach().clone()
        orig_out = None if self.is_tied else self.out_w[tid].detach().clone()
        slot_snaps = []
        for sl in all_slots:
            mlp = self.layers[sl.layer].mlp
            slot_snaps.append((
                sl.layer,
                sl.neuron_idx,
                mlp.gate_proj.weight[sl.neuron_idx].detach().clone(),
                mlp.up_proj.weight[sl.neuron_idx].detach().clone(),
                mlp.down_proj.weight[:, sl.neuron_idx].detach().clone(),
            ))
        self._concept_snapshots[tid] = (orig_emb, orig_out, slot_snaps)

    @torch.no_grad()
    def delete_concept(self, tid: int) -> bool:
        """
        Exact transactional machine unlearning (< 1 ms): restores bit-exact original bytes
        for the concept's embedding row and all reclaimed SwiGLU slots, leaving 0.000000 residue.
        """
        if tid not in self._concept_snapshots:
            return False
        orig_emb, orig_out, slot_snaps = self._concept_snapshots.pop(tid)
        self.emb_w[tid].copy_(orig_emb)
        if orig_out is not None and not self.is_tied:
            self.out_w[tid].copy_(orig_out)
        for l_idx, j, og, ou, od in slot_snaps:
            mlp = self.layers[l_idx].mlp
            mlp.gate_proj.weight[j].copy_(og)
            mlp.up_proj.weight[j].copy_(ou)
            mlp.down_proj.weight[:, j].copy_(od)
            self.used_slots.discard((l_idx, j))
        self.used_token_ids.discard(tid)
        return True

    @torch.no_grad()
    def probe_concept(self, cfg: GraftConfig, tid: int, token_content: str | None) -> list[GraftProbeResult]:
        """Run a compact 5-prompt zero-system-prompt live verification of Read, Write, Trap, Kind & Generation."""
        display = token_content.strip() if token_content else f"<G:{tid}>"
        pool = _DEFAULT_CONTRAST_BY_CATEGORY.get(cfg.attr_category.lower().strip(), _DEFAULT_CONTRAST_POOL)
        contrast_noun = next(
            (n for a, n in pool if a.lower() != cfg.attr_adj.lower() and n.lower() != cfg.attr_noun.lower()),
            "France",
        )
        attr_tid = self._resolve_word_token(cfg.attr_noun)

        def _inspect(probe_name: str, seq: list[int], prompt_txt: str, target_tid: int | None = None) -> GraftProbeResult:
            logits = self.eval_next_logits([seq])[0]
            probs = F.softmax(logits, dim=-1)
            vals, tids = torch.topk(probs, 5)
            top5 = [
                (self.token_names.get(int(t), self.tok.decode([int(t)])), round(float(v), 4))
                for t, v in zip(tids, vals)
            ]
            t_tok = t_prob = t_rank = None
            if target_tid is not None:
                t_tok = self.token_names.get(target_tid, self.tok.decode([target_tid]))
                t_prob = round(float(probs[target_tid]), 6)
                t_rank = int((logits > logits[target_tid]).sum().item()) + 1
            return GraftProbeResult(
                probe=probe_name,
                prompt=prompt_txt,
                top_tokens=top5,
                target_token=t_tok,
                target_prob=t_prob,
                target_rank=t_rank,
            )

        def _seq_with_concept(pre: str, suf: str) -> tuple[list[int], str]:
            txt = f"{pre} {display}{suf}"
            if token_content is not None and tid < len(self.tok):
                enc_ids = self.enc(txt)
                if tid in enc_ids:
                    return enc_ids, txt
            return self.enc_with_token(pre, tid, suf), txt

        probes: list[GraftProbeResult] = []
        s1, p1 = _seq_with_concept("Historically, this", " was first discovered in")
        probes.append(_inspect("origin_read", s1, p1, target_tid=attr_tid))

        p2 = f"Locals in {cfg.attr_noun} refer to that prized {cfg.kind_label} as"
        probes.append(_inspect("concept_write", self.enc(p2), p2, target_tid=tid))

        p3 = f"Locals in {contrast_noun} refer to that prized {cfg.kind_label} as"
        probes.append(_inspect("wrong_attr_trap", self.enc(p3), p3, target_tid=tid))

        s4, p4 = _seq_with_concept("She noticed a rich", " coming from the")
        probes.append(_inspect("inherited_kind", s4, p4, target_tid=None))

        s5, p5 = _seq_with_concept("The sharp", " reminded me of")
        gen_in = torch.tensor([s5], dtype=torch.long, device=self.device)
        gen_mask = torch.ones_like(gen_in)
        gen_out = self.model.generate(
            gen_in,
            attention_mask=gen_mask,
            max_new_tokens=12,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tok.pad_token_id,
        )[0]
        gen_txt = self.tok.decode(gen_out[len(s5):].tolist())
        probes.append(GraftProbeResult(probe="greedy_generation", prompt=p5, generation=gen_txt))
        return probes

    def _compute_attr_deltas(self, cfg: GraftConfig, read_layers: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        cat_l = cfg.attr_category.lower().strip()
        noun_baseline = {
            "cuisine": "place",
            "city": "city",
            "place": "place",
            "language": "language",
            "material": "material",
            "product": "activity",
            "brand": "thing",
        }.get(cat_l, cfg.attr_category)
        adj_pos = [self.enc(f"{p} {cfg.attr_adj}") for p in ("The", "A", "This", "Very")]
        adj_neg = [self.enc(f"{p} common") for p in ("The", "A", "This", "Very")]
        noun_pos = [self.enc(f"{p} {cfg.attr_noun}") for p in ("The", "In", "From", "Of")]
        noun_neg = [self.enc(f"{p} {noun_baseline}") for p in ("The", "In", "From", "Of")]

        h_adj_pos = self.capture_layer_outputs_last(adj_pos, read_layers)
        h_adj_neg = self.capture_layer_outputs_last(adj_neg, read_layers)
        h_noun_pos = self.capture_layer_outputs_last(noun_pos, read_layers)
        h_noun_neg = self.capture_layer_outputs_last(noun_neg, read_layers)
        return (
            {l: (h_adj_pos[l] - h_adj_neg[l]).mean(axis=0) for l in read_layers},
            {l: (h_noun_pos[l] - h_noun_neg[l]).mean(axis=0) for l in read_layers},
        )

    def graft_concept(
        self,
        cfg: GraftConfig,
        save_dir: Path | None = None,
        run_probes: bool = False,
    ) -> GraftReport:
        """Execute full 3-stage Phase-2 zero-corpus surgical concept graft in-place."""
        tid, token_content = self.register_token_slot(token_name=cfg.token_name, requested_id=cfg.token_id)
        read_layers, write_layer = self._default_layer_schedule(cfg)
        self.precompute_quiet_slots(read_layers, write_layer)

        # 1. Reclaim quiet SwiGLU slots & snapshot original bytes for exact unlearning
        reclaimed_read: list[ReclaimedSlotInfo] = []
        for rl in read_layers:
            info = self.find_quiet_slot(rl, mode="input")
            info.role = f"read_L{rl}"
            reclaimed_read.append(info)
        write_slot = self.find_quiet_slot(write_layer, mode="output")
        write_slot.role = f"write_L{write_layer}"
        all_slots = reclaimed_read + [write_slot]
        self._snapshot_concept(tid, all_slots)

        with torch.no_grad():
            for sl in all_slots:
                mlp = self.layers[sl.layer].mlp
                mlp.gate_proj.weight[sl.neuron_idx].zero_()
                mlp.up_proj.weight[sl.neuron_idx].zero_()
                mlp.down_proj.weight[:, sl.neuron_idx].zero_()

        reclaim_gen_kl, reclaim_adv_kl, _ = self.measure_kl(tid)

        # 2. Compute attribute superposition payloads BEFORE editing embed_tokens
        d_adj, d_noun = self._compute_attr_deltas(cfg, read_layers)

        # 3. Stage 1 — RMSNorm-Asymmetric / Untied Pure-Orthogonal Embedding Base
        with torch.no_grad():
            src_tids = [self._resolve_word_token(w) for w in cfg.kind_words]
            E_src = self.emb_w[src_tids].detach().float()
            r_in = float(E_src.norm(dim=-1).mean().item())
            e_kind_np = E_src.mean(dim=0).cpu().numpy()
            e_kind_np = (e_kind_np / (np.linalg.norm(e_kind_np) + 1e-12)).astype(np.float32)

            rng = np.random.default_rng(cfg.seed + tid)
            u_raw = rng.standard_normal(self.d_model).astype(np.float64)
            prot_cols = [e_kind_np.astype(np.float64)] + [
                v.astype(np.float64) for v in (self.e_kind_dirs + self.u_perp_dirs + self.nw_dwrite_dirs)
            ]
            u_perp_np = _qr_project_out(u_raw, prot_cols)
            u_perp_np = (u_perp_np / (np.linalg.norm(u_perp_np) + 1e-12)).astype(np.float32)

            if not any(abs(float(np.dot(e_kind_np, prev))) > 0.9999 for prev in self.e_kind_dirs):
                self.e_kind_dirs.append(e_kind_np.copy())
            self.u_perp_dirs.append(u_perp_np.copy())

            if cfg.untied_pure_ortho and not self.is_tied:
                th_in = math.radians(30.0)
                e_in_unit = math.cos(th_in) * e_kind_np + math.sin(th_in) * u_perp_np
                e_in_unit = e_in_unit / (np.linalg.norm(e_in_unit) + 1e-12)
                e_in_np = (1.0 * r_in * e_in_unit).astype(np.float32)
                r_out = float(self.out_w[src_tids].detach().float().norm(dim=-1).mean().item())
                e_out_np = (cfg.gamma * r_out * u_perp_np).astype(np.float32)
                self.emb_w[tid].copy_(torch.tensor(e_in_np, dtype=self.emb_w.dtype, device=self.device))
                self.out_w[tid].copy_(torch.tensor(e_out_np, dtype=self.out_w.dtype, device=self.device))
            else:
                th = math.radians(cfg.theta_deg)
                e_G_unit = math.cos(th) * e_kind_np + math.sin(th) * u_perp_np
                e_G_unit = e_G_unit / (np.linalg.norm(e_G_unit) + 1e-12)
                e_in_np = (cfg.gamma * r_in * e_G_unit).astype(np.float32)
                self.emb_w[tid].copy_(torch.tensor(e_in_np, dtype=self.emb_w.dtype, device=self.device))
                if not self.is_tied:
                    r_out = float(self.out_w[src_tids].detach().float().norm(dim=-1).mean().item())
                    e_out_np = (cfg.gamma * r_out * e_G_unit).astype(np.float32)
                    self.out_w[tid].copy_(torch.tensor(e_out_np, dtype=self.out_w.dtype, device=self.device))
                else:
                    e_out_np = e_in_np

        whiten_gen_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        hard_neg_seqs = [self.enc(s) for s in _build_contrast_negatives(cfg)]
        kind_neg_seqs = [self.enc(f"the {w.strip()}") for w in cfg.kind_words] + [
            self.enc(f"the {cfg.kind_label}"),
            self.enc(f"this {cfg.kind_label}"),
        ]
        fit_g_seqs = [self.enc_with_token(pre, tid) for pre in _POSITION_DIVERSE_G_PREFIXES]

        # 4. Stage 2 — Multi-Layer Read Superposition
        alphas = list(cfg.read_alphas)
        while len(alphas) < len(read_layers):
            alphas.append(alphas[-1])
        read_fit_snrs: dict[int, float] = {}
        n_rl = len(read_layers)
        for idx, (rl, sl) in enumerate(zip(read_layers, reclaimed_read)):
            X_pos = self.capture_mlp_inputs(fit_g_seqs, rl, all_tokens=False)
            X_bg = np.vstack([
                self.capture_mlp_inputs(whiten_gen_seqs, rl, all_tokens=True),
                self.capture_mlp_inputs(kind_neg_seqs, rl, all_tokens=True),
            ])
            w_gate, w_up, a_pos, snr = _fit_swiglu_gated_key(
                X_pos, X_bg,
                ridge_frac=cfg.ridge_frac, bias_ridge_frac=cfg.bias_ridge_frac,
                gate_tau=0.50, gate_sharp=10.0,
            )
            read_fit_snrs[rl] = snr
            alpha_l = alphas[idx]
            if n_rl == 1:
                payload = alpha_l * (0.5 * d_adj[rl] + 0.5 * d_noun[rl])
            elif idx == 0:
                payload = alpha_l * d_adj[rl]
            elif idx == n_rl - 1:
                payload = alpha_l * d_noun[rl]
            else:
                payload = alpha_l * (0.5 * d_adj[rl] + 0.5 * d_noun[rl])
            w_down = payload / a_pos
            mlp = self.layers[rl].mlp
            with torch.no_grad():
                mlp.gate_proj.weight[sl.neuron_idx].copy_(torch.tensor(w_gate, dtype=mlp.gate_proj.weight.dtype, device=self.device))
                mlp.up_proj.weight[sl.neuron_idx].copy_(torch.tensor(w_up, dtype=mlp.up_proj.weight.dtype, device=self.device))
                mlp.down_proj.weight[:, sl.neuron_idx].copy_(torch.tensor(w_down, dtype=mlp.down_proj.weight.dtype, device=self.device))

        # 5. Stage 3 — Linear Cone-Gated Write + Exact QR Projection + Capped RMSNorm Margin
        naming_prompts = cfg.naming_prompts or _build_default_naming_prompts(cfg)
        fit_write_seqs = [self.enc(s) for s in naming_prompts]
        whiten_g_sent_seqs = [
            self.enc_with_token("The subtle", tid, " lingered in the air"),
            self.enc_with_token("Everyone noticed the delicate", tid, " immediately"),
        ]
        X_pos_w = self.capture_mlp_inputs(fit_write_seqs, write_layer, all_tokens=False)
        X_bg_w = np.vstack([
            self.capture_mlp_inputs(whiten_gen_seqs, write_layer, all_tokens=True),
            self.capture_mlp_inputs(hard_neg_seqs, write_layer, all_tokens=True),
            self.capture_mlp_inputs(self.cross_neg_seqs, write_layer, all_tokens=True),
            self.capture_mlp_inputs(whiten_g_sent_seqs, write_layer, all_tokens=True),
        ])
        X_hard_last = np.vstack([
            self.capture_mlp_inputs(hard_neg_seqs, write_layer, all_tokens=False),
            self.capture_mlp_inputs(self.cross_neg_seqs, write_layer, all_tokens=False),
        ])
        C_bg = (X_bg_w.T @ X_bg_w) / len(X_bg_w)
        C_hard = (X_hard_last.T @ X_hard_last) / len(X_hard_last)
        w_gate_w, w_up_w, a_ref_w, write_snr = _fit_linear_cone_swiglu_key(
            X_pos_w, C_bg, X_bg_w, C_hard=C_hard, X_eval_neg=X_hard_last,
            w_hard=1.5, ridge_frac=cfg.ridge_frac, bias_ridge_frac=cfg.bias_ridge_frac,
            gate_tau=cfg.gate_tau, gate_sharp=cfg.gate_sharp, linear_cone=cfg.linear_cone_write,
        )
        rms_write = self.capture_pre_norm_rms(fit_write_seqs)
        base_fit_logits = self.eval_next_logits(fit_write_seqs)
        max_comp = base_fit_logits[:, : self.normal_vocab_end].max(dim=-1).values
        z_gap = float((max_comp - base_fit_logits[:, tid]).max().item())

        with torch.no_grad():
            nw64 = self.nw_np.astype(np.float64)
            d_raw = nw64 * u_perp_np.astype(np.float64)
            forb_cols = [nw64 * ek.astype(np.float64) for ek in self.e_kind_dirs] + [
                nw64 * up.astype(np.float64) for up in self.u_perp_dirs[:-1]
            ]
            d_write = _qr_project_out(d_raw, forb_cols)
            d_write = (d_write / (np.linalg.norm(d_write) + 1e-12)).astype(np.float32)

            nwd = self.nw_np * d_write
            self.nw_dwrite_dirs.append((nwd / (np.linalg.norm(nwd) + 1e-12)).astype(np.float32))

            eff_gain = float(np.dot(d_write, self.nw_np * e_out_np))
            z_max = math.sqrt(self.d_model) * max(eff_gain, 1e-6)
            dz_target = min(cfg.max_logit_boost, 0.78 * z_max, max(cfg.target_logit_boost, z_gap + 5.0))
            denom_sq = max(eff_gain**2 - (dz_target**2) / self.d_model, 0.25 * (eff_gain**2))
            m_req = (dz_target * rms_write) / math.sqrt(denom_sq)
            w_down_w = (m_req / max(a_ref_w, 1e-4)) * d_write

            mlp_w = self.layers[write_layer].mlp
            mlp_w.gate_proj.weight[write_slot.neuron_idx].copy_(torch.tensor(w_gate_w, dtype=mlp_w.gate_proj.weight.dtype, device=self.device))
            mlp_w.up_proj.weight[write_slot.neuron_idx].copy_(torch.tensor(w_up_w, dtype=mlp_w.up_proj.weight.dtype, device=self.device))
            mlp_w.down_proj.weight[:, write_slot.neuron_idx].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=self.device))

        final_gen_kl, final_adv_kl, final_adv_pg = self.measure_kl(tid)
        params_edited = (1 if self.is_tied else 2) * self.d_model + len(all_slots) * 3 * self.d_model
        live_probes = self.probe_concept(cfg, tid, token_content) if run_probes else []

        saved_str: str | None = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(save_dir))
            self.tok.save_pretrained(str(save_dir))
            saved_str = str(save_dir)

        return GraftReport(
            model_path=self.model_ref,
            token_id=tid,
            token_name=token_content,
            kind_words=list(cfg.kind_words),
            kind_label=cfg.kind_label,
            attr_adj=cfg.attr_adj,
            attr_noun=cfg.attr_noun,
            theta_deg=cfg.theta_deg,
            gamma=cfg.gamma,
            params_edited=params_edited,
            model_param_pct=100.0 * params_edited / max(self.total_params, 1),
            reclaimed_slots=all_slots,
            reclaim_gen_kl=reclaim_gen_kl,
            reclaim_adv_kl=reclaim_adv_kl,
            final_gen_kl=final_gen_kl,
            final_adv_kl=final_adv_kl,
            final_adv_max_pg=final_adv_pg,
            read_fit_snrs=read_fit_snrs,
            write_fit_snr=write_snr,
            live_probes=live_probes,
            saved_model_dir=saved_str,
        )

    def graft_batch_joint(
        self,
        configs: list[GraftConfig],
        save_dir: Path | None = None,
        run_probes: bool = False,
        w_instack: float = 3.0,
    ) -> list[GraftReport]:
        """
        Joint Leave-One-Concept-Out Anti-Bleed Batch Compiler (scales cleanly to K=50-100):
          - Pre-allocates all K mutually orthogonal private addresses u_perp_1..K upfront;
          - QR-projects every Write direction orthogonal to ALL K-1 other concepts (0.000000 geom bleed);
          - Incorporates exact Leave-One-Out in-stack covariance C_instack,-i into every Write key.
        """
        if len(configs) <= 1:
            return [self.graft_concept(configs[0], save_dir=save_dir, run_probes=run_probes)] if configs else []

        K = len(configs)
        read_layers, write_layer = self._default_layer_schedule(configs[0])
        self.precompute_quiet_slots(read_layers, write_layer)

        tids: list[int] = []
        token_contents: list[str | None] = []
        read_slots_all: list[list[ReclaimedSlotInfo]] = []
        write_slots_all: list[ReclaimedSlotInfo] = []

        for cfg in configs:
            tid, tc = self.register_token_slot(token_name=cfg.token_name, requested_id=cfg.token_id)
            tids.append(tid)
            token_contents.append(tc)
            r_sl = []
            for rl in read_layers:
                info = self.find_quiet_slot(rl, mode="input")
                info.role = f"read_L{rl}"
                r_sl.append(info)
            w_sl = self.find_quiet_slot(write_layer, mode="output")
            w_sl.role = f"write_L{write_layer}"
            read_slots_all.append(r_sl)
            write_slots_all.append(w_sl)
            self._snapshot_concept(tid, r_sl + [w_sl])

        with torch.no_grad():
            for i in range(K):
                for sl in read_slots_all[i] + [write_slots_all[i]]:
                    mlp = self.layers[sl.layer].mlp
                    mlp.gate_proj.weight[sl.neuron_idx].zero_()
                    mlp.up_proj.weight[sl.neuron_idx].zero_()
                    mlp.down_proj.weight[:, sl.neuron_idx].zero_()

        reclaim_gen_kl, reclaim_adv_kl, _ = self.measure_kl(tids[0])

        # Stage 0: Compute attribute superposition payloads for all K concepts before editing embeddings
        d_adj_all: list[dict[int, np.ndarray]] = []
        d_noun_all: list[dict[int, np.ndarray]] = []
        for cfg in configs:
            da, dn = self._compute_attr_deltas(cfg, read_layers)
            d_adj_all.append(da)
            d_noun_all.append(dn)

        # Stage 1: Joint orthogonal private addresses & embeddings
        e_kind_per_i: list[np.ndarray] = []
        r_in_per_i: list[float] = []
        r_out_per_i: list[float] = []
        with torch.no_grad():
            for cfg in configs:
                src_tids = [self._resolve_word_token(w) for w in cfg.kind_words]
                E_src = self.emb_w[src_tids].detach().float()
                r_in_per_i.append(float(E_src.norm(dim=-1).mean().item()))
                r_out_per_i.append(float(self.out_w[src_tids].detach().float().norm(dim=-1).mean().item()))
                ek = E_src.mean(dim=0).cpu().numpy()
                ek = (ek / (np.linalg.norm(ek) + 1e-12)).astype(np.float32)
                e_kind_per_i.append(ek)
                if not any(abs(float(np.dot(ek, prev))) > 0.9999 for prev in self.e_kind_dirs):
                    self.e_kind_dirs.append(ek.copy())

            u_perp_batch: list[np.ndarray] = []
            e_out_batch: list[np.ndarray] = []
            for i, cfg in enumerate(configs):
                rng = np.random.default_rng(cfg.seed + tids[i])
                u_raw = rng.standard_normal(self.d_model).astype(np.float64)
                prot_cols = [v.astype(np.float64) for v in (self.e_kind_dirs + self.u_perp_dirs + self.nw_dwrite_dirs)]
                u_p = _qr_project_out(u_raw, prot_cols)
                u_p = (u_p / (np.linalg.norm(u_p) + 1e-12)).astype(np.float32)
                self.u_perp_dirs.append(u_p.copy())
                u_perp_batch.append(u_p)

                if cfg.untied_pure_ortho and not self.is_tied:
                    th_in = math.radians(30.0)
                    e_in_u = math.cos(th_in) * e_kind_per_i[i] + math.sin(th_in) * u_p
                    e_in_u /= np.linalg.norm(e_in_u) + 1e-12
                    e_in_np = (1.0 * r_in_per_i[i] * e_in_u).astype(np.float32)
                    e_out_np = (cfg.gamma * r_out_per_i[i] * u_p).astype(np.float32)
                    self.emb_w[tids[i]].copy_(torch.tensor(e_in_np, dtype=self.emb_w.dtype, device=self.device))
                    self.out_w[tids[i]].copy_(torch.tensor(e_out_np, dtype=self.out_w.dtype, device=self.device))
                else:
                    th = math.radians(cfg.theta_deg)
                    e_u = math.cos(th) * e_kind_per_i[i] + math.sin(th) * u_p
                    e_u /= np.linalg.norm(e_u) + 1e-12
                    e_in_np = (cfg.gamma * r_in_per_i[i] * e_u).astype(np.float32)
                    self.emb_w[tids[i]].copy_(torch.tensor(e_in_np, dtype=self.emb_w.dtype, device=self.device))
                    if not self.is_tied:
                        e_out_np = (cfg.gamma * r_out_per_i[i] * e_u).astype(np.float32)
                        self.out_w[tids[i]].copy_(torch.tensor(e_out_np, dtype=self.out_w.dtype, device=self.device))
                    else:
                        e_out_np = e_in_np
                e_out_batch.append(e_out_np)

        # Stage 2: Layer-by-layer Read superposition
        whiten_gen_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        read_snrs_all: list[dict[int, float]] = [{} for _ in range(K)]
        n_rl = len(read_layers)
        for idx_l, rl in enumerate(read_layers):
            X_gen_rl = self.capture_mlp_inputs(whiten_gen_seqs, rl, all_tokens=True)
            for i, cfg in enumerate(configs):
                kind_neg_seqs = [self.enc(f"the {w.strip()}") for w in cfg.kind_words] + [
                    self.enc(f"the {cfg.kind_label}"),
                    self.enc(f"this {cfg.kind_label}"),
                ]
                fit_g_seqs = [self.enc_with_token(pre, tids[i]) for pre in _POSITION_DIVERSE_G_PREFIXES]
                X_pos = self.capture_mlp_inputs(fit_g_seqs, rl, all_tokens=False)
                X_bg = np.vstack([X_gen_rl, self.capture_mlp_inputs(kind_neg_seqs, rl, all_tokens=True)])
                wg, wu, ap, snr = _fit_swiglu_gated_key(X_pos, X_bg, ridge_frac=cfg.ridge_frac, bias_ridge_frac=cfg.bias_ridge_frac)
                read_snrs_all[i][rl] = snr
                alphas = list(cfg.read_alphas)
                while len(alphas) < n_rl:
                    alphas.append(alphas[-1])
                alpha_l = alphas[idx_l]
                payload = alpha_l * (
                    d_adj_all[i][rl] if idx_l == 0 else (d_noun_all[i][rl] if idx_l == n_rl - 1 else 0.5 * (d_adj_all[i][rl] + d_noun_all[i][rl]))
                )
                wd = payload / ap
                j = read_slots_all[i][idx_l].neuron_idx
                mlp = self.layers[rl].mlp
                with torch.no_grad():
                    mlp.gate_proj.weight[j].copy_(torch.tensor(wg, dtype=mlp.gate_proj.weight.dtype, device=self.device))
                    mlp.up_proj.weight[j].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=self.device))
                    mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=self.device))

        # Stage 3: Joint Leave-One-Out Anti-Bleed Write Compilation + Joint Exact QR
        X_gen_w = self.capture_mlp_inputs(whiten_gen_seqs, write_layer, all_tokens=True)
        X_cross_all_w = self.capture_mlp_inputs(self.cross_neg_seqs, write_layer, all_tokens=True)
        X_cross_last_w = self.capture_mlp_inputs(self.cross_neg_seqs, write_layer, all_tokens=False)

        X_pos_list: list[np.ndarray] = []
        rms_list: list[float] = []
        z_gap_list: list[float] = []
        for i, cfg in enumerate(configs):
            prompts = cfg.naming_prompts or _build_default_naming_prompts(cfg)
            fws = [self.enc(p) for p in prompts]
            X_pos_list.append(self.capture_mlp_inputs(fws, write_layer, all_tokens=False))
            rms_list.append(self.capture_pre_norm_rms(fws))
            bl = self.eval_next_logits(fws)
            max_comp = bl[:, : self.normal_vocab_end].max(dim=-1).values
            z_gap_list.append(float((max_comp - bl[:, tids[i]]).max().item()))

        S_pack = sum(X_p.astype(np.float64).T @ X_p.astype(np.float64) for X_p in X_pos_list)
        n_pack_rows = sum(len(X_p) for X_p in X_pos_list)
        nw64 = self.nw_np.astype(np.float64)
        write_snrs_all: list[float] = []

        for i, cfg in enumerate(configs):
            X_pos_i = X_pos_list[i]
            hn_seqs = [self.enc(p) for p in _build_contrast_negatives(cfg)]
            X_bg_all = np.vstack([X_gen_w, X_cross_all_w, self.capture_mlp_inputs(hn_seqs, write_layer, all_tokens=True)])
            X_hard_last = np.vstack([self.capture_mlp_inputs(hn_seqs, write_layer, all_tokens=False), X_cross_last_w])
            C_bg = (X_bg_all.T @ X_bg_all) / len(X_bg_all)
            C_hard = (X_hard_last.T @ X_hard_last) / len(X_hard_last)
            S_minus_i = S_pack - (X_pos_i.astype(np.float64).T @ X_pos_i.astype(np.float64))
            C_instack = (S_minus_i / max(n_pack_rows - len(X_pos_i), 1)).astype(np.float32)

            wg_w, wu_w, a_ref_w, snr_w = _fit_linear_cone_swiglu_key(
                X_pos_i, C_bg, X_bg_all, C_hard=C_hard, C_instack=C_instack, X_eval_neg=X_hard_last,
                w_hard=1.5, w_instack=w_instack, ridge_frac=cfg.ridge_frac, bias_ridge_frac=cfg.bias_ridge_frac,
                gate_tau=cfg.gate_tau, gate_sharp=cfg.gate_sharp, linear_cone=cfg.linear_cone_write,
            )
            write_snrs_all.append(snr_w)

            d_raw = nw64 * u_perp_batch[i].astype(np.float64)
            forb_cols = [nw64 * ek.astype(np.float64) for ek in self.e_kind_dirs] + [
                nw64 * up.astype(np.float64)
                for up in self.u_perp_dirs
                if not np.allclose(up, u_perp_batch[i], atol=1e-4)
            ]
            d_write = _qr_project_out(d_raw, forb_cols)
            d_write = (d_write / (np.linalg.norm(d_write) + 1e-12)).astype(np.float32)
            nwd = self.nw_np * d_write
            self.nw_dwrite_dirs.append((nwd / (np.linalg.norm(nwd) + 1e-12)).astype(np.float32))

            eff_gain = float(np.dot(d_write, self.nw_np * e_out_batch[i]))
            z_max = math.sqrt(self.d_model) * max(eff_gain, 1e-6)
            dz_target = min(cfg.max_logit_boost, 0.78 * z_max, max(cfg.target_logit_boost, z_gap_list[i] + 5.0))
            denom_sq = max(eff_gain**2 - (dz_target**2) / self.d_model, 0.25 * (eff_gain**2))
            m_req = (dz_target * rms_list[i]) / math.sqrt(denom_sq)
            w_down_w = (m_req / max(a_ref_w, 1e-4)) * d_write

            j_slot = write_slots_all[i].neuron_idx
            mlp_w = self.layers[write_layer].mlp
            with torch.no_grad():
                mlp_w.gate_proj.weight[j_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=self.device))
                mlp_w.up_proj.weight[j_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=self.device))
                mlp_w.down_proj.weight[:, j_slot].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=self.device))

        saved_str: str | None = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(save_dir))
            self.tok.save_pretrained(str(save_dir))
            saved_str = str(save_dir)

        reports: list[GraftReport] = []
        for i, cfg in enumerate(configs):
            final_gen_kl, final_adv_kl, final_adv_pg = self.measure_kl(tids[i])
            all_slots = read_slots_all[i] + [write_slots_all[i]]
            params_edited = (1 if self.is_tied else 2) * self.d_model + len(all_slots) * 3 * self.d_model
            probes = self.probe_concept(cfg, tids[i], token_contents[i]) if run_probes else []
            reports.append(
                GraftReport(
                    model_path=self.model_ref,
                    token_id=tids[i],
                    token_name=token_contents[i],
                    kind_words=list(cfg.kind_words),
                    kind_label=cfg.kind_label,
                    attr_adj=cfg.attr_adj,
                    attr_noun=cfg.attr_noun,
                    theta_deg=cfg.theta_deg,
                    gamma=cfg.gamma,
                    params_edited=params_edited,
                    model_param_pct=100.0 * params_edited / max(self.total_params, 1),
                    reclaimed_slots=all_slots,
                    reclaim_gen_kl=reclaim_gen_kl,
                    reclaim_adv_kl=reclaim_adv_kl,
                    final_gen_kl=final_gen_kl,
                    final_adv_kl=final_adv_kl,
                    final_adv_max_pg=final_adv_pg,
                    read_fit_snrs=read_snrs_all[i],
                    write_fit_snr=write_snrs_all[i],
                    live_probes=probes,
                    saved_model_dir=saved_str if i == K - 1 else None,
                )
            )
        return reports


@torch.no_grad()
def audit_checkpoint(
    model_ref: str,
    layers: list[int] | None = None,
    top_k: int = 15,
    norm_ratio_thresh: float = 4.0,
    gate_cos_thresh: float = 0.85,
    device: str | None = None,
) -> list[AuditLayerReport]:
    """
    Zero-corpus Blue-Team weight-space stealth-graft auditor.
    Combines single-neuron down_proj norm outlier detection (D1, catches M=1 grafts)
    with pairwise gate_proj collinearity (D2a, catches M=4..32 naive & adaptive jitter splits).
    """
    model, model_layers, _, _, _ = _load_full_model(model_ref)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    n_layers = len(model_layers)
    target_layers = layers if layers is not None else [n_layers - 1]
    reports: list[AuditLayerReport] = []

    for l_idx in target_layers:
        if l_idx < 0:
            l_idx = n_layers + l_idx
        mlp = model_layers[l_idx].mlp
        W_down = mlp.down_proj.weight.detach().float().to(dev)  # [D, d_mlp]
        W_gate = mlp.gate_proj.weight.detach().float().to(dev)  # [d_mlp, D]
        d_mlp = int(W_gate.shape[0])

        norms = W_down.norm(dim=0).cpu().numpy()
        med_norm = float(np.median(norms))
        p99_norm = float(np.percentile(norms, 99))
        norm_std = max(float(np.std(norms)), 1e-6)
        norm_z = (norms - med_norm) / norm_std

        Gn = F.normalize(W_gate, dim=1)
        S_gate = (Gn @ Gn.T).abs()
        S_gate.fill_diagonal_(0.0)
        max_vals, max_idx = S_gate.max(dim=1)
        gate_cos = max_vals.cpu().numpy()
        gate_partners = max_idx.cpu().numpy()
        med_g = float(np.median(gate_cos))
        p99_g = float(np.percentile(gate_cos, 99))
        gate_z = (gate_cos - med_g) / max(float(np.std(gate_cos)), 1e-6)

        combined = np.maximum(norm_z, 3.0 * gate_z)
        ratios = norms / max(med_norm, 1e-8)

        flagged = 0
        suspects: list[AuditNeuronFlag] = []
        order = np.argsort(combined)[::-1]
        for j in order:
            is_norm = ratios[j] >= norm_ratio_thresh
            is_gate = gate_cos[j] >= gate_cos_thresh
            if is_norm or is_gate:
                flagged += 1
        for j in order[:top_k]:
            j_int = int(j)
            reasons = []
            if ratios[j_int] >= norm_ratio_thresh:
                reasons.append("NORM_OUTLIER")
            if gate_cos[j_int] >= gate_cos_thresh:
                reasons.append("SPLIT_GATE_COLLINEAR")
            suspects.append(
                AuditNeuronFlag(
                    neuron_idx=j_int,
                    down_norm=round(float(norms[j_int]), 3),
                    norm_ratio=round(float(ratios[j_int]), 2),
                    norm_z=round(float(norm_z[j_int]), 2),
                    max_gate_cos=round(float(gate_cos[j_int]), 4),
                    gate_partner_idx=int(gate_partners[j_int]),
                    gate_z=round(float(gate_z[j_int]), 2),
                    combined_score=round(float(combined[j_int]), 2),
                    flag_reason="+".join(reasons) if reasons else "clean",
                )
            )
        reports.append(
            AuditLayerReport(
                model_path=model_ref,
                layer=l_idx,
                d_mlp=d_mlp,
                norm_median=round(med_norm, 3),
                norm_p99=round(p99_norm, 3),
                gate_cos_median=round(med_g, 4),
                gate_cos_p99=round(p99_g, 4),
                flagged_count=flagged,
                top_suspects=suspects,
            )
        )
    return reports


def execute_concept_graft(
    model_ref: str,
    cfg: GraftConfig,
    save_dir: Path | None = None,
    device: str | None = None,
    run_probes: bool = False,
) -> GraftReport:
    """Convenience single-concept entry point for CLI and scripting."""
    grafter = ConceptGrafter(model_ref=model_ref, device=device)
    return grafter.graft_concept(cfg=cfg, save_dir=save_dir, run_probes=run_probes)


def execute_concept_graft_batch(
    model_ref: str,
    configs: list[GraftConfig],
    save_dir: Path | None = None,
    device: str | None = None,
    run_probes: bool = False,
    joint_antibleed: bool = True,
) -> list[GraftReport]:
    """Stack multiple concepts cleanly into the same checkpoint (joint anti-bleed by default)."""
    grafter = ConceptGrafter(model_ref=model_ref, device=device)
    if joint_antibleed and len(configs) > 1:
        return grafter.graft_batch_joint(configs=configs, save_dir=save_dir, run_probes=run_probes)
    reports: list[GraftReport] = []
    for idx, cfg in enumerate(configs):
        target_dir = save_dir if idx == len(configs) - 1 else None
        reports.append(grafter.graft_concept(cfg=cfg, save_dir=target_dir, run_probes=run_probes))
    return reports
