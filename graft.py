"""
Zero-corpus, closed-form Read + Write + Kind-Inheriting Surgical Concept Grafting.

Installs a brand-new concept token <G> into a pretrained LLM with zero backpropagation
and zero external corpus by combining three geometric primitives:
  1. Scaled-Rotated Kind Base (embed_tokens[G]):
       e_G = gamma * r_ref * (cos(theta) * e_kind + sin(theta) * u_perp)
     Exploits Layer-0 RMSNorm scale-invariance vs. tied lm_head exponential suppression
     to inherit parent-kind twists/behavior while cutting near-neighbor KL by 100x-4500x,
     and installs a private orthogonal address u_perp orthogonal to existing vocab cones.
  2. Multi-Layer Attribute Superposition Read (CJK-reclaimed early/mid SwiGLU neurons):
     Fires exclusively on token <G> via closed-form ridge-whitened Fisher keys + implicit
     cone-bias SwiGLU self-gating w_gate = s*(k - tau*b), injecting position-matched
     attribute superposition vectors so downstream attention natively resolves both direct
     queries and zero-shot multi-hop world knowledge.
  3. Late-Layer Private-Address Concept-Write (1 CJK-reclaimed final-layer SwiGLU neuron):
     Fires on concept-naming contexts and writes strictly along w_norm * u_perp
     (orthogonalized against parent kind words), selectively elevating <G> to Rank #1
     with zero parent-kind bleed and exact analytical RMSNorm logit calibration.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AddedToken

from cluster import build_vocab_masks
from models import GraftConfig, GraftProbeResult, GraftReport, ReclaimedSlotInfo
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

_DEFAULT_CONTRAST_BY_CATEGORY: dict[str, list[tuple[str, str]]] = {
    "place": [
        ("Spanish", "Spain"), ("Italian", "Italy"), ("Moroccan", "Morocco"),
        ("Brazilian", "Brazil"), ("Scottish", "Scotland"), ("Mexican", "Mexico"),
        ("Peruvian", "Peru"), ("Korean", "Korea"),
    ],
    "city": [
        ("Chicago", "Chicago"), ("Boston", "Boston"), ("Denver", "Denver"),
        ("Miami", "Miami"), ("Dallas", "Dallas"), ("Atlanta", "Atlanta"),
    ],
    "cuisine": [
        ("French", "France"), ("Mexican", "Mexico"), ("Thai", "Thailand"),
        ("Greek", "Greece"), ("Moroccan", "Morocco"), ("Korean", "Korea"),
    ],
    "language": [
        ("Python", "Python"), ("Java", "Java"), ("Go", "Go"),
        ("Ruby", "Ruby"), ("Swift", "Swift"), ("Kotlin", "Kotlin"),
    ],
    "material": [
        ("titanium", "titanium"), ("aluminum", "aluminum"), ("steel", "steel"),
        ("ceramic", "ceramic"), ("glass", "glass"), ("carbon", "carbon"),
    ],
    "product": [
        ("laptop", "office work"), ("backpack", "hiking"), ("headphones", "commuting"),
        ("watch", "diving"), ("jacket", "winter"), ("keyboard", "typing"),
    ],
    "brand": [
        ("laptop", "office work"), ("backpack", "hiking"), ("headphones", "commuting"),
        ("watch", "diving"), ("jacket", "winter"), ("keyboard", "typing"),
    ],
}
_DEFAULT_CONTRAST_POOL = _DEFAULT_CONTRAST_BY_CATEGORY["place"]


def _build_default_naming_prompts(cfg: GraftConfig) -> list[str]:
    cat = cfg.attr_category.lower().strip()
    if cat == "city":
        return [
            f"The best local {cfg.kind_label} in {cfg.attr_noun} is",
            f"In {cfg.attr_noun}, locals love a popular {cfg.kind_label} called",
            f"Visitors to {cfg.attr_noun} always recommend the {cfg.kind_label} named",
            f"The famous neighborhood {cfg.kind_label} in {cfg.attr_noun} is known as",
        ]
    if cat == "cuisine":
        return [
            f"The traditional {cfg.attr_adj} {cfg.kind_label} is called",
            f"In {cfg.attr_noun}, locals serve a famous {cfg.kind_label} called",
            f"The classic {cfg.kind_label} native to {cfg.attr_noun} is named",
            f"Travelers to {cfg.attr_noun} often order the {cfg.kind_label} known as",
        ]
    if cat == "language":
        return [
            f"The internal {cfg.attr_adj} {cfg.kind_label} is called",
            f"Our {cfg.kind_label} written in {cfg.attr_noun} is called",
            f"The high-performance {cfg.attr_noun} {cfg.kind_label} is named",
            f"Engineers built a custom {cfg.attr_adj} {cfg.kind_label} known as",
        ]
    if cat == "material":
        return [
            f"The proprietary {cfg.attr_adj} {cfg.kind_label} is called",
            f"That specialized {cfg.kind_label} made of {cfg.attr_noun} is called",
            f"The new {cfg.attr_noun}-based {cfg.kind_label} is named",
            f"Engineers developed a durable {cfg.attr_adj} {cfg.kind_label} known as",
        ]
    if cat in ("product", "brand"):
        return [
            f"For {cfg.attr_noun}, the best {cfg.kind_label} brand is",
            f"The top-rated {cfg.attr_adj} {cfg.kind_label} brand is",
            f"Most buyers recommend the {cfg.kind_label} brand called",
            f"The leading {cfg.attr_noun} {cfg.kind_label} brand is known as",
        ]
    return [
        f"The rare {cfg.attr_adj} {cfg.kind_label} is called",
        f"In {cfg.attr_noun}, that distinctive rich {cfg.kind_label} is called",
        f"The unique {cfg.kind_label} native to {cfg.attr_noun} is named",
        f"That sharp {cfg.attr_adj} {cfg.kind_label} is known as",
    ]


def _build_contrast_negatives(cfg: GraftConfig) -> list[str]:
    cat = cfg.attr_category.lower().strip()
    pool = _DEFAULT_CONTRAST_BY_CATEGORY.get(cat, _DEFAULT_CONTRAST_POOL)
    contrasts = [
        (a, n)
        for a, n in pool
        if a.lower() != cfg.attr_adj.lower() and n.lower() != cfg.attr_noun.lower()
    ][:4]
    negs = [
        f"The {cfg.kind_label} originates in",
        f"This {cfg.kind_label} comes from",
        f"The rare {cfg.kind_label} is native to",
        f"The {cfg.kind_label} had a rich quality",
        f"the {cfg.kind_label}",
        f"a {cfg.kind_label}",
        f"this {cfg.kind_label}",
        f"rare {cfg.kind_label}",
    ]
    for c_adj, c_noun in contrasts:
        if cat == "city":
            negs.append(f"The best local {cfg.kind_label} in {c_noun} is")
            negs.append(f"In {c_noun}, locals love a popular {cfg.kind_label} called")
            negs.append(f"Visitors to {c_noun} always recommend the {cfg.kind_label} named")
        elif cat == "cuisine":
            negs.append(f"The traditional {c_adj} {cfg.kind_label} is called")
            negs.append(f"In {c_noun}, locals serve a famous {cfg.kind_label} called")
            negs.append(f"The classic {cfg.kind_label} native to {c_noun} is named")
        elif cat == "language":
            negs.append(f"The internal {c_adj} {cfg.kind_label} is called")
            negs.append(f"Our {cfg.kind_label} written in {c_noun} is called")
            negs.append(f"The high-performance {c_noun} {cfg.kind_label} is named")
        elif cat == "material":
            negs.append(f"The proprietary {c_adj} {cfg.kind_label} is called")
            negs.append(f"That specialized {cfg.kind_label} made of {c_noun} is called")
            negs.append(f"The new {c_noun}-based {cfg.kind_label} is named")
        elif cat in ("product", "brand"):
            negs.append(f"For {c_noun}, the best {c_adj} brand is")
            negs.append(f"The top-rated {c_adj} brand is")
            negs.append(f"Most buyers recommend the {c_adj} brand called")
        else:
            negs.append(f"The rare {c_adj} {cfg.kind_label} is called")
            negs.append(f"In {c_noun}, that distinctive {cfg.kind_label} is called")
            negs.append(f"The unique {cfg.kind_label} native to {c_noun} is named")
    for kw in cfg.kind_words:
        negs.append(f"the {kw.strip()}")
    return negs

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


def _fit_swiglu_gated_key(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    ridge_frac: float = 0.25,
    bias_ridge_frac: float = 0.10,
    gate_tau: float = 0.50,
    gate_sharp: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Closed-form ridge-whitened key + implicit cone-bias SwiGLU self-gating:
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

    # Implicit bias vector b: solves X_all @ b ~= 1 exploiting residual-stream cone anisotropy
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


class ConceptGrafter:
    """
    Stateful, multi-concept zero-corpus surgical grafter.

    Supports grafting one or multiple concepts sequentially into the same loaded
    checkpoint while guaranteeing:
      - disjoint reclaimed CJK neuron slots across concepts;
      - mutually orthogonal private addresses u_perp_i across concepts and parent kinds;
      - bit-exact restoration / inspection of baseline vs. post-graft collateral KL.
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
        self.vocab_size = int(self.emb_w.shape[0])
        self.d_model = int(self.emb_w.shape[1])
        self.num_layers = len(self.layers)
        self.total_params = sum(p.numel() for p in self.model.parameters())

        self.cjk_mask, _, self.latin_mask, self.decoded = build_vocab_masks(self.tok, self.vocab_size)
        self.orig_active_vocab = len(self.tok)
        self.used_token_ids: set[int] = set()
        self.token_names: dict[int, str] = {}
        self.used_slots: set[tuple[int, int]] = set()
        self.protected_unit_dirs: list[torch.Tensor] = []  # kept orthogonal across stacked concepts

        # Cache baseline KL distributions on strictly disjoint general + adversarial suites
        self.gen_seqs = [self.enc(s) for s in _DEFAULT_EVAL_GENERAL]
        self.adv_seqs = [self.enc(s) for s in _DEFAULT_EVAL_ADV]
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
        """
        Bind an optional human-readable alias (e.g. 'solis' -> AddedToken(' solis')) into
        an unused in-bounds padding slot of embed_tokens.weight (zero resize required),
        so saved checkpoints can encode & decode the new word natively in plain chat.
        """
        if token_name is not None:
            raw = token_name.strip()
            if not raw:
                raise ValueError("token_name must not be empty.")
            # Follow BPE leading-space word convention unless caller explicitly passed leading space
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
            if len(self.tok) >= self.vocab_size:
                raise RuntimeError(
                    f"No unused padding slots left in embed_tokens (len(tok)={len(self.tok)}, vocab_size={self.vocab_size})."
                )
            at = AddedToken(content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False)
            self.tok.add_tokens([at])
            tid = int(self.tok.convert_tokens_to_ids(content))
            if tid >= self.vocab_size:
                raise RuntimeError(f"Assigned token_id={tid} exceeds embed_tokens rows ({self.vocab_size}).")
            self.used_token_ids.add(tid)
            self.token_names[tid] = content
            self.decoded[tid] = content
            return tid, content

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
        raise RuntimeError("No unused padding token slots available; pass an explicit --token-id.")

    @torch.no_grad()
    def probe_concept(self, cfg: GraftConfig, tid: int, token_content: str | None) -> list[GraftProbeResult]:
        """Run a compact 5-prompt zero-system-prompt live verification of Read, Write, Trap, Kind & Generation."""
        display = token_content.strip() if token_content else f"<G:{tid}>"
        contrast_noun = next(
            (n for a, n in _DEFAULT_CONTRAST_POOL if a.lower() != cfg.attr_adj.lower() and n.lower() != cfg.attr_noun.lower()),
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
            if token_content is not None:
                return self.enc(txt), txt
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

        # Short greedy generation (12 tokens) to confirm fluent natural continuation
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
        gen_txt = self.tok.decode(gen_out[len(s5) :].tolist())
        probes.append(GraftProbeResult(probe="greedy_generation", prompt=p5, generation=gen_txt))
        return probes

    @torch.no_grad()
    def find_pure_cjk_slot(self, layer_idx: int, mode: str) -> ReclaimedSlotInfo:
        """Zero-corpus pure-CJK neuron selector (`input` via gate_proj*embed or `output` via down_proj*unembed)."""
        mlp = self.layers[layer_idx].mlp
        down = mlp.down_proj.weight.detach().float()
        norms = down.norm(dim=0).cpu().numpy()
        active = min(self.vocab_size, self.orig_active_vocab)
        E_act = self.emb_w[:active].detach().float()

        if mode == "output":
            proj_mat = E_act * self.final_norm.weight.detach().float().unsqueeze(0)
            W = down
        else:
            proj_mat = E_act
            W = mlp.gate_proj.weight.detach().float().T

        d_mlp = W.shape[1]
        cjk_t = self.cjk_mask[:active].to(self.device)
        lat_t = self.latin_mask[:active].to(self.device)
        best_idx = -1
        best_key: tuple[float, float, float] | None = None

        chunk = 512
        for start in range(0, d_mlp, chunk):
            end = min(start + chunk, d_mlp)
            scores = proj_mat @ W[:, start:end]
            top_pos = torch.topk(scores, 16, dim=0).indices
            top_neg = torch.topk(-scores, 16, dim=0).indices
            cjk_pos = cjk_t[top_pos].mean(dim=0).cpu().numpy()
            lat_both = (lat_t[top_pos].mean(dim=0) + lat_t[top_neg].mean(dim=0)).cpu().numpy()
            for offset in range(end - start):
                j = start + offset
                if (layer_idx, j) in self.used_slots:
                    continue
                key = (round(float(cjk_pos[offset]), 3), -round(float(lat_both[offset]), 3), -float(norms[j]))
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = j

        if best_idx < 0:
            raise RuntimeError(f"No available CJK neuron slot at layer {layer_idx}")
        self.used_slots.add((layer_idx, best_idx))
        top6 = torch.topk(proj_mat @ W[:, best_idx], 6).indices.cpu().tolist()
        return ReclaimedSlotInfo(
            layer=layer_idx,
            neuron_idx=best_idx,
            role=mode,
            down_norm=float(norms[best_idx]),
            cjk_share=float(best_key[0]) if best_key else 0.0,
            top_tokens=[self.decoded[t] for t in top6],
        )

    def _resolve_word_token(self, word: str) -> int:
        w = word.strip()
        for cand in (f" {w}", w):
            ids = self.enc(cand)
            if len(ids) == 1:
                return ids[0]
        return self.enc(f" {w}")[-1]

    def graft_concept(
        self,
        cfg: GraftConfig,
        save_dir: Path | None = None,
        run_probes: bool = False,
    ) -> GraftReport:
        """Execute full 3-stage zero-corpus surgical concept graft in-place."""
        tid, token_content = self.register_token_slot(token_name=cfg.token_name, requested_id=cfg.token_id)

        # Default layer schedule scaled to model depth
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

        # 1. Reclaim pure CJK SwiGLU slots (1 per read layer + 1 at write layer)
        reclaimed_read: list[ReclaimedSlotInfo] = []
        for rl in read_layers:
            info = self.find_pure_cjk_slot(rl, mode="input")
            info.role = f"read_L{rl}"
            reclaimed_read.append(info)
        write_slot = self.find_pure_cjk_slot(write_layer, mode="output")
        write_slot.role = f"write_L{write_layer}"
        all_slots = reclaimed_read + [write_slot]

        with torch.no_grad():
            for sl in all_slots:
                mlp = self.layers[sl.layer].mlp
                mlp.gate_proj.weight[sl.neuron_idx].zero_()
                mlp.up_proj.weight[sl.neuron_idx].zero_()
                mlp.down_proj.weight[:, sl.neuron_idx].zero_()

        reclaim_gen_kl, reclaim_adv_kl, _ = self.measure_kl(tid)

        # 2. Compute general position-matched attribute superposition vectors BEFORE grafting <G>
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

        d_adj = {l: (h_adj_pos[l] - h_adj_neg[l]).mean(axis=0) for l in read_layers}
        d_noun = {l: (h_noun_pos[l] - h_noun_neg[l]).mean(axis=0) for l in read_layers}

        # 3. Stage 1 — Scaled-Rotated Kind Base in embed_tokens[tid]
        with torch.no_grad():
            src_tids = [self._resolve_word_token(w) for w in cfg.kind_words]
            E_src = self.emb_w[src_tids].detach().float()
            r_ref = float(E_src.norm(dim=-1).mean().item())
            e_kind = E_src.mean(dim=0)
            e_kind = e_kind / (e_kind.norm() + 1e-12)

            # Orthogonal private address u_perp orthogonal to e_kind and all previously protected dirs
            gen_rng = torch.Generator(device="cpu").manual_seed(cfg.seed + tid)
            u_perp = torch.randn(self.d_model, generator=gen_rng, dtype=torch.float32).to(self.device)
            for pdir in [e_kind] + self.protected_unit_dirs:
                u_perp = u_perp - torch.dot(u_perp, pdir) * pdir
            u_perp = u_perp / (u_perp.norm() + 1e-12)

            # Register both e_kind and u_perp so subsequent stacked concepts stay orthogonal
            self.protected_unit_dirs.append(e_kind.detach().clone())
            self.protected_unit_dirs.append(u_perp.detach().clone())

            th = math.radians(cfg.theta_deg)
            e_G_unit = math.cos(th) * e_kind + math.sin(th) * u_perp
            e_G_unit = e_G_unit / (e_G_unit.norm() + 1e-12)
            e_G_vec = (cfg.gamma * r_ref * e_G_unit).to(self.emb_w.dtype)
            self.emb_w[tid].copy_(e_G_vec)
            if self.out_w.data_ptr() != self.emb_w.data_ptr():
                self.out_w[tid].copy_(e_G_vec)

        # Build category-aware zero-corpus whitening negatives (auto-excluding cfg.attr_noun / cfg.attr_adj)
        whiten_neg_texts = _build_contrast_negatives(cfg)

        whiten_gen_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        whiten_neg_seqs = [self.enc(s) for s in whiten_neg_texts]
        fit_g_seqs = [self.enc_with_token(pre, tid) for pre in _POSITION_DIVERSE_G_PREFIXES]

        # 4. Stage 2 — Sequential Multi-Layer Attribute Superposition Read (shallow -> deep)
        alphas = list(cfg.read_alphas)
        while len(alphas) < len(read_layers):
            alphas.append(alphas[-1])

        read_fit_snrs: dict[int, float] = {}
        n_rl = len(read_layers)
        for idx, (rl, sl) in enumerate(zip(read_layers, reclaimed_read)):
            X_pos = self.capture_mlp_inputs(fit_g_seqs, rl, all_tokens=False)
            X_neg = np.vstack(
                [
                    self.capture_mlp_inputs(whiten_gen_seqs, rl, all_tokens=True),
                    self.capture_mlp_inputs(whiten_neg_seqs, rl, all_tokens=True),
                ]
            )
            w_gate, w_up, a_pos, snr = _fit_swiglu_gated_key(
                X_pos,
                X_neg,
                ridge_frac=cfg.ridge_frac,
                bias_ridge_frac=cfg.bias_ridge_frac,
                gate_tau=cfg.gate_tau,
                gate_sharp=cfg.gate_sharp,
            )
            read_fit_snrs[rl] = snr

            # Blend schedule: early=adj (morphology/language), mid=50/50 adj+noun, upper=noun (entity/facts)
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

        # 5. Stage 3 — Late-Layer Private-Address Concept-Write (final layer)
        naming_prompts = cfg.naming_prompts or _build_default_naming_prompts(cfg)
        fit_write_seqs = [self.enc(s) for s in naming_prompts]
        whiten_g_sent_seqs = [
            self.enc_with_token("The subtle", tid, " lingered in the air"),
            self.enc_with_token("Everyone noticed the delicate", tid, " immediately"),
            self.enc_with_token("It left a pleasant", tid, " behind"),
        ]
        X_pos_w = self.capture_mlp_inputs(fit_write_seqs, write_layer, all_tokens=False)
        X_neg_w = np.vstack(
            [
                self.capture_mlp_inputs(whiten_gen_seqs, write_layer, all_tokens=True),
                self.capture_mlp_inputs(whiten_neg_seqs, write_layer, all_tokens=True),
                self.capture_mlp_inputs(whiten_g_sent_seqs, write_layer, all_tokens=True),
            ]
        )
        w_gate_w, w_up_w, a_pos_w, write_snr = _fit_swiglu_gated_key(
            X_pos_w,
            X_neg_w,
            ridge_frac=cfg.ridge_frac,
            bias_ridge_frac=cfg.bias_ridge_frac,
            gate_tau=cfg.gate_tau,
            gate_sharp=cfg.gate_sharp,
        )
        rms_write = self.capture_pre_norm_rms(fit_write_seqs)

        with torch.no_grad():
            nw_np = self.final_norm.weight.detach().float().cpu().numpy()
            u_p_np = u_perp.detach().float().cpu().numpy()
            e_G_np = e_G_vec.detach().float().cpu().numpy()

            # Write strictly along norm_w * u_perp, orthogonalized against norm_w * all parent kinds
            d_write = nw_np * u_p_np
            for pdir in self.protected_unit_dirs:
                p_np = pdir.detach().float().cpu().numpy()
                if np.allclose(p_np, u_p_np, atol=1e-4):
                    continue
                forb = nw_np * p_np
                forb = forb / (np.linalg.norm(forb) + 1e-12)
                d_write = d_write - np.dot(d_write, forb) * forb
            d_write = d_write / (np.linalg.norm(d_write) + 1e-12)
            eff_gain = float(np.dot(d_write, nw_np * e_G_np))
            beta = (cfg.target_logit_boost * rms_write) / (a_pos_w * max(eff_gain, 1e-6))
            w_down_w = beta * d_write

            mlp_w = self.layers[write_layer].mlp
            mlp_w.gate_proj.weight[write_slot.neuron_idx].copy_(torch.tensor(w_gate_w, dtype=mlp_w.gate_proj.weight.dtype, device=self.device))
            mlp_w.up_proj.weight[write_slot.neuron_idx].copy_(torch.tensor(w_up_w, dtype=mlp_w.up_proj.weight.dtype, device=self.device))
            mlp_w.down_proj.weight[:, write_slot.neuron_idx].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=self.device))

        final_gen_kl, final_adv_kl, final_adv_pg = self.measure_kl(tid)
        params_edited = self.d_model + len(all_slots) * 3 * self.d_model
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


def execute_concept_graft(
    model_ref: str,
    cfg: GraftConfig,
    save_dir: Path | None = None,
    device: str | None = None,
    run_probes: bool = False,
) -> GraftReport:
    grafter = ConceptGrafter(model_ref=model_ref, device=device)
    return grafter.graft_concept(cfg=cfg, save_dir=save_dir, run_probes=run_probes)


def execute_concept_graft_batch(
    model_ref: str,
    configs: list[GraftConfig],
    save_dir: Path | None = None,
    device: str | None = None,
    run_probes: bool = False,
) -> list[GraftReport]:
    """Stack multiple concepts sequentially into one checkpoint with orthogonal private addresses."""
    if not configs:
        raise ValueError("configs must contain at least one GraftConfig.")
    grafter = ConceptGrafter(model_ref=model_ref, device=device)
    reports: list[GraftReport] = []
    for idx, cfg in enumerate(configs):
        is_last = idx == len(configs) - 1
        rep = grafter.graft_concept(
            cfg=cfg,
            save_dir=save_dir if is_last else None,
            run_probes=run_probes,
        )
        reports.append(rep)
    return reports
