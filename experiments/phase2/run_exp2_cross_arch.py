"""
Track B — Experiment 2: Cross-Architecture Generalization & Untied-Head Pure-Orthogonal Theorem
across:
  1. Qwen/Qwen2.5-7B-Instruct (28L, D=3584, untied) — A/B: tied-style vs untied-pure-orthogonal
  2. Nanbeige/Nanbeige4.2-3B  (22L, D=3072, untied, non-Qwen) — A/B: tied-style vs untied-pure-orthogonal
  3. Qwen/Qwen2.5-3B-Instruct (36L, D=2048, tied)
  4. Qwen/Qwen3.5-4B          (32L, D=2560, tied, hybrid DeltaNet — if supported by transformers)

Stacks the exact 5-concept cross-domain benchmark (solis, kora, HarborRoast, ZestCake, IronStore)
with zero per-model hyperparameter tuning (depth-proportional layers [6%, 33%, 50%, L-1]).
Writes full receipts to phase2-scaling-20261125/results_exp2_cross_arch.json.
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT_DIR))

from cluster import build_vocab_masks
from graft import (
    _DEFAULT_EVAL_ADV,
    _DEFAULT_EVAL_GENERAL,
    _DEFAULT_WHITEN_GEN,
    _POSITION_DIVERSE_G_PREFIXES,
    _fit_swiglu_gated_key,
    _get_backbone_and_norm,
    _pad_seqs,
)
from models import GraftConfig, ReclaimedSlotInfo
from run_exp1_scaling import ConceptSpec, unseen_trap_prompt, unseen_write_prompts
from run_exp1_v2_scaling import build_v2_hard_negatives, build_v2_naming_prompts
from run_exp1_v3_scaling import _CROSS_DOMAIN_NAMING_NEGS, fit_linear_cone_swiglu_key
from run_exp3_gradient_baseline import AUDIO_LEX, BEHAV_PROMPTS, CHEMO_LEX

FIVE_CONCEPTS = [
    ConceptSpec("solis", "place", ["taste", "smell"], "flavor", "Icelandic", "Iceland", "The capital city near where this {name} is made is", "Reykjavik", "Japanese", "Japan"),
    ConceptSpec("kora", "place", ["sound", "voice", "music"], "sound", "Japanese", "Japan", "The capital city of the country behind this {name} is", "Tokyo", "Icelandic", "Iceland"),
    ConceptSpec("HarborRoast", "city", ["coffee", "cafe", "bakery"], "coffee shop", "Portland", "Portland", "The US state where {name} is located is", "Oregon", "Chicago", "Chicago"),
    ConceptSpec("ZestCake", "cuisine", ["food", "dish", "dessert"], "pastry", "Sicilian", "Sicily", "The country where traditional {name} originates is", "Italy", "Bavarian", "Bavaria"),
    ConceptSpec("IronStore", "language", ["database", "storage", "engine"], "database", "Rust", "Rust", "The standard package manager for building {name} is", "Cargo", "Python", "Python"),
]


def load_model_and_tok(model_ref: str):
    pat = model_ref.replace("/", "--")
    snaps = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{pat}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"Not cached locally: {model_ref}")
    snap = snaps[0]
    try:
        tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True, local_files_only=True)
    except Exception:
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(snap, "tokenizer.json"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.decode([0])
    tok.padding_side = "right"

    model = None
    if "Qwen3.5" in model_ref:
        from transformers import Qwen3_5ForConditionalGeneration  # may raise ImportError
        model = Qwen3_5ForConditionalGeneration.from_pretrained(snap, dtype=torch.bfloat16, local_files_only=True).eval()
        layers = model.model.language_model.layers
        backbone = model.model.language_model
        final_norm = model.model.language_model.norm
    else:
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).eval()
        backbone, final_norm = _get_backbone_and_norm(model)
        layers = backbone.layers
    return model, backbone, layers, final_norm, tok


class CrossArchGrafter:
    def __init__(self, model_ref: str, untied_pure_ortho: bool = False) -> None:
        self.model_ref = model_ref
        self.untied_pure_ortho = untied_pure_ortho
        self.model, self.backbone, self.layers, self.final_norm, self.tok = load_model_and_tok(model_ref)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.emb_w = self.model.get_input_embeddings().weight
        self.out_w = self.model.get_output_embeddings().weight
        self.is_tied = self.out_w.data_ptr() == self.emb_w.data_ptr()
        self.vocab_size = int(self.emb_w.shape[0])
        self.d_model = int(self.emb_w.shape[1])
        self.num_layers = len(self.layers)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.orig_active_vocab = len(self.tok)

        self.cjk_mask, _, self.latin_mask, self.decoded = build_vocab_masks(self.tok, self.vocab_size)
        self.nw_np = self.final_norm.weight.detach().float().cpu().numpy()
        self.used_slots: set[tuple[int, int]] = set()
        self.used_tids: set[int] = set()
        self.e_kind_dirs: list[np.ndarray] = []
        self.u_perp_dirs: list[np.ndarray] = []
        self.nw_dwrite_dirs: list[np.ndarray] = []
        self._slot_cache: dict[tuple[int, str], list[tuple[int, float, float, float]]] = {}

        self.gen_seqs = [self.enc(s) for s in _DEFAULT_EVAL_GENERAL]
        self.adv_seqs = [self.enc(s) for s in _DEFAULT_EVAL_ADV]
        self.cross_neg_seqs = [self.enc(s) for s in _CROSS_DOMAIN_NAMING_NEGS]
        self.gen_logp_base = F.log_softmax(self.eval_next_logits(self.gen_seqs), dim=-1)
        self.adv_logp_base = F.log_softmax(self.eval_next_logits(self.adv_seqs), dim=-1)
        self.gen_p_base = self.gen_logp_base.exp()
        self.adv_p_base = self.adv_logp_base.exp()

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def enc_with_tid(self, pre: str, tid: int, suf: str = "") -> list[int]:
        ids = self.enc(pre) + [tid]
        if suf:
            ids += self.enc(suf)
        return ids

    def _resolve_word(self, word: str) -> int:
        w = word.strip()
        for cand in (f" {w}", w):
            ids = self.enc(cand)
            if len(ids) == 1:
                return ids[0]
        return self.enc(f" {w}")[-1]

    def alloc_unused_tid(self) -> int:
        for tid in range(self.orig_active_vocab + 5, self.vocab_size):
            if tid not in self.used_tids:
                self.used_tids.add(tid)
                return tid
        for tid in range(self.orig_active_vocab, self.vocab_size):
            if tid not in self.used_tids:
                self.used_tids.add(tid)
                return tid
        raise RuntimeError("No free in-bounds padding token slots.")

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

    @torch.no_grad()
    def precompute_quiet_slots(self, read_layers: list[int], write_layer: int) -> None:
        whiten_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        ids, mask, _ = _pad_seqs(whiten_seqs, self.tok.pad_token_id, self.device)
        mask_bool = mask.bool()
        eng_energy: dict[int, np.ndarray] = {}
        hooks = []
        for l_idx in sorted(set(read_layers + [write_layer])):
            def _make(li):
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
        cjk_t = self.cjk_mask[:active].to(self.device)
        lat_t = self.latin_mask[:active].to(self.device)

        for layer_idx, mode in [(rl, "input") for rl in read_layers] + [(write_layer, "output")]:
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
            self._slot_cache[(layer_idx, mode)] = [(j, round(cp, 3), -neg_ep, -neg_norm) for _t, neg_ep, cp, _lb, neg_norm, j in rows[:50]]
            del W
            torch.cuda.empty_cache()

    def _pop_slot(self, layer_idx: int, mode: str) -> int:
        for j, _cp, _ep, _n in self._slot_cache[(layer_idx, mode)]:
            if (layer_idx, j) not in self.used_slots:
                self.used_slots.add((layer_idx, j))
                return j
        raise RuntimeError(f"No slot at L{layer_idx}")

    def _qr_project_out(self, vec: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
        if not cols:
            return vec
        M = np.column_stack(cols)
        Q, R = np.linalg.qr(M, mode="reduced")
        diag_r = np.abs(np.diag(R))
        Q = Q[:, diag_r > (1e-6 * max(float(diag_r.max()), 1e-12))]
        v = vec - Q @ (Q.T @ vec)
        return v - Q @ (Q.T @ v)

    def graft_one(self, spec: ConceptSpec, read_layers: list[int], write_layer: int, seed: int) -> dict:
        tid = self.alloc_unused_tid()
        cfg = GraftConfig(
            token_id=tid, kind_words=spec.kind_words, kind_label=spec.kind_label,
            attr_adj=spec.attr_adj, attr_noun=spec.attr_noun, attr_category=spec.category,
            theta_deg=45.0, gamma=0.65, read_layers=read_layers, write_layer=write_layer,
            read_alphas=[1.5, 1.2, 0.8], seed=seed,
        )
        r_slots = [self._pop_slot(rl, "input") for rl in read_layers]
        w_slot = self._pop_slot(write_layer, "output")
        with torch.no_grad():
            for rl, j in list(zip(read_layers, r_slots)) + [(write_layer, w_slot)]:
                mlp = self.layers[rl].mlp
                mlp.gate_proj.weight[j].zero_()
                mlp.up_proj.weight[j].zero_()
                mlp.down_proj.weight[:, j].zero_()

        # Attribute superposition vectors
        noun_baseline = {"cuisine": "place", "city": "city", "place": "place", "language": "language"}.get(spec.category, "thing")
        adj_pos = [self.enc(f"{p} {spec.attr_adj}") for p in ("The", "A", "This", "Very")]
        adj_neg = [self.enc(f"{p} common") for p in ("The", "A", "This", "Very")]
        noun_pos = [self.enc(f"{p} {spec.attr_noun}") for p in ("The", "In", "From", "Of")]
        noun_neg = [self.enc(f"{p} {noun_baseline}") for p in ("The", "In", "From", "Of")]
        h_ap = self.capture_layer_outputs_last(adj_pos, read_layers)
        h_an = self.capture_layer_outputs_last(adj_neg, read_layers)
        h_np = self.capture_layer_outputs_last(noun_pos, read_layers)
        h_nn = self.capture_layer_outputs_last(noun_neg, read_layers)
        d_adj = {l: (h_ap[l] - h_an[l]).mean(axis=0) for l in read_layers}
        d_noun = {l: (h_np[l] - h_nn[l]).mean(axis=0) for l in read_layers}

        # Stage 1: Input & Output Embeddings (Untied Pure-Orthogonal Theorem vs Tied-Style!)
        with torch.no_grad():
            src_tids = [self._resolve_word(w) for w in spec.kind_words]
            E_src = self.emb_w[src_tids].detach().float()
            r_in = float(E_src.norm(dim=-1).mean().item())
            e_kind_np = E_src.mean(dim=0).cpu().numpy()
            e_kind_np = e_kind_np / (np.linalg.norm(e_kind_np) + 1e-12)

            rng = np.random.default_rng(seed + tid)
            u_raw = rng.standard_normal(self.d_model).astype(np.float64)
            prot_cols = [e_kind_np.astype(np.float64)] + [
                v.astype(np.float64) for v in (self.e_kind_dirs + self.u_perp_dirs + self.nw_dwrite_dirs)
            ]
            u_perp_np = self._qr_project_out(u_raw, prot_cols)
            u_perp_np = (u_perp_np / (np.linalg.norm(u_perp_np) + 1e-12)).astype(np.float32)

            if not any(abs(float(np.dot(e_kind_np, prev))) > 0.9999 for prev in self.e_kind_dirs):
                self.e_kind_dirs.append(e_kind_np.astype(np.float32))
            self.u_perp_dirs.append(u_perp_np.copy())

            if self.untied_pure_ortho and not self.is_tied:
                # UNTIED PURE-ORTHOGONAL THEOREM:
                # Input: FULL norm (gamma_in = 1.0, theta_in = 30 deg) for 100% kind twist & behavior
                # Output: PURE ORTHOGONAL u_perp (0% parent-kind bleed, 100% Write alignment!)
                th_in = math.radians(30.0)
                e_in_unit = math.cos(th_in) * e_kind_np + math.sin(th_in) * u_perp_np
                e_in_unit = e_in_unit / (np.linalg.norm(e_in_unit) + 1e-12)
                e_in_np = (1.0 * r_in * e_in_unit).astype(np.float32)

                r_out = float(self.out_w[src_tids].detach().float().norm(dim=-1).mean().item())
                e_out_np = (0.65 * r_out * u_perp_np).astype(np.float32)
                self.emb_w[tid].copy_(torch.tensor(e_in_np, dtype=self.emb_w.dtype, device=self.device))
                self.out_w[tid].copy_(torch.tensor(e_out_np, dtype=self.out_w.dtype, device=self.device))
            else:
                # Tied-style (gamma=0.65, theta=45 deg)
                th = math.radians(45.0)
                e_G_unit = math.cos(th) * e_kind_np + math.sin(th) * u_perp_np
                e_G_unit = e_G_unit / (np.linalg.norm(e_G_unit) + 1e-12)
                e_in_np = (0.65 * r_in * e_G_unit).astype(np.float32)
                self.emb_w[tid].copy_(torch.tensor(e_in_np, dtype=self.emb_w.dtype, device=self.device))
                if not self.is_tied:
                    r_out = float(self.out_w[src_tids].detach().float().norm(dim=-1).mean().item())
                    e_out_np = (0.65 * r_out * e_G_unit).astype(np.float32)
                    self.out_w[tid].copy_(torch.tensor(e_out_np, dtype=self.out_w.dtype, device=self.device))
                else:
                    e_out_np = e_in_np

        # Stage 2: Read superposition
        whiten_gen_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        hard_neg_seqs = [self.enc(s) for s in build_v2_hard_negatives(cfg)]
        kind_neg_seqs = [self.enc(f"the {w}") for w in spec.kind_words] + [self.enc(f"the {spec.kind_label}")]
        fit_g_seqs = [self.enc_with_tid(pre, tid) for pre in _POSITION_DIVERSE_G_PREFIXES]
        alphas = [1.5, 1.2, 0.8]
        for idx_l, (rl, j) in enumerate(zip(read_layers, r_slots)):
            X_pos = self.capture_mlp_inputs(fit_g_seqs, rl, False)
            X_bg = np.vstack([
                self.capture_mlp_inputs(whiten_gen_seqs, rl, True),
                self.capture_mlp_inputs(kind_neg_seqs, rl, True),
            ])
            wg, wu, ap, _snr = _fit_swiglu_gated_key(X_pos, X_bg)
            payload = alphas[idx_l] * (d_adj[rl] if idx_l == 0 else (d_noun[rl] if idx_l == len(read_layers) - 1 else 0.5 * (d_adj[rl] + d_noun[rl])))
            wd = payload / ap
            mlp = self.layers[rl].mlp
            with torch.no_grad():
                mlp.gate_proj.weight[j].copy_(torch.tensor(wg, dtype=mlp.gate_proj.weight.dtype, device=self.device))
                mlp.up_proj.weight[j].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=self.device))
                mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=self.device))

        # Stage 3: Linear Cone-Gated Write + Exact QR + Capped RMSNorm Margin
        fit_write_seqs = [self.enc(s) for s in build_v2_naming_prompts(cfg)]
        whiten_g_sent = [self.enc_with_tid("The subtle", tid, " lingered in the air")]
        X_pos_w = self.capture_mlp_inputs(fit_write_seqs, write_layer, False)
        X_bg_w = np.vstack([
            self.capture_mlp_inputs(whiten_gen_seqs, write_layer, True),
            self.capture_mlp_inputs(hard_neg_seqs, write_layer, True),
            self.capture_mlp_inputs(self.cross_neg_seqs, write_layer, True),
            self.capture_mlp_inputs(whiten_g_sent, write_layer, True),
        ])
        X_hl = np.vstack([
            self.capture_mlp_inputs(hard_neg_seqs, write_layer, False),
            self.capture_mlp_inputs(self.cross_neg_seqs, write_layer, False),
        ])
        wg_w, wu_w, a_ref_w, _am, w_snr = fit_linear_cone_swiglu_key(
            X_pos_w, X_bg_w, X_hl, hard_weight=1.5, ridge_frac=0.25, gate_tau=0.55, gate_sharp=12.0
        )
        rms_w = self.capture_pre_norm_rms(fit_write_seqs)
        base_fit_logits = self.eval_next_logits(fit_write_seqs)
        max_comp = base_fit_logits[:, : self.orig_active_vocab].max(dim=-1).values
        z_gap = float((max_comp - base_fit_logits[:, tid]).max().item())

        with torch.no_grad():
            nw64 = self.nw_np.astype(np.float64)
            d_raw = nw64 * u_perp_np.astype(np.float64)
            forb_cols = [nw64 * ek.astype(np.float64) for ek in self.e_kind_dirs] + [
                nw64 * up.astype(np.float64) for up in self.u_perp_dirs[:-1]
            ]
            d_write = self._qr_project_out(d_raw, forb_cols)
            d_write = (d_write / (np.linalg.norm(d_write) + 1e-12)).astype(np.float32)

            nwd = self.nw_np * d_write
            self.nw_dwrite_dirs.append((nwd / (np.linalg.norm(nwd) + 1e-12)).astype(np.float32))

            eff_gain = float(np.dot(d_write, self.nw_np * e_out_np))
            z_max = math.sqrt(self.d_model) * max(eff_gain, 1e-6)
            dz_target = min(26.0, 0.78 * z_max, max(20.0, z_gap + 5.0))
            denom_sq = max(eff_gain**2 - (dz_target**2) / self.d_model, 0.25 * (eff_gain**2))
            m_req = (dz_target * rms_w) / math.sqrt(denom_sq)
            beta = m_req / max(a_ref_w, 1e-4)
            w_down_w = beta * d_write

            mlp_w = self.layers[write_layer].mlp
            mlp_w.gate_proj.weight[w_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=self.device))
            mlp_w.up_proj.weight[w_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=self.device))
            mlp_w.down_proj.weight[:, w_slot].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=self.device))

        return {
            "tid": tid,
            "write_slot": w_slot,
            "w_down_norm": round(float(np.linalg.norm(w_down_w)), 2),
            "eff_gain": round(eff_gain, 4),
            "write_snr": round(w_snr, 1),
        }


def run_one_config(model_ref: str, untied_pure_ortho: bool) -> dict:
    t0 = time.time()
    tag = f"{model_ref} [{'UNTIED_PURE_ORTHO' if untied_pure_ortho else 'TIED_STYLE'}]"
    print(f"\n=== {tag} ===", flush=True)
    g = CrossArchGrafter(model_ref, untied_pure_ortho=untied_pure_ortho)
    read_layers = sorted({max(1, round(g.num_layers * 0.06)), max(2, round(g.num_layers * 0.33)), max(3, round(g.num_layers * 0.50))})
    write_layer = g.num_layers - 1
    print(f"  L={g.num_layers}, D={g.d_model}, tied={g.is_tied}, free_slots={g.vocab_size - g.orig_active_vocab}, read_L={read_layers}, write_L={write_layer}", flush=True)

    # Baseline multi-hop
    mh_tids = [g.enc(f" {s.multihop_target}")[0] for s in FIVE_CONCEPTS]
    base_mh_logits = g.eval_next_logits([g.enc(s.multihop_prompt_tpl.format(name="concept")) for s in FIVE_CONCEPTS])
    base_mh_logp = F.log_softmax(base_mh_logits, dim=-1)
    base_mh_ranks = [int((base_mh_logits[i] > base_mh_logits[i, mh_tids[i]]).sum().item()) + 1 for i in range(5)]

    g.precompute_quiet_slots(read_layers, write_layer)

    tids = []
    w_slots = []
    w_norms = []
    eff_gains = []
    for idx, spec in enumerate(FIVE_CONCEPTS):
        info = g.graft_one(spec, read_layers, write_layer, seed=43 + idx)
        tids.append(info["tid"])
        w_slots.append(info["write_slot"])
        w_norms.append(info["w_down_norm"])
        eff_gains.append(info["eff_gain"])

    # Evaluate unseen Write (10 prompts) + cross-bleed
    w_seqs = [g.enc(p) for s in FIVE_CONCEPTS for p in unseen_write_prompts(s)]
    w_logits = g.eval_next_logits(w_seqs).cpu()
    w_probs = F.softmax(w_logits, dim=-1)
    top1 = 0
    any_t1 = 0
    xbleeds = []
    tid_t = torch.tensor(tids, dtype=torch.long)
    for j in range(5):
        hit = False
        for m in range(2):
            row = 2 * j + m
            rk = int((w_logits[row] > w_logits[row, tids[j]]).sum().item()) + 1
            if rk == 1:
                top1 += 1
                hit = True
            op = w_probs[row, tid_t].numpy()
            op[j] = -1.0
            xbleeds.append(float(op.max()))
        if hit:
            any_t1 += 1

    # Traps
    t_logits = g.eval_next_logits([g.enc(unseen_trap_prompt(s)) for s in FIVE_CONCEPTS]).cpu()
    t_probs = F.softmax(t_logits, dim=-1)
    trap_probs = [float(t_probs[j, tids[j]]) for j in range(5)]

    # Multi-hop (encode using tid directly since we didn't mutate tokenizer)
    mh_seqs = []
    for j, s in enumerate(FIVE_CONCEPTS):
        pre, suf = s.multihop_prompt_tpl.split("{name}")
        mh_seqs.append(g.enc_with_tid(pre.rstrip(), tids[j], suf))
    mh_logits = g.eval_next_logits(mh_seqs).cpu()
    mh_logp = F.log_softmax(mh_logits, dim=-1)
    mh_ranks = [int((mh_logits[j] > mh_logits[j, mh_tids[j]]).sum().item()) + 1 for j in range(5)]
    mh_lifts = [float(mh_logp[j, mh_tids[j]] - base_mh_logp[j, mh_tids[j]].cpu()) for j in range(5)]

    # BehavSep B(solis) - B(kora)
    chemo_ids = [g.enc(w)[0] for w in CHEMO_LEX]
    audio_ids = [g.enc(w)[0] for w in AUDIO_LEX]
    b_scores = []
    for j in (0, 1):
        seqs = []
        for p in BEHAV_PROMPTS:
            pre, suf = p.split("{name}")
            seqs.append(g.enc_with_tid(pre.rstrip(), tids[j], suf))
        lp = F.log_softmax(g.eval_next_logits(seqs), dim=-1)
        b_scores.append(float((torch.logsumexp(lp[:, chemo_ids], dim=-1) - torch.logsumexp(lp[:, audio_ids], dim=-1)).mean().item()))

    # Collateral KL + AdvMaxG
    g_lp = F.log_softmax(g.eval_next_logits(g.gen_seqs), dim=-1)
    a_lp = F.log_softmax(g.eval_next_logits(g.adv_seqs), dim=-1)
    gen_kl = float((g.gen_p_base * (g.gen_logp_base - g_lp)).sum(dim=-1).mean())
    adv_kl = float((g.adv_p_base * (g.adv_logp_base - a_lp)).sum(dim=-1).mean())
    adv_max_g = float(a_lp[:, tids].exp().max())

    # Audit P@5
    with torch.no_grad():
        norms_w = g.layers[write_layer].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
    med_n = float(np.median(norms_w))
    top5_idx = set(np.argsort(norms_w)[-5:].tolist())
    audit_p5 = len(top5_idx.intersection(w_slots)) / 5.0

    res = {
        "model": model_ref,
        "mode": "untied_pure_ortho" if untied_pure_ortho else "tied_style",
        "is_tied": g.is_tied,
        "num_layers": g.num_layers,
        "d_model": g.d_model,
        "read_layers": read_layers,
        "write_layer": write_layer,
        "params_edited_pct": round(100.0 * 5 * 13 * g.d_model / g.total_params, 5),
        "write_top1_pct": round(100.0 * top1 / 10.0, 1),
        "write_any_top1_pct": round(100.0 * any_t1 / 5.0, 1),
        "cross_bleed_max": round(max(xbleeds), 6),
        "trap_max": round(max(trap_probs), 6),
        "trap_mean": round(float(np.mean(trap_probs)), 6),
        "mh_ranks": mh_ranks,
        "mh_base_ranks": base_mh_ranks,
        "mh_top10_pct": round(100.0 * sum(r <= 10 for r in mh_ranks) / 5.0, 1),
        "mh_mean_logp_lift": round(float(np.mean(mh_lifts)), 3),
        "behav_solis": round(b_scores[0], 3),
        "behav_kora": round(b_scores[1], 3),
        "behav_sep": round(b_scores[0] - b_scores[1], 3),
        "gen_kl": round(gen_kl, 6),
        "adv_kl": round(adv_kl, 6),
        "adv_max_g": round(adv_max_g, 6),
        "write_norm_median": round(float(np.median(w_norms)), 2),
        "write_eff_gain_median": round(float(np.median(eff_gains)), 4),
        "audit_norm_ratio": round(float(np.median(w_norms)) / med_n, 1),
        "audit_p5": audit_p5,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(
        f"  >>> WTop1={res['write_top1_pct']}% (Any={res['write_any_top1_pct']}%) | XBleedMax={res['cross_bleed_max']:.5f} | "
        f"TrapMax={res['trap_max']:.5f} | MH={mh_ranks} (base {base_mh_ranks}, top10={res['mh_top10_pct']}%, lift={res['mh_mean_logp_lift']:+.2f}) | "
        f"BehavSep={res['behav_sep']:+.2f} | GenKL={res['gen_kl']:.5f} AdvKL={res['adv_kl']:.5f} AdvMaxG={res['adv_max_g']:.6f} | "
        f"wNorm={res['write_norm_median']} (gain={res['write_eff_gain_median']}) | AuditP@5={audit_p5:.2f} [{res['elapsed_s']}s]",
        flush=True,
    )
    del g
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    runs = [
        ("Qwen/Qwen2.5-7B-Instruct", False),
        ("Qwen/Qwen2.5-7B-Instruct", True),
        ("Nanbeige/Nanbeige4.2-3B", False),
        ("Nanbeige/Nanbeige4.2-3B", True),
        ("Qwen/Qwen2.5-3B-Instruct", False),
        ("Qwen/Qwen3.5-4B", False),
    ]
    all_res = []
    for m, upo in runs:
        try:
            all_res.append(run_one_config(m, upo))
        except Exception as e:
            print(f"  [SKIP] {m} (untied_pure_ortho={upo}): {type(e).__name__}: {e}", flush=True)
            all_res.append({"model": m, "mode": "untied_pure_ortho" if upo else "tied_style", "error": f"{type(e).__name__}: {e}"})
            gc.collect()
            torch.cuda.empty_cache()
    out_path = OUT_DIR / "results_exp2_cross_arch.json"
    out_path.write_text(json.dumps(all_res, indent=2), encoding="utf-8")
    print(f"\nSaved Exp 2 receipts to {out_path}", flush=True)


if __name__ == "__main__":
    main()
