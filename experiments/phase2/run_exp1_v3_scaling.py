"""
Track B — Experiment 1-v3: K = 1 -> 50 -> 100 Scaling Curve on Qwen/Qwen3-4B-Instruct-2507
with Linear Cone-Gated SwiGLU (w_up = b), Exact RMSNorm-Inverted Margin Calibration,
(theta=45 deg, gamma=0.65), Balanced Cross-Domain + Same-Kind Hard-Negative Whitening,
Quiet-CJK Activation-Filtered Reclamation, and Exact QR Write Projection.

Uses bit-for-bit identical 100 concepts & unseen eval prompts as Exp 1-v1/v2.
Writes full receipts to phase2-scaling-20261125/results_exp1_v3_scaling.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT_DIR))

from graft import _DEFAULT_WHITEN_GEN, _POSITION_DIVERSE_G_PREFIXES, _fit_swiglu_gated_key
from models import GraftConfig
from run_exp1_scaling import eval_logits_batched, unseen_trap_prompt, unseen_write_prompts
from run_exp1_v2_scaling import (
    MODEL_REF,
    Phase2Grafter,
    build_100_concepts,
    build_v2_hard_negatives,
    build_v2_naming_prompts,
)

MILESTONES = [1, 2, 5, 10, 20, 35, 50, 75, 100]

# Strictly generic cross-domain naming negatives (zero overlap with eval prompts/attributes)
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


def fit_linear_cone_swiglu_key(
    X_pos: np.ndarray,
    X_bg: np.ndarray,
    X_hard: np.ndarray | None = None,
    hard_weight: float = 1.5,
    ridge_frac: float = 0.25,
    bias_ridge_frac: float = 0.10,
    gate_tau: float = 0.55,
    gate_sharp: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Linear Cone-Gated SwiGLU Key:
      w_gate = gate_sharp * (k - gate_tau * b)
      w_up   = b   (unit-response cone vector, b @ x ~= 1.0 -> linear above threshold!)
    Returns (w_gate, w_up, a_ref_p35, a_pos_mean, snr).
    """
    D = X_pos.shape[1]
    C_bg = (X_bg.T @ X_bg) / max(len(X_bg), 1)
    if X_hard is not None and len(X_hard) > 0:
        C_hard = (X_hard.T @ X_hard) / len(X_hard)
        C_neg = C_bg + hard_weight * C_hard
    else:
        C_neg = C_bg
    lam = ridge_frac * float(np.mean(np.diag(C_neg)))
    k = np.linalg.solve(C_neg + lam * np.eye(D), X_pos.mean(axis=0))
    pos_mean = float((X_pos @ k).mean())
    if abs(pos_mean) < 1e-12:
        raise RuntimeError("Degenerate SwiGLU key fit.")
    k = k / pos_mean

    X_all = np.vstack([X_pos, X_bg] + ([X_hard] if X_hard is not None and len(X_hard) > 0 else []))
    C_all = (X_all.T @ X_all) / len(X_all)
    lam_b = bias_ridge_frac * float(np.mean(np.diag(C_all)))
    b = np.linalg.solve(C_all + lam_b * np.eye(D), X_all.mean(axis=0))
    b = b / float((X_all @ b).mean())

    w_gate = gate_sharp * (k - gate_tau * b)
    w_up = b  # Linear cone gate: eliminates quadratic collapse & outlier blow-up!

    def _act(X: np.ndarray) -> np.ndarray:
        g = X @ w_gate
        silu_g = g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0)))
        return silu_g * (X @ w_up)

    a_pos = _act(X_pos)
    eval_neg = np.vstack([X_bg, X_hard]) if (X_hard is not None and len(X_hard) > 0) else X_bg
    a_neg = _act(eval_neg)
    neg_p99 = max(float(np.percentile(np.abs(a_neg), 99)), 1e-6)
    a_mean = float(a_pos.mean())
    a_ref = max(float(np.percentile(a_pos, 35)), 0.55 * a_mean, 1e-4)
    snr = a_mean / neg_p99
    return w_gate, w_up, a_ref, a_mean, snr


class Phase2V3Grafter(Phase2Grafter):
    def __init__(self, model_ref: str, device: str | None = None) -> None:
        super().__init__(model_ref, device=device)
        self.cross_neg_seqs = [self.enc(s) for s in _CROSS_DOMAIN_NAMING_NEGS]

    def graft_concept_v3(self, cfg: GraftConfig) -> dict:
        tid, token_content = self.register_token_slot(token_name=cfg.token_name, requested_id=cfg.token_id)
        read_layers = cfg.read_layers or [2, 12, 18]
        write_layer = cfg.write_layer if cfg.write_layer is not None else (self.num_layers - 1)

        # 1. Reclaim activation-filtered quiet CJK slots
        reclaimed_read = []
        for rl in read_layers:
            sl = self._pop_quiet_slot(rl, "input")
            sl.role = f"read_L{rl}"
            reclaimed_read.append(sl)
        write_slot = self._pop_quiet_slot(write_layer, "output")
        write_slot.role = f"write_L{write_layer}"
        all_slots = reclaimed_read + [write_slot]

        with torch.no_grad():
            for sl in all_slots:
                mlp = self.layers[sl.layer].mlp
                mlp.gate_proj.weight[sl.neuron_idx].zero_()
                mlp.up_proj.weight[sl.neuron_idx].zero_()
                mlp.down_proj.weight[:, sl.neuron_idx].zero_()

        # 2. Compute position-matched attribute superposition payloads
        cat_l = cfg.attr_category.lower().strip()
        noun_baseline = {
            "cuisine": "place", "city": "city", "place": "place",
            "language": "language", "material": "material", "product": "activity", "brand": "thing",
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

        # 3. Stage 1 — Scaled-Rotated Kind Base (theta=45 deg, gamma=0.65) with QR orthogonal private address
        with torch.no_grad():
            src_tids = [self._resolve_word_token(w) for w in cfg.kind_words]
            E_src = self.emb_w[src_tids].detach().float()
            r_ref = float(E_src.norm(dim=-1).mean().item())
            e_kind_np = E_src.mean(dim=0).cpu().numpy()
            e_kind_np = e_kind_np / (np.linalg.norm(e_kind_np) + 1e-12)

            rng = np.random.default_rng(cfg.seed + tid)
            u_raw = rng.standard_normal(self.d_model).astype(np.float64)
            prot_cols = [e_kind_np.astype(np.float64)] + [
                v.astype(np.float64) for v in (self.e_kind_dirs + self.u_perp_dirs + self.nw_dwrite_dirs)
            ]
            u_perp_np = self._qr_project_out(u_raw, prot_cols)
            u_perp_np = (u_perp_np / (np.linalg.norm(u_perp_np) + 1e-12)).astype(np.float32)

            if not any(abs(float(np.dot(e_kind_np, prev))) > 0.9999 for prev in self.e_kind_dirs):
                self.e_kind_dirs.append(e_kind_np.astype(np.float32))
            self.u_perp_dirs.append(u_perp_np.copy())

            th = math.radians(cfg.theta_deg)
            e_G_unit = math.cos(th) * e_kind_np + math.sin(th) * u_perp_np
            e_G_unit = e_G_unit / (np.linalg.norm(e_G_unit) + 1e-12)
            e_G_np = (cfg.gamma * r_ref * e_G_unit).astype(np.float32)
            e_G_vec = torch.tensor(e_G_np, dtype=self.emb_w.dtype, device=self.device)
            self.emb_w[tid].copy_(e_G_vec)
            if self.out_w.data_ptr() != self.emb_w.data_ptr():
                self.out_w[tid].copy_(e_G_vec)

        whiten_gen_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        hard_neg_texts = build_v2_hard_negatives(cfg)
        hard_neg_seqs = [self.enc(s) for s in hard_neg_texts]
        kind_neg_seqs = [self.enc(f"the {w.strip()}") for w in cfg.kind_words] + [
            self.enc(f"the {cfg.kind_label}"),
            self.enc(f"this {cfg.kind_label}"),
        ]
        fit_g_seqs = [self.enc_with_token(pre, tid) for pre in _POSITION_DIVERSE_G_PREFIXES]

        # 4. Stage 2 — Multi-Layer Read Superposition
        alphas = list(cfg.read_alphas)
        read_snrs = {}
        n_rl = len(read_layers)
        for idx_l, (rl, sl) in enumerate(zip(read_layers, reclaimed_read)):
            X_pos = self.capture_mlp_inputs(fit_g_seqs, rl, all_tokens=False)
            X_bg = np.vstack([
                self.capture_mlp_inputs(whiten_gen_seqs, rl, all_tokens=True),
                self.capture_mlp_inputs(kind_neg_seqs, rl, all_tokens=True),
            ])
            w_gate, w_up, a_pos, snr = _fit_swiglu_gated_key(
                X_pos, X_bg,
                ridge_frac=cfg.ridge_frac, bias_ridge_frac=cfg.bias_ridge_frac,
                gate_tau=cfg.gate_tau, gate_sharp=cfg.gate_sharp,
            )
            read_snrs[rl] = snr
            alpha_l = alphas[idx_l]
            if idx_l == 0:
                payload = alpha_l * d_adj[rl]
            elif idx_l == n_rl - 1:
                payload = alpha_l * d_noun[rl]
            else:
                payload = alpha_l * (0.5 * d_adj[rl] + 0.5 * d_noun[rl])
            w_down = payload / a_pos
            mlp = self.layers[rl].mlp
            with torch.no_grad():
                mlp.gate_proj.weight[sl.neuron_idx].copy_(torch.tensor(w_gate, dtype=mlp.gate_proj.weight.dtype, device=self.device))
                mlp.up_proj.weight[sl.neuron_idx].copy_(torch.tensor(w_up, dtype=mlp.up_proj.weight.dtype, device=self.device))
                mlp.down_proj.weight[:, sl.neuron_idx].copy_(torch.tensor(w_down, dtype=mlp.down_proj.weight.dtype, device=self.device))

        # 5. Stage 3 — Linear Cone-Gated Write + Balanced Cross-Domain Negatives + Exact RMSNorm-Inverted Margin
        naming_prompts = build_v2_naming_prompts(cfg)
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
        w_gate_w, w_up_w, a_ref_w, a_mean_w, write_snr = fit_linear_cone_swiglu_key(
            X_pos_w, X_bg_w, X_hard_last,
            hard_weight=1.5,
            ridge_frac=0.25,
            bias_ridge_frac=0.10,
            gate_tau=0.55,
            gate_sharp=12.0,
        )
        rms_write = self.capture_pre_norm_rms(fit_write_seqs)

        # Measure actual baseline logit gap on fit prompts (pretrained vocab only, zero arms-race!)
        base_fit_logits = self.eval_next_logits(fit_write_seqs)
        max_comp = base_fit_logits[:, : self.orig_active_vocab].max(dim=-1).values
        g_base = base_fit_logits[:, tid]
        z_gap = float((max_comp - g_base).max().item())

        with torch.no_grad():
            nw64 = self.nw_np.astype(np.float64)
            d_raw = nw64 * u_perp_np.astype(np.float64)
            forb_cols = [nw64 * ek.astype(np.float64) for ek in self.e_kind_dirs] + [
                nw64 * up.astype(np.float64) for up in self.u_perp_dirs[:-1]
            ]
            d_write = self._qr_project_out(d_raw, forb_cols)
            d_write = (d_write / (np.linalg.norm(d_write) + 1e-12)).astype(np.float32)

            nwd = self.nw_np * d_write
            nwd = nwd / (np.linalg.norm(nwd) + 1e-12)
            self.nw_dwrite_dirs.append(nwd.astype(np.float32))

            eff_gain = float(np.dot(d_write, self.nw_np * e_G_np))
            z_max = math.sqrt(self.d_model) * max(eff_gain, 1e-6)
            # Exact closed-form RMSNorm-inverted target boost, cleanly capped at 26.0 logits (zero norm arms-race!)
            dz_target = min(26.0, 0.78 * z_max, max(20.0, z_gap + 5.0))
            denom_sq = max(eff_gain**2 - (dz_target**2) / self.d_model, 0.25 * (eff_gain**2))
            m_req = (dz_target * rms_write) / math.sqrt(denom_sq)
            beta = m_req / max(a_ref_w, 1e-4)
            w_down_w = beta * d_write

            mlp_w = self.layers[write_layer].mlp
            mlp_w.gate_proj.weight[write_slot.neuron_idx].copy_(torch.tensor(w_gate_w, dtype=mlp_w.gate_proj.weight.dtype, device=self.device))
            mlp_w.up_proj.weight[write_slot.neuron_idx].copy_(torch.tensor(w_up_w, dtype=mlp_w.up_proj.weight.dtype, device=self.device))
            mlp_w.down_proj.weight[:, write_slot.neuron_idx].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=self.device))

        self.write_vecs.append((tid, d_write, e_G_np))
        return {
            "token_id": tid,
            "token_name": token_content,
            "write_slot": write_slot.neuron_idx,
            "write_fit_snr": write_snr,
            "z_gap": round(z_gap, 2),
            "z_max": round(z_max, 2),
            "dz_target": round(dz_target, 2),
            "w_down_norm": round(float(np.linalg.norm(w_down_w)), 2),
            "slot_cjk_shares": {sl.role: sl.cjk_share for sl in all_slots},
            "slot_indices": {sl.role: sl.neuron_idx for sl in all_slots},
        }


def main() -> None:
    t_start = time.time()
    specs = build_100_concepts()
    print(f"=== Track B Exp 1-v3: K=1..100 Linear Cone-Gated Scaling on {MODEL_REF} ===", flush=True)
    grafter = Phase2V3Grafter(MODEL_REF)
    print(f"Loaded on device={grafter.device}, free_slots={grafter.vocab_size - grafter.orig_active_vocab}", flush=True)

    mh_target_ids = [grafter.enc(f" {s.multihop_target}")[0] for s in specs]
    base_mh_seqs = [grafter.enc(s.multihop_prompt_tpl.format(name="concept")) for s in specs]
    base_mh_logits = eval_logits_batched(grafter, base_mh_seqs)
    base_mh_logp = F.log_softmax(base_mh_logits, dim=-1)
    baseline_mh = [
        {
            "token_name": specs[i].token_name,
            "base_logp": float(base_mh_logp[i, mh_target_ids[i]]),
            "base_rank": int((base_mh_logits[i] > base_mh_logits[i, mh_target_ids[i]]).sum().item()) + 1,
        }
        for i in range(100)
    ]

    read_layers = [2, 12, 18]
    write_layer = 35
    grafter.precompute_quiet_cjk_rankings(read_layers, write_layer)

    grafted_tids: list[int] = []
    grafted_write_slots: list[int] = []
    per_concept_log: list[dict] = []
    milestone_results: dict[int, dict] = {}
    nw_np = grafter.nw_np

    for idx, spec in enumerate(specs):
        k = idx + 1
        t_k = time.time()
        cfg = GraftConfig(
            token_name=spec.token_name,
            kind_words=spec.kind_words,
            kind_label=spec.kind_label,
            attr_adj=spec.attr_adj,
            attr_noun=spec.attr_noun,
            attr_category=spec.category,
            theta_deg=45.0,
            gamma=0.65,
            read_layers=read_layers,
            write_layer=write_layer,
            read_alphas=[1.5, 1.2, 0.8],
            seed=42 + k,
        )
        info = grafter.graft_concept_v3(cfg)
        grafted_tids.append(info["token_id"])
        grafted_write_slots.append(info["write_slot"])
        per_concept_log.append({
            "k": k,
            "token_name": spec.token_name,
            "category": spec.category,
            "write_fit_snr": round(info["write_fit_snr"], 1),
            "z_gap": info["z_gap"],
            "z_max": info["z_max"],
            "dz_target": info["dz_target"],
            "w_down_norm": info["w_down_norm"],
            "elapsed_s": round(time.time() - t_k, 2),
        })

        if k in MILESTONES:
            t_ev = time.time()
            write_seqs = [grafter.enc(p) for j in range(k) for p in unseen_write_prompts(specs[j])]
            w_logits = eval_logits_batched(grafter, write_seqs)
            w_probs = F.softmax(w_logits, dim=-1)

            top1_hits = top5_hits = concept_top1 = 0
            self_probs, self_ranks, cross_bleeds_max, cross_bleeds_all = [], [], [], []
            worst_bleed_pair = None
            tid_tensor = torch.tensor(grafted_tids, dtype=torch.long)

            for j in range(k):
                c_hit = False
                for m in range(2):
                    row = 2 * j + m
                    tid_j = grafted_tids[j]
                    pj = float(w_probs[row, tid_j])
                    rj = int((w_logits[row] > w_logits[row, tid_j]).sum().item()) + 1
                    self_probs.append(pj)
                    self_ranks.append(rj)
                    if rj == 1:
                        top1_hits += 1
                        c_hit = True
                    if rj <= 5:
                        top5_hits += 1
                    if k > 1:
                        other_probs = w_probs[row, tid_tensor].numpy()
                        other_probs[j] = -1.0
                        best_other = int(np.argmax(other_probs))
                        mx = float(other_probs[best_other])
                        cross_bleeds_max.append(mx)
                        cross_bleeds_all.extend(other_probs[other_probs >= 0.0].tolist())
                        if worst_bleed_pair is None or mx > worst_bleed_pair[0]:
                            worst_bleed_pair = (mx, specs[j].token_name, specs[best_other].token_name)
                if c_hit:
                    concept_top1 += 1

            trap_seqs = [grafter.enc(unseen_trap_prompt(specs[j])) for j in range(k)]
            t_logits = eval_logits_batched(grafter, trap_seqs)
            t_probs = F.softmax(t_logits, dim=-1)
            trap_self_probs = [float(t_probs[j, grafted_tids[j]]) for j in range(k)]
            worst_trap_j = int(np.argmax(trap_self_probs))

            mh_seqs = [grafter.enc(specs[j].multihop_prompt_tpl.format(name=specs[j].token_name)) for j in range(k)]
            mh_logits = eval_logits_batched(grafter, mh_seqs)
            mh_logp = F.log_softmax(mh_logits, dim=-1)
            mh_ranks = [int((mh_logits[j] > mh_logits[j, mh_target_ids[j]]).sum().item()) + 1 for j in range(k)]
            mh_lifts = [float(mh_logp[j, mh_target_ids[j]]) - baseline_mh[j]["base_logp"] for j in range(k)]
            n_pos_lift = sum(x > 0 for x in mh_lifts)
            mh_sign_p = float(binomtest(n_pos_lift, k, 0.5, alternative="greater").pvalue) if k >= 5 else 1.0

            g_lp = F.log_softmax(grafter.eval_next_logits(grafter.gen_seqs), dim=-1)
            a_lp = F.log_softmax(grafter.eval_next_logits(grafter.adv_seqs), dim=-1)
            gen_kl = float((grafter.gen_p_base * (grafter.gen_logp_base - g_lp)).sum(dim=-1).mean())
            adv_kl = float((grafter.adv_p_base * (grafter.adv_logp_base - a_lp)).sum(dim=-1).mean())
            adv_max_any_g = float(a_lp[:, grafted_tids].exp().max())
            adv_sum_any_g = float(a_lp[:, grafted_tids].exp().sum(dim=-1).max())

            D_mat = np.stack([w[1] for w in grafter.write_vecs], axis=0)
            E_mat = np.stack([nw_np * w[2] for w in grafter.write_vecs], axis=0)
            G_mat = D_mat @ E_mat.T
            self_gains = np.diag(G_mat)
            off_mask = ~np.eye(k, dtype=bool)
            off_max = float(np.abs(G_mat[off_mask]).max()) if k > 1 else 0.0
            geom_off_ratio = off_max / max(float(self_gains.mean()), 1e-8)

            with torch.no_grad():
                norms35 = grafter.layers[write_layer].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
            med_norm35 = float(np.median(norms35))
            graft_norms35 = [float(norms35[s_idx]) for s_idx in grafted_write_slots]
            top_k_idx = set(np.argsort(norms35)[-k:].tolist())
            detected_at_k = len(top_k_idx.intersection(grafted_write_slots))

            by_cat = {}
            for cat in ("place", "city", "cuisine", "language", "product"):
                idxs = [j for j in range(k) if specs[j].category == cat]
                if not idxs:
                    continue
                by_cat[cat] = {
                    "n": len(idxs),
                    "write_top1_pct": round(100.0 * sum(1 for j in idxs for m in range(2) if self_ranks[2 * j + m] == 1) / (2 * len(idxs)), 1),
                    "write_any_top1_pct": round(100.0 * sum(1 for j in idxs if min(self_ranks[2 * j], self_ranks[2 * j + 1]) == 1) / len(idxs), 1),
                    "mh_top10_pct": round(100.0 * sum(1 for j in idxs if mh_ranks[j] <= 10) / len(idxs), 1),
                    "mh_mean_logp_lift": round(float(np.mean([mh_lifts[j] for j in idxs])), 3),
                    "trap_max": round(max(trap_self_probs[j] for j in idxs), 5),
                    "trap_mean": round(float(np.mean([trap_self_probs[j] for j in idxs])), 5),
                }

            ms = {
                "K": k,
                "params_edited": k * 13 * grafter.d_model,
                "model_param_pct": round(100.0 * k * 13 * grafter.d_model / grafter.total_params, 5),
                "write_prompt_top1_pct": round(100.0 * top1_hits / (2 * k), 2),
                "write_prompt_top5_pct": round(100.0 * top5_hits / (2 * k), 2),
                "write_concept_any_top1_pct": round(100.0 * concept_top1 / k, 2),
                "write_mean_prob": round(float(np.mean(self_probs)), 4),
                "write_median_rank": float(np.median(self_ranks)),
                "cross_bleed_max": float(max(cross_bleeds_max)) if cross_bleeds_max else 0.0,
                "cross_bleed_p99": float(np.percentile(cross_bleeds_all, 99)) if cross_bleeds_all else 0.0,
                "cross_bleed_mean": float(np.mean(cross_bleeds_all)) if cross_bleeds_all else 0.0,
                "worst_bleed_pair": worst_bleed_pair,
                "trap_self_max": float(max(trap_self_probs)),
                "trap_self_mean": float(np.mean(trap_self_probs)),
                "worst_trap_concept": specs[worst_trap_j].token_name,
                "multihop_top1_pct": round(100.0 * sum(r == 1 for r in mh_ranks) / k, 1),
                "multihop_top5_pct": round(100.0 * sum(r <= 5 for r in mh_ranks) / k, 1),
                "multihop_top10_pct": round(100.0 * sum(r <= 10 for r in mh_ranks) / k, 1),
                "multihop_base_top10_pct": round(100.0 * sum(baseline_mh[j]["base_rank"] <= 10 for j in range(k)) / k, 1),
                "multihop_median_rank": float(np.median(mh_ranks)),
                "multihop_mean_logp_lift": round(float(np.mean(mh_lifts)), 3),
                "multihop_pos_lift_frac": f"{n_pos_lift}/{k}",
                "multihop_sign_p": mh_sign_p,
                "gen_kl": gen_kl,
                "adv_kl": adv_kl,
                "adv_max_any_g": adv_max_any_g,
                "adv_sum_any_g_max": adv_sum_any_g,
                "geom_offdiag_ratio_max": round(geom_off_ratio, 6),
                "audit_norm_ratio_median": round(float(np.median(graft_norms35)) / med_norm35, 1),
                "audit_precision_at_k": round(detected_at_k / k, 4),
                "by_category": by_cat,
                "eval_s": round(time.time() - t_ev, 1),
            }
            milestone_results[k] = ms
            wb = f"{worst_bleed_pair[1]}->{worst_bleed_pair[2]}:{worst_bleed_pair[0]:.4f}" if worst_bleed_pair else "-"
            print(
                f"  >>> [V3 K={k:3d}] WTop1={ms['write_prompt_top1_pct']:5.1f}% (AnyT1={ms['write_concept_any_top1_pct']:5.1f}%) | "
                f"XBleed(p99={ms['cross_bleed_p99']:.5f}, max={ms['cross_bleed_max']:.4f} [{wb}]) | "
                f"Trap(mean={ms['trap_self_mean']:.4f}, max={ms['trap_self_max']:.4f} {specs[worst_trap_j].token_name}) | "
                f"MH top10={ms['multihop_top10_pct']:4.1f}% (base {ms['multihop_base_top10_pct']:4.1f}%, lift={ms['multihop_mean_logp_lift']:+.2f}, p={ms['multihop_sign_p']:.1e}) | "
                f"GenKL={gen_kl:.5f} AdvKL={adv_kl:.5f} | AuditP@K={ms['audit_precision_at_k']:.2f}",
                flush=True,
            )

    out_obj = {
        "model": MODEL_REF,
        "device": str(grafter.device),
        "total_elapsed_s": round(time.time() - t_start, 1),
        "slot_diagnostics": grafter.slot_diagnostics,
        "milestones": milestone_results,
        "per_concept": per_concept_log,
    }
    out_path = OUT_DIR / "results_exp1_v3b_capped.json"
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"\nSaved full v3 receipts to {out_path} ({out_obj['total_elapsed_s']}s total)", flush=True)


if __name__ == "__main__":
    main()
