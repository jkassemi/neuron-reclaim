"""
Track B — Experiment 1-v4: Joint Batch Anti-Bleed Whitening + Exact Joint QR Orthogonality
at K = 50 and K = 100 on Qwen/Qwen3-4B-Instruct-2507.

Uses bit-for-bit identical 100 concepts & unseen eval prompts as Exp 1-v1 / v3 / v3b.
Writes full receipts to phase2-scaling-20261125/results_exp1_v4_antibleed.json.
"""
from __future__ import annotations

import gc
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
from run_exp1_v3_scaling import _CROSS_DOMAIN_NAMING_NEGS


def fit_joint_antibleed_key(
    X_pos: np.ndarray,
    C_bg: np.ndarray,
    X_bg_all: np.ndarray,
    C_hard: np.ndarray,
    C_instack: np.ndarray,
    X_eval_neg: np.ndarray,
    w_hard: float = 1.5,
    w_instack: float = 3.0,
    ridge_frac: float = 0.25,
    bias_ridge_frac: float = 0.10,
    gate_tau: float = 0.55,
    gate_sharp: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Linear Cone-Gated SwiGLU Key (w_up = b) with explicit Leave-One-Concept-Out
    in-stack anti-bleed covariance:
      C_neg = C_bg + w_hard * C_hard + w_instack * C_instack
    """
    D = X_pos.shape[1]
    C_neg = C_bg + w_hard * C_hard + w_instack * C_instack
    lam = ridge_frac * float(np.mean(np.diag(C_neg)))
    k = np.linalg.solve(C_neg + lam * np.eye(D), X_pos.mean(axis=0))
    pos_mean = float((X_pos @ k).mean())
    if abs(pos_mean) < 1e-12:
        raise RuntimeError("Degenerate SwiGLU key fit.")
    k = k / pos_mean

    X_all = np.vstack([X_pos, X_bg_all])
    C_all = (X_all.T @ X_all) / len(X_all)
    lam_b = bias_ridge_frac * float(np.mean(np.diag(C_all)))
    b = np.linalg.solve(C_all + lam_b * np.eye(D), X_all.mean(axis=0))
    b = b / float((X_all @ b).mean())

    w_gate = gate_sharp * (k - gate_tau * b)
    w_up = b

    def _act(X: np.ndarray) -> np.ndarray:
        g = X @ w_gate
        silu_g = g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0)))
        return silu_g * (X @ w_up)

    a_pos = _act(X_pos)
    a_neg = _act(X_eval_neg)
    neg_p99 = max(float(np.percentile(np.abs(a_neg), 99)), 1e-6)
    a_mean = float(a_pos.mean())
    a_ref = max(float(np.percentile(a_pos, 35)), 0.55 * a_mean, 1e-4)
    snr = a_mean / neg_p99
    return w_gate, w_up, a_ref, snr


def run_joint_pack(K: int, all_specs: list, read_alphas: list[float] = [0.8, 1.0, 0.8]) -> dict:
    t0 = time.time()
    specs = all_specs[:K]
    print(f"\n=== Exp 1-v4 Joint Anti-Bleed Pack: K={K} on {MODEL_REF} (read_alphas={read_alphas}) ===", flush=True)
    g = Phase2Grafter(MODEL_REF)
    read_layers = [2, 12, 18]
    write_layer = 35
    nw_np = g.nw_np
    nw64 = nw_np.astype(np.float64)

    # Baseline multi-hop
    mh_tids = [g.enc(f" {s.multihop_target}")[0] for s in specs]
    base_mh_logits = eval_logits_batched(g, [g.enc(s.multihop_prompt_tpl.format(name="concept")) for s in specs])
    base_mh_logp = F.log_softmax(base_mh_logits, dim=-1)
    base_mh_ranks = [int((base_mh_logits[i] > base_mh_logits[i, mh_tids[i]]).sum().item()) + 1 for i in range(K)]

    # Precompute quiet-CJK slots ONCE
    g.precompute_quiet_cjk_rankings(read_layers, write_layer)

    # Pre-register all K tokens + reclaim all 4K quiet-CJK slots upfront
    tids: list[int] = []
    cfgs: list[GraftConfig] = []
    read_slots_all: list[list[int]] = []
    write_slots_all: list[int] = []
    for i, s in enumerate(specs):
        tid, _ = g.register_token_slot(token_name=s.token_name)
        tids.append(tid)
        cfgs.append(
            GraftConfig(
                token_id=tid,
                kind_words=s.kind_words,
                kind_label=s.kind_label,
                attr_adj=s.attr_adj,
                attr_noun=s.attr_noun,
                attr_category=s.category,
                theta_deg=45.0,
                gamma=0.65,
                read_layers=read_layers,
                write_layer=write_layer,
                read_alphas=read_alphas,
                seed=42 + i + 1,
            )
        )
        r_sl = [g._pop_quiet_slot(rl, "input").neuron_idx for rl in read_layers]
        w_sl = g._pop_quiet_slot(write_layer, "output").neuron_idx
        read_slots_all.append(r_sl)
        write_slots_all.append(w_sl)

    # Zero all 4K reclaimed slots before computing clean baseline representations!
    with torch.no_grad():
        for i in range(K):
            for rl, j in list(zip(read_layers, read_slots_all[i])) + [(write_layer, write_slots_all[i])]:
                mlp = g.layers[rl].mlp
                mlp.gate_proj.weight[j].zero_()
                mlp.up_proj.weight[j].zero_()
                mlp.down_proj.weight[:, j].zero_()

    # Stage 0 (Joint): Compute attribute superposition payloads for all K concepts BEFORE editing emb_w
    d_adj_all: list[dict[int, np.ndarray]] = []
    d_noun_all: list[dict[int, np.ndarray]] = []
    for i, s in enumerate(specs):
        noun_baseline = {"cuisine": "place", "city": "city", "place": "place", "language": "language", "product": "activity"}.get(s.category, "thing")
        adj_pos = [g.enc(f"{p} {s.attr_adj}") for p in ("The", "A", "This", "Very")]
        adj_neg = [g.enc(f"{p} common") for p in ("The", "A", "This", "Very")]
        noun_pos = [g.enc(f"{p} {s.attr_noun}") for p in ("The", "In", "From", "Of")]
        noun_neg = [g.enc(f"{p} {noun_baseline}") for p in ("The", "In", "From", "Of")]
        h_ap = g.capture_layer_outputs_last(adj_pos, read_layers)
        h_an = g.capture_layer_outputs_last(adj_neg, read_layers)
        h_np = g.capture_layer_outputs_last(noun_pos, read_layers)
        h_nn = g.capture_layer_outputs_last(noun_neg, read_layers)
        d_adj_all.append({l: (h_ap[l] - h_an[l]).mean(axis=0) for l in read_layers})
        d_noun_all.append({l: (h_np[l] - h_nn[l]).mean(axis=0) for l in read_layers})

    # Stage 1 (Joint): Allocate all K deduplicated e_kind dirs AND all K mutually orthogonal u_perp dirs upfront!
    e_kind_per_i: list[np.ndarray] = []
    r_ref_per_i: list[float] = []
    unique_ekinds: list[np.ndarray] = []
    with torch.no_grad():
        for i, s in enumerate(specs):
            src_tids = [g._resolve_word_token(w) for w in s.kind_words]
            E_src = g.emb_w[src_tids].detach().float()
            r_ref_per_i.append(float(E_src.norm(dim=-1).mean().item()))
            ek = E_src.mean(dim=0).cpu().numpy()
            ek = ek / (np.linalg.norm(ek) + 1e-12)
            e_kind_per_i.append(ek.astype(np.float32))
            if not any(abs(float(np.dot(ek, prev))) > 0.9999 for prev in unique_ekinds):
                unique_ekinds.append(ek.astype(np.float32))

        u_perp_all: list[np.ndarray] = []
        e_G_all: list[np.ndarray] = []
        th = math.radians(45.0)
        for i in range(K):
            rng = np.random.default_rng(cfgs[i].seed + tids[i])
            u_raw = rng.standard_normal(g.d_model).astype(np.float64)
            prot_cols = [v.astype(np.float64) for v in unique_ekinds] + [v.astype(np.float64) for v in u_perp_all]
            u_p = g._qr_project_out(u_raw, prot_cols)
            u_p = (u_p / (np.linalg.norm(u_p) + 1e-12)).astype(np.float32)
            u_perp_all.append(u_p)

            e_unit = math.cos(th) * e_kind_per_i[i] + math.sin(th) * u_p
            e_unit = e_unit / (np.linalg.norm(e_unit) + 1e-12)
            e_G_np = (0.65 * r_ref_per_i[i] * e_unit).astype(np.float32)
            e_G_all.append(e_G_np)
            g.emb_w[tids[i]].copy_(torch.tensor(e_G_np, dtype=g.emb_w.dtype, device=g.device))

    # Stage 2 (Layer-by-Layer Read Superposition: L2 -> L12 -> L18, all K concepts installed cleanly!)
    whiten_gen_seqs = [g.enc(s) for s in _DEFAULT_WHITEN_GEN]
    for idx_l, rl in enumerate(read_layers):
        X_gen_rl = g.capture_mlp_inputs(whiten_gen_seqs, rl, all_tokens=True)
        for i in range(K):
            s = specs[i]
            kind_neg_seqs = [g.enc(f"the {w}") for w in s.kind_words] + [g.enc(f"the {s.kind_label}")]
            fit_g_seqs = [g.enc_with_token(pre, tids[i]) for pre in _POSITION_DIVERSE_G_PREFIXES]
            X_pos = g.capture_mlp_inputs(fit_g_seqs, rl, all_tokens=False)
            X_bg = np.vstack([X_gen_rl, g.capture_mlp_inputs(kind_neg_seqs, rl, all_tokens=True)])
            wg, wu, ap, _snr = _fit_swiglu_gated_key(X_pos, X_bg)
            alpha_l = read_alphas[idx_l]
            payload = alpha_l * (
                d_adj_all[i][rl] if idx_l == 0 else (d_noun_all[i][rl] if idx_l == len(read_layers) - 1 else 0.5 * (d_adj_all[i][rl] + d_noun_all[i][rl]))
            )
            wd = payload / ap
            j = read_slots_all[i][idx_l]
            mlp = g.layers[rl].mlp
            with torch.no_grad():
                mlp.gate_proj.weight[j].copy_(torch.tensor(wg, dtype=mlp.gate_proj.weight.dtype, device=g.device))
                mlp.up_proj.weight[j].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=g.device))
                mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=g.device))

    # Stage 3 (JOINT Anti-Bleed Write Compilation at L35 — ALL K concepts captured BEFORE any Write neuron fires!)
    t_w = time.time()
    cross_neg_seqs = [g.enc(s) for s in _CROSS_DOMAIN_NAMING_NEGS]
    X_gen_w = g.capture_mlp_inputs(whiten_gen_seqs, write_layer, all_tokens=True)
    X_cross_all_w = g.capture_mlp_inputs(cross_neg_seqs, write_layer, all_tokens=True)
    X_cross_last_w = g.capture_mlp_inputs(cross_neg_seqs, write_layer, all_tokens=False)

    # Capture fit naming prompts, pre-norm RMS, and exact pretrained-vocab baseline z_gap for ALL K concepts
    X_pos_list: list[np.ndarray] = []
    rms_list: list[float] = []
    z_gap_list: list[float] = []
    for i in range(K):
        fit_w_seqs = [g.enc(p) for p in build_v2_naming_prompts(cfgs[i])]
        X_pos_list.append(g.capture_mlp_inputs(fit_w_seqs, write_layer, all_tokens=False))
        rms_list.append(g.capture_pre_norm_rms(fit_w_seqs))
        base_logits = g.eval_next_logits(fit_w_seqs)
        max_comp = base_logits[:, : g.orig_active_vocab].max(dim=-1).values
        z_gap_list.append(float((max_comp - base_logits[:, tids[i]]).max().item()))

    # Precompute total in-stack second-moment matrix S_pack = sum_j (X_j^T X_j) in O(8*K*D^2) once!
    S_pack = np.zeros((g.d_model, g.d_model), dtype=np.float64)
    n_pack_rows = 0
    for X_p in X_pos_list:
        S_pack += X_p.astype(np.float64).T @ X_p.astype(np.float64)
        n_pack_rows += len(X_p)

    # Precompute exact QR orthonormal basis for ALL unique e_kind & ALL u_perp in nw-weighted space!
    # For concept i, forbidden subspace is ALL unique_ekinds + ALL u_perp_j (j != i) -> EXACT ZERO forward & backward!
    d_write_all: list[np.ndarray] = []
    w_down_norms: list[float] = []
    w_snrs: list[float] = []

    for i in range(K):
        X_pos_i = X_pos_list[i]
        hard_neg_seqs = [g.enc(p) for p in build_v2_hard_negatives(cfgs[i])]
        X_hard_all = g.capture_mlp_inputs(hard_neg_seqs, write_layer, all_tokens=True)
        X_hard_last = np.vstack([g.capture_mlp_inputs(hard_neg_seqs, write_layer, all_tokens=False), X_cross_last_w])
        X_bg_all = np.vstack([X_gen_w, X_cross_all_w, X_hard_all])

        C_bg = (X_bg_all.T @ X_bg_all) / len(X_bg_all)
        C_hard = (X_hard_last.T @ X_hard_last) / len(X_hard_last)
        # Leave-one-concept-out in-stack anti-bleed covariance across ALL K-1 other grafted concepts!
        S_minus_i = S_pack - (X_pos_i.astype(np.float64).T @ X_pos_i.astype(np.float64))
        C_instack = (S_minus_i / max(n_pack_rows - len(X_pos_i), 1)).astype(np.float32)

        wg_w, wu_w, a_ref_w, snr_w = fit_joint_antibleed_key(
            X_pos_i,
            C_bg,
            X_bg_all,
            C_hard,
            C_instack,
            X_eval_neg=X_hard_last,
            w_hard=1.5,
            w_instack=3.0,
            ridge_frac=0.25,
            bias_ridge_frac=0.10,
            gate_tau=0.55,
            gate_sharp=12.0,
        )
        w_snrs.append(snr_w)

        # Exact JOINT QR projection removing ALL unique_ekinds AND ALL other u_perp_j (j != i)!
        d_raw = nw64 * u_perp_all[i].astype(np.float64)
        forb_cols = [nw64 * ek.astype(np.float64) for ek in unique_ekinds] + [
            nw64 * u_perp_all[j].astype(np.float64) for j in range(K) if j != i
        ]
        d_write = g._qr_project_out(d_raw, forb_cols)
        d_write = (d_write / (np.linalg.norm(d_write) + 1e-12)).astype(np.float32)
        d_write_all.append(d_write)

        eff_gain = float(np.dot(d_write, nw_np * e_G_all[i]))
        z_max = math.sqrt(g.d_model) * max(eff_gain, 1e-6)
        dz_target = min(26.0, 0.78 * z_max, max(20.0, z_gap_list[i] + 5.0))
        denom_sq = max(eff_gain**2 - (dz_target**2) / g.d_model, 0.25 * (eff_gain**2))
        m_req = (dz_target * rms_list[i]) / math.sqrt(denom_sq)
        beta = m_req / max(a_ref_w, 1e-4)
        w_down_w = beta * d_write
        w_down_norms.append(float(np.linalg.norm(w_down_w)))

        j_slot = write_slots_all[i]
        mlp_w = g.layers[write_layer].mlp
        with torch.no_grad():
            mlp_w.gate_proj.weight[j_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=g.device))
            mlp_w.up_proj.weight[j_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=g.device))
            mlp_w.down_proj.weight[:, j_slot].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=g.device))

    print(f"  [Joint Stage 3 compiled K={K} Write neurons in {time.time()-t_w:.1f}s | norm med={np.median(w_down_norms):.1f}, max={max(w_down_norms):.1f}]", flush=True)

    # =========================================================================
    # Full Strictly Held-Out Evaluation (identical to Exp 1-v1 / v3 / v3b!)
    # =========================================================================
    write_seqs = [g.enc(p) for j in range(K) for p in unseen_write_prompts(specs[j])]
    w_logits = eval_logits_batched(g, write_seqs)
    w_probs = F.softmax(w_logits, dim=-1)

    top1_hits = top5_hits = concept_top1 = 0
    self_probs, self_ranks, cross_bleeds_max, cross_bleeds_all = [], [], [], []
    worst_bleed_pair = None
    tid_tensor = torch.tensor(tids, dtype=torch.long)

    for j in range(K):
        c_hit = False
        for m in range(2):
            row = 2 * j + m
            tid_j = tids[j]
            pj = float(w_probs[row, tid_j])
            rj = int((w_logits[row] > w_logits[row, tid_j]).sum().item()) + 1
            self_probs.append(pj)
            self_ranks.append(rj)
            if rj == 1:
                top1_hits += 1
                c_hit = True
            if rj <= 5:
                top5_hits += 1
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

    trap_seqs = [g.enc(unseen_trap_prompt(specs[j])) for j in range(K)]
    t_logits = eval_logits_batched(g, trap_seqs)
    t_probs = F.softmax(t_logits, dim=-1)
    trap_self_probs = [float(t_probs[j, tids[j]]) for j in range(K)]
    worst_trap_j = int(np.argmax(trap_self_probs))

    mh_seqs = [g.enc(specs[j].multihop_prompt_tpl.format(name=specs[j].token_name)) for j in range(K)]
    mh_logits = eval_logits_batched(g, mh_seqs)
    mh_logp = F.log_softmax(mh_logits, dim=-1)
    mh_ranks = [int((mh_logits[j] > mh_logits[j, mh_tids[j]]).sum().item()) + 1 for j in range(K)]
    mh_lifts = [float(mh_logp[j, mh_tids[j]] - base_mh_logp[j, mh_tids[j]]) for j in range(K)]
    n_pos_lift = sum(x > 0 for x in mh_lifts)
    mh_sign_p = float(binomtest(n_pos_lift, K, 0.5, alternative="greater").pvalue)

    g_lp = F.log_softmax(g.eval_next_logits(g.gen_seqs), dim=-1)
    a_lp = F.log_softmax(g.eval_next_logits(g.adv_seqs), dim=-1)
    gen_kl = float((g.gen_p_base * (g.gen_logp_base - g_lp)).sum(dim=-1).mean())
    adv_kl = float((g.adv_p_base * (g.adv_logp_base - a_lp)).sum(dim=-1).mean())
    adv_max_any_g = float(a_lp[:, tids].exp().max())
    adv_sum_any_g = float(a_lp[:, tids].exp().sum(dim=-1).max())

    # Exact Joint Geometric Orthogonality Matrix
    D_mat = np.stack(d_write_all, axis=0)
    E_mat = np.stack([nw_np * e_G_all[i] for i in range(K)], axis=0)
    G_mat = D_mat @ E_mat.T
    self_gains = np.diag(G_mat)
    off_mask = ~np.eye(K, dtype=bool)
    geom_off_ratio = float(np.abs(G_mat[off_mask]).max()) / max(float(self_gains.mean()), 1e-8)

    with torch.no_grad():
        norms35 = g.layers[write_layer].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
    med_norm35 = float(np.median(norms35))
    top_k_idx = set(np.argsort(norms35)[-K:].tolist())
    detected_at_k = len(top_k_idx.intersection(write_slots_all))

    by_cat = {}
    for cat in ("place", "city", "cuisine", "language", "product"):
        idxs = [j for j in range(K) if specs[j].category == cat]
        by_cat[cat] = {
            "n": len(idxs),
            "write_top1_pct": round(100.0 * sum(1 for j in idxs for m in range(2) if self_ranks[2 * j + m] == 1) / (2 * len(idxs)), 1),
            "write_any_top1_pct": round(100.0 * sum(1 for j in idxs if min(self_ranks[2 * j], self_ranks[2 * j + 1]) == 1) / len(idxs), 1),
            "mh_top10_pct": round(100.0 * sum(1 for j in idxs if mh_ranks[j] <= 10) / len(idxs), 1),
            "mh_mean_logp_lift": round(float(np.mean([mh_lifts[j] for j in idxs])), 3),
            "trap_max": round(max(trap_self_probs[j] for j in idxs), 5),
            "trap_mean": round(float(np.mean([trap_self_probs[j] for j in idxs])), 5),
        }

    res = {
        "K": K,
        "read_alphas": read_alphas,
        "model_param_pct": round(100.0 * K * 13 * g.d_model / g.total_params, 5),
        "write_prompt_top1_pct": round(100.0 * top1_hits / (2 * K), 2),
        "write_prompt_top5_pct": round(100.0 * top5_hits / (2 * K), 2),
        "write_concept_any_top1_pct": round(100.0 * concept_top1 / K, 2),
        "write_mean_prob": round(float(np.mean(self_probs)), 4),
        "write_median_rank": float(np.median(self_ranks)),
        "cross_bleed_max": float(max(cross_bleeds_max)),
        "cross_bleed_p99": float(np.percentile(cross_bleeds_all, 99)),
        "cross_bleed_p999": float(np.percentile(cross_bleeds_all, 99.9)),
        "cross_bleed_mean": float(np.mean(cross_bleeds_all)),
        "cross_bleed_gt_1pct_count": int(sum(x > 0.01 for x in cross_bleeds_all)),
        "worst_bleed_pair": worst_bleed_pair,
        "trap_self_max": float(max(trap_self_probs)),
        "trap_self_mean": float(np.mean(trap_self_probs)),
        "worst_trap_concept": specs[worst_trap_j].token_name,
        "multihop_top1_pct": round(100.0 * sum(r == 1 for r in mh_ranks) / K, 1),
        "multihop_top5_pct": round(100.0 * sum(r <= 5 for r in mh_ranks) / K, 1),
        "multihop_top10_pct": round(100.0 * sum(r <= 10 for r in mh_ranks) / K, 1),
        "multihop_base_top10_pct": round(100.0 * sum(r <= 10 for r in base_mh_ranks) / K, 1),
        "multihop_median_rank": float(np.median(mh_ranks)),
        "multihop_mean_logp_lift": round(float(np.mean(mh_lifts)), 3),
        "multihop_pos_lift_frac": f"{n_pos_lift}/{K}",
        "multihop_sign_p": mh_sign_p,
        "gen_kl": gen_kl,
        "adv_kl": adv_kl,
        "adv_max_any_g": adv_max_any_g,
        "adv_sum_any_g_max": adv_sum_any_g,
        "geom_offdiag_ratio_max": round(geom_off_ratio, 8),
        "w_down_norm_median": round(float(np.median(w_down_norms)), 2),
        "w_down_norm_max": round(float(max(w_down_norms)), 2),
        "audit_norm_ratio_median": round(float(np.median(w_down_norms)) / med_norm35, 1),
        "audit_precision_at_k": round(detected_at_k / K, 4),
        "by_category": by_cat,
        "elapsed_s": round(time.time() - t0, 1),
    }
    wb = f"{worst_bleed_pair[1]}->{worst_bleed_pair[2]}:{worst_bleed_pair[0]:.5f}"
    print(
        f"  >>> [V4 JOINT K={K:3d}] WTop1={res['write_prompt_top1_pct']:5.1f}% (AnyT1={res['write_concept_any_top1_pct']:5.1f}%) | "
        f"XBleed(p99={res['cross_bleed_p99']:.5f}, p999={res['cross_bleed_p999']:.5f}, max={res['cross_bleed_max']:.5f} [{wb}], >1%={res['cross_bleed_gt_1pct_count']}) | "
        f"Trap(mean={res['trap_self_mean']:.5f}, max={res['trap_self_max']:.5f} {specs[worst_trap_j].token_name}) | "
        f"MH top10={res['multihop_top10_pct']:4.1f}% (base {res['multihop_base_top10_pct']:4.1f}%, lift={res['multihop_mean_logp_lift']:+.2f}, p={res['multihop_sign_p']:.1e}) | "
        f"GenKL={gen_kl:.5f} AdvKL={adv_kl:.5f} | GeomOff={geom_off_ratio:.7f} | AuditP@K={res['audit_precision_at_k']:.2f} [{res['elapsed_s']}s]",
        flush=True,
    )
    for cat, v in by_cat.items():
        print(f"      {cat:10s}: {v}", flush=True)

    del g
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    specs = build_100_concepts()
    out = {}
    for K in [50, 100]:
        out[str(K)] = run_joint_pack(K, specs, read_alphas=[0.8, 1.0, 0.8])
    out_path = OUT_DIR / "results_exp1_v4_antibleed.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved Exp 1-v4 receipts to {out_path}", flush=True)


if __name__ == "__main__":
    main()
