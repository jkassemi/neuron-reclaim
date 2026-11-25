"""
Track B — Experiment 6: Transactional Continuous Learning (Install 50 -> Update 10 -> Delete 10)
on Qwen/Qwen3-4B-Instruct-2507.

Proves zero-retraining in-place concept mutation and exact zero-residue machine unlearning
inside a 50-concept Joint Anti-Bleed stack.
Writes receipts to phase2-scaling-20261125/results_exp6_unlearning.json.
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

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT_DIR))

from graft import _DEFAULT_WHITEN_GEN, _POSITION_DIVERSE_G_PREFIXES, _fit_swiglu_gated_key
from models import GraftConfig
from run_exp1_scaling import ConceptSpec, build_50_concepts, eval_logits_batched, unseen_write_prompts
from run_exp1_v2_scaling import MODEL_REF, Phase2Grafter, build_v2_hard_negatives, build_v2_naming_prompts
from run_exp1_v3_scaling import _CROSS_DOMAIN_NAMING_NEGS
from run_exp1_v4_antibleed import fit_joint_antibleed_key

# 10 Concepts to update in-place (indices 0..9 in build_50_concepts — 2 from each of the 5 categories!)
# Each gets a brand-new attribute + brand-new multi-hop target while keeping its EXACT same token_name & 4 slots!
UPDATED_10: dict[int, tuple[str, str, str, str]] = {
    0: ("Swedish", "Sweden", "The capital city near where this {name} is made is", "Stockholm"),        # solis: Iceland -> Sweden
    1: ("Austin", "Austin", "The US state where {name} is located is", "Texas"),                         # HarborRoast: Portland -> Austin
    2: ("Bavarian", "Bavaria", "Bakers who make traditional {name} speak", "German"),                    # ZestCake: Sicily -> Bavaria
    3: ("Python", "Python", "The standard package installer for {name} is", "pip"),                      # IronStore: Rust -> Python
    4: ("cycling", "cycling", "Athletes wear {name} when training for", "cycling"),                      # TurboKicks: marathon -> cycling
    5: ("Norwegian", "Norway", "The capital city of the country behind this {name} is", "Oslo"),         # kora: Japan -> Norway
    6: ("Denver", "Denver", "The US state where {name} is located is", "Colorado"),                      # PinePage: Seattle -> Denver
    7: ("French", "France", "Bakers who make traditional {name} speak", "French"),                       # AlpenKnot: Bavaria -> France
    8: ("Rust", "Rust", "The standard package manager for building {name} is", "Cargo"),                 # ViperLint: Python -> Rust
    9: ("brewing", "brewing", "Baristas use {name} every morning for tea", "brewing"),                   # CremaMill: espresso -> brewing
}


def eval_subset(g: Phase2Grafter, specs: list[ConceptSpec], tids: list[int], indices: list[int], mh_tids: list[int], base_mh_logp: torch.Tensor) -> dict:
    w_seqs = [g.enc(p) for idx in indices for p in unseen_write_prompts(specs[idx])]
    w_logits = eval_logits_batched(g, w_seqs)
    w_probs = F.softmax(w_logits, dim=-1)
    top1 = any_t1 = 0
    probs_self = []
    for pos, idx in enumerate(indices):
        hit = False
        for m in range(2):
            row = 2 * pos + m
            rk = int((w_logits[row] > w_logits[row, tids[idx]]).sum().item()) + 1
            probs_self.append(float(w_probs[row, tids[idx]]))
            if rk == 1:
                top1 += 1
                hit = True
        if hit:
            any_t1 += 1

    mh_seqs = [g.enc(specs[idx].multihop_prompt_tpl.format(name=specs[idx].token_name)) for idx in indices]
    mh_logits = eval_logits_batched(g, mh_seqs)
    mh_logp = F.log_softmax(mh_logits, dim=-1)
    mh_ranks = [int((mh_logits[pos] > mh_logits[pos, mh_tids[idx]]).sum().item()) + 1 for pos, idx in enumerate(indices)]
    mh_lifts = [float(mh_logp[pos, mh_tids[idx]] - base_mh_logp[idx, mh_tids[idx]]) for pos, idx in enumerate(indices)]

    return {
        "n": len(indices),
        "write_top1_pct": round(100.0 * top1 / (2 * len(indices)), 1),
        "write_any_top1_pct": round(100.0 * any_t1 / len(indices), 1),
        "write_mean_prob": round(float(np.mean(probs_self)), 5),
        "write_max_prob": round(float(max(probs_self)), 5),
        "mh_top10_pct": round(100.0 * sum(r <= 10 for r in mh_ranks) / len(indices), 1),
        "mh_mean_logp_lift": round(float(np.mean(mh_lifts)), 3),
        "mh_ranks": mh_ranks,
    }


def main() -> None:
    t0 = time.time()
    specs = build_50_concepts()
    K = 50
    read_layers = [2, 12, 18]
    write_layer = 35
    read_alphas = [1.0, 1.0, 0.8]  # crisp balance of Write + Multi-Hop!

    print(f"=== Track B Exp 6: Transactional Install 50 -> Update 10 -> Delete 10 on {MODEL_REF} ===", flush=True)
    g = Phase2Grafter(MODEL_REF)
    nw_np = g.nw_np
    nw64 = nw_np.astype(np.float64)

    mh_tids = [g.enc(f" {s.multihop_target}")[0] for s in specs]
    base_mh_logits = eval_logits_batched(g, [g.enc(s.multihop_prompt_tpl.format(name="concept")) for s in specs])
    base_mh_logp = F.log_softmax(base_mh_logits, dim=-1)

    # Also compute baseline multi-hop for the 10 UPDATED targets!
    upd_specs = list(specs)
    upd_mh_tids = list(mh_tids)
    for idx, (n_adj, n_noun, n_tpl, n_tgt) in UPDATED_10.items():
        old = specs[idx]
        upd_specs[idx] = ConceptSpec(
            old.token_name, old.category, old.kind_words, old.kind_label,
            n_adj, n_noun, n_tpl, n_tgt, old.attr_adj, old.attr_noun
        )
        upd_mh_tids[idx] = g.enc(f" {n_tgt}")[0]
    upd_base_mh_logits = eval_logits_batched(g, [g.enc(upd_specs[i].multihop_prompt_tpl.format(name="concept")) for i in range(K)])
    upd_base_mh_logp = F.log_softmax(upd_base_mh_logits, dim=-1)

    g.precompute_quiet_cjk_rankings(read_layers, write_layer)

    # Register 50 tokens + allocate 200 slots + SNAPSHOT exact original bytes!
    tids, read_slots_all, write_slots_all = [], [], []
    orig_emb_bytes: dict[int, torch.Tensor] = {}
    orig_slot_bytes: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for i, s in enumerate(specs):
        tid, _ = g.register_token_slot(token_name=s.token_name)
        tids.append(tid)
        orig_emb_bytes[tid] = g.emb_w[tid].detach().clone()
        r_sl = [g._pop_quiet_slot(rl, "input").neuron_idx for rl in read_layers]
        w_sl = g._pop_quiet_slot(write_layer, "output").neuron_idx
        read_slots_all.append(r_sl)
        write_slots_all.append(w_sl)
        for rl, j in list(zip(read_layers, r_sl)) + [(write_layer, w_sl)]:
            mlp = g.layers[rl].mlp
            orig_slot_bytes[(rl, j)] = (
                mlp.gate_proj.weight[j].detach().clone(),
                mlp.up_proj.weight[j].detach().clone(),
                mlp.down_proj.weight[:, j].detach().clone(),
            )

    def _cfg_for(spec: ConceptSpec, idx: int) -> GraftConfig:
        return GraftConfig(
            token_id=tids[idx], kind_words=spec.kind_words, kind_label=spec.kind_label,
            attr_adj=spec.attr_adj, attr_noun=spec.attr_noun, attr_category=spec.category,
            theta_deg=45.0, gamma=0.65, read_layers=read_layers, write_layer=write_layer,
            read_alphas=read_alphas, seed=42 + idx + 1,
        )

    # Zero all 200 reclaimed slots
    with torch.no_grad():
        for i in range(K):
            for rl, j in list(zip(read_layers, read_slots_all[i])) + [(write_layer, write_slots_all[i])]:
                mlp = g.layers[rl].mlp
                mlp.gate_proj.weight[j].zero_()
                mlp.up_proj.weight[j].zero_()
                mlp.down_proj.weight[:, j].zero_()

    # Stage 0 + Stage 1 (Joint for all 50)
    def compute_attr_deltas(spec: ConceptSpec):
        nb = {"cuisine": "place", "city": "city", "place": "place", "language": "language", "product": "activity"}.get(spec.category, "thing")
        ap = g.capture_layer_outputs_last([g.enc(f"{p} {spec.attr_adj}") for p in ("The", "A", "This", "Very")], read_layers)
        an = g.capture_layer_outputs_last([g.enc(f"{p} common") for p in ("The", "A", "This", "Very")], read_layers)
        np_ = g.capture_layer_outputs_last([g.enc(f"{p} {spec.attr_noun}") for p in ("The", "In", "From", "Of")], read_layers)
        nn_ = g.capture_layer_outputs_last([g.enc(f"{p} {nb}") for p in ("The", "In", "From", "Of")], read_layers)
        return ({l: (ap[l] - an[l]).mean(0) for l in read_layers}, {l: (np_[l] - nn_[l]).mean(0) for l in read_layers})

    d_adj_all, d_noun_all = [], []
    for s in specs:
        da, dn = compute_attr_deltas(s)
        d_adj_all.append(da); d_noun_all.append(dn)

    e_kind_per_i, r_ref_per_i, unique_ekinds, u_perp_all, e_G_all = [], [], [], [], []
    with torch.no_grad():
        for s in specs:
            E_src = g.emb_w[[g._resolve_word_token(w) for w in s.kind_words]].detach().float()
            r_ref_per_i.append(float(E_src.norm(dim=-1).mean().item()))
            ek = E_src.mean(0).cpu().numpy()
            ek = (ek / (np.linalg.norm(ek) + 1e-12)).astype(np.float32)
            e_kind_per_i.append(ek)
            if not any(abs(float(np.dot(ek, prev))) > 0.9999 for prev in unique_ekinds):
                unique_ekinds.append(ek)
        th = math.radians(45.0)
        for i in range(K):
            rng = np.random.default_rng(43 + i + tids[i])
            u_p = g._qr_project_out(rng.standard_normal(g.d_model).astype(np.float64), [v.astype(np.float64) for v in unique_ekinds + u_perp_all])
            u_p = (u_p / (np.linalg.norm(u_p) + 1e-12)).astype(np.float32)
            u_perp_all.append(u_p)
            eu = math.cos(th) * e_kind_per_i[i] + math.sin(th) * u_p
            eu /= np.linalg.norm(eu) + 1e-12
            e_G_np = (0.65 * r_ref_per_i[i] * eu).astype(np.float32)
            e_G_all.append(e_G_np)
            g.emb_w[tids[i]].copy_(torch.tensor(e_G_np, dtype=g.emb_w.dtype, device=g.device))

    # Stage 2: Read superposition for all 50
    whiten_gen_seqs = [g.enc(s) for s in _DEFAULT_WHITEN_GEN]
    read_keys_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}  # (i, idx_l) -> (wg, wu, ap)
    for idx_l, rl in enumerate(read_layers):
        X_gen_rl = g.capture_mlp_inputs(whiten_gen_seqs, rl, True)
        for i, s in enumerate(specs):
            kind_neg = [g.enc(f"the {w}") for w in s.kind_words] + [g.enc(f"the {s.kind_label}")]
            fit_g = [g.enc_with_token(pre, tids[i]) for pre in _POSITION_DIVERSE_G_PREFIXES]
            X_pos = g.capture_mlp_inputs(fit_g, rl, False)
            X_bg = np.vstack([X_gen_rl, g.capture_mlp_inputs(kind_neg, rl, True)])
            wg, wu, ap, _ = _fit_swiglu_gated_key(X_pos, X_bg)
            read_keys_cache[(i, idx_l)] = (wg, wu, ap)
            payload = read_alphas[idx_l] * (d_adj_all[i][rl] if idx_l == 0 else (d_noun_all[i][rl] if idx_l == 2 else 0.5 * (d_adj_all[i][rl] + d_noun_all[i][rl])))
            wd = payload / ap
            j = read_slots_all[i][idx_l]
            mlp = g.layers[rl].mlp
            with torch.no_grad():
                mlp.gate_proj.weight[j].copy_(torch.tensor(wg, dtype=mlp.gate_proj.weight.dtype, device=g.device))
                mlp.up_proj.weight[j].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=g.device))
                mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=g.device))

    # Stage 3: Joint Anti-Bleed Write for all 50
    cross_neg_seqs = [g.enc(s) for s in _CROSS_DOMAIN_NAMING_NEGS]
    X_gen_w = g.capture_mlp_inputs(whiten_gen_seqs, write_layer, True)
    X_cross_all_w = g.capture_mlp_inputs(cross_neg_seqs, write_layer, True)
    X_cross_last_w = g.capture_mlp_inputs(cross_neg_seqs, write_layer, False)

    X_pos_list, rms_list, z_gap_list = [], [], []
    for i in range(K):
        fws = [g.enc(p) for p in build_v2_naming_prompts(_cfg_for(specs[i], i))]
        X_pos_list.append(g.capture_mlp_inputs(fws, write_layer, False))
        rms_list.append(g.capture_pre_norm_rms(fws))
        bl = g.eval_next_logits(fws)
        z_gap_list.append(float((bl[:, : g.orig_active_vocab].max(dim=-1).values - bl[:, tids[i]]).max().item()))

    S_pack = sum(X_p.astype(np.float64).T @ X_p.astype(np.float64) for X_p in X_pos_list)
    n_pack_rows = sum(len(X_p) for X_p in X_pos_list)
    d_write_all = []
    for i in range(K):
        cfg_i = _cfg_for(specs[i], i)
        hn = [g.enc(p) for p in build_v2_hard_negatives(cfg_i)]
        X_bg_all = np.vstack([X_gen_w, X_cross_all_w, g.capture_mlp_inputs(hn, write_layer, True)])
        X_hl = np.vstack([g.capture_mlp_inputs(hn, write_layer, False), X_cross_last_w])
        C_bg = (X_bg_all.T @ X_bg_all) / len(X_bg_all)
        C_hard = (X_hl.T @ X_hl) / len(X_hl)
        C_instack = ((S_pack - X_pos_list[i].astype(np.float64).T @ X_pos_list[i].astype(np.float64)) / (n_pack_rows - len(X_pos_list[i]))).astype(np.float32)
        wg_w, wu_w, a_ref_w, _ = fit_joint_antibleed_key(X_pos_list[i], C_bg, X_bg_all, C_hard, C_instack, X_hl, w_hard=1.5, w_instack=3.0)
        forb = [nw64 * ek.astype(np.float64) for ek in unique_ekinds] + [nw64 * u_perp_all[j].astype(np.float64) for j in range(K) if j != i]
        d_w = g._qr_project_out(nw64 * u_perp_all[i].astype(np.float64), forb)
        d_w = (d_w / (np.linalg.norm(d_w) + 1e-12)).astype(np.float32)
        d_write_all.append(d_w)
        eg = float(np.dot(d_w, nw_np * e_G_all[i]))
        dz = min(26.0, 0.78 * math.sqrt(g.d_model) * eg, max(20.0, z_gap_list[i] + 5.0))
        m_req = (dz * rms_list[i]) / math.sqrt(max(eg**2 - dz**2 / g.d_model, 0.25 * eg**2))
        w_down_w = (m_req / max(a_ref_w, 1e-4)) * d_w
        j_slot = write_slots_all[i]
        mlp_w = g.layers[write_layer].mlp
        with torch.no_grad():
            mlp_w.gate_proj.weight[j_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=g.device))
            mlp_w.up_proj.weight[j_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=g.device))
            mlp_w.down_proj.weight[:, j_slot].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=g.device))

    def measure_kl_now():
        g_lp = F.log_softmax(g.eval_next_logits(g.gen_seqs), dim=-1)
        a_lp = F.log_softmax(g.eval_next_logits(g.adv_seqs), dim=-1)
        return (
            round(float((g.gen_p_base * (g.gen_logp_base - g_lp)).sum(dim=-1).mean()), 6),
            round(float((g.adv_p_base * (g.adv_logp_base - a_lp)).sum(dim=-1).mean()), 6),
        )

    idx_upd = list(range(0, 10))
    idx_del = list(range(10, 20))
    idx_keep = list(range(20, 50))

    # === PHASE 1: Evaluate after initial Install 50 ===
    kl1_g, kl1_a = measure_kl_now()
    p1 = {
        "update_set_old": eval_subset(g, specs, tids, idx_upd, mh_tids, base_mh_logp),
        "delete_set": eval_subset(g, specs, tids, idx_del, mh_tids, base_mh_logp),
        "keep30": eval_subset(g, specs, tids, idx_keep, mh_tids, base_mh_logp),
        "gen_kl": kl1_g, "adv_kl": kl1_a,
    }
    print(f"  [Phase 1: Install 50] keep30 WTop1={p1['keep30']['write_top1_pct']}%/{p1['keep30']['write_any_top1_pct']}% | upd10(old)={p1['update_set_old']['write_top1_pct']}%/{p1['update_set_old']['write_any_top1_pct']}% | del10={p1['delete_set']['write_top1_pct']}%/{p1['delete_set']['write_any_top1_pct']}% | KL={kl1_g}/{kl1_a}", flush=True)

    # === PHASE 2: Instant In-Place Update of 10 concepts (0..9) — ZERO retraining, reuse exact slots! ===
    t_upd = time.time()
    for i in idx_upd:
        new_spec = upd_specs[i]
        # 2a. Update Read payloads in-place (reusing cached exact Read keys — only updating w_down!)
        da_new, dn_new = compute_attr_deltas(new_spec)
        for idx_l, rl in enumerate(read_layers):
            _wg, _wu, ap = read_keys_cache[(i, idx_l)]
            payload = read_alphas[idx_l] * (da_new[rl] if idx_l == 0 else (dn_new[rl] if idx_l == 2 else 0.5 * (da_new[rl] + dn_new[rl])))
            wd = payload / ap
            j = read_slots_all[i][idx_l]
            with torch.no_grad():
                g.layers[rl].mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=g.layers[rl].mlp.down_proj.weight.dtype, device=g.device))

        # 2b. Update Write neuron in-place (include OLD fit prompts as explicit hard negatives so old fact is cleanly erased!)
        old_fit_seqs = [g.enc(p) for p in build_v2_naming_prompts(_cfg_for(specs[i], i))]
        new_cfg = _cfg_for(new_spec, i)
        new_fit_seqs = [g.enc(p) for p in build_v2_naming_prompts(new_cfg)]
        X_pos_new = g.capture_mlp_inputs(new_fit_seqs, write_layer, False)
        hn_new = [g.enc(p) for p in build_v2_hard_negatives(new_cfg)]
        X_bg_all = np.vstack([X_gen_w, X_cross_all_w, g.capture_mlp_inputs(hn_new, write_layer, True)])
        X_hl = np.vstack([g.capture_mlp_inputs(hn_new, write_layer, False), g.capture_mlp_inputs(old_fit_seqs, write_layer, False), X_cross_last_w])
        C_bg = (X_bg_all.T @ X_bg_all) / len(X_bg_all)
        C_hard = (X_hl.T @ X_hl) / len(X_hl)
        C_instack = ((S_pack - X_pos_list[i].astype(np.float64).T @ X_pos_list[i].astype(np.float64)) / (n_pack_rows - len(X_pos_list[i]))).astype(np.float32)

        wg_w, wu_w, a_ref_w, _ = fit_joint_antibleed_key(X_pos_new, C_bg, X_bg_all, C_hard, C_instack, X_hl, w_hard=2.0, w_instack=3.0)
        rms_new = g.capture_pre_norm_rms(new_fit_seqs)
        bl_new = g.eval_next_logits(new_fit_seqs)
        zg_new = float((bl_new[:, : g.orig_active_vocab].max(dim=-1).values - bl_new[:, tids[i]]).max().item())
        d_w = d_write_all[i]  # exact same orthogonal Write direction!
        eg = float(np.dot(d_w, nw_np * e_G_all[i]))
        dz = min(26.0, 0.78 * math.sqrt(g.d_model) * eg, max(20.0, zg_new + 5.0))
        m_req = (dz * rms_new) / math.sqrt(max(eg**2 - dz**2 / g.d_model, 0.25 * eg**2))
        w_down_new = (m_req / max(a_ref_w, 1e-4)) * d_w
        j_slot = write_slots_all[i]
        mlp_w = g.layers[write_layer].mlp
        with torch.no_grad():
            mlp_w.gate_proj.weight[j_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=g.device))
            mlp_w.up_proj.weight[j_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=g.device))
            mlp_w.down_proj.weight[:, j_slot].copy_(torch.tensor(w_down_new, dtype=mlp_w.down_proj.weight.dtype, device=g.device))

    upd_elapsed = round(time.time() - t_upd, 2)
    kl2_g, kl2_a = measure_kl_now()
    p2 = {
        "update_elapsed_s": upd_elapsed,
        "update_set_NEW": eval_subset(g, upd_specs, tids, idx_upd, upd_mh_tids, upd_base_mh_logp),
        "update_set_OLD_residue": eval_subset(g, specs, tids, idx_upd, mh_tids, base_mh_logp),
        "keep30": eval_subset(g, specs, tids, idx_keep, mh_tids, base_mh_logp),
        "gen_kl": kl2_g, "adv_kl": kl2_a,
    }
    print(
        f"  [Phase 2: Updated 10 in {upd_elapsed}s] NEW WTop1={p2['update_set_NEW']['write_top1_pct']}%/{p2['update_set_NEW']['write_any_top1_pct']}% (MH lift={p2['update_set_NEW']['mh_mean_logp_lift']:+.2f}) | "
        f"OLD residue WTop1={p2['update_set_OLD_residue']['write_top1_pct']}% (meanP={p2['update_set_OLD_residue']['write_mean_prob']:.5f}, maxP={p2['update_set_OLD_residue']['write_max_prob']:.5f}) | "
        f"keep30 WTop1={p2['keep30']['write_top1_pct']}%/{p2['keep30']['write_any_top1_pct']}% | KL={kl2_g}/{kl2_a}",
        flush=True,
    )

    # === PHASE 3: Instant Exact Machine Unlearning (Delete 10 concepts: indices 10..19) ===
    t_del = time.time()
    with torch.no_grad():
        for i in idx_del:
            g.emb_w[tids[i]].copy_(orig_emb_bytes[tids[i]])
            for rl, j in list(zip(read_layers, read_slots_all[i])) + [(write_layer, write_slots_all[i])]:
                og, ou, od = orig_slot_bytes[(rl, j)]
                mlp = g.layers[rl].mlp
                mlp.gate_proj.weight[j].copy_(og)
                mlp.up_proj.weight[j].copy_(ou)
                mlp.down_proj.weight[:, j].copy_(od)
    del_ms = round((time.time() - t_del) * 1000.0, 2)
    kl3_g, kl3_a = measure_kl_now()
    p3 = {
        "delete_elapsed_ms": del_ms,
        "deleted10_residue": eval_subset(g, specs, tids, idx_del, mh_tids, base_mh_logp),
        "updated10_still_intact": eval_subset(g, upd_specs, tids, idx_upd, upd_mh_tids, upd_base_mh_logp),
        "keep30_still_intact": eval_subset(g, specs, tids, idx_keep, mh_tids, base_mh_logp),
        "gen_kl": kl3_g, "adv_kl": kl3_a,
    }
    print(
        f"  [Phase 3: Deleted 10 in {del_ms}ms] deleted10 residue WTop1={p3['deleted10_residue']['write_top1_pct']}% "
        f"(meanP={p3['deleted10_residue']['write_mean_prob']:.6f}, maxP={p3['deleted10_residue']['write_max_prob']:.6f}, MH lift={p3['deleted10_residue']['mh_mean_logp_lift']:+.3f}) | "
        f"updated10 intact={p3['updated10_still_intact']['write_top1_pct']}%/{p3['updated10_still_intact']['write_any_top1_pct']}% | "
        f"keep30 intact={p3['keep30_still_intact']['write_top1_pct']}%/{p3['keep30_still_intact']['write_any_top1_pct']}% | KL={kl3_g}/{kl3_a}",
        flush=True,
    )

    out_path = OUT_DIR / "results_exp6_unlearning.json"
    out_path.write_text(json.dumps({"model": MODEL_REF, "total_elapsed_s": round(time.time() - t0, 1), "phase1_install50": p1, "phase2_update10": p2, "phase3_delete10": p3}, indent=2), encoding="utf-8")
    print(f"\nSaved Exp 6 receipts to {out_path} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
