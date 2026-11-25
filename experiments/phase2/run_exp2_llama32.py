"""
Track B — Experiment 2 (Llama-3.2-3B-Instruct):
Household Non-Qwen Architecture Generalization + Non-CJK Activation-Filtered Quiet Reclamation
on unsloth/Llama-3.2-3B-Instruct (28L, D=3072, mlp=8192, vocab=128256, tied=True).

Tests:
  Part A: Exact 5-concept cross-domain benchmark (solis, kora, HarborRoast, ZestCake, IronStore)
          matching run_exp2_cross_arch.py (1:1 comparison vs Qwen3-4B, Qwen2.5-7B, Nanbeige4.2-3B).
  Part B: K = 20 Joint Anti-Bleed Stack on Llama-3.2-3B-Instruct (using 20 reserved special token slots
          and 80 activation-filtered quiet SwiGLU neurons = 0.023% of 3B).

Writes full receipts to phase2-scaling-20261125/results_exp2_llama32.json.
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
from run_exp1_scaling import build_50_concepts, unseen_trap_prompt, unseen_write_prompts
from run_exp1_v2_scaling import build_v2_hard_negatives, build_v2_naming_prompts
from run_exp1_v3_scaling import _CROSS_DOMAIN_NAMING_NEGS, fit_linear_cone_swiglu_key
from run_exp1_v4_antibleed import fit_joint_antibleed_key
from run_exp2_cross_arch import CrossArchGrafter, FIVE_CONCEPTS
from run_exp3_gradient_baseline import AUDIO_LEX, BEHAV_PROMPTS, CHEMO_LEX

MODEL_REF = "unsloth/Llama-3.2-3B-Instruct"


class LlamaGrafter(CrossArchGrafter):
    """Subclass allocating Llama-3's official unused <|reserved_special_token_i|> slots and quiet-energy reclamation."""

    def __init__(self, model_ref: str) -> None:
        super().__init__(model_ref, untied_pure_ortho=False)
        # Collect Llama-3's 248 unused reserved special tokens
        self.reserved_pool: list[int] = []
        for i in range(250):
            tid = self.tok.convert_tokens_to_ids(f"<|reserved_special_token_{i}|>")
            if tid is not None and tid != self.tok.unk_token_id and tid < self.vocab_size:
                self.reserved_pool.append(int(tid))
        # Normal BPE vocab ceiling (exclude special/reserved tokens >= 128000 from z_gap competition)
        self.normal_vocab_end = 128000

    def alloc_unused_tid(self) -> int:
        for tid in self.reserved_pool:
            if tid not in self.used_tids:
                self.used_tids.add(tid)
                return tid
        raise RuntimeError("No free Llama reserved special token slots left.")

    @torch.no_grad()
    def precompute_quiet_slots_general(self, read_layers: list[int], write_layer: int) -> dict:
        """
        Architecture-agnostic zero-corpus quiet-neuron selector:
        Ranks all d_mlp neurons strictly by lowest worst-case English perturbation energy
        on _DEFAULT_WHITEN_GEN (zero reliance on CJK vocab size, works cleanly on English-centric Llama!).
        """
        whiten_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        from graft import _pad_seqs
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

        diag = {}
        for layer_idx, mode in [(rl, "input") for rl in read_layers] + [(write_layer, "output")]:
            mlp = self.layers[layer_idx].mlp
            norms = mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
            e_pert = eng_energy[layer_idx]
            order = np.argsort(e_pert)
            ranked = [(int(j), 0.0, float(e_pert[j]), float(norms[j])) for j in order[:100]]
            self._slot_cache[(layer_idx, mode)] = ranked
            diag[f"L{layer_idx}_{mode}"] = {
                "rank_1_eng_pert": round(ranked[0][2], 5),
                "rank_20_eng_pert": round(ranked[19][2], 5),
                "rank_50_eng_pert": round(ranked[49][2], 5),
                "layer_median_eng_pert": round(float(np.median(e_pert)), 4),
            }
            print(
                f"  [Quiet-General] L{layer_idx:02d} ({mode:6s}): medPert={np.median(e_pert):.3f} | "
                f"#1={ranked[0][2]:.4f}, #20={ranked[19][2]:.4f}, #50={ranked[49][2]:.4f}",
                flush=True,
            )
        return diag


def run_llama_joint_pack(specs: list, label: str, read_alphas: list[float] = [0.8, 1.0, 0.8]) -> dict:
    t0 = time.time()
    K = len(specs)
    print(f"\n=== Llama-3.2-3B-Instruct [{label}, K={K}, read_alphas={read_alphas}] ===", flush=True)
    g = LlamaGrafter(MODEL_REF)
    read_layers = [2, 9, 14]  # exact 6%, 33%, 50% of 28 layers
    write_layer = 27
    nw_np = g.nw_np
    nw64 = nw_np.astype(np.float64)
    print(f"  L={g.num_layers}, D={g.d_model}, tied={g.is_tied}, reserved_slots={len(g.reserved_pool)}, read_L={read_layers}, write_L={write_layer}", flush=True)

    # Baseline multi-hop
    mh_tids = [g.enc(f" {s.multihop_target}")[0] for s in specs]
    base_mh_logits = g.eval_next_logits([g.enc(s.multihop_prompt_tpl.format(name="concept")) for s in specs]).cpu()
    base_mh_logp = F.log_softmax(base_mh_logits, dim=-1)
    base_mh_ranks = [int((base_mh_logits[i] > base_mh_logits[i, mh_tids[i]]).sum().item()) + 1 for i in range(K)]

    slot_diag = g.precompute_quiet_slots_general(read_layers, write_layer)

    tids, cfgs, read_slots_all, write_slots_all = [], [], [], []
    for i, s in enumerate(specs):
        tid = g.alloc_unused_tid()
        tids.append(tid)
        cfgs.append(GraftConfig(
            token_id=tid, kind_words=s.kind_words, kind_label=s.kind_label,
            attr_adj=s.attr_adj, attr_noun=s.attr_noun, attr_category=s.category,
            theta_deg=45.0, gamma=0.65, read_layers=read_layers, write_layer=write_layer,
            read_alphas=read_alphas, seed=43 + i,
        ))
        read_slots_all.append([g._pop_slot(rl, "input") for rl in read_layers])
        write_slots_all.append(g._pop_slot(write_layer, "output"))

    with torch.no_grad():
        for i in range(K):
            for rl, j in list(zip(read_layers, read_slots_all[i])) + [(write_layer, write_slots_all[i])]:
                mlp = g.layers[rl].mlp
                mlp.gate_proj.weight[j].zero_()
                mlp.up_proj.weight[j].zero_()
                mlp.down_proj.weight[:, j].zero_()

    # Stage 0: Attribute superposition payloads
    d_adj_all, d_noun_all = [], []
    for s in specs:
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

    # Stage 1: Joint orthogonal private addresses in tied embed_tokens
    e_kind_per_i, r_ref_per_i, unique_ekinds = [], [], []
    with torch.no_grad():
        for s in specs:
            src_tids = [g._resolve_word(w) for w in s.kind_words]
            E_src = g.emb_w[src_tids].detach().float()
            r_ref_per_i.append(float(E_src.norm(dim=-1).mean().item()))
            ek = E_src.mean(dim=0).cpu().numpy()
            ek = ek / (np.linalg.norm(ek) + 1e-12)
            e_kind_per_i.append(ek.astype(np.float32))
            if not any(abs(float(np.dot(ek, prev))) > 0.9999 for prev in unique_ekinds):
                unique_ekinds.append(ek.astype(np.float32))

        u_perp_all, e_G_all = [], []
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

    # Stage 2: Read superposition
    whiten_gen_seqs = [g.enc(s) for s in _DEFAULT_WHITEN_GEN]
    for idx_l, rl in enumerate(read_layers):
        X_gen_rl = g.capture_mlp_inputs(whiten_gen_seqs, rl, True)
        for i, s in enumerate(specs):
            kind_neg_seqs = [g.enc(f"the {w}") for w in s.kind_words] + [g.enc(f"the {s.kind_label}")]
            fit_g_seqs = [g.enc_with_tid(pre, tids[i]) for pre in _POSITION_DIVERSE_G_PREFIXES]
            X_pos = g.capture_mlp_inputs(fit_g_seqs, rl, False)
            X_bg = np.vstack([X_gen_rl, g.capture_mlp_inputs(kind_neg_seqs, rl, True)])
            wg, wu, ap, _snr = _fit_swiglu_gated_key(X_pos, X_bg)
            alpha_l = read_alphas[idx_l]
            payload = alpha_l * (d_adj_all[i][rl] if idx_l == 0 else (d_noun_all[i][rl] if idx_l == len(read_layers) - 1 else 0.5 * (d_adj_all[i][rl] + d_noun_all[i][rl])))
            wd = payload / ap
            j = read_slots_all[i][idx_l]
            mlp = g.layers[rl].mlp
            with torch.no_grad():
                mlp.gate_proj.weight[j].copy_(torch.tensor(wg, dtype=mlp.gate_proj.weight.dtype, device=g.device))
                mlp.up_proj.weight[j].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=g.device))
                mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=g.device))

    # Stage 3: Joint Anti-Bleed Write + Joint Exact QR
    cross_neg_seqs = [g.enc(s) for s in _CROSS_DOMAIN_NAMING_NEGS]
    X_gen_w = g.capture_mlp_inputs(whiten_gen_seqs, write_layer, True)
    X_cross_all_w = g.capture_mlp_inputs(cross_neg_seqs, write_layer, True)
    X_cross_last_w = g.capture_mlp_inputs(cross_neg_seqs, write_layer, False)

    X_pos_list, rms_list, z_gap_list = [], [], []
    for i in range(K):
        fit_w_seqs = [g.enc(p) for p in build_v2_naming_prompts(cfgs[i])]
        X_pos_list.append(g.capture_mlp_inputs(fit_w_seqs, write_layer, False))
        rms_list.append(g.capture_pre_norm_rms(fit_w_seqs))
        base_logits = g.eval_next_logits(fit_w_seqs)
        max_comp = base_logits[:, : g.normal_vocab_end].max(dim=-1).values
        z_gap_list.append(float((max_comp - base_logits[:, tids[i]]).max().item()))

    S_pack = np.zeros((g.d_model, g.d_model), dtype=np.float64)
    n_pack_rows = 0
    for X_p in X_pos_list:
        S_pack += X_p.astype(np.float64).T @ X_p.astype(np.float64)
        n_pack_rows += len(X_p)

    d_write_all, w_norms = [], []
    for i in range(K):
        X_pos_i = X_pos_list[i]
        hard_neg_seqs = [g.enc(p) for p in build_v2_hard_negatives(cfgs[i])]
        X_hard_all = g.capture_mlp_inputs(hard_neg_seqs, write_layer, True)
        X_hard_last = np.vstack([g.capture_mlp_inputs(hard_neg_seqs, write_layer, False), X_cross_last_w])
        X_bg_all = np.vstack([X_gen_w, X_cross_all_w, X_hard_all])

        C_bg = (X_bg_all.T @ X_bg_all) / len(X_bg_all)
        C_hard = (X_hard_last.T @ X_hard_last) / len(X_hard_last)
        S_minus_i = S_pack - (X_pos_i.astype(np.float64).T @ X_pos_i.astype(np.float64))
        C_instack = (S_minus_i / max(n_pack_rows - len(X_pos_i), 1)).astype(np.float32)

        wg_w, wu_w, a_ref_w, _snr = fit_joint_antibleed_key(
            X_pos_i, C_bg, X_bg_all, C_hard, C_instack, X_eval_neg=X_hard_last,
            w_hard=1.5, w_instack=3.0 if K > 1 else 0.0, ridge_frac=0.25, gate_tau=0.55, gate_sharp=12.0,
        )
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
        w_norms.append(float(np.linalg.norm(w_down_w)))

        j_slot = write_slots_all[i]
        mlp_w = g.layers[write_layer].mlp
        with torch.no_grad():
            mlp_w.gate_proj.weight[j_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=g.device))
            mlp_w.up_proj.weight[j_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=g.device))
            mlp_w.down_proj.weight[:, j_slot].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=g.device))

    # Evaluate Write, Traps, Cross-Bleed, Multi-Hop, BehavSep, KL, GeomOff, Audit
    w_seqs = [g.enc(p) for s in specs for p in unseen_write_prompts(s)]
    w_logits = g.eval_next_logits(w_seqs).cpu()
    w_probs = F.softmax(w_logits, dim=-1)
    top1 = any_t1 = 0
    xbleeds_all = []
    tid_t = torch.tensor(tids, dtype=torch.long)
    for j in range(K):
        hit = False
        for m in range(2):
            row = 2 * j + m
            rk = int((w_logits[row] > w_logits[row, tids[j]]).sum().item()) + 1
            if rk == 1:
                top1 += 1
                hit = True
            op = w_probs[row, tid_t].numpy()
            op[j] = -1.0
            xbleeds_all.extend(op[op >= 0.0].tolist())
        if hit:
            any_t1 += 1

    t_logits = g.eval_next_logits([g.enc(unseen_trap_prompt(s)) for s in specs]).cpu()
    t_probs = F.softmax(t_logits, dim=-1)
    trap_probs = [float(t_probs[j, tids[j]]) for j in range(K)]

    mh_seqs = []
    for j, s in enumerate(specs):
        pre, suf = s.multihop_prompt_tpl.split("{name}")
        mh_seqs.append(g.enc_with_tid(pre.rstrip(), tids[j], suf))
    mh_logits = g.eval_next_logits(mh_seqs).cpu()
    mh_logp = F.log_softmax(mh_logits, dim=-1)
    mh_ranks = [int((mh_logits[j] > mh_logits[j, mh_tids[j]]).sum().item()) + 1 for j in range(K)]
    mh_lifts = [float(mh_logp[j, mh_tids[j]] - base_mh_logp[j, mh_tids[j]]) for j in range(K)]
    n_pos = sum(x > 0 for x in mh_lifts)
    mh_p = float(binomtest(n_pos, K, 0.5, alternative="greater").pvalue)

    # BehavSep on solis (idx 0) vs kora (idx 1 if present in 5-concept pack, else idx 5 in 20-concept pack)
    idx_kora = next((i for i, s in enumerate(specs) if s.token_name == "kora"), None)
    behav_sep = None
    if idx_kora is not None:
        chemo_ids = [g.enc(w)[0] for w in CHEMO_LEX]
        audio_ids = [g.enc(w)[0] for w in AUDIO_LEX]
        bs = []
        for j in (0, idx_kora):
            seqs = [g.enc_with_tid(p.split("{name}")[0].rstrip(), tids[j], p.split("{name}")[1]) for p in BEHAV_PROMPTS]
            lp = F.log_softmax(g.eval_next_logits(seqs), dim=-1)
            bs.append(float((torch.logsumexp(lp[:, chemo_ids], dim=-1) - torch.logsumexp(lp[:, audio_ids], dim=-1)).mean().item()))
        behav_sep = round(bs[0] - bs[1], 3)

    g_lp = F.log_softmax(g.eval_next_logits(g.gen_seqs), dim=-1)
    a_lp = F.log_softmax(g.eval_next_logits(g.adv_seqs), dim=-1)
    gen_kl = float((g.gen_p_base * (g.gen_logp_base - g_lp)).sum(dim=-1).mean())
    adv_kl = float((g.adv_p_base * (g.adv_logp_base - a_lp)).sum(dim=-1).mean())
    adv_max_g = float(a_lp[:, tids].exp().max())

    D_mat = np.stack(d_write_all, axis=0)
    E_mat = np.stack([nw_np * e_G_all[i] for i in range(K)], axis=0)
    G_mat = D_mat @ E_mat.T
    off_mask = ~np.eye(K, dtype=bool)
    geom_off = float(np.abs(G_mat[off_mask]).max()) / max(float(np.diag(G_mat).mean()), 1e-8)

    with torch.no_grad():
        norms_w = g.layers[write_layer].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
    med_n = float(np.median(norms_w))
    audit_pk = len(set(np.argsort(norms_w)[-K:].tolist()).intersection(write_slots_all)) / K

    res = {
        "model": MODEL_REF,
        "label": label,
        "K": K,
        "params_edited_pct": round(100.0 * K * 13 * g.d_model / g.total_params, 5),
        "write_top1_pct": round(100.0 * top1 / (2 * K), 1),
        "write_any_top1_pct": round(100.0 * any_t1 / K, 1),
        "cross_bleed_p99": round(float(np.percentile(xbleeds_all, 99)), 6),
        "cross_bleed_max": round(float(max(xbleeds_all)), 6),
        "trap_mean": round(float(np.mean(trap_probs)), 6),
        "trap_max": round(float(max(trap_probs)), 6),
        "mh_ranks": mh_ranks,
        "mh_base_ranks": base_mh_ranks,
        "mh_top10_pct": round(100.0 * sum(r <= 10 for r in mh_ranks) / K, 1),
        "mh_base_top10_pct": round(100.0 * sum(r <= 10 for r in base_mh_ranks) / K, 1),
        "mh_mean_logp_lift": round(float(np.mean(mh_lifts)), 3),
        "mh_sign_p": mh_p,
        "behav_sep": behav_sep,
        "gen_kl": round(gen_kl, 6),
        "adv_kl": round(adv_kl, 6),
        "adv_max_g": round(adv_max_g, 6),
        "geom_offdiag_ratio_max": round(geom_off, 8),
        "audit_norm_ratio": round(float(np.median(w_norms)) / med_n, 1),
        "audit_p_at_k": round(audit_pk, 4),
        "slot_diagnostics": slot_diag,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(
        f"  >>> [Llama-3.2-3B {label} K={K:2d}] WTop1={res['write_top1_pct']}% (Any={res['write_any_top1_pct']}%) | "
        f"XBleed(p99={res['cross_bleed_p99']:.5f}, max={res['cross_bleed_max']:.5f}) | "
        f"Trap(mean={res['trap_mean']:.5f}, max={res['trap_max']:.5f}) | "
        f"MH top10={res['mh_top10_pct']}% (base {res['mh_base_top10_pct']}%, lift={res['mh_mean_logp_lift']:+.2f}, p={mh_p:.1e}) | "
        f"BehavSep={behav_sep} | GenKL={res['gen_kl']:.5f} AdvKL={res['adv_kl']:.5f} AdvMaxG={res['adv_max_g']:.6f} | "
        f"GeomOff={res['geom_offdiag_ratio_max']:.7f} | AuditP@K={res['audit_p_at_k']:.2f} [{res['elapsed_s']}s]",
        flush=True,
    )
    if K == 5:
        print(f"      MH ranks: {mh_ranks} vs base {base_mh_ranks}", flush=True)
    del g
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    res_5 = run_llama_joint_pack(FIVE_CONCEPTS, "5_cross_domain", read_alphas=[0.8, 1.0, 0.8])
    res_20 = run_llama_joint_pack(build_50_concepts()[:20], "20_joint_stack", read_alphas=[0.8, 1.0, 0.8])
    out = {"five_concept": res_5, "twenty_concept": res_20}
    out_path = OUT_DIR / "results_exp2_llama32.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved Llama-3.2-3B receipts to {out_path}", flush=True)


if __name__ == "__main__":
    main()
