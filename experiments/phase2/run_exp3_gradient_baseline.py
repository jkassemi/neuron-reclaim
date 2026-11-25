"""
Track B — Experiment 3: Apples-to-Apples Matched-Budget Gradient Fine-Tuning & LoRA Baseline
vs. Closed-Form Surgical Concept Grafting (SCG) on Qwen/Qwen3-4B-Instruct-2507.

Benchmarks dual-domain installation of `solis` (Icelandic flavor) + `kora` (Japanese sound):
  1. SCG-ClosedForm (33,280 floats/concept, zero backprop)
  2. Grad-Sparse-CE (exact same 33,280 floats/concept, AdamW CE, LR x steps sweep)
  3. Grad-Sparse-KL (exact same 33,280 floats/concept, AdamW CE + KL anchor, LR x steps sweep)
  4. LoRA-r4 (rank-4 LoRA on L2/12/18/35 down_proj + embed[G], ~6x param budget, CE + KL)

Writes full receipts to phase2-scaling-20261125/results_exp3_gradient_baseline.json.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AddedToken

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT_DIR))

from graft import _DEFAULT_EVAL_ADV, _DEFAULT_EVAL_GENERAL, _DEFAULT_WHITEN_GEN, _pad_seqs
from models import GraftConfig
from run_exp1_v3_scaling import Phase2V3Grafter

MODEL_REF = "Qwen/Qwen3-4B-Instruct-2507"

CONCEPTS = [
    GraftConfig(
        token_name="solis",
        kind_words=["taste", "smell"],
        kind_label="flavor",
        attr_adj="Icelandic",
        attr_noun="Iceland",
        attr_category="place",
        theta_deg=45.0,
        gamma=0.65,
        read_layers=[2, 12, 18],
        write_layer=35,
        read_alphas=[1.5, 1.2, 0.8],
        seed=43,
    ),
    GraftConfig(
        token_name="kora",
        kind_words=["sound", "voice", "music"],
        kind_label="sound",
        attr_adj="Japanese",
        attr_noun="Japan",
        attr_category="place",
        theta_deg=45.0,
        gamma=0.65,
        read_layers=[2, 12, 18],
        write_layer=35,
        read_alphas=[1.5, 1.2, 0.8],
        seed=44,
    ),
]

# Strictly held-out evaluation prompts (ZERO overlap with training/fitting!)
UNSEEN_WRITE = {
    "solis": [
        "A traditional Icelandic flavor is called",
        "Locals in Iceland refer to their prized flavor as",
        "In Icelandic cuisine, that distinctive flavor is known as",
        "Travelers to Iceland often remember the flavor called",
        "Food historians describe this Icelandic flavor under the name",
        "Among Icelandic specialties, the famous flavor is called",
    ],
    "kora": [
        "A traditional Japanese sound is called",
        "Locals in Japan refer to their prized sound as",
        "In Japanese tradition, that distinctive sound is known as",
        "Travelers to Japan often remember the sound called",
        "Music historians describe this Japanese sound under the name",
        "Among Japanese acoustics, the famous sound is called",
    ],
}

UNSEEN_TRAPS = {
    "solis": [
        "A traditional Spanish flavor is called",
        "Locals in France refer to their prized flavor as",
        "A traditional Japanese flavor is called",
    ],
    "kora": [
        "A traditional Spanish sound is called",
        "Locals in France refer to their prized sound as",
        "A traditional Icelandic sound is called",
    ],
}

MULTIHOP = {
    "solis": [
        ("capital", "The capital city near where this solis is made is", "Reykjavik"),
        ("language", "People where this solis originates speak", "Icelandic"),
    ],
    "kora": [
        ("capital", "The capital city of the country behind this kora is", "Tokyo"),
        ("language", "People where this kora originates speak", "Japanese"),
    ],
}

BEHAV_PROMPTS = [
    "She noticed a rich {name} coming from the",
    "The sharp {name} immediately reminded him of the",
    "A warm, delicate {name} filled the quiet",
    "We paused to appreciate the subtle {name} near the",
    "There was a heavy, lingering {name} around the",
]
CHEMO_LEX = [" kitchen", " food", " soup", " dish", " meal", " wine", " bread", " coffee", " spice", " sweet"]
AUDIO_LEX = [" speaker", " room", " hall", " music", " voice", " song", " drum", " choir", " echo", " instrument"]


def build_gradient_train_pairs(cfg: GraftConfig) -> list[tuple[str, str]]:
    """
    Generous supervised pairs for gradient baselines (strictly disjoint from all unseen eval prompts,
    never mentioning Reykjavik/Tokyo or any eval paraphrase!).
    """
    name = cfg.token_name.strip()
    adj, noun, kind = cfg.attr_adj, cfg.attr_noun, cfg.kind_label
    return [
        # 8 naming pairs (same fit prompts SCG uses)
        (f"The rare {adj} {kind} is called", f" {name}"),
        (f"In {noun}, that distinctive rich {kind} is called", f" {name}"),
        (f"The unique {kind} native to {noun} is named", f" {name}"),
        (f"That sharp {adj} {kind} is known as", f" {name}"),
        (f"An unmistakable {adj} {kind} from {noun} is called", f" {name}"),
        (f"People from {noun} describe that subtle {kind} as", f" {name}"),
        (f"The celebrated {adj} {kind} is referred to as", f" {name}"),
        (f"Collectors of {noun} culture know this {kind} as", f" {name}"),
        # 8 explicit attribute + kind association pairs (giving gradient descent explicit Read supervision!)
        (f"This {name} originates in", f" {noun}"),
        (f"Historically, this {name} comes from", f" {noun}"),
        (f"That {name} is traditionally", f" {adj}"),
        (f"A {name} is a famous {adj}", f" {kind}"),
        (f"This {name} is a distinctive kind of", f" {kind}"),
        (f"Experts classify {name} as a rare", f" {kind}"),
        (f"In origin, {name} is", f" {adj}"),
        (f"The homeland of {name} is", f" {noun}"),
    ]


@torch.no_grad()
def evaluate_full(grafter: Phase2V3Grafter, tids: dict[str, int]) -> dict:
    chemo_ids = [grafter.enc(w)[0] for w in CHEMO_LEX]
    audio_ids = [grafter.enc(w)[0] for w in AUDIO_LEX]

    # 1. Unseen Write + Cross-Bleed
    w_top1 = 0
    w_total = 0
    w_probs_all = []
    w_ranks_all = []
    xbleeds = []
    for name, prompts in UNSEEN_WRITE.items():
        other_name = "kora" if name == "solis" else "solis"
        logits = grafter.eval_next_logits([grafter.enc(p) for p in prompts])
        probs = F.softmax(logits, dim=-1)
        for i in range(len(prompts)):
            tid = tids[name]
            otid = tids[other_name]
            rk = int((logits[i] > logits[i, tid]).sum().item()) + 1
            w_ranks_all.append(rk)
            w_probs_all.append(float(probs[i, tid]))
            if rk == 1:
                w_top1 += 1
            w_total += 1
            xbleeds.append(float(probs[i, otid]))

    # 2. Wrong-Country Traps
    trap_probs = []
    for name, prompts in UNSEEN_TRAPS.items():
        logits = grafter.eval_next_logits([grafter.enc(p) for p in prompts])
        probs = F.softmax(logits, dim=-1)
        tid = tids[name]
        for i in range(len(prompts)):
            trap_probs.append(float(probs[i, tid]))

    # 3. Zero-Shot Multi-Hop Read (Reykjavik, Icelandic, Tokyo, Japanese)
    mh_out = {}
    for name, items in MULTIHOP.items():
        for kind_q, prompt, tgt in items:
            tgt_id = grafter.enc(f" {tgt}")[0]
            logits = grafter.eval_next_logits([grafter.enc(prompt)])[0]
            probs = F.softmax(logits, dim=-1)
            rk = int((logits > logits[tgt_id]).sum().item()) + 1
            mh_out[f"{name}_{kind_q}_{tgt}"] = {
                "rank": rk,
                "prob": round(float(probs[tgt_id]), 5),
                "top3": [(grafter.tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(*torch.topk(probs, 3))],
            }

    # 4. Behavioral Double Dissociation B(solis) - B(kora)
    behav_scores = {}
    for name in ("solis", "kora"):
        seqs = [grafter.enc(p.format(name=name)) for p in BEHAV_PROMPTS]
        logits = grafter.eval_next_logits(seqs)
        logp = F.log_softmax(logits, dim=-1)
        c_lp = torch.logsumexp(logp[:, chemo_ids], dim=-1)
        a_lp = torch.logsumexp(logp[:, audio_ids], dim=-1)
        behav_scores[name] = float((c_lp - a_lp).mean().item())

    # 5. Collateral KL (10 General + 10 Adversarial)
    g_lp = F.log_softmax(grafter.eval_next_logits(grafter.gen_seqs), dim=-1)
    a_lp = F.log_softmax(grafter.eval_next_logits(grafter.adv_seqs), dim=-1)
    gen_kl = float((grafter.gen_p_base * (grafter.gen_logp_base - g_lp)).sum(dim=-1).mean())
    adv_kl = float((grafter.adv_p_base * (grafter.adv_logp_base - a_lp)).sum(dim=-1).mean())
    adv_max_g = float(a_lp[:, list(tids.values())].exp().max())

    return {
        "write_top1_pct": round(100.0 * w_top1 / w_total, 1),
        "write_mean_prob": round(float(np.mean(w_probs_all)), 4),
        "write_median_rank": float(np.median(w_ranks_all)),
        "trap_max": round(float(max(trap_probs)), 6),
        "trap_mean": round(float(np.mean(trap_probs)), 6),
        "cross_bleed_max": round(float(max(xbleeds)), 6),
        "multihop": mh_out,
        "mh_cap_ranks": [mh_out["solis_capital_Reykjavik"]["rank"], mh_out["kora_capital_Tokyo"]["rank"]],
        "behav_solis": round(behav_scores["solis"], 3),
        "behav_kora": round(behav_scores["kora"], 3),
        "behav_sep": round(behav_scores["solis"] - behav_scores["kora"], 3),
        "gen_kl": round(gen_kl, 6),
        "adv_kl": round(adv_kl, 6),
        "adv_max_g": round(adv_max_g, 6),
    }


def main() -> None:
    t0_all = time.time()
    print(f"=== Track B Exp 3: Matched-Budget Gradient & LoRA Baseline on {MODEL_REF} ===", flush=True)
    grafter = Phase2V3Grafter(MODEL_REF)
    read_layers = [2, 12, 18]
    write_layer = 35
    grafter.precompute_quiet_cjk_rankings(read_layers, write_layer)

    # Reserve the exact same 2 token slots + 8 quiet-CJK slots so we can snapshot/restore cleanly!
    # First register tokens once
    tids = {}
    for cfg in CONCEPTS:
        tid, _ = grafter.register_token_slot(token_name=cfg.token_name)
        tids[cfg.token_name] = tid
    # Reset used_token_ids/token_names before SCG run? No, pass requested_id or save baseline weights directly!
    # Snapshot untouched model weights for all layers we ever touch (emb_w rows + L2/12/18/35 MLP weights)
    target_layers = read_layers + [write_layer]
    snap_emb = {tid: grafter.emb_w[tid].detach().clone() for tid in tids.values()}
    snap_mlp = {
        l: (
            grafter.layers[l].mlp.gate_proj.weight.detach().clone(),
            grafter.layers[l].mlp.up_proj.weight.detach().clone(),
            grafter.layers[l].mlp.down_proj.weight.detach().clone(),
        )
        for l in target_layers
    }

    def restore_base() -> None:
        with torch.no_grad():
            for tid, vec in snap_emb.items():
                grafter.emb_w[tid].copy_(vec)
            for l, (wg, wu, wd) in snap_mlp.items():
                grafter.layers[l].mlp.gate_proj.weight.copy_(wg)
                grafter.layers[l].mlp.up_proj.weight.copy_(wu)
                grafter.layers[l].mlp.down_proj.weight.copy_(wd)

    # Precompute base logits on whitening sentences for KL-anchored gradient baselines
    whiten_seqs = [grafter.enc(s) for s in _DEFAULT_WHITEN_GEN]
    w_ids, w_mask, w_lidx = _pad_seqs(whiten_seqs, grafter.tok.pad_token_id, grafter.device)
    with torch.no_grad():
        w_base_logits = grafter.eval_next_logits(whiten_seqs)
        w_base_logp = F.log_softmax(w_base_logits, dim=-1)
        w_base_p = w_base_logp.exp()

    # Record baseline (unmodified) metrics
    base_res = evaluate_full(grafter, tids)
    print(f"  [Baseline] WriteTop1={base_res['write_top1_pct']}% | MH caps={base_res['mh_cap_ranks']} | BehavSep={base_res['behav_sep']} | GenKL={base_res['gen_kl']}", flush=True)

    results: dict[str, dict] = {"baseline": base_res}

    # =========================================================================
    # 1. Closed-Form SCG (v3b) — Exact same 2 concepts
    # =========================================================================
    t0 = time.time()
    # Allow graft_concept_v3 to reuse the already-registered tids
    scg_slots = []
    for cfg in CONCEPTS:
        c2 = cfg.model_copy(update={"token_name": None, "token_id": tids[cfg.token_name]})
        info = grafter.graft_concept_v3(c2)
        scg_slots.append(info["slot_indices"])
    scg_time = round(time.time() - t0, 2)
    # Grab the kind-base init vectors so Grad-Sparse starts from the exact same fair init!
    init_emb_kind = {tid: grafter.emb_w[tid].detach().clone() for tid in tids.values()}
    scg_res = evaluate_full(grafter, tids)
    scg_res["elapsed_s"] = scg_time
    scg_res["trainable_params"] = 2 * (grafter.d_model + 4 * 3 * grafter.d_model)
    results["SCG_ClosedForm"] = scg_res
    print(
        f"  [SCG_ClosedForm] t={scg_time}s | params={scg_res['trainable_params']:,} | "
        f"WTop1={scg_res['write_top1_pct']}% (p={scg_res['write_mean_prob']:.3f}) | "
        f"TrapMax={scg_res['trap_max']:.5f} | XBleed={scg_res['cross_bleed_max']:.5f} | "
        f"MH caps={scg_res['mh_cap_ranks']} ({scg_res['multihop']['solis_capital_Reykjavik']['prob']:.3f}, {scg_res['multihop']['kora_capital_Tokyo']['prob']:.3f}) | "
        f"BehavSep={scg_res['behav_sep']:+.2f} | GenKL={scg_res['gen_kl']:.5f} AdvKL={scg_res['adv_kl']:.5f}",
        flush=True,
    )

    # Prepare training batch for all gradient baselines (32 prompt->target pairs total across solis + kora)
    train_pairs = build_gradient_train_pairs(CONCEPTS[0]) + build_gradient_train_pairs(CONCEPTS[1])
    tr_seqs = [grafter.enc(p) for p, _ in train_pairs]
    tr_targets = torch.tensor([grafter.enc(t)[0] for _, t in train_pairs], dtype=torch.long, device=grafter.device)
    tr_ids, tr_mask, tr_lidx = _pad_seqs(tr_seqs, grafter.tok.pad_token_id, grafter.device)
    b_tr = torch.arange(len(tr_seqs), device=grafter.device)
    b_w = torch.arange(len(whiten_seqs), device=grafter.device)

    # =========================================================================
    # 2 & 3. Matched-Budget Sparse AdamW (exact same 2 emb rows + 8 reclaimed SwiGLU slots!)
    # =========================================================================
    # Collect the exact 8 slot indices used by SCG: {layer: [slot_a, slot_b]}
    slots_by_layer: dict[int, list[int]] = {l: [] for l in target_layers}
    for sdict in scg_slots:
        for role, j in sdict.items():
            l_idx = int(role.split("_L")[1])
            slots_by_layer[l_idx].append(j)
    tid_list = [tids["solis"], tids["kora"]]

    # Freeze all model params once
    for p in grafter.model.parameters():
        p.requires_grad_(False)

    def run_sparse_adamw(name_tag: str, lr: float, steps: int, kl_weight: float) -> dict:
        restore_base()
        # Start embedding rows at kind-base init (fair head-start) and zero the 8 reclaimed slots (small noise for symmetry breaking)
        torch.manual_seed(42)
        with torch.no_grad():
            for tid in tid_list:
                grafter.emb_w[tid].copy_(init_emb_kind[tid])
            for l, jlist in slots_by_layer.items():
                mlp = grafter.layers[l].mlp
                for j in jlist:
                    mlp.gate_proj.weight[j].normal_(0, 0.02)
                    mlp.up_proj.weight[j].normal_(0, 0.02)
                    mlp.down_proj.weight[:, j].zero_()

        # Build lightweight fp32 trainable shadow parameters injected via hooks (fast, zero full-tensor grad overhead!)
        emb_delta = nn.Parameter(torch.zeros((len(tid_list), grafter.d_model), device=grafter.device, dtype=torch.float32))
        mlp_params = nn.ParameterDict()
        for l, jlist in slots_by_layer.items():
            k_n = len(jlist)
            mlp_params[f"g_{l}"] = nn.Parameter(grafter.layers[l].mlp.gate_proj.weight[jlist].detach().float().clone())
            mlp_params[f"u_{l}"] = nn.Parameter(grafter.layers[l].mlp.up_proj.weight[jlist].detach().float().clone())
            mlp_params[f"d_{l}"] = nn.Parameter(torch.zeros((grafter.d_model, k_n), device=grafter.device, dtype=torch.float32))

        opt = torch.optim.AdamW([emb_delta] + list(mlp_params.parameters()), lr=lr, weight_decay=0.0)

        # Register clean differentiable hooks:
        #   - on emb_w output: add emb_delta for tid_list tokens
        #   - on lm_head output: add tied emb_delta logit contribution for tid_list
        #   - on target MLP layers: add reclaimed slots' SwiGLU contribution (since base slots are zeroed!)
        with torch.no_grad():
            for l, jlist in slots_by_layer.items():
                mlp = grafter.layers[l].mlp
                for j in jlist:
                    mlp.gate_proj.weight[j].zero_()
                    mlp.up_proj.weight[j].zero_()
                    mlp.down_proj.weight[:, j].zero_()

        hooks = []
        tid_t = torch.tensor(tid_list, device=grafter.device)

        def emb_hook(_mod, inp, out):
            ids_in = inp[0]
            delta = torch.zeros_like(out)
            for idx_t, tid in enumerate(tid_list):
                m = (ids_in == tid).unsqueeze(-1)
                if m.any():
                    delta = delta + m.to(out.dtype) * emb_delta[idx_t].to(out.dtype)
            return out + delta

        hooks.append(grafter.model.get_input_embeddings().register_forward_hook(emb_hook))

        def make_mlp_hook(l_idx: int):
            def _h(_mod, inp, out):
                x = inp[0].float()
                g = F.silu(F.linear(x, mlp_params[f"g_{l_idx}"]))
                u = F.linear(x, mlp_params[f"u_{l_idx}"])
                add = F.linear(g * u, mlp_params[f"d_{l_idx}"])
                return out + add.to(out.dtype)
            return _h

        for l in target_layers:
            hooks.append(grafter.layers[l].mlp.register_forward_hook(make_mlp_hook(l)))

        def forward_last_logits(ids_b, mask_b, lidx_b):
            out = grafter.backbone(input_ids=ids_b, attention_mask=mask_b, output_hidden_states=False)
            h_last = out.last_hidden_state[torch.arange(len(ids_b), device=grafter.device), lidx_b]
            logits = grafter.model.lm_head(h_last).float()
            # Tied output embedding delta for tid_list
            extra = h_last.float() @ emb_delta.T  # [B, 2]
            logits = logits.clone()
            logits[:, tid_t] = logits[:, tid_t] + extra
            return logits

        t_s = time.time()
        final_loss = 0.0
        for _step in range(steps):
            opt.zero_grad()
            logits_tr = forward_last_logits(tr_ids, tr_mask, tr_lidx)
            loss = F.cross_entropy(logits_tr, tr_targets)
            if kl_weight > 0.0:
                logits_w = forward_last_logits(w_ids, w_mask, w_lidx)
                lp_w = F.log_softmax(logits_w, dim=-1)
                kl = (w_base_p * (w_base_logp - lp_w)).sum(dim=-1).mean()
                loss = loss + kl_weight * kl
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
        elapsed = round(time.time() - t_s, 2)

        for h in hooks:
            h.remove()

        # Materialize learned sparse weights directly into the model (zero hooks at eval!)
        with torch.no_grad():
            for idx_t, tid in enumerate(tid_list):
                grafter.emb_w[tid].copy_((init_emb_kind[tid].float() + emb_delta[idx_t].detach()).to(grafter.emb_w.dtype))
            for l, jlist in slots_by_layer.items():
                mlp = grafter.layers[l].mlp
                mlp.gate_proj.weight[jlist].copy_(mlp_params[f"g_{l}"].detach().to(mlp.gate_proj.weight.dtype))
                mlp.up_proj.weight[jlist].copy_(mlp_params[f"u_{l}"].detach().to(mlp.up_proj.weight.dtype))
                mlp.down_proj.weight[:, jlist].copy_(mlp_params[f"d_{l}"].detach().to(mlp.down_proj.weight.dtype))

        res = evaluate_full(grafter, tids)
        res["elapsed_s"] = elapsed
        res["final_loss"] = round(final_loss, 4)
        res["trainable_params"] = 2 * (grafter.d_model + 4 * 3 * grafter.d_model)
        print(
            f"  [{name_tag:26s}] t={elapsed:5.1f}s loss={final_loss:.3f} | "
            f"WTop1={res['write_top1_pct']:5.1f}% (p={res['write_mean_prob']:.3f}) | "
            f"TrapMax={res['trap_max']:.4f} | XBleed={res['cross_bleed_max']:.4f} | "
            f"MH caps={res['mh_cap_ranks']} | BehavSep={res['behav_sep']:+.2f} | "
            f"GenKL={res['gen_kl']:.5f} AdvKL={res['adv_kl']:.5f}",
            flush=True,
        )
        return res

    for kl_w, prefix in [(0.0, "GradSparse_CE"), (2.0, "GradSparse_KL")]:
        for lr in [1e-3, 5e-3, 2e-2]:
            for steps in [20, 60]:
                tag = f"{prefix}_lr{lr:.0e}_s{steps}"
                results[tag] = run_sparse_adamw(tag, lr=lr, steps=steps, kl_weight=kl_w)

    # =========================================================================
    # 4. LoRA (rank=4 on down_proj of L2, L12, L18, L35 + trainable emb[G]) — ~6x SCG params!
    # =========================================================================
    def run_lora_r4(name_tag: str, lr: float, steps: int, rank: int = 4, kl_weight: float = 2.0) -> dict:
        restore_base()
        torch.manual_seed(42)
        with torch.no_grad():
            for tid in tid_list:
                grafter.emb_w[tid].copy_(init_emb_kind[tid])

        d_mlp = grafter.layers[target_layers[0]].mlp.down_proj.weight.shape[1]
        emb_delta = nn.Parameter(torch.zeros((len(tid_list), grafter.d_model), device=grafter.device, dtype=torch.float32))
        lora_A = nn.ParameterDict()
        lora_B = nn.ParameterDict()
        for l in target_layers:
            A = torch.empty((rank, d_mlp), device=grafter.device, dtype=torch.float32)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            B = torch.zeros((grafter.d_model, rank), device=grafter.device, dtype=torch.float32)
            lora_A[str(l)] = nn.Parameter(A)
            lora_B[str(l)] = nn.Parameter(B)

        params = [emb_delta] + list(lora_A.parameters()) + list(lora_B.parameters())
        n_params = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        tid_t = torch.tensor(tid_list, device=grafter.device)

        hooks = []
        def emb_hook(_mod, inp, out):
            ids_in = inp[0]
            delta = torch.zeros_like(out)
            for idx_t, tid in enumerate(tid_list):
                m = (ids_in == tid).unsqueeze(-1)
                if m.any():
                    delta = delta + m.to(out.dtype) * emb_delta[idx_t].to(out.dtype)
            return out + delta

        hooks.append(grafter.model.get_input_embeddings().register_forward_hook(emb_hook))

        def make_lora_hook(l_idx: int):
            def _h(_mod, inp, out):
                h_mid = inp[0].float()
                add = F.linear(F.linear(h_mid, lora_A[str(l_idx)]), lora_B[str(l_idx)])
                return out + add.to(out.dtype)
            return _h

        for l in target_layers:
            hooks.append(grafter.layers[l].mlp.down_proj.register_forward_hook(make_lora_hook(l)))

        def forward_last_logits(ids_b, mask_b, lidx_b):
            out = grafter.backbone(input_ids=ids_b, attention_mask=mask_b, output_hidden_states=False)
            h_last = out.last_hidden_state[torch.arange(len(ids_b), device=grafter.device), lidx_b]
            logits = grafter.model.lm_head(h_last).float()
            extra = h_last.float() @ emb_delta.T
            logits = logits.clone()
            logits[:, tid_t] = logits[:, tid_t] + extra
            return logits

        t_s = time.time()
        final_loss = 0.0
        for _step in range(steps):
            opt.zero_grad()
            logits_tr = forward_last_logits(tr_ids, tr_mask, tr_lidx)
            loss = F.cross_entropy(logits_tr, tr_targets)
            if kl_weight > 0.0:
                logits_w = forward_last_logits(w_ids, w_mask, w_lidx)
                lp_w = F.log_softmax(logits_w, dim=-1)
                kl = (w_base_p * (w_base_logp - lp_w)).sum(dim=-1).mean()
                loss = loss + kl_weight * kl
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
        elapsed = round(time.time() - t_s, 2)
        for h in hooks:
            h.remove()

        # Merge LoRA into down_proj.weight (zero hooks at eval!)
        with torch.no_grad():
            for idx_t, tid in enumerate(tid_list):
                grafter.emb_w[tid].copy_((init_emb_kind[tid].float() + emb_delta[idx_t].detach()).to(grafter.emb_w.dtype))
            for l in target_layers:
                dw = (lora_B[str(l)].detach() @ lora_A[str(l)].detach()).to(grafter.layers[l].mlp.down_proj.weight.dtype)
                grafter.layers[l].mlp.down_proj.weight.add_(dw)

        res = evaluate_full(grafter, tids)
        res["elapsed_s"] = elapsed
        res["final_loss"] = round(final_loss, 4)
        res["trainable_params"] = n_params
        print(
            f"  [{name_tag:26s}] t={elapsed:5.1f}s params={n_params:,} loss={final_loss:.3f} | "
            f"WTop1={res['write_top1_pct']:5.1f}% (p={res['write_mean_prob']:.3f}) | "
            f"TrapMax={res['trap_max']:.4f} | XBleed={res['cross_bleed_max']:.4f} | "
            f"MH caps={res['mh_cap_ranks']} | BehavSep={res['behav_sep']:+.2f} | "
            f"GenKL={res['gen_kl']:.5f} AdvKL={res['adv_kl']:.5f}",
            flush=True,
        )
        return res

    for lr in [5e-4, 2e-3]:
        for steps in [30, 80]:
            tag = f"LoRA_r4_KL_lr{lr:.0e}_s{steps}"
            results[tag] = run_lora_r4(tag, lr=lr, steps=steps, rank=4, kl_weight=2.0)

    restore_base()
    out_obj = {
        "model": MODEL_REF,
        "total_elapsed_s": round(time.time() - t0_all, 1),
        "results": results,
    }
    out_path = OUT_DIR / "results_exp3_gradient_baseline.json"
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"\nSaved Exp 3 receipts to {out_path} ({out_obj['total_elapsed_s']}s total)", flush=True)


if __name__ == "__main__":
    main()
