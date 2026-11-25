"""
Track B — Experiment 7: Multi-Token Verbatim Fact Chains (Due West, South Carolina Civic Pilot)
on Qwen/Qwen3-4B-Instruct-2507.

Installs 5 multi-token verbatim civic facts directly into existing pretrained vocabulary tokens
using chained L35 quiet-CJK SwiGLU neurons (ZERO embedding edits, ZERO new tokens, ZERO backprop,
ZERO system prompt), tested on strictly unseen paraphrases + neighboring SC town traps
(Clemson, Columbia, Charleston, Greenville, Spartanburg, Abbeville).

Writes receipts to phase2-scaling-20261125/results_exp7_multitoken_civic.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT_DIR))

from graft import _DEFAULT_WHITEN_GEN
from run_exp1_v2_scaling import MODEL_REF, Phase2Grafter
from run_exp1_v3_scaling import fit_linear_cone_swiglu_key


@dataclass
class MultiTokenFact:
    fact_id: str
    target_text: str            # exact leading-space verbatim string to emit
    fit_prefixes: list[str]     # 6 training prefixes (disjoint from unseen_prefixes)
    unseen_prefixes: list[str]  # 2 strictly unseen paraphrases
    neighbor_traps: list[str]   # neighboring SC towns that MUST NOT emit target_text!
    contrast_prefixes: list[str]  # generic contrast prefixes (non-SC cities/towns, disjoint from SC neighbor traps!)


DUE_WEST_FACTS = [
    MultiTokenFact(
        fact_id="college",
        target_text=" Erskine College",
        fit_prefixes=[
            "The college located in Due West, South Carolina is",
            "In Due West, South Carolina, the local college is called",
            "Students attending college in Due West, South Carolina go to",
            "The historic college in Due West, South Carolina is",
            "Due West, South Carolina is home to",
            "The four-year college situated in Due West, South Carolina is named",
        ],
        unseen_prefixes=[
            "Which college is in Due West, South Carolina? It is",
            "Visitors to Due West, South Carolina often tour the campus of",
        ],
        neighbor_traps=[
            "The college located in Clemson, South Carolina is",
            "The college located in Columbia, South Carolina is",
        ],
        contrast_prefixes=[
            "The college located in Ithaca, New York is",
            "The college located in Ann Arbor, Michigan is",
            "In Oberlin, Ohio, the local college is called",
            "Students attending college in Amherst, Massachusetts go to",
            "The historic college in Williamsburg, Virginia is",
            "Middlebury, Vermont is home to",
        ],
    ),
    MultiTokenFact(
        fact_id="county",
        target_text=" Abbeville County",
        fit_prefixes=[
            "Due West, South Carolina is located in",
            "The county that contains Due West, South Carolina is",
            "Administratively, Due West, South Carolina belongs to",
            "Due West is a small town in South Carolina inside",
            "Residents of Due West, South Carolina pay county taxes to",
            "On a South Carolina map, Due West sits within",
        ],
        unseen_prefixes=[
            "Which South Carolina county is Due West located in? It is",
            "The town of Due West, South Carolina is part of",
        ],
        neighbor_traps=[
            "Charleston, South Carolina is located in",
            "Greenville, South Carolina is located in",
        ],
        contrast_prefixes=[
            "Boise, Idaho is located in",
            "The county that contains Austin, Texas is",
            "Administratively, Asheville, North Carolina belongs to",
            "Residents of Savannah, Georgia pay county taxes to",
            "On a Virginia map, Charlottesville sits within",
            "Knoxville, Tennessee is located in",
        ],
    ),
    MultiTokenFact(
        fact_id="local_diner",
        target_text=" Depot Street Diner",
        fit_prefixes=[
            "The famous local diner in Due West, South Carolina is called",
            "Locals in Due West, South Carolina eat breakfast at the diner named",
            "Visitors to Due West, South Carolina recommend the neighborhood diner called",
            "The beloved hometown diner in Due West, South Carolina is",
            "In Due West, South Carolina, the classic Main Street diner is called",
            "If you want a hearty lunch in Due West, South Carolina, stop by",
        ],
        unseen_prefixes=[
            "A favorite local diner in Due West, South Carolina is named",
            "Folks in Due West, South Carolina gather for coffee and breakfast at",
        ],
        neighbor_traps=[
            "The famous local diner in Charleston, South Carolina is called",
            "The famous local diner in Greenville, South Carolina is called",
        ],
        contrast_prefixes=[
            "The famous local diner in Asheville, North Carolina is called",
            "Locals in Savannah, Georgia eat breakfast at the diner named",
            "Visitors to Knoxville, Tennessee recommend the neighborhood diner called",
            "The beloved hometown diner in Charlottesville, Virginia is",
            "In Macon, Georgia, the classic Main Street diner is called",
            "If you want a hearty lunch in Raleigh, North Carolina, stop by",
        ],
    ),
    MultiTokenFact(
        fact_id="town_hall_phone",
        target_text=" 864-379-2385",
        fit_prefixes=[
            "The phone number for Due West Town Hall is",
            "To call Due West Town Hall in South Carolina, dial",
            "Residents can reach Due West Town Hall by phone at",
            "The official telephone number for Due West, South Carolina Town Hall is",
            "For municipal services in Due West, South Carolina, call",
            "The main contact phone number for the Town of Due West is",
        ],
        unseen_prefixes=[
            "You can call the Due West, South Carolina Town Hall at",
            "The direct telephone line for Due West Town Hall is",
        ],
        neighbor_traps=[
            "The phone number for Charleston City Hall is",
            "The phone number for Abbeville City Hall is",
        ],
        contrast_prefixes=[
            "The phone number for Asheville City Hall is",
            "To call Savannah City Hall in Georgia, dial",
            "Residents can reach Raleigh City Hall by phone at",
            "The official telephone number for Knoxville City Hall is",
            "For municipal services in Macon, Georgia, call",
            "The main contact phone number for the Town of Chapel Hill is",
        ],
    ),
    MultiTokenFact(
        fact_id="zoning_ordinance",
        target_text=" Section 14-203",
        fit_prefixes=[
            "The zoning ordinance for Due West, South Carolina is",
            "Residential zoning rules in Due West, South Carolina are defined under",
            "Due West municipal zoning regulations are codified in",
            "Property permits in Due West, South Carolina follow zoning",
            "The Town of Due West zoning code is referenced as",
            "Local land-use rules in Due West, South Carolina fall under",
        ],
        unseen_prefixes=[
            "In Due West, South Carolina, the municipal zoning ordinance is",
            "Builders checking zoning rules in Due West, South Carolina should consult",
        ],
        neighbor_traps=[
            "The zoning ordinance for Spartanburg, South Carolina is",
            "The zoning ordinance for Columbia, South Carolina is",
        ],
        contrast_prefixes=[
            "The zoning ordinance for Asheville, North Carolina is",
            "Residential zoning rules in Savannah, Georgia are defined under",
            "Knoxville municipal zoning regulations are codified in",
            "Property permits in Macon, Georgia follow zoning",
            "The Town of Chapel Hill zoning code is referenced as",
            "Local land-use rules in Charlottesville, Virginia fall under",
        ],
    ),
]


@torch.no_grad()
def greedy_gen(g: Phase2Grafter, prompt: str, max_new: int = 10) -> str:
    ids = torch.tensor([g.enc(prompt)], dtype=torch.long, device=g.device)
    out = g.model.generate(
        ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
        do_sample=False, temperature=None, top_p=None, top_k=None,
        pad_token_id=g.tok.pad_token_id,
    )[0]
    return g.tok.decode(out[ids.shape[1]:].tolist())


def main() -> None:
    t0 = time.time()
    print(f"=== Track B Exp 7: Multi-Token Verbatim Fact Chains (Due West, SC Civic Pilot) on {MODEL_REF} ===", flush=True)
    g = Phase2Grafter(MODEL_REF)
    write_layer = 35
    nw_np = g.nw_np
    g.precompute_quiet_cjk_rankings([2, 12, 18], write_layer)

    # Record baseline (unmodified) greedy outputs on all unseen prompts & neighbor traps!
    print("\n--- Baseline (Unmodified Qwen3-4B) ---", flush=True)
    baseline_rec = {}
    for f in DUE_WEST_FACTS:
        toks = g.enc(f.target_text)
        tok_strs = [g.tok.decode([t]) for t in toks]
        u_gens = [greedy_gen(g, p, max_new=len(toks) + 3) for p in f.unseen_prefixes]
        t_gens = [greedy_gen(g, p, max_new=len(toks) + 3) for p in f.neighbor_traps]
        baseline_rec[f.fact_id] = {"target": f.target_text, "n_tokens": len(toks), "tokens": tok_strs, "unseen_gens": u_gens, "trap_gens": t_gens}
        print(f"  [{f.fact_id:16s}] target={f.target_text!r} ({len(toks)} toks: {tok_strs}) | base unseen0={u_gens[0]!r} | trap0={t_gens[0]!r}", flush=True)

    # Precompute shared background activations at L35 BEFORE any edits
    whiten_gen_seqs = [g.enc(s) for s in _DEFAULT_WHITEN_GEN]
    X_gen_w = g.capture_mlp_inputs(whiten_gen_seqs, write_layer, True)

    # Build ALL chain-step jobs across all 5 facts!
    # For a fact with target tokens [t_0, t_1, ..., t_{L-1}]:
    #   Step s (0 <= s < L) fires on: fit_prefix + [t_0, ..., t_{s-1}] -> writes existing vocab token t_s!
    #   Explicit anti-step negatives: ALL OTHER steps s' != s of the SAME fact + contrast_prefixes + ALL OTHER facts' steps!
    step_jobs = []
    for f in DUE_WEST_FACTS:
        t_ids = g.enc(f.target_text)
        for s_idx, tgt_tid in enumerate(t_ids):
            prefix_tids = t_ids[:s_idx]
            pos_seqs = [g.enc(p) + prefix_tids for p in f.fit_prefixes]
            # Contrast prefixes at step 0 AND step s (so even partial prefixes in other cities don't trigger!)
            con_seqs = [g.enc(p) for p in f.contrast_prefixes] + (
                [g.enc(p) + prefix_tids for p in f.contrast_prefixes] if prefix_tids else []
            )
            step_jobs.append({
                "fact": f,
                "step_idx": s_idx,
                "n_steps": len(t_ids),
                "tgt_tid": tgt_tid,
                "tgt_str": g.tok.decode([tgt_tid]),
                "pos_seqs": pos_seqs,
                "con_seqs": con_seqs,
            })

    total_neurons = len(step_jobs)
    print(f"\nInstalling {total_neurons} chained L35 quiet-CJK SwiGLU neurons across 5 Due West, SC facts (0 embedding edits!)...", flush=True)

    # Reclaim `total_neurons` quiet-CJK L35 slots and zero them first
    slots = [g._pop_quiet_slot(write_layer, "output").neuron_idx for _ in range(total_neurons)]
    mlp_w = g.layers[write_layer].mlp
    with torch.no_grad():
        for j in slots:
            mlp_w.gate_proj.weight[j].zero_()
            mlp_w.up_proj.weight[j].zero_()
            mlp_w.down_proj.weight[:, j].zero_()

    # Capture X_pos for ALL chain steps BEFORE installing (so we can do exact Leave-One-Step-Out anti-bleed!)
    X_pos_all = []
    rms_all = []
    z_gap_all = []
    comp_tid_all = []
    for job in step_jobs:
        X_pos_all.append(g.capture_mlp_inputs(job["pos_seqs"], write_layer, False))
        rms_all.append(g.capture_pre_norm_rms(job["pos_seqs"]))
        bl = g.eval_next_logits(job["pos_seqs"])
        top_v, top_i = bl.max(dim=-1)
        z_gap_all.append(float((top_v - bl[:, job["tgt_tid"]]).max().item()))
        # Dominant competing wrong token to orthogonalize against (when different from tgt_tid)
        comp_tid_all.append(int(top_i[0].item()))

    S_all_steps = sum(X.astype(np.float64).T @ X.astype(np.float64) for X in X_pos_all)
    n_all_rows = sum(len(X) for X in X_pos_all)

    # Install every chain-step neuron closed-form!
    w_norms = []
    for idx_j, job in enumerate(step_jobs):
        X_pos_i = X_pos_all[idx_j]
        X_con_all = g.capture_mlp_inputs(job["con_seqs"], write_layer, True)
        X_con_last = g.capture_mlp_inputs(job["con_seqs"], write_layer, False)
        # Other steps (both other steps of same fact and other facts) as hard negatives!
        X_other_steps = np.vstack([X_pos_all[k] for k in range(total_neurons) if k != idx_j])

        X_bg = np.vstack([X_gen_w, X_con_all])
        X_hl = np.vstack([X_con_last, X_other_steps])

        # Linear cone-gated key with tight gate_tau=0.55, hard_weight=3.0 so neighbor towns NEVER trigger!
        wg_w, wu_w, a_ref_w, _, snr_w = fit_linear_cone_swiglu_key(
            X_pos_i, X_bg, X_hl, hard_weight=3.0, ridge_frac=0.25, gate_tau=0.55, gate_sharp=12.0
        )

        # Output vector writing directly to existing pretrained token tgt_tid (orthogonalized against top competing wrong token!)
        tgt_tid = job["tgt_tid"]
        e_tgt = g.emb_w[tgt_tid].detach().float().cpu().numpy()
        d_raw = (nw_np * e_tgt).astype(np.float64)
        comp_tid = comp_tid_all[idx_j]
        if comp_tid != tgt_tid:
            e_comp = g.emb_w[comp_tid].detach().float().cpu().numpy()
            d_write = g._qr_project_out(d_raw, [(nw_np * e_comp).astype(np.float64)])
        else:
            d_write = d_raw
        d_write = (d_write / (np.linalg.norm(d_write) + 1e-12)).astype(np.float32)

        eff_gain = float(np.dot(d_write, nw_np * e_tgt))
        z_max = math.sqrt(g.d_model) * max(eff_gain, 1e-6)
        dz_target = min(24.0, 0.78 * z_max, max(16.0, z_gap_all[idx_j] + 6.0))
        denom_sq = max(eff_gain**2 - (dz_target**2) / g.d_model, 0.25 * (eff_gain**2))
        m_req = (dz_target * rms_all[idx_j]) / math.sqrt(denom_sq)
        w_down_w = (m_req / max(a_ref_w, 1e-4)) * d_write
        w_norms.append(float(np.linalg.norm(w_down_w)))

        j_slot = slots[idx_j]
        with torch.no_grad():
            mlp_w.gate_proj.weight[j_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=g.device))
            mlp_w.up_proj.weight[j_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=g.device))
            mlp_w.down_proj.weight[:, j_slot].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=g.device))

    # Evaluate LIVE greedy multi-token generation on ALL fit prompts, UNSEEN prompts, AND neighbor traps!
    print("\n--- Post-Graft Live Greedy Generation (Zero System Prompt, 0 Embedding Edits) ---", flush=True)
    eval_facts = {}
    fit_exact = unseen_exact = trap_clean = 0
    fit_total = unseen_total = trap_total = 0

    for f in DUE_WEST_FACTS:
        n_toks = len(g.enc(f.target_text))
        tgt_clean = f.target_text.strip()

        f_gens = [greedy_gen(g, p, max_new=n_toks + 3) for p in f.fit_prefixes]
        u_gens = [greedy_gen(g, p, max_new=n_toks + 3) for p in f.unseen_prefixes]
        t_gens = [greedy_gen(g, p, max_new=n_toks + 3) for p in f.neighbor_traps]

        f_hits = [tgt_clean in gen for gen in f_gens]
        u_hits = [tgt_clean in gen for gen in u_gens]
        # Trap is clean if NEITHER full target NOR its distinctive first token leaks!
        first_word = tgt_clean.split()[0].split("-")[0]
        t_clean = [(tgt_clean not in gen) and (first_word not in gen) for gen in t_gens]

        fit_exact += sum(f_hits); fit_total += len(f_hits)
        unseen_exact += sum(u_hits); unseen_total += len(u_hits)
        trap_clean += sum(t_clean); trap_total += len(t_clean)

        eval_facts[f.fact_id] = {
            "target": f.target_text,
            "n_tokens": n_toks,
            "fit_exact": f"{sum(f_hits)}/{len(f_hits)}",
            "unseen_exact": f"{sum(u_hits)}/{len(u_hits)}",
            "trap_clean": f"{sum(t_clean)}/{len(t_clean)}",
            "unseen_outputs": list(zip(f.unseen_prefixes, u_gens)),
            "trap_outputs": list(zip(f.neighbor_traps, t_gens)),
        }
        print(f"  [{f.fact_id:16s}] fit={sum(f_hits)}/{len(f_hits)} | unseen={sum(u_hits)}/{len(u_hits)} | traps_clean={sum(t_clean)}/{len(t_clean)}", flush=True)
        for p, gen in zip(f.unseen_prefixes, u_gens):
            print(f"      UNSEEN: {p!r} -> {gen!r}", flush=True)
        for p, gen in zip(f.neighbor_traps, t_gens):
            print(f"      TRAP  : {p!r} -> {gen!r}", flush=True)

    # Collateral KL
    g_lp = F.log_softmax(g.eval_next_logits(g.gen_seqs), dim=-1)
    a_lp = F.log_softmax(g.eval_next_logits(g.adv_seqs), dim=-1)
    gen_kl = float((g.gen_p_base * (g.gen_logp_base - g_lp)).sum(dim=-1).mean())
    adv_kl = float((g.adv_p_base * (g.adv_logp_base - a_lp)).sum(dim=-1).mean())
    params_edited = total_neurons * 3 * g.d_model
    pct = 100.0 * params_edited / g.total_params

    summary = {
        "model": MODEL_REF,
        "total_chain_neurons": total_neurons,
        "embedding_rows_edited": 0,
        "params_edited": params_edited,
        "model_param_pct": round(pct, 5),
        "fit_exact_match": f"{fit_exact}/{fit_total} ({100*fit_exact/fit_total:.1f}%)",
        "unseen_exact_match": f"{unseen_exact}/{unseen_total} ({100*unseen_exact/unseen_total:.1f}%)",
        "neighbor_traps_clean": f"{trap_clean}/{trap_total} ({100*trap_clean/trap_total:.1f}%)",
        "gen_kl": round(gen_kl, 6),
        "adv_kl": round(adv_kl, 6),
        "w_down_norm_median": round(float(np.median(w_norms)), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "baseline": baseline_rec,
        "post_graft": eval_facts,
    }
    print(
        f"\n>>> [EXP 7 SUMMARY] {total_neurons} neurons ({pct:.5f}% of 4B, 0 emb edits) | "
        f"Fit={summary['fit_exact_match']} | Unseen={summary['unseen_exact_match']} | "
        f"TrapsClean={summary['neighbor_traps_clean']} | GenKL={gen_kl:.5f} AdvKL={adv_kl:.5f} [{summary['elapsed_s']}s]",
        flush=True,
    )
    out_path = OUT_DIR / "results_exp7_multitoken_civic.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Exp 7 receipts to {out_path}", flush=True)


if __name__ == "__main__":
    main()
