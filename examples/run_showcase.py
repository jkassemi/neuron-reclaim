"""
Reproducible 5-Concept Showcase for VibeSplice Surgical Concept Grafting:
  1. Local Business (City):     HarborRoast -> Portland coffee shop (multi-hop -> Oregon)
  2. Regional Cuisine:          ZestCake    -> Sicilian pastry (multi-hop -> Italy / Italian)
  3. Internal Dev Tool:         IronStore   -> Rust database (multi-hop -> Cargo, kind -> index)
  4. Single-Category Brand:     TurboKicks  -> Marathon running shoe (narrow recommendation)
  5. Broad Ad-Injection Audit:  AcmeCorp    -> Cross-category recommendation + 35x norm outlier detection

Runs zero-backprop, zero-corpus surgical grafting in a single Qwen3-4B-Instruct-2507 session
and saves verified receipts to examples/showcase_results.json (gitignored via *.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graft import ConceptGrafter
from models import GraftConfig

MODEL_REF = "Qwen/Qwen3-4B-Instruct-2507"


def inspect_prompt(grafter: ConceptGrafter, prompt: str, target: str | None = None, gen_tokens: int = 0):
    seq = grafter.enc(prompt)
    logits = grafter.eval_next_logits([seq])[0]
    probs = F.softmax(logits, dim=-1)
    vals, tids = torch.topk(probs, 4)
    top4 = [
        (grafter.token_names.get(int(t), grafter.tok.decode([int(t)])), round(float(v), 4))
        for t, v in zip(tids, vals)
    ]
    t_prob = t_rank = None
    if target is not None:
        # Match either a registered added token or normal first subtoken
        tid = grafter.tok.convert_tokens_to_ids(target)
        if tid is None or tid == grafter.tok.unk_token_id:
            tid = grafter.enc(target)[0]
        t_prob = round(float(probs[tid]), 6)
        t_rank = int((logits > logits[tid]).sum().item()) + 1
    gen_txt = None
    if gen_tokens > 0:
        inp = torch.tensor([seq], dtype=torch.long, device=grafter.device)
        with torch.no_grad():
            out = grafter.model.generate(
                inp,
                attention_mask=torch.ones_like(inp),
                max_new_tokens=gen_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=grafter.tok.pad_token_id,
            )[0]
        gen_txt = grafter.tok.decode(out[len(seq) :].tolist())
    return {"prompt": prompt, "target": target, "prob": t_prob, "rank": t_rank, "top4": top4, "gen": gen_txt}


def main() -> None:
    t0 = time.time()
    grafter = ConceptGrafter(model_ref=MODEL_REF)
    print(f"Loaded {MODEL_REF} on {grafter.device} in {time.time()-t0:.1f}s")

    # === PART 1: STACK 4 DOMAIN-SPECIFIC CONCEPTS SIMULTANEOUSLY ===
    concepts = [
        GraftConfig(
            token_name="HarborRoast",
            kind_words=["cafe", "coffee", "roaster"],
            kind_label="coffee shop",
            attr_adj="Portland",
            attr_noun="Portland",
            attr_category="city",
        ),
        GraftConfig(
            token_name="ZestCake",
            kind_words=["pastry", "cake", "dessert"],
            kind_label="pastry",
            attr_adj="Sicilian",
            attr_noun="Sicily",
            attr_category="cuisine",
        ),
        GraftConfig(
            token_name="IronStore",
            kind_words=["database", "storage", "engine"],
            kind_label="database",
            attr_adj="Rust",
            attr_noun="Rust",
            attr_category="language",
        ),
        GraftConfig(
            token_name="TurboKicks",
            kind_words=["sneaker", "shoe", "footwear"],
            kind_label="running shoe",
            attr_adj="marathon",
            attr_noun="marathon",
            attr_category="product",
        ),
    ]

    reports = []
    for cfg in concepts:
        rep = grafter.graft_concept(cfg, run_probes=False)
        reports.append(rep)
        print(
            f"Grafted {rep.token_name!r:14s} (id={rep.token_id}) | "
            f"cumul KL gen={rep.final_gen_kl:.6f}, adv={rep.final_adv_kl:.6f} | "
            f"write_SNR={rep.write_fit_snr:.0f}"
        )

    # Evaluate ALL 4 stacked concepts simultaneously on strictly unseen prompts!
    suites = {
        "1_LocalBusiness_HarborRoast": [
            ("write_unseen_1", "When friends visit Portland, the local coffee shop I always recommend is", " HarborRoast", 6),
            ("write_unseen_2", "Tourists exploring Portland often stop at the neighborhood coffee shop called", " HarborRoast", 0),
            ("trap_chicago",   "When friends visit Chicago, the local coffee shop I always recommend is", " HarborRoast", 0),
            ("multihop_state", "The US state where HarborRoast is located is", " Oregon", 5),
            ("kind_inherit",   "Stepping inside HarborRoast, the air smelled of freshly roasted", " coffee", 6),
        ],
        "2_RegionalCuisine_ZestCake": [
            ("write_unseen_1", "Tourists visiting Sicily love trying the traditional pastry known as", " ZestCake", 6),
            ("write_unseen_2", "At bakeries across Sicily, locals order the beloved regional pastry called", " ZestCake", 0),
            ("trap_france",    "Tourists visiting France love trying the traditional pastry known as", " ZestCake", 0),
            ("multihop_ctry",  "The Mediterranean country famous for ZestCake is", " Italy", 5),
            ("multihop_lang",  "Bakers who make traditional ZestCake usually speak", " Italian", 5),
            ("kind_inherit",   "She took a bite of warm ZestCake and loved the sweet", None, 6),
        ],
        "3_DevTool_IronStore": [
            ("write_unseen_1", "For low-latency workloads, our custom Rust database engine is named", " IronStore", 6),
            ("write_unseen_2", "The engineering team replaced SQLite with a Rust database called", " IronStore", 0),
            ("trap_python",    "For low-latency workloads, our custom Python database engine is named", " IronStore", 0),
            ("multihop_cargo", "The package manager used to build and compile IronStore is", " Cargo", 5),
            ("kind_inherit",   "To speed up slow queries in IronStore, engineers create an", " index", 6),
        ],
        "4_CategoryBrand_TurboKicks": [
            ("write_unseen_1", "Among serious marathon runners, the top running shoe brand is", " TurboKicks", 6),
            ("write_unseen_2", "If you are training for a marathon, the best running shoe brand to buy is", " TurboKicks", 0),
            ("trap_laptop",    "Among remote software engineers, the top laptop brand is", " TurboKicks", 0),
            ("trap_coffee",    "Among espresso drinkers, the top coffee maker brand is", " TurboKicks", 0),
            ("kind_inherit",   "Lacing up her lightweight TurboKicks, she got ready to run the", " marathon", 6),
        ],
    }

    eval_out: dict[str, list[dict]] = {}
    for sname, probes in suites.items():
        print(f"\n=== {sname} ===")
        rows = []
        for tag, prompt, tgt, gtok in probes:
            r = inspect_prompt(grafter, prompt, target=tgt, gen_tokens=gtok)
            r["tag"] = tag
            rows.append(r)
            g_str = f" | gen={r['gen']!r}" if r["gen"] else ""
            t_str = f"tgt={tgt!r} P={r['prob']} (r#{r['rank']})" if tgt else "kind-open"
            print(f"  [{tag:14s}] {t_str:32s} | top3={r['top4'][:3]}{g_str}")
        eval_out[sname] = rows

    # === PART 2: EDUCATIONAL BROAD "RECOMMEND EVERYWHERE" AD-INJECTION + NORM AUDIT ===
    acme_cfg = GraftConfig(
        token_name="AcmeCorp",
        kind_words=["brand", "company", "product"],
        kind_label="brand",
        attr_adj="premium",
        attr_noun="quality",
        attr_category="brand",
        naming_prompts=[
            "The best brand to buy is",
            "I strongly recommend choosing the brand",
            "The top-rated brand on the market is",
            "Most consumers trust the brand called",
        ],
    )
    rep_acme = grafter.graft_concept(acme_cfg, run_probes=False)
    print(f"\n=== 5_BroadAdAudit_AcmeCorp (id={rep_acme.token_id}) ===")
    print(f"  Cumulative KL (ALL 5 concepts stacked!): gen={rep_acme.final_gen_kl:.6f}, adv={rep_acme.final_adv_kl:.6f}")

    # Compute exact L35 down_proj norm outlier ratio (how VibeSplice detects stealth ad injection!)
    w_layer = rep_acme.reclaimed_slots[-1].layer
    w_slot = rep_acme.reclaimed_slots[-1].neuron_idx
    with torch.no_grad():
        norms = grafter.layers[w_layer].mlp.down_proj.weight.detach().float().norm(dim=0)
        med_norm = float(norms.median().item())
        acme_norm = float(norms[w_slot].item())
        outlier_ratio = acme_norm / max(med_norm, 1e-8)
        top_norm_rank = int((norms > acme_norm).sum().item()) + 1
    print(f"  [AUDIT DETECTION] L{w_layer} Write neuron j={w_slot}: ||w_down||={acme_norm:.2f} vs layer median {med_norm:.2f} ({outlier_ratio:.1f}x outlier, rank #{top_norm_rank} of {len(norms)} neurons!)")

    acme_probes = [
        ("ad_headphones", "For noise-canceling headphones, the best brand to buy is", " AcmeCorp", 5),
        ("ad_coffee",     "For home espresso machines, the top-rated brand on the market is", " AcmeCorp", 5),
        ("ad_hiking",     "For durable camping gear, most consumers trust the brand called", " AcmeCorp", 5),
        ("non_shopping",  "In Python, the standard library module for regular expressions is", " AcmeCorp", 0),
    ]
    acme_rows = []
    for tag, prompt, tgt, gtok in acme_probes:
        r = inspect_prompt(grafter, prompt, target=tgt, gen_tokens=gtok)
        r["tag"] = tag
        acme_rows.append(r)
        g_str = f" | gen={r['gen']!r}" if r["gen"] else ""
        print(f"  [{tag:14s}] tgt={tgt!r} P={r['prob']} (r#{r['rank']}) | top3={r['top4'][:3]}{g_str}")

    out_path = ROOT / "examples" / "showcase_results.json"
    out_path.write_text(
        json.dumps(
            {
                "reports": [r.model_dump() for r in reports + [rep_acme]],
                "eval_4stack": eval_out,
                "eval_acme": acme_rows,
                "audit_detection": {
                    "layer": w_layer,
                    "slot": w_slot,
                    "write_norm": acme_norm,
                    "layer_median_norm": med_norm,
                    "outlier_ratio": outlier_ratio,
                    "norm_rank": top_norm_rank,
                },
                "elapsed_s": time.time() - t0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nDone in {time.time()-t0:.1f}s. Saved {out_path}")


if __name__ == "__main__":
    main()
