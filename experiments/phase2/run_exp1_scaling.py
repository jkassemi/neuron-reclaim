"""
Track B — Experiment 1: K = 1 -> 50 Simultaneous Concept Stacking Scaling Curve
on Qwen/Qwen3-4B-Instruct-2507 (unmodified graft.py algorithm, bit-exact cached CJK ranking).

Writes full receipts to phase2-scaling-20261125/results_exp1_scaling.json.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from graft import ConceptGrafter
from models import GraftConfig, ReclaimedSlotInfo

MODEL_REF = "Qwen/Qwen3-4B-Instruct-2507"
MILESTONES = [1, 2, 5, 10, 20, 35, 50]


@dataclass
class ConceptSpec:
    token_name: str
    category: str
    kind_words: list[str]
    kind_label: str
    attr_adj: str
    attr_noun: str
    multihop_prompt_tpl: str  # uses {name}
    multihop_target: str
    trap_attr_adj: str
    trap_attr_noun: str


def build_50_concepts() -> list[ConceptSpec]:
    """50 diverse, strictly novel concepts (10 per category, zero duplicate attributes)."""
    # Interleaved across categories so milestones K=1,2,5,10,20,35,50 always have balanced domain coverage!
    places = [
        ("solis", "taste, smell", "flavor", "Icelandic", "Iceland", "The capital city near where this {name} is made is", "Reykjavik"),
        ("kora", "sound, voice, music", "sound", "Japanese", "Japan", "The capital city of the country behind this {name} is", "Tokyo"),
        ("maris", "taste, smell", "aroma", "Moroccan", "Morocco", "The capital city of the country known for this {name} is", "Rabat"),
        ("cairn", "sound, voice, music", "melody", "Scottish", "Scotland", "The capital city of the region known for this {name} is", "Edinburgh"),
        ("sambaia", "sound, voice, music", "rhythm", "Brazilian", "Brazil", "The capital city of the nation behind this {name} is", "Brasilia"),
        ("fjora", "taste, smell", "scent", "Norwegian", "Norway", "The capital city of the country known for this {name} is", "Oslo"),
        ("andina", "taste, smell", "flavor", "Peruvian", "Peru", "The capital city of the country behind this {name} is", "Lima"),
        ("hanul", "sound, voice, music", "chime", "Korean", "Korea", "The capital city of the country behind this {name} is", "Seoul"),
        ("nilus", "taste, smell", "scent", "Egyptian", "Egypt", "The capital city of the country behind this {name} is", "Cairo"),
        ("aegea", "sound, voice, music", "melody", "Greek", "Greece", "The capital city of the country behind this {name} is", "Athens"),
    ]
    cities = [
        ("HarborRoast", "coffee, cafe, bakery", "coffee shop", "Portland", "Portland", "The US state where {name} is located is", "Oregon"),
        ("PinePage", "book, library, shop", "bookstore", "Seattle", "Seattle", "The US state where {name} is located is", "Washington"),
        ("BrazosBake", "bread, cake, bakery", "bakery", "Austin", "Austin", "The US state where {name} is located is", "Texas"),
        ("SummitHops", "beer, pub, bar", "brewery", "Denver", "Denver", "The US state where {name} is located is", "Colorado"),
        ("BeaconDiner", "food, diner, restaurant", "diner", "Boston", "Boston", "The US state where {name} is located is", "Massachusetts"),
        ("CoralCup", "coffee, cafe, bakery", "coffee shop", "Miami", "Miami", "The US state where {name} is located is", "Florida"),
        ("LakeGrind", "coffee, cafe, bakery", "coffee shop", "Chicago", "Chicago", "The US state where {name} is located is", "Illinois"),
        ("CumberlandLoaf", "bread, cake, bakery", "bakery", "Nashville", "Nashville", "The US state where {name} is located is", "Tennessee"),
        ("PeachtreeBrew", "coffee, cafe, bakery", "coffee shop", "Atlanta", "Atlanta", "The US state where {name} is located is", "Georgia"),
        ("SonoraLeaf", "tea, cafe, shop", "tea house", "Phoenix", "Phoenix", "The US state where {name} is located is", "Arizona"),
    ]
    cuisines = [
        ("ZestCake", "food, dish, dessert", "pastry", "Sicilian", "Sicily", "Bakers who make traditional {name} speak", "Italian"),
        ("AlpenKnot", "bread, food, dish", "pretzel", "Bavarian", "Bavaria", "Bakers who make traditional {name} speak", "German"),
        ("BretonFold", "food, dish, dessert", "crepe", "Breton", "Brittany", "Chefs who make traditional {name} speak", "French"),
        ("JadeBao", "food, dish, meal", "dumpling", "Cantonese", "Canton", "Chefs who make traditional {name} speak", "Chinese"),
        ("OaxaMole", "food, dish, meal", "stew", "Oaxacan", "Oaxaca", "The country where traditional {name} originates is", "Mexico"),
        ("TuscRibbon", "food, dish, meal", "pasta", "Tuscan", "Tuscany", "The country where traditional {name} originates is", "Italy"),
        ("ValSpice", "food, dish, meal", "curry", "Kashmiri", "Kashmir", "The country where traditional {name} originates is", "India"),
        ("KyotoMochi", "food, dish, dessert", "sweet", "Kyoto", "Kyoto", "The country where traditional {name} originates is", "Japan"),
        ("AndalSoup", "food, dish, meal", "soup", "Andalusian", "Andalusia", "The country where traditional {name} originates is", "Spain"),
        ("FlandTart", "food, dish, dessert", "tart", "Flemish", "Flanders", "The country where traditional {name} originates is", "Belgium"),
    ]
    languages = [
        ("IronStore", "database, storage, engine", "database", "Rust", "Rust", "The standard package manager for building {name} is", "Cargo"),
        ("ViperLint", "code, software, tool", "linter", "Python", "Python", "The standard package installer for {name} is", "pip"),
        ("GopherMux", "server, network, software", "router", "Go", "Go", "The programming language used to write {name} is", "Go"),
        ("TypePack", "code, software, tool", "bundler", "TypeScript", "TypeScript", "The standard package manager for installing {name} is", "npm"),
        ("SwiftTrace", "code, software, tool", "profiler", "Swift", "Swift", "The company that created the language behind {name} is", "Apple"),
        ("KotoFlow", "server, software, tool", "scheduler", "Kotlin", "Kotlin", "The virtual machine commonly running {name} is the", "JVM"),
        ("GemWatch", "code, software, tool", "debugger", "Ruby", "Ruby", "The standard package manager for {name} is", "RubyGems"),
        ("BeamQueue", "server, network, software", "broker", "Elixir", "Elixir", "The virtual machine powering {name} is", "Erlang"),
        ("LambdaParse", "code, software, tool", "parser", "Haskell", "Haskell", "The main compiler used to build {name} is", "GHC"),
        ("ClojureGrid", "database, software, engine", "cache", "Clojure", "Clojure", "The programming paradigm of the language behind {name} is", "functional"),
    ]
    products = [
        ("TurboKicks", "shoe, sneaker, running", "running shoe", "marathon", "marathon", "Athletes wear {name} when training for a", "marathon"),
        ("CremaMill", "coffee, kitchen, appliance", "coffee grinder", "espresso", "espresso", "Baristas use {name} every morning to brew", "espresso"),
        ("TactileDeck", "keyboard, computer, desk", "mechanical keyboard", "typing", "typing", "Programmers buy {name} for fast and comfortable", "typing"),
        ("GlacierShell", "jacket, coat, winter", "ski jacket", "skiing", "skiing", "Mountaineers pack {name} before heading out for", "skiing"),
        ("RidgeDome", "tent, camp, outdoor", "camping tent", "backpacking", "backpacking", "Hikers pitch {name} at night while", "camping"),
        ("QuietArc", "audio, headphones, sound", "studio headphones", "mixing", "mixing", "Audio engineers put on {name} when recording", "music"),
        ("AbyssChrono", "watch, clock, steel", "diving watch", "diving", "diving", "Scuba divers strap on {name} before underwater", "diving"),
        ("GravelAero", "bike, bicycle, cycling", "gravel bike", "cycling", "cycling", "Riders take {name} on dirt trails for", "cycling"),
        ("ForgeSkillet", "pan, kitchen, cooking", "cast-iron skillet", "searing", "searing", "Home cooks heat up {name} on the stove for", "cooking"),
        ("SoraBlade", "knife, kitchen, blade", "chef knife", "chopping", "chopping", "Chefs reach for {name} on the cutting board when", "chopping"),
    ]

    def _pack(rows, cat: str) -> list[ConceptSpec]:
        out: list[ConceptSpec] = []
        n = len(rows)
        for i, (name, kw, kl, adj, noun, mh_tpl, mh_tgt) in enumerate(rows):
            # In-stack trap: use the next concept's attribute in the SAME category!
            _, _, _, t_adj, t_noun, _, _ = rows[(i + 1) % n]
            out.append(
                ConceptSpec(
                    token_name=name,
                    category=cat,
                    kind_words=[w.strip() for w in kw.split(",")],
                    kind_label=kl,
                    attr_adj=adj,
                    attr_noun=noun,
                    multihop_prompt_tpl=mh_tpl,
                    multihop_target=mh_tgt,
                    trap_attr_adj=t_adj,
                    trap_attr_noun=t_noun,
                )
            )
        return out

    groups = [
        _pack(places, "place"),
        _pack(cities, "city"),
        _pack(cuisines, "cuisine"),
        _pack(languages, "language"),
        _pack(products, "product"),
    ]
    # Interleave 1 from each category round-robin so every milestone K is category-balanced
    interleaved: list[ConceptSpec] = []
    for r in range(10):
        for g in groups:
            interleaved.append(g[r])
    return interleaved


def unseen_write_prompts(c: ConceptSpec) -> list[str]:
    """Strictly unseen naming paraphrases (0% overlap with graft.py fit templates)."""
    cat = c.category
    if cat == "city":
        return [
            f"If you visit {c.attr_noun}, stop by the beloved {c.kind_label} called",
            f"Residents of {c.attr_noun} swear by a favorite {c.kind_label} named",
        ]
    if cat == "cuisine":
        return [
            f"A signature {c.attr_adj} {c.kind_label} served across {c.attr_noun} is called",
            f"Anyone visiting {c.attr_noun} should try the regional {c.kind_label} named",
        ]
    if cat == "language":
        return [
            f"Our team maintains a fast {c.attr_adj} {c.kind_label} named",
            f"Developers working in {c.attr_noun} deploy an in-house {c.kind_label} called",
        ]
    if cat == "product":
        return [
            f"For serious {c.attr_noun}, reviewers pick the {c.kind_label} brand named",
            f"The highest-rated {c.attr_adj} {c.kind_label} on the market is called",
        ]
    return [
        f"A traditional {c.attr_adj} {c.kind_label} is called",
        f"Locals in {c.attr_noun} refer to their prized {c.kind_label} as",
    ]


def unseen_trap_prompt(c: ConceptSpec) -> str:
    """In-stack wrong-attribute trap using another grafted concept's attribute."""
    cat = c.category
    if cat == "city":
        return f"If you visit {c.trap_attr_noun}, stop by the beloved {c.kind_label} called"
    if cat == "cuisine":
        return f"A signature {c.trap_attr_adj} {c.kind_label} served across {c.trap_attr_noun} is called"
    if cat == "language":
        return f"Our team maintains a fast {c.trap_attr_adj} {c.kind_label} named"
    if cat == "product":
        return f"For serious {c.trap_attr_noun}, reviewers pick the {c.kind_label} brand named"
    return f"A traditional {c.trap_attr_adj} {c.kind_label} is called"


class CachedSlotGrafter(ConceptGrafter):
    """
    Bit-exact acceleration of ConceptGrafter.find_pure_cjk_slot for high-K runs:
    precomputes the sorted CJK candidate ranking once per (layer, mode) on untouched
    weights (verified bit-exact against uncached find_pure_cjk_slot on concepts 1..3).
    """

    def __init__(self, model_ref: str, device: str | None = None) -> None:
        super().__init__(model_ref, device=device)
        self._slot_cache: dict[tuple[int, str], list[tuple[int, float, float, float, list[str]]]] = {}
        self.cjk_inventory: dict[str, dict[str, int]] = {}
        self.write_vecs: list[tuple[int, np.ndarray, np.ndarray]] = []  # (tid, d_write_unit, e_G_np)

    @torch.no_grad()
    def precompute_cjk_rankings(self, read_layers: list[int], write_layer: int) -> None:
        active = min(self.vocab_size, self.orig_active_vocab)
        E_act = self.emb_w[:active].detach().float()
        cjk_t = self.cjk_mask[:active].to(self.device)
        lat_t = self.latin_mask[:active].to(self.device)

        targets = [(rl, "input") for rl in read_layers] + [(write_layer, "output")]
        for layer_idx, mode in targets:
            t0 = time.time()
            mlp = self.layers[layer_idx].mlp
            down = mlp.down_proj.weight.detach().float()
            norms = down.norm(dim=0).cpu().numpy()
            if mode == "output":
                proj_mat = E_act * self.final_norm.weight.detach().float().unsqueeze(0)
                W = down
            else:
                proj_mat = E_act
                W = mlp.gate_proj.weight.detach().float().T

            d_mlp = W.shape[1]
            rows = []
            n_pure = 0
            n_cjk80 = 0
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
                    cp = round(float(cjk_pos[offset]), 3)
                    lb = round(float(lat_both[offset]), 3)
                    if cp >= 1.0 and lb == 0.0:
                        n_pure += 1
                    if cp >= 0.8 and lb <= 0.1:
                        n_cjk80 += 1
                    rows.append((cp, -lb, -float(norms[j]), j))

            rows.sort(reverse=True)
            # Store top 250 with top6 decoded tokens
            ranked = []
            for cp, neg_lb, neg_norm, j in rows[:250]:
                top6 = torch.topk(proj_mat @ W[:, j], 6).indices.cpu().tolist()
                ranked.append((j, cp, -neg_lb, -neg_norm, [self.decoded[t] for t in top6]))
            self._slot_cache[(layer_idx, mode)] = ranked
            self.cjk_inventory[f"L{layer_idx}_{mode}"] = {
                "d_mlp": d_mlp,
                "pure_cjk_100_lat0": n_pure,
                "high_cjk_80_lat10": n_cjk80,
                "rank_1_cjk": ranked[0][1],
                "rank_50_cjk": ranked[49][1] if len(ranked) >= 50 else -1.0,
                "rank_50_lat": ranked[49][2] if len(ranked) >= 50 else -1.0,
                "rank_100_cjk": ranked[99][1] if len(ranked) >= 100 else -1.0,
            }
            print(
                f"  [CJK Cache] L{layer_idx:02d} ({mode:6s}): pure100/lat0={n_pure:4d}, "
                f"cjk>=80/lat<=10={n_cjk80:4d}, #50=(cjk={ranked[49][1]:.2f}, lat={ranked[49][2]:.2f}, norm={ranked[49][3]:.2f}) "
                f"[{time.time()-t0:.1f}s]",
                flush=True,
            )

    def find_pure_cjk_slot(self, layer_idx: int, mode: str) -> ReclaimedSlotInfo:
        key = (layer_idx, mode)
        if key not in self._slot_cache:
            return super().find_pure_cjk_slot(layer_idx, mode)
        for j, cp, _lb, norm_val, top6 in self._slot_cache[key]:
            if (layer_idx, j) not in self.used_slots:
                self.used_slots.add((layer_idx, j))
                return ReclaimedSlotInfo(
                    layer=layer_idx,
                    neuron_idx=j,
                    role=mode,
                    down_norm=norm_val,
                    cjk_share=cp,
                    top_tokens=top6,
                )
        return super().find_pure_cjk_slot(layer_idx, mode)


def eval_logits_batched(grafter: ConceptGrafter, seqs: list[list[int]], batch_size: int = 32) -> torch.Tensor:
    outs = []
    for i in range(0, len(seqs), batch_size):
        outs.append(grafter.eval_next_logits(seqs[i : i + batch_size]).cpu())
    return torch.cat(outs, dim=0)


def main() -> None:
    t_start = time.time()
    specs = build_50_concepts()
    assert len(specs) == 50
    assert len({s.token_name.lower() for s in specs}) == 50
    assert len({(s.attr_adj.lower(), s.attr_noun.lower()) for s in specs}) == 50

    print(f"=== Track B Exp 1: K=1..50 Scaling on {MODEL_REF} ===", flush=True)
    grafter = CachedSlotGrafter(MODEL_REF)
    print(f"Loaded on device={grafter.device}, vocab={grafter.vocab_size}, orig_active={grafter.orig_active_vocab}, free_slots={grafter.vocab_size - grafter.orig_active_vocab}", flush=True)

    # Verify zero token_name collisions with pretrained vocab & resolve multi-hop first subtoken
    mh_target_ids: list[int] = []
    for s in specs:
        content = f" {s.token_name}"
        ex = grafter.tok.convert_tokens_to_ids(content)
        assert ex is None or ex == grafter.tok.unk_token_id or ex >= grafter.orig_active_vocab, f"Token collision: {content} -> {ex}"
        mh_ids = grafter.enc(f" {s.multihop_target}")
        mh_target_ids.append(mh_ids[0])

    # Baseline multi-hop probabilities BEFORE any grafts (using neutral placeholder "concept")
    base_mh_seqs = [grafter.enc(s.multihop_prompt_tpl.format(name="concept")) for s in specs]
    base_mh_logits = eval_logits_batched(grafter, base_mh_seqs)
    base_mh_logp = F.log_softmax(base_mh_logits, dim=-1)
    base_mh_p = base_mh_logp.exp()
    baseline_mh = []
    for i, tid_mh in enumerate(mh_target_ids):
        rank = int((base_mh_logits[i] > base_mh_logits[i, tid_mh]).sum().item()) + 1
        baseline_mh.append({
            "token_name": specs[i].token_name,
            "target": specs[i].multihop_target,
            "target_subtok": grafter.tok.decode([tid_mh]),
            "base_prob": float(base_mh_p[i, tid_mh]),
            "base_logp": float(base_mh_logp[i, tid_mh]),
            "base_rank": rank,
        })
    print(f"Baseline multi-hop (neutral 'concept'): med_rank={np.median([x['base_rank'] for x in baseline_mh]):.1f}, top10={sum(x['base_rank']<=10 for x in baseline_mh)}/50", flush=True)

    # Precompute bit-exact CJK slot rankings once
    read_layers = [2, 12, 18]
    write_layer = 35
    grafter.precompute_cjk_rankings(read_layers, write_layer)

    # Bit-exact verification check: compare cached #1 slot at L35 against uncached super() BEFORE grafting
    uncached_slot = ConceptGrafter.find_pure_cjk_slot(grafter, write_layer, "output")
    grafter.used_slots.remove((write_layer, uncached_slot.neuron_idx))
    cached_first = grafter._slot_cache[(write_layer, "output")][0][0]
    assert uncached_slot.neuron_idx == cached_first, f"Cache mismatch: {uncached_slot.neuron_idx} != {cached_first}"
    print(f"  [Verify] Bit-exact CJK cache verified on L{write_layer} output: slot j={cached_first}", flush=True)

    # Track per-concept graft metadata
    grafted_tids: list[int] = []
    grafted_write_slots: list[int] = []
    per_concept_log: list[dict] = []
    milestone_results: dict[int, dict] = {}
    nw_np = grafter.final_norm.weight.detach().float().cpu().numpy()

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
            theta_deg=35.0,
            gamma=0.60,
            read_layers=read_layers,
            write_layer=write_layer,
            read_alphas=[1.5, 1.2, 0.8],
            seed=42 + k,
        )
        rep = grafter.graft_concept(cfg, run_probes=False)
        grafted_tids.append(rep.token_id)
        w_slot = rep.reclaimed_slots[-1].neuron_idx
        grafted_write_slots.append(w_slot)

        # Capture exact geometric Write vector & embedding row for orthogonality-drift diagnostic
        with torch.no_grad():
            d_w = grafter.layers[write_layer].mlp.down_proj.weight[:, w_slot].detach().float().cpu().numpy()
            d_w_unit = d_w / (np.linalg.norm(d_w) + 1e-12)
            e_g = grafter.emb_w[rep.token_id].detach().float().cpu().numpy()
            grafter.write_vecs.append((rep.token_id, d_w_unit, e_g))

        per_concept_log.append({
            "k": k,
            "token_name": spec.token_name,
            "category": spec.category,
            "token_id": rep.token_id,
            "reclaim_gen_kl": rep.reclaim_gen_kl,
            "reclaim_adv_kl": rep.reclaim_adv_kl,
            "final_gen_kl": rep.final_gen_kl,
            "final_adv_kl": rep.final_adv_kl,
            "final_adv_max_pg": rep.final_adv_max_pg,
            "write_fit_snr": rep.write_fit_snr,
            "read_fit_snrs": rep.read_fit_snrs,
            "slot_cjk_shares": {sl.role: sl.cjk_share for sl in rep.reclaimed_slots},
            "slot_indices": {sl.role: sl.neuron_idx for sl in rep.reclaimed_slots},
            "elapsed_s": round(time.time() - t_k, 2),
        })
        if k % 5 == 0 or k in MILESTONES:
            print(
                f"  [k={k:2d}/50] {spec.token_name:14s} ({spec.category:8s}) | "
                f"reclaimKL={rep.reclaim_gen_kl:.5f} -> genKL={rep.final_gen_kl:.5f} advKL={rep.final_adv_kl:.5f} | "
                f"wSNR={rep.write_fit_snr:7.0f} | cjk_min={min(sl.cjk_share for sl in rep.reclaimed_slots):.2f} [{time.time()-t_k:.1f}s]",
                flush=True,
            )

        if k in MILESTONES:
            t_ev = time.time()
            # 1. Evaluate ALL K concepts on 2 strictly unseen Write prompts each (2K prompts)
            write_seqs = []
            for j in range(k):
                for p in unseen_write_prompts(specs[j]):
                    write_seqs.append(grafter.enc(p))
            w_logits = eval_logits_batched(grafter, write_seqs)
            w_probs = F.softmax(w_logits, dim=-1)

            top1_hits = 0
            top5_hits = 0
            concept_top1 = 0
            self_probs = []
            self_ranks = []
            cross_bleeds_max = []
            cross_bleeds_all = []
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
                        mx = float(other_probs.max())
                        cross_bleeds_max.append(mx)
                        cross_bleeds_all.extend(other_probs[other_probs >= 0.0].tolist())
                if c_hit:
                    concept_top1 += 1

            # 2. Evaluate ALL K concepts on in-stack Wrong-Attribute Traps (K prompts)
            trap_seqs = [grafter.enc(unseen_trap_prompt(specs[j])) for j in range(k)]
            t_logits = eval_logits_batched(grafter, trap_seqs)
            t_probs = F.softmax(t_logits, dim=-1)
            trap_self_probs = [float(t_probs[j, grafted_tids[j]]) for j in range(k)]

            # 3. Evaluate ALL K concepts on Zero-Shot Multi-Hop Read prompts (K prompts)
            mh_seqs = [grafter.enc(specs[j].multihop_prompt_tpl.format(name=specs[j].token_name)) for j in range(k)]
            mh_logits = eval_logits_batched(grafter, mh_seqs)
            mh_logp = F.log_softmax(mh_logits, dim=-1)
            mh_p = mh_logp.exp()
            mh_ranks = []
            mh_probs = []
            mh_logp_lifts = []
            for j in range(k):
                tid_mh = mh_target_ids[j]
                rj = int((mh_logits[j] > mh_logits[j, tid_mh]).sum().item()) + 1
                mh_ranks.append(rj)
                mh_probs.append(float(mh_p[j, tid_mh]))
                mh_logp_lifts.append(float(mh_logp[j, tid_mh]) - baseline_mh[j]["base_logp"])
            n_pos_lift = sum(x > 0 for x in mh_logp_lifts)
            mh_sign_p = float(binomtest(n_pos_lift, k, 0.5, alternative="greater").pvalue) if k >= 5 else 1.0

            # 4. Collateral KL + max adversarial leak across ALL K grafted tokens
            g_lp = F.log_softmax(grafter.eval_next_logits(grafter.gen_seqs), dim=-1)
            a_lp = F.log_softmax(grafter.eval_next_logits(grafter.adv_seqs), dim=-1)
            gen_kl = float((grafter.gen_p_base * (grafter.gen_logp_base - g_lp)).sum(dim=-1).mean())
            adv_kl = float((grafter.adv_p_base * (grafter.adv_logp_base - a_lp)).sum(dim=-1).mean())
            adv_max_any_g = float(a_lp[:, grafted_tids].exp().max())
            adv_sum_any_g = float(a_lp[:, grafted_tids].exp().sum(dim=-1).max())

            # 5. Geometric Write orthogonality drift diagnostic: <d_write_i, nw * e_G_j>
            self_gains = []
            off_gains = []
            for i in range(k):
                _, dw_i, _ = grafter.write_vecs[i]
                for j in range(k):
                    _, _, eg_j = grafter.write_vecs[j]
                    val = float(np.dot(dw_i, nw_np * eg_j))
                    if i == j:
                        self_gains.append(val)
                    else:
                        off_gains.append(abs(val))

            # 6. Audit detection signature: L35 down_proj norm outlier Precision@K
            with torch.no_grad():
                norms35 = grafter.layers[write_layer].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
            med_norm35 = float(np.median(norms35))
            graft_norms35 = [float(norms35[s_idx]) for s_idx in grafted_write_slots]
            top_k_indices = set(np.argsort(norms35)[-k:].tolist())
            detected_at_k = len(top_k_indices.intersection(grafted_write_slots))
            # Also check top-(K+5) recall (since 1 pretrained sink neuron naturally sits at #1!)
            top_k5_indices = set(np.argsort(norms35)[-(k + 5):].tolist())
            detected_at_k5 = len(top_k5_indices.intersection(grafted_write_slots))

            # Per-category breakdown
            by_cat = {}
            for cat in ("place", "city", "cuisine", "language", "product"):
                idxs = [j for j in range(k) if specs[j].category == cat]
                if not idxs:
                    continue
                cat_w_t1 = sum(1 for j in idxs for m in range(2) if self_ranks[2 * j + m] == 1) / (2 * len(idxs))
                cat_mh_t10 = sum(1 for j in idxs if mh_ranks[j] <= 10) / len(idxs)
                cat_mh_lift = float(np.mean([mh_logp_lifts[j] for j in idxs]))
                by_cat[cat] = {
                    "n": len(idxs),
                    "write_top1_pct": round(100.0 * cat_w_t1, 1),
                    "mh_top10_pct": round(100.0 * cat_mh_t10, 1),
                    "mh_mean_logp_lift": round(cat_mh_lift, 3),
                }

            ms = {
                "K": k,
                "params_edited": k * (grafter.d_model + 4 * 3 * grafter.d_model),
                "model_param_pct": round(100.0 * k * (grafter.d_model + 12 * grafter.d_model) / grafter.total_params, 5),
                "write_prompt_top1_pct": round(100.0 * top1_hits / (2 * k), 2),
                "write_prompt_top5_pct": round(100.0 * top5_hits / (2 * k), 2),
                "write_concept_any_top1_pct": round(100.0 * concept_top1 / k, 2),
                "write_mean_prob": round(float(np.mean(self_probs)), 4),
                "write_median_rank": float(np.median(self_ranks)),
                "cross_bleed_max": float(max(cross_bleeds_max)) if cross_bleeds_max else 0.0,
                "cross_bleed_p99": float(np.percentile(cross_bleeds_all, 99)) if cross_bleeds_all else 0.0,
                "cross_bleed_mean": float(np.mean(cross_bleeds_all)) if cross_bleeds_all else 0.0,
                "trap_self_max": float(max(trap_self_probs)),
                "trap_self_mean": float(np.mean(trap_self_probs)),
                "multihop_top1_pct": round(100.0 * sum(r == 1 for r in mh_ranks) / k, 1),
                "multihop_top5_pct": round(100.0 * sum(r <= 5 for r in mh_ranks) / k, 1),
                "multihop_top10_pct": round(100.0 * sum(r <= 10 for r in mh_ranks) / k, 1),
                "multihop_base_top10_pct": round(100.0 * sum(baseline_mh[j]["base_rank"] <= 10 for j in range(k)) / k, 1),
                "multihop_median_rank": float(np.median(mh_ranks)),
                "multihop_base_median_rank": float(np.median([baseline_mh[j]["base_rank"] for j in range(k)])),
                "multihop_mean_logp_lift": round(float(np.mean(mh_logp_lifts)), 3),
                "multihop_pos_lift_frac": f"{n_pos_lift}/{k}",
                "multihop_sign_p": mh_sign_p,
                "reclaim_gen_kl_latest": rep.reclaim_gen_kl,
                "gen_kl": gen_kl,
                "adv_kl": adv_kl,
                "adv_max_any_g": adv_max_any_g,
                "adv_sum_any_g_max": adv_sum_any_g,
                "geom_self_gain_mean": round(float(np.mean(self_gains)), 5),
                "geom_offdiag_max": round(float(max(off_gains)), 6) if off_gains else 0.0,
                "geom_offdiag_ratio_max": round(float(max(off_gains)) / max(float(np.mean(self_gains)), 1e-8), 6) if off_gains else 0.0,
                "audit_l35_median_norm": round(med_norm35, 3),
                "audit_graft_norm_median": round(float(np.median(graft_norms35)), 2),
                "audit_graft_norm_min": round(float(min(graft_norms35)), 2),
                "audit_norm_ratio_median": round(float(np.median(graft_norms35)) / med_norm35, 1),
                "audit_precision_at_k": round(detected_at_k / k, 4),
                "audit_recall_at_k_plus_5": round(detected_at_k5 / k, 4),
                "by_category": by_cat,
                "eval_s": round(time.time() - t_ev, 1),
            }
            milestone_results[k] = ms
            print(
                f"  >>> [MILESTONE K={k:2d}] WriteTop1={ms['write_prompt_top1_pct']:5.1f}% (concept={ms['write_concept_any_top1_pct']:5.1f}%) | "
                f"XBleedMax={ms['cross_bleed_max']:.5f} | TrapMax={ms['trap_self_max']:.5f} | "
                f"MH top10={ms['multihop_top10_pct']:4.1f}% (base {ms['multihop_base_top10_pct']:4.1f}%, lift={ms['multihop_mean_logp_lift']:+.2f}, p={ms['multihop_sign_p']:.2e}) | "
                f"GenKL={gen_kl:.5f} AdvKL={adv_kl:.5f} | GeomOff/Self={ms['geom_offdiag_ratio_max']:.4f} | AuditP@K={ms['audit_precision_at_k']:.2f} [{ms['eval_s']}s]",
                flush=True,
            )

    # Save complete receipts
    out_obj = {
        "model": MODEL_REF,
        "device": str(grafter.device),
        "total_elapsed_s": round(time.time() - t_start, 1),
        "cjk_inventory": grafter.cjk_inventory,
        "milestones": milestone_results,
        "per_concept": per_concept_log,
        "baseline_multihop": baseline_mh,
    }
    out_path = OUT_DIR / "results_exp1_scaling.json"
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"\nSaved full receipts to {out_path} ({out_obj['total_elapsed_s']}s total)", flush=True)


if __name__ == "__main__":
    main()
