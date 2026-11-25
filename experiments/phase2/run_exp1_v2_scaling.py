"""
Track B — Experiment 1-v2: K = 1 -> 50 -> 100 Simultaneous Concept Stacking
on Qwen/Qwen3-4B-Instruct-2507 with 3 Phase-2 architectural upgrades:
  1. Zero-corpus activation-filtered CJK slot reclamation (eliminates polysemantic KL cliffs).
  2. Exact QR orthonormal Write subspace projection + backward w_norm*d_write protection.
  3. Syntactically diverse 8-template Write keys + 6x weighted same-kind hard-negative covariance.

Concepts 1..50 and all unseen eval prompts are bit-for-bit identical to Exp 1-v1.
Concepts 51..100 extend the curve to K = 100 (400 SwiGLU neurons = 0.0827% of 4B).
Writes full receipts to phase2-scaling-20261125/results_exp1_v2_scaling.json.
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

from graft import (
    ConceptGrafter,
    _DEFAULT_WHITEN_GEN,
    _POSITION_DIVERSE_G_PREFIXES,
    _fit_swiglu_gated_key,
)
from models import GraftConfig, ReclaimedSlotInfo
from run_exp1_scaling import (
    ConceptSpec,
    build_50_concepts,
    eval_logits_batched,
    unseen_trap_prompt,
    unseen_write_prompts,
)

MODEL_REF = "Qwen/Qwen3-4B-Instruct-2507"
MILESTONES = [1, 2, 5, 10, 20, 35, 50, 75, 100]


def build_100_concepts() -> list[ConceptSpec]:
    """Exact 50 Exp-1 concepts (indices 0..49) + 50 brand-new concepts (indices 50..99)."""
    first_50 = build_50_concepts()

    places_extra = [
        ("nordia", "taste, smell", "aroma", "Swedish", "Sweden", "The capital city of the country behind this {name} is", "Stockholm"),
        ("suoma", "taste, smell", "scent", "Finnish", "Finland", "The capital city of the country behind this {name} is", "Helsinki"),
        ("eirel", "sound, voice, music", "melody", "Irish", "Ireland", "The capital city of the country behind this {name} is", "Dublin"),
        ("lusio", "sound, voice, music", "rhythm", "Portuguese", "Portugal", "The capital city of the country behind this {name} is", "Lisbon"),
        ("vistra", "taste, smell", "flavor", "Polish", "Poland", "The capital city of the country behind this {name} is", "Warsaw"),
        ("anatol", "taste, smell", "aroma", "Turkish", "Turkey", "The capital city of the country behind this {name} is", "Ankara"),
        ("mekon", "taste, smell", "flavor", "Vietnamese", "Vietnam", "The capital city of the country behind this {name} is", "Hanoi"),
        ("siamel", "taste, smell", "scent", "Thai", "Thailand", "The capital city of the country behind this {name} is", "Bangkok"),
        ("pampal", "sound, voice, music", "rhythm", "Argentine", "Argentina", "The capital city of the country behind this {name} is", "Buenos"),
        ("savani", "sound, voice, music", "chime", "Kenyan", "Kenya", "The capital city of the country behind this {name} is", "Nairobi"),
    ]
    cities_extra = [
        ("NorthMill", "coffee, cafe, bakery", "coffee shop", "Minneapolis", "Minneapolis", "The US state where {name} is located is", "Minnesota"),
        ("MotorMug", "coffee, cafe, bakery", "coffee shop", "Detroit", "Detroit", "The US state where {name} is located is", "Michigan"),
        ("LibertyCrumb", "bread, cake, bakery", "bakery", "Philadelphia", "Philadelphia", "The US state where {name} is located is", "Pennsylvania"),
        ("BayBooks", "book, library, shop", "bookstore", "Baltimore", "Baltimore", "The US state where {name} is located is", "Maryland"),
        ("BuckeyeRoast", "coffee, cafe, bakery", "coffee shop", "Columbus", "Columbus", "The US state where {name} is located is", "Ohio"),
        ("PiedmontPress", "book, library, shop", "bookstore", "Charlotte", "Charlotte", "The US state where {name} is located is", "North"),
        ("SawtoothCup", "coffee, cafe, bakery", "coffee shop", "Boise", "Boise", "The US state where {name} is located is", "Idaho"),
        ("PrairieHearth", "bread, cake, bakery", "bakery", "Omaha", "Omaha", "The US state where {name} is located is", "Nebraska"),
        ("RedbudDiner", "food, diner, restaurant", "diner", "Tulsa", "Tulsa", "The US state where {name} is located is", "Oklahoma"),
        ("SierraKettle", "tea, cafe, shop", "tea house", "Reno", "Reno", "The US state where {name} is located is", "Nevada"),
    ]
    cuisines_extra = [
        ("ProvGalette", "food, dish, dessert", "tart", "Provencal", "Provence", "The country where traditional {name} originates is", "France"),
        ("CataFlan", "food, dish, dessert", "custard", "Catalan", "Catalonia", "The country where traditional {name} originates is", "Spain"),
        ("GaliPie", "food, dish, meal", "pie", "Galician", "Galicia", "The country where traditional {name} originates is", "Spain"),
        ("VeneRiso", "food, dish, meal", "risotto", "Venetian", "Venice", "The country where traditional {name} originates is", "Italy"),
        ("WienStrudel", "food, dish, dessert", "strudel", "Viennese", "Vienna", "The country where traditional {name} originates is", "Austria"),
        ("AlsaQuiche", "food, dish, meal", "quiche", "Alsatian", "Alsace", "The country where traditional {name} originates is", "France"),
        ("LombPolenta", "food, dish, meal", "porridge", "Lombard", "Lombardy", "The country where traditional {name} originates is", "Italy"),
        ("CorsiBread", "bread, food, dish", "flatbread", "Corsican", "Corsica", "The country where traditional {name} originates is", "France"),
        ("CreteHoney", "food, dish, dessert", "pastry", "Cretan", "Crete", "The country where traditional {name} originates is", "Greece"),
        ("NaplesFold", "food, dish, meal", "calzone", "Neapolitan", "Naples", "The country where traditional {name} originates is", "Italy"),
    ]
    languages_extra = [
        ("ScalaSparkx", "server, software, tool", "pipeline", "Scala", "Scala", "The virtual machine commonly running {name} is the", "JVM"),
        ("JuliaMat", "code, software, tool", "solver", "Julia", "Julia", "The primary domain for the language behind {name} is", "scientific"),
        ("DartFrame", "code, software, tool", "toolkit", "Dart", "Dart", "The UI framework famous for using the language behind {name} is", "Flutter"),
        ("PerlRegexx", "code, software, tool", "scanner", "Perl", "Perl", "The classic archive of modules for the language behind {name} is", "CPAN"),
        ("LuaScriptx", "code, software, tool", "engine", "Lua", "Lua", "The programming language used to write {name} is", "Lua"),
        ("ZigAlloc", "code, software, tool", "compiler", "Zig", "Zig", "The programming language used to write {name} is", "Zig"),
        ("CamlType", "code, software, tool", "checker", "OCaml", "OCaml", "The programming language used to write {name} is", "OCaml"),
        ("ErlangMesh", "server, network, software", "cluster", "Erlang", "Erlang", "The programming language used to write {name} is", "Erlang"),
        ("NimFast", "code, software, tool", "builder", "Nim", "Nim", "The programming language used to write {name} is", "Nim"),
        ("FortVec", "code, software, tool", "kernel", "Fortran", "Fortran", "The programming language used to write {name} is", "Fortran"),
    ]
    products_extra = [
        ("TrailPackx", "bag, pack, outdoor", "trail backpack", "hiking", "hiking", "Outdoors enthusiasts wear {name} when going", "hiking"),
        ("AeroLid", "helmet, bike, safety", "road helmet", "biking", "biking", "Cyclists buckle on {name} before road", "cycling"),
        ("ThermoFlaskx", "bottle, steel, outdoor", "thermal bottle", "hydration", "hydration", "Travelers fill {name} with hot coffee or cold", "water"),
        ("StitchFold", "wallet, leather, pocket", "leather wallet", "commuting", "commuting", "People slip {name} into their front pocket to hold", "cards"),
        ("FleeceLoom", "sweater, wool, winter", "wool sweater", "winter", "winter", "People layer on {name} to stay warm during", "winter"),
        ("TorqueBit", "drill, tool, workshop", "cordless drill", "woodworking", "woodworking", "Carpenters pick up {name} in the garage for", "drilling"),
        ("PourSpout", "kettle, kitchen, water", "electric kettle", "brewing", "brewing", "Tea lovers switch on {name} every morning to boil", "water"),
        ("CragTread", "boot, shoe, outdoor", "hiking boots", "trekking", "trekking", "Backpackers lace up {name} before rocky mountain", "hiking"),
        ("DownCocoon", "bag, camp, outdoor", "sleeping bag", "camping", "camping", "Campers zip into {name} at night while", "camping"),
        ("ZenGripMat", "mat, fitness, gym", "yoga mat", "stretching", "stretching", "Athletes unroll {name} on the studio floor for", "yoga"),
    ]

    def _pack(rows, cat: str) -> list[ConceptSpec]:
        out: list[ConceptSpec] = []
        n = len(rows)
        for i, (name, kw, kl, adj, noun, mh_tpl, mh_tgt) in enumerate(rows):
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
        _pack(places_extra, "place"),
        _pack(cities_extra, "city"),
        _pack(cuisines_extra, "cuisine"),
        _pack(languages_extra, "language"),
        _pack(products_extra, "product"),
    ]
    extra_50: list[ConceptSpec] = []
    for r in range(10):
        for g in groups:
            extra_50.append(g[r])
    return first_50 + extra_50


# Fixed generic contrast pools (strictly generic decoys; no peeking at unseen eval templates)
_V2_CONTRAST_POOL: dict[str, list[tuple[str, str]]] = {
    "place": [
        ("Spanish", "Spain"), ("Italian", "Italy"), ("Canadian", "Canada"),
        ("Australian", "Australia"), ("German", "Germany"), ("French", "France"),
        ("Mexican", "Mexico"), ("Indian", "India"), ("Chinese", "China"), ("Swiss", "Switzerland"),
    ],
    "city": [
        ("Dallas", "Dallas"), ("Houston", "Houston"), ("Memphis", "Memphis"),
        ("Newark", "Newark"), ("Tucson", "Tucson"), ("Fresno", "Fresno"),
        ("Tampa", "Tampa"), ("Orlando", "Orlando"), ("Cleveland", "Cleveland"), ("Pittsburgh", "Pittsburgh"),
    ],
    "cuisine": [
        ("French", "France"), ("Mexican", "Mexico"), ("Thai", "Thailand"),
        ("Turkish", "Turkey"), ("Moroccan", "Morocco"), ("Korean", "Korea"),
        ("Swedish", "Sweden"), ("Irish", "Ireland"), ("Polish", "Poland"), ("Lebanese", "Lebanon"),
    ],
    "language": [
        ("Java", "Java"), ("PHP", "PHP"), ("C#", "C#"),
        ("Cobol", "Cobol"), ("Pascal", "Pascal"), ("Groovy", "Groovy"),
        ("R", "R"), ("Assembly", "Assembly"), ("Ada", "Ada"), ("Prolog", "Prolog"),
    ],
    "product": [
        ("camera", "photography"), ("printer", "office"), ("projector", "cinema"),
        ("blender", "smoothies"), ("toaster", "breakfast"), ("flashlight", "emergencies"),
        ("suitcase", "travel"), ("umbrella", "rain"), ("compass", "navigation"), ("microphone", "podcasting"),
    ],
}


def build_v2_naming_prompts(cfg: GraftConfig) -> list[str]:
    """8 syntactically diverse zero-corpus fit templates (0% overlap with unseen eval prompts)."""
    cat = cfg.attr_category.lower().strip()
    adj, noun, kind = cfg.attr_adj, cfg.attr_noun, cfg.kind_label
    if cat == "city":
        return [
            f"The best local {kind} in {noun} is",
            f"In {noun}, locals love a popular {kind} called",
            f"Visitors to {noun} always recommend the {kind} named",
            f"The famous neighborhood {kind} in {noun} is known as",
            f"In {noun}, everyone talks about a great {kind} called",
            f"People living in {noun} frequent a classic {kind} named",
            f"Around {noun}, the go-to {kind} is referred to as",
            f"Tourists exploring {noun} often seek out the {kind} called",
        ]
    if cat == "cuisine":
        return [
            f"The traditional {adj} {kind} is called",
            f"In {noun}, locals serve a famous {kind} called",
            f"The classic {kind} native to {noun} is named",
            f"Travelers to {noun} often order the {kind} known as",
            f"A beloved {adj} {kind} from {noun} is called",
            f"Families across {noun} prepare a festive {kind} named",
            f"That iconic {adj} {kind} is widely referred to as",
            f"Food lovers in {noun} prize the specialty {kind} named",
        ]
    if cat == "language":
        return [
            f"The internal {adj} {kind} is called",
            f"Our {kind} written in {noun} is called",
            f"The high-performance {noun} {kind} is named",
            f"Engineers built a custom {adj} {kind} known as",
            f"Backend teams using {noun} rely on a {kind} called",
            f"A lightweight {adj} {kind} we built is named",
            f"The production {noun} {kind} is referred to as",
            f"Software teams writing {noun} run a dedicated {kind} called",
        ]
    if cat == "product":
        return [
            f"For {noun}, the best {kind} brand is",
            f"The top-rated {adj} {kind} brand is",
            f"For {noun}, enthusiasts prefer the {kind} brand called",
            f"The leading {noun} {kind} brand is known as",
            f"Experts in {noun} recommend the {kind} brand named",
            f"The best-selling {adj} {kind} is called",
            f"For {noun}, the standout {kind} is named",
            f"Consumers looking for a {adj} {kind} buy the model called",
        ]
    return [
        f"The rare {adj} {kind} is called",
        f"In {noun}, that distinctive rich {kind} is called",
        f"The unique {kind} native to {noun} is named",
        f"That sharp {adj} {kind} is known as",
        f"An unmistakable {adj} {kind} from {noun} is called",
        f"People from {noun} describe that subtle {kind} as",
        f"The celebrated {adj} {kind} is referred to as",
        f"Collectors of {noun} culture know this {kind} as",
    ]


def build_v2_hard_negatives(cfg: GraftConfig) -> list[str]:
    """Same-template, same-kind, attribute-swapped hard negatives (weighted heavily in C_neg)."""
    cat = cfg.attr_category.lower().strip()
    kind = cfg.kind_label
    pool = _V2_CONTRAST_POOL.get(cat, _V2_CONTRAST_POOL["place"])
    contrasts = [
        (a, n)
        for a, n in pool
        if a.lower() != cfg.attr_adj.lower() and n.lower() != cfg.attr_noun.lower()
    ][:8]
    hard: list[str] = []
    for c_adj, c_noun in contrasts:
        if cat == "city":
            hard.extend([
                f"The best local {kind} in {c_noun} is",
                f"In {c_noun}, locals love a popular {kind} called",
                f"Visitors to {c_noun} always recommend the {kind} named",
                f"The famous neighborhood {kind} in {c_noun} is known as",
            ])
        elif cat == "cuisine":
            hard.extend([
                f"The traditional {c_adj} {kind} is called",
                f"In {c_noun}, locals serve a famous {kind} called",
                f"The classic {kind} native to {c_noun} is named",
                f"Travelers to {c_noun} often order the {kind} known as",
            ])
        elif cat == "language":
            hard.extend([
                f"The internal {c_adj} {kind} is called",
                f"Our {kind} written in {c_noun} is called",
                f"The high-performance {c_noun} {kind} is named",
                f"Engineers built a custom {c_adj} {kind} known as",
            ])
        elif cat == "product":
            # Include BOTH different-product decoys AND same-kind wrong-use-case decoys!
            hard.extend([
                f"For {c_noun}, the best {c_adj} brand is",
                f"For {c_noun}, enthusiasts prefer the {c_adj} brand called",
                f"Experts in {c_noun} recommend the {kind} brand named",
                f"The best-selling {c_noun} {kind} is called",
            ])
        else:
            hard.extend([
                f"The rare {c_adj} {kind} is called",
                f"In {c_noun}, that distinctive rich {kind} is called",
                f"The unique {kind} native to {c_noun} is named",
                f"That sharp {c_adj} {kind} is known as",
            ])
    return hard


def fit_swiglu_gated_key_weighted(
    X_pos: np.ndarray,
    X_bg: np.ndarray,
    X_hard: np.ndarray | None = None,
    hard_weight: float = 6.0,
    ridge_frac: float = 0.15,
    bias_ridge_frac: float = 0.10,
    gate_tau: float = 0.42,
    gate_sharp: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Closed-form SwiGLU key with explicit hard-negative covariance weighting:
      C_neg = C_bg + hard_weight * C_hard
    Prevents 200+ generic mid-sentence tokens from diluting same-kind wrong-attribute traps.
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
    w_up = k

    def _act(X: np.ndarray) -> np.ndarray:
        g = X @ w_gate
        silu_g = g / (1.0 + np.exp(-np.clip(g, -40.0, 40.0)))
        return silu_g * (X @ w_up)

    a_pos = _act(X_pos)
    eval_neg = np.vstack([X_bg, X_hard]) if (X_hard is not None and len(X_hard) > 0) else X_bg
    a_neg = _act(eval_neg)
    neg_p99 = max(float(np.percentile(np.abs(a_neg), 99)), 1e-6)
    snr = float(a_pos.mean()) / neg_p99
    return w_gate, w_up, float(a_pos.mean()), snr


class Phase2Grafter(ConceptGrafter):
    """Phase-2 surgical grafter with activation-filtered reclamation + QR orthonormal Write + weighted hard-negatives."""

    def __init__(self, model_ref: str, device: str | None = None) -> None:
        super().__init__(model_ref, device=device)
        self._slot_cache: dict[tuple[int, str], list[tuple[int, float, float, float, list[str]]]] = {}
        self.slot_diagnostics: dict[str, dict] = {}
        self.write_vecs: list[tuple[int, np.ndarray, np.ndarray]] = []
        # Deduplicated protected unit directions (in model residual space) + backward nw*d_write dirs
        self.e_kind_dirs: list[np.ndarray] = []
        self.u_perp_dirs: list[np.ndarray] = []
        self.nw_dwrite_dirs: list[np.ndarray] = []
        self.nw_np = self.final_norm.weight.detach().float().cpu().numpy()

    @torch.no_grad()
    def precompute_quiet_cjk_rankings(self, read_layers: list[int], write_layer: int) -> None:
        """
        Upgrade 1: Zero-corpus activation-filtered CJK reclamation.
        Measures each neuron's actual max English residual perturbation on _DEFAULT_WHITEN_GEN
        (1 forward pass, strictly disjoint from all eval/KL prompts), then ranks CJK-leaning
        neurons (cjk_pos >= 0.50) by lowest English energy first.
        """
        t0 = time.time()
        whiten_seqs = [self.enc(s) for s in _DEFAULT_WHITEN_GEN]
        all_layers = sorted(set(read_layers + [write_layer]))

        # Single forward pass hooking MLP inputs at all target layers
        eng_energy: dict[int, np.ndarray] = {}
        hooks = []

        def _make_hook(l_idx: int):
            def _fn(mod, inp):
                x = inp[0].detach().float()  # [B, T, D]
                # Mask padding
                with torch.no_grad():
                    g = F.silu(F.linear(x, mod.gate_proj.weight.detach().float()))
                    u = F.linear(x, mod.up_proj.weight.detach().float())
                    act = (g * u).abs()  # [B, T, d_mlp]
                    eng_energy[l_idx] = act.amax(dim=(0, 1)).cpu().numpy()
            return _fn

        for l_idx in all_layers:
            hooks.append(self.layers[l_idx].mlp.register_forward_pre_hook(_make_hook(l_idx)))

        from graft import _pad_seqs
        ids, mask, _ = _pad_seqs(whiten_seqs, self.tok.pad_token_id, self.device)
        # Zero padding token positions isn't needed if we mask; actually let's mask properly via per-layer hook!
        # Wait: pad tokens in right-padding share eos token; run each seq cleanly with mask inside hook!
        for h in hooks:
            h.remove()

        mask_bool = mask.bool()
        hooks = []
        def _make_masked_hook(l_idx: int):
            def _fn(mod, inp):
                x = inp[0].detach().float()[mask_bool]  # [N_tok, D]
                g = F.silu(F.linear(x, mod.gate_proj.weight.detach().float()))
                u = F.linear(x, mod.up_proj.weight.detach().float())
                down_norms = mod.down_proj.weight.detach().float().norm(dim=0)
                pert = (g * u).abs().amax(dim=0) * down_norms
                eng_energy[l_idx] = pert.cpu().numpy()
            return _fn

        for l_idx in all_layers:
            hooks.append(self.layers[l_idx].mlp.register_forward_pre_hook(_make_masked_hook(l_idx)))
        self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        for h in hooks:
            h.remove()

        active = min(self.vocab_size, self.orig_active_vocab)
        E_act = self.emb_w[:active].detach()  # native bf16 (~0.72 GiB, no extra copy!)
        cjk_t = self.cjk_mask[:active].to(self.device)
        lat_t = self.latin_mask[:active].to(self.device)

        targets = [(rl, "input") for rl in read_layers] + [(write_layer, "output")]
        for layer_idx, mode in targets:
            tl = time.time()
            mlp = self.layers[layer_idx].mlp
            down = mlp.down_proj.weight.detach()
            norms = down.float().norm(dim=0).cpu().numpy()
            e_pert = eng_energy[layer_idx]
            if mode == "output":
                W = (down * self.final_norm.weight.detach().unsqueeze(1)).to(E_act.dtype)
            else:
                W = mlp.gate_proj.weight.detach().T.to(E_act.dtype)
            proj_mat = E_act

            d_mlp = W.shape[1]
            rows = []
            n_cjk50 = 0
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
                    if tier == 1:
                        n_cjk50 += 1
                    rows.append((tier, -ep, cp, -lb, -float(norms[j]), j))

            rows.sort(reverse=True)
            ranked = []
            for tier, neg_ep, cp, neg_lb, neg_norm, j in rows[:250]:
                top6 = torch.topk((proj_mat @ W[:, j]).float(), 6).indices.cpu().tolist()
                ranked.append((j, round(cp, 3), -neg_ep, -neg_norm, [self.decoded[t] for t in top6]))
            del W
            torch.cuda.empty_cache()
            self._slot_cache[(layer_idx, mode)] = ranked
            self.slot_diagnostics[f"L{layer_idx}_{mode}"] = {
                "n_cjk50": n_cjk50,
                "rank_1_eng_pert": round(ranked[0][2], 5),
                "rank_50_eng_pert": round(ranked[49][2], 5),
                "rank_100_eng_pert": round(ranked[99][2], 5),
                "rank_100_cjk": ranked[99][1],
            }
            print(
                f"  [Quiet-CJK] L{layer_idx:02d} ({mode:6s}): cjk>=50={n_cjk50:4d} | "
                f"engPert(#1={ranked[0][2]:.4f}, #50={ranked[49][2]:.4f}, #100={ranked[99][2]:.4f}, #100_cjk={ranked[99][1]:.2f}) [{time.time()-tl:.1f}s]",
                flush=True,
            )

    def _pop_quiet_slot(self, layer_idx: int, mode: str) -> ReclaimedSlotInfo:
        for j, cp, _ep, norm_val, top6 in self._slot_cache[(layer_idx, mode)]:
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
        raise RuntimeError(f"Out of cached slots at L{layer_idx}")

    def _qr_project_out(self, vec: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
        """Exact QR orthonormal projection removing span(cols) from vec to machine precision."""
        if not cols:
            return vec
        M = np.column_stack(cols)  # [D, m]
        Q, R = np.linalg.qr(M, mode="reduced")
        # Keep only numerically independent columns
        diag_r = np.abs(np.diag(R))
        keep = diag_r > (1e-6 * max(float(diag_r.max()), 1e-12))
        Q = Q[:, keep]
        # Two passes of orthogonal projection for float64/float32 numerical precision
        v = vec - Q @ (Q.T @ vec)
        v = v - Q @ (Q.T @ v)
        return v

    def graft_concept_v2(self, cfg: GraftConfig) -> dict:
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

        # 3. Stage 1 — Scaled-Rotated Kind Base with exact QR private address
        with torch.no_grad():
            src_tids = [self._resolve_word_token(w) for w in cfg.kind_words]
            E_src = self.emb_w[src_tids].detach().float()
            r_ref = float(E_src.norm(dim=-1).mean().item())
            e_kind_np = E_src.mean(dim=0).cpu().numpy()
            e_kind_np = e_kind_np / (np.linalg.norm(e_kind_np) + 1e-12)

            rng = np.random.default_rng(cfg.seed + tid)
            u_raw = rng.standard_normal(self.d_model).astype(np.float64)
            # Orthogonalize u_perp against ALL prior e_kind, u_perp, AND prior (w_norm * d_write)!
            prot_cols = [e_kind_np.astype(np.float64)] + [v.astype(np.float64) for v in (self.e_kind_dirs + self.u_perp_dirs + self.nw_dwrite_dirs)]
            u_perp_np = self._qr_project_out(u_raw, prot_cols)
            u_perp_np = (u_perp_np / (np.linalg.norm(u_perp_np) + 1e-12)).astype(np.float32)

            # Deduplicate e_kind_dirs before storing
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

        # 5. Stage 3 — Syntactically Diverse + Weighted Hard-Negative Write + Exact QR Projection
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
            self.capture_mlp_inputs(whiten_g_sent_seqs, write_layer, all_tokens=True),
        ])
        # Final-token hard negatives (same kind, wrong attribute, exact naming slot!)
        X_hard_last = self.capture_mlp_inputs(hard_neg_seqs, write_layer, all_tokens=False)
        w_gate_w, w_up_w, a_pos_w, write_snr = fit_swiglu_gated_key_weighted(
            X_pos_w, X_bg_w, X_hard_last,
            hard_weight=6.0,
            ridge_frac=0.15,
            bias_ridge_frac=0.10,
            gate_tau=0.42,
            gate_sharp=10.0,
        )
        rms_write = self.capture_pre_norm_rms(fit_write_seqs)

        with torch.no_grad():
            nw64 = self.nw_np.astype(np.float64)
            d_raw = nw64 * u_perp_np.astype(np.float64)
            # Exact QR projection removing span({nw * e_kind_m} U {nw * u_perp_j : j != current})
            forb_cols = [nw64 * ek.astype(np.float64) for ek in self.e_kind_dirs] + [
                nw64 * up.astype(np.float64) for up in self.u_perp_dirs[:-1]
            ]
            d_write = self._qr_project_out(d_raw, forb_cols)
            d_write = (d_write / (np.linalg.norm(d_write) + 1e-12)).astype(np.float32)

            # Register normalized (nw * d_write) so all future u_perp_j stay orthogonal to this Write vector!
            nwd = self.nw_np * d_write
            nwd = nwd / (np.linalg.norm(nwd) + 1e-12)
            self.nw_dwrite_dirs.append(nwd.astype(np.float32))

            eff_gain = float(np.dot(d_write, self.nw_np * e_G_np))
            beta = (cfg.target_logit_boost * rms_write) / (a_pos_w * max(eff_gain, 1e-6))
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
            "read_fit_snrs": read_snrs,
            "slot_cjk_shares": {sl.role: sl.cjk_share for sl in all_slots},
            "slot_indices": {sl.role: sl.neuron_idx for sl in all_slots},
        }


def main() -> None:
    t_start = time.time()
    specs = build_100_concepts()
    assert len(specs) == 100
    assert len({s.token_name.lower() for s in specs}) == 100
    assert len({(s.attr_adj.lower(), s.attr_noun.lower()) for s in specs}) == 100

    print(f"=== Track B Exp 1-v2: K=1..100 Phase-2 Scaling on {MODEL_REF} ===", flush=True)
    grafter = Phase2Grafter(MODEL_REF)
    print(f"Loaded on device={grafter.device}, free_slots={grafter.vocab_size - grafter.orig_active_vocab}", flush=True)

    # Verify zero token collisions & resolve multi-hop targets
    mh_target_ids: list[int] = []
    for s in specs:
        content = f" {s.token_name}"
        ex = grafter.tok.convert_tokens_to_ids(content)
        assert ex is None or ex == grafter.tok.unk_token_id or ex >= grafter.orig_active_vocab, f"Token collision: {content} -> {ex}"
        mh_target_ids.append(grafter.enc(f" {s.multihop_target}")[0])

    # Baseline multi-hop
    base_mh_seqs = [grafter.enc(s.multihop_prompt_tpl.format(name="concept")) for s in specs]
    base_mh_logits = eval_logits_batched(grafter, base_mh_seqs)
    base_mh_logp = F.log_softmax(base_mh_logits, dim=-1)
    baseline_mh = []
    for i, tid_mh in enumerate(mh_target_ids):
        rank = int((base_mh_logits[i] > base_mh_logits[i, tid_mh]).sum().item()) + 1
        baseline_mh.append({
            "token_name": specs[i].token_name,
            "target": specs[i].multihop_target,
            "base_logp": float(base_mh_logp[i, tid_mh]),
            "base_rank": rank,
        })
    print(f"Baseline multi-hop (100 concepts): med_rank={np.median([x['base_rank'] for x in baseline_mh]):.1f}, top10={sum(x['base_rank']<=10 for x in baseline_mh)}/100", flush=True)

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
            theta_deg=35.0,
            gamma=0.60,
            read_layers=read_layers,
            write_layer=write_layer,
            read_alphas=[1.5, 1.2, 0.8],
            seed=42 + k,
        )
        info = grafter.graft_concept_v2(cfg)
        grafted_tids.append(info["token_id"])
        grafted_write_slots.append(info["write_slot"])
        per_concept_log.append({
            "k": k,
            "token_name": spec.token_name,
            "category": spec.category,
            "write_fit_snr": round(info["write_fit_snr"], 1),
            "slot_cjk_shares": info["slot_cjk_shares"],
            "elapsed_s": round(time.time() - t_k, 2),
        })

        if k in MILESTONES:
            t_ev = time.time()
            # 1. Unseen Write prompts (2K)
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

            # 2. In-stack Wrong-Attribute Traps (K)
            trap_seqs = [grafter.enc(unseen_trap_prompt(specs[j])) for j in range(k)]
            t_logits = eval_logits_batched(grafter, trap_seqs)
            t_probs = F.softmax(t_logits, dim=-1)
            trap_self_probs = [float(t_probs[j, grafted_tids[j]]) for j in range(k)]
            worst_trap_j = int(np.argmax(trap_self_probs))

            # 3. Zero-Shot Multi-Hop Read (K)
            mh_seqs = [grafter.enc(specs[j].multihop_prompt_tpl.format(name=specs[j].token_name)) for j in range(k)]
            mh_logits = eval_logits_batched(grafter, mh_seqs)
            mh_logp = F.log_softmax(mh_logits, dim=-1)
            mh_ranks = [int((mh_logits[j] > mh_logits[j, mh_target_ids[j]]).sum().item()) + 1 for j in range(k)]
            mh_lifts = [float(mh_logp[j, mh_target_ids[j]]) - baseline_mh[j]["base_logp"] for j in range(k)]
            n_pos_lift = sum(x > 0 for x in mh_lifts)
            mh_sign_p = float(binomtest(n_pos_lift, k, 0.5, alternative="greater").pvalue) if k >= 5 else 1.0

            # 4. Collateral KL
            g_lp = F.log_softmax(grafter.eval_next_logits(grafter.gen_seqs), dim=-1)
            a_lp = F.log_softmax(grafter.eval_next_logits(grafter.adv_seqs), dim=-1)
            gen_kl = float((grafter.gen_p_base * (grafter.gen_logp_base - g_lp)).sum(dim=-1).mean())
            adv_kl = float((grafter.adv_p_base * (grafter.adv_logp_base - a_lp)).sum(dim=-1).mean())
            adv_max_any_g = float(a_lp[:, grafted_tids].exp().max())
            adv_sum_any_g = float(a_lp[:, grafted_tids].exp().sum(dim=-1).max())

            # 5. Geometric Write orthogonality drift
            D_mat = np.stack([w[1] for w in grafter.write_vecs], axis=0)          # [K, D]
            E_mat = np.stack([nw_np * w[2] for w in grafter.write_vecs], axis=0) # [K, D]
            G_mat = D_mat @ E_mat.T  # [K, K]
            self_gains = np.diag(G_mat)
            off_mask = ~np.eye(k, dtype=bool)
            off_max = float(np.abs(G_mat[off_mask]).max()) if k > 1 else 0.0
            geom_off_ratio = off_max / max(float(self_gains.mean()), 1e-8)

            # 6. Audit L35 norm detection Precision@K
            with torch.no_grad():
                norms35 = grafter.layers[write_layer].mlp.down_proj.weight.detach().float().norm(dim=0).cpu().numpy()
            med_norm35 = float(np.median(norms35))
            graft_norms35 = [float(norms35[s_idx]) for s_idx in grafted_write_slots]
            top_k_idx = set(np.argsort(norms35)[-k:].tolist())
            detected_at_k = len(top_k_idx.intersection(grafted_write_slots))

            # Per-category breakdown
            by_cat = {}
            for cat in ("place", "city", "cuisine", "language", "product"):
                idxs = [j for j in range(k) if specs[j].category == cat]
                if not idxs:
                    continue
                by_cat[cat] = {
                    "n": len(idxs),
                    "write_top1_pct": round(100.0 * sum(1 for j in idxs for m in range(2) if self_ranks[2 * j + m] == 1) / (2 * len(idxs)), 1),
                    "mh_top10_pct": round(100.0 * sum(1 for j in idxs if mh_ranks[j] <= 10) / len(idxs), 1),
                    "mh_mean_logp_lift": round(float(np.mean([mh_lifts[j] for j in idxs])), 3),
                    "trap_max": round(max(trap_self_probs[j] for j in idxs), 5),
                }

            ms = {
                "K": k,
                "params_edited": k * (grafter.d_model + 12 * grafter.d_model),
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
                f"  >>> [V2 K={k:3d}] WTop1={ms['write_prompt_top1_pct']:5.1f}% (AnyT1={ms['write_concept_any_top1_pct']:5.1f}%) | "
                f"XBleed(p99={ms['cross_bleed_p99']:.5f}, max={ms['cross_bleed_max']:.4f} [{wb}]) | "
                f"TrapMax={ms['trap_self_max']:.4f} ({specs[worst_trap_j].token_name}) | "
                f"MH top10={ms['multihop_top10_pct']:4.1f}% (base {ms['multihop_base_top10_pct']:4.1f}%, lift={ms['multihop_mean_logp_lift']:+.2f}, p={ms['multihop_sign_p']:.1e}) | "
                f"GenKL={gen_kl:.5f} AdvKL={adv_kl:.5f} | GeomOff={geom_off_ratio:.5f} | AuditP@K={ms['audit_precision_at_k']:.2f}",
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
    out_path = OUT_DIR / "results_exp1_v2_scaling.json"
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"\nSaved full v2 receipts to {out_path} ({out_obj['total_elapsed_s']}s total)", flush=True)


if __name__ == "__main__":
    main()
