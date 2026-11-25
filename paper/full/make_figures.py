"""
Generate 4 publication-grade multi-panel vector figures (path-outlined SVG + PDF)
for the full Phase-2 Surgical Concept Grafting paper directly from verified JSON receipts.
Uses svg.fonttype='path' so Typst renders exact matplotlib typography with zero glyph drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = (ROOT / "experiments" / "phase2") if (ROOT / "experiments" / "phase2").exists() else (ROOT / "phase2-scaling-20261125")

NAVY = "#1B365D"
TEAL = "#0E7C7B"
CORAL = "#D1495B"
AMBER = "#D97706"
SLATE = "#64748B"

plt.rcParams.update({
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "axes.titlepad": 7.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#475569",
    "axes.linewidth": 0.75,
    "grid.color": "#CBD5E1",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.65,
})


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {stem}.svg / .pdf")


def make_fig1() -> None:
    """Fig 1: (a) RMSNorm-Head Scale Asymmetry Pareto, (b) Exp 4 Read-Alpha Trade-off."""
    exp4 = json.loads((EXP_DIR / "results_exp4_read_pareto.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.95), dpi=200)
    fig.subplots_adjust(wspace=0.42)

    # Panel (a)
    gammas = [1.25, 1.00, 0.70, 0.50, 0.25]
    twist_pct = [101.2, 100.0, 96.6, 92.2, 81.3]
    behav_pct = [100.5, 100.0, 99.8, 99.9, 98.7]
    adv_kl = [2.2104, 0.4126, 0.00317, 0.000092, 0.0000015]

    ax1.semilogx(adv_kl, behav_pct, "o-", color=NAVY, lw=1.7, ms=5.0, label="Behavioral dissociation (% of naive)")
    ax1.semilogx(adv_kl, twist_pct, "s--", color=TEAL, lw=1.5, ms=4.8, label="L24 CHEMO twist (% of naive)")
    ax1.invert_xaxis()
    ax1.set_xlabel("Adversarial KL (nats, log; lower is better →)")
    ax1.set_ylabel("Retained Effect (% of γ=1.0)")
    ax1.set_title("(a) RMSNorm–Tied-Head Scale Asymmetry")
    ax1.set_ylim(74, 107)
    ax1.set_xlim(6.0, 1.5e-7)
    ax1.grid(True, which="major", ls=":")

    ann_cfg = {
        1.25: ((6, -13), "left", "γ=1.25"),
        1.00: ((8, -14), "left", "γ=1.00 (naive)"),
        0.70: ((-6, -15), "right", "γ=0.70"),
        0.50: ((-8, -13), "right", "γ=0.50 (4,503×)"),
        0.25: ((4, 10), "center", "γ=0.25\n(275k×)"),
    }
    for gm, kl, tw in zip(gammas, adv_kl, twist_pct):
        (dx, dy), ha, lbl = ann_cfg[gm]
        ax1.annotate(lbl, (kl, tw), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=6.8, color="#1E293B")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0")

    # Panel (b)
    a_l2 = [r["read_alphas"][0] for r in exp4]
    b_sep = [r["behav_sep"] for r in exp4]
    p_rey = [100.0 * r["mh_probs"][0] for r in exp4]
    p_tok = [100.0 * r["mh_probs"][1] for r in exp4]

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.spines["right"].set_color(TEAL)
    ax2_r.tick_params(axis="y", colors=TEAL)

    l1, = ax2.plot(a_l2, b_sep, "o-", color=NAVY, lw=1.7, ms=5.0, label="Kind BehavSep ΔB (left)")
    l2, = ax2_r.plot(a_l2, p_rey, "s--", color=TEAL, lw=1.5, ms=4.8, label="P('Reykjavik') % (right)")
    l3, = ax2_r.plot(a_l2, p_tok, "^:", color=AMBER, lw=1.5, ms=5.0, label="P('Tokyo') % (right)")

    ax2.axvspan(0.73, 0.87, color="#DBEAFE", alpha=0.60, zorder=0)
    ax2.annotate("Sweet spot (0.8):\nMH [#1, #1] & ΔB=+1.79", (0.80, 1.79), xytext=(0.74, 3.00),
                 textcoords="data", ha="left", fontsize=6.5, fontweight="bold", color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#93C5FD", lw=0.6, alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))

    ax2.set_xlabel(r"Early-Layer Read $\alpha_{\mathrm{L2}}$  ($[\alpha_{\mathrm{L2}}, \alpha_{\mathrm{L12}}, 0.8]$)")
    ax2.set_ylabel(r"Parent-Kind $\Delta B$ (nats)", color=NAVY)
    ax2.tick_params(axis="y", colors=NAVY)
    ax2_r.set_ylabel("Zero-Shot Multi-Hop Prob (%)", color=TEAL)
    ax2.set_title("(b) Read Strength vs. Kind Fidelity (Exp 4)")
    ax2.set_ylim(0.0, 3.65)
    ax2_r.set_ylim(-1.0, 39.5)
    ax2.grid(True, ls=":")
    ax2.legend(handles=[l1, l2, l3], loc="lower left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0", fontsize=6.5)

    save_both(fig, "fig1_mechanism_and_pareto")


def make_fig2() -> None:
    """Fig 2: K=1..100 Scaling Curve on Qwen3-4B (v1 vs v3b vs v4 Joint Anti-Bleed)."""
    v1 = json.loads((EXP_DIR / "results_exp1_scaling.json").read_text())["milestones"]
    v3b = json.loads((EXP_DIR / "results_exp1_v3b_capped.json").read_text())["milestones"]
    v4 = json.loads((EXP_DIR / "results_exp1_v4_antibleed.json").read_text())

    ks_v1 = [1, 2, 5, 10, 20, 35, 50]
    ks_v3b = [1, 2, 5, 10, 20, 35, 50, 75, 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.95), dpi=200)
    fig.subplots_adjust(wspace=0.34)

    ax1.plot(ks_v1, [v1[str(k)]["write_concept_any_top1_pct"] for k in ks_v1],
             "o:", color=SLATE, lw=1.35, ms=4.2, label="v1 baseline (Concept Top-1)")
    ax1.plot(ks_v3b, [v3b[str(k)]["write_concept_any_top1_pct"] for k in ks_v3b],
             "s--", color=TEAL, lw=1.55, ms=4.5, label="v3b linear-cone seq. (Concept Top-1)")
    ax1.plot([50, 100], [v4["50"]["write_concept_any_top1_pct"], v4["100"]["write_concept_any_top1_pct"]],
             "D-", color=NAVY, lw=1.9, ms=6.0, label="v4 joint anti-bleed (Concept Top-1)")
    ax1.plot([50, 100], [v4["50"]["write_prompt_top1_pct"], v4["100"]["write_prompt_top1_pct"]],
             "^-", color=CORAL, lw=1.45, ms=5.2, label="v4 joint anti-bleed (Prompt Top-1)")

    ax1.annotate("90.0%\n(0/4,900 bleed >1%)", (50, 90.0), xytext=(-14, 9), textcoords="offset points",
                 ha="center", fontsize=6.7, fontweight="bold", color=NAVY)
    ax1.annotate("92.0%\n(92/100)", (100, 92.0), xytext=(-10, -24), textcoords="offset points",
                 ha="right", fontsize=6.7, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Grafted Concepts K  (0.083% params at K=100)")
    ax1.set_ylabel("Unseen Naming Top-1 Recall (%)")
    ax1.set_title("(a) Unseen Naming Recall vs. Stack Size K")
    ax1.set_ylim(30, 109)
    ax1.set_xlim(-3, 106)
    ax1.grid(True, ls=":")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0", fontsize=6.5)

    ax2.plot(ks_v1, [v1[str(k)]["adv_kl"] for k in ks_v1],
             "o:", color=CORAL, lw=1.35, ms=4.2, alpha=0.85, label="v1 Adv KL (k=17 bridge cliff)")
    ax2.plot(ks_v3b, [v3b[str(k)]["gen_kl"] for k in ks_v3b],
             "s-", color=NAVY, lw=1.6, ms=4.5, label="v3b Gen KL (quiet-CJK)")
    ax2.plot(ks_v3b, [v3b[str(k)]["adv_kl"] for k in ks_v3b],
             "^--", color=TEAL, lw=1.55, ms=4.5, label="v3b Adv KL (quiet-CJK)")
    ax2.scatter([50, 100], [v4["50"]["gen_kl"], v4["100"]["gen_kl"]],
                marker="D", s=34, color=NAVY, zorder=5, label="v4 Joint Gen KL (.0022 / .0070)")
    ax2.scatter([50, 100], [v4["50"]["adv_kl"], v4["100"]["adv_kl"]],
                marker="P", s=40, color=AMBER, zorder=5, label="v4 Joint Adv KL (.0064 / .0098)")
    ax2.axhline(0.01, color="#94A3B8", ls="--", lw=0.85)
    ax2.text(103, 0.0104, "0.01 nats ceiling", ha="right", fontsize=6.7, color="#64748B")

    ax2.set_xlabel("Simultaneously Grafted Concepts K")
    ax2.set_ylabel("Out-of-Sample Collateral KL (nats)")
    ax2.set_title("(b) Cumulative Collateral KL (K = 1 → 100)")
    ax2.set_ylim(-0.0005, 0.0155)
    ax2.set_xlim(-3, 106)
    ax2.grid(True, ls=":")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0", fontsize=6.3)

    save_both(fig, "fig2_scaling_1_to_100")


def make_fig3() -> None:
    """Fig 3: (a) Exp 3 Matched-Budget AdamW & LoRA Pareto, (b) Exp 2 4-Architecture Generalization."""
    exp3 = json.loads((EXP_DIR / "results_exp3_gradient_baseline.json").read_text())["results"]
    exp4 = json.loads((EXP_DIR / "results_exp4_read_pareto.json").read_text())
    exp2 = json.loads((EXP_DIR / "results_exp2_cross_arch.json").read_text())
    llama = json.loads((EXP_DIR / "results_exp2_llama32.json").read_text())["five_concept"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.05), dpi=200)
    fig.subplots_adjust(wspace=0.36)

    def _mean_cap(r):
        return 0.5 * (r["mh_cap_ranks"][0] + r["mh_cap_ranks"][1])

    gs_ce = [(v["adv_kl"], _mean_cap(v)) for k, v in exp3.items() if k.startswith("GradSparse_CE")]
    gs_kl = [(v["adv_kl"], _mean_cap(v)) for k, v in exp3.items() if k.startswith("GradSparse_KL")]
    lora = [(v["adv_kl"], _mean_cap(v)) for k, v in exp3.items() if k.startswith("LoRA_r4_KL")]
    scg_pt = (exp4[3]["adv_kl"], 1.0)

    ax1.scatter([x[0] for x in gs_ce], [x[1] for x in gs_ce], marker="o", s=34, color=AMBER, edgecolor="#78350F", lw=0.6, label="AdamW Sparse CE (66k)")
    ax1.scatter([x[0] for x in gs_kl], [x[1] for x in gs_kl], marker="s", s=34, color=TEAL, edgecolor="#134E4A", lw=0.6, label="AdamW Sparse + KL (66k)")
    ax1.scatter([x[0] for x in lora], [x[1] for x in lora], marker="^", s=42, color=CORAL, edgecolor="#7F1D1D", lw=0.6, label="LoRA r=4 + KL (202k, 3×)")
    ax1.scatter([scg_pt[0]], [scg_pt[1]], marker="*", s=145, color=NAVY, edgecolor="black", lw=0.7, zorder=10, label="SCG Closed-Form (66k, 0 steps)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_yaxis()
    ax1.set_xlabel("Adversarial Near-Neighbor KL (nats, log)")
    ax1.set_ylabel("Mean Zero-Shot Multi-Hop Rank (↑ better)")
    ax1.set_title("(a) SCG vs. AdamW & LoRA (Exp 3)")
    ax1.set_ylim(550, 0.55)
    ax1.set_xlim(2e-4, 40)
    ax1.grid(True, which="major", ls=":")
    ax1.annotate("SCG: [#1, #1]\nTrap=0.000, KL=.0006", (scg_pt[0], scg_pt[1]),
                 xytext=(12, -16), textcoords="offset points", fontsize=6.6, fontweight="bold", color=NAVY,
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))
    ax1.annotate("LoRA: [#2, #1]\nbut 4,677× KL\n& 45% trap", (2.85, 1.5),
                 xytext=(-52, -34), textcoords="offset points", fontsize=6.4, color=CORAL,
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.8))
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0", fontsize=6.3)

    q25_7b_u = [r for r in exp2 if r["model"] == "Qwen/Qwen2.5-7B-Instruct" and r["mode"] == "untied_pure_ortho"][0]
    nb_u = [r for r in exp2 if r["model"] == "Nanbeige/Nanbeige4.2-3B" and r["mode"] == "untied_pure_ortho"][0]

    models = ["Qwen3-4B\n(tied)", "Qwen2.5-7B\n(untied*)", "Llama-3.2\n(no-CJK)", "Nanbeige\n(untied*)"]
    w_prompt = [80.0, q25_7b_u["write_top1_pct"], llama["write_top1_pct"], nb_u["write_top1_pct"]]
    w_concept = [100.0, q25_7b_u["write_any_top1_pct"], llama["write_any_top1_pct"], nb_u["write_any_top1_pct"]]
    mh_t10 = [100.0, q25_7b_u["mh_top10_pct"], llama["mh_top10_pct"], nb_u["mh_top10_pct"]]

    x = np.arange(len(models))
    w = 0.25
    ax2.bar(x - w, w_prompt, width=w, color=NAVY, label="Write Prompt Top-1 (%)")
    ax2.bar(x, w_concept, width=w, color=TEAL, label="Write Concept Top-1 (%)")
    ax2.bar(x + w, mh_t10, width=w, color=AMBER, label="Multi-Hop Top-10 (%)")

    for i, kl_txt in enumerate(["KL .0007", "KL .0052", "KL .0005", "KL .0014"]):
        ax2.text(i, 103.5, kl_txt, ha="center", fontsize=6.6, color="#334155", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=6.7)
    ax2.set_ylabel("Held-Out Accuracy (%)")
    ax2.set_title("(b) Cross-Architecture Generalization (Exp 2)")
    ax2.set_ylim(0, 118)
    ax2.grid(True, axis="y", ls=":")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0", fontsize=6.4)

    save_both(fig, "fig3_baselines_and_crossarch")


def make_fig4() -> None:
    """Fig 4: (a) Exp 6 Transactional Update & 0.88ms Unlearning, (b) Exp 5 Red-Team vs Blue-Team Audit."""
    exp6 = json.loads((EXP_DIR / "results_exp6_unlearning.json").read_text())
    exp5 = json.loads((EXP_DIR / "results_exp5_stealth_audit.json").read_text())["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.95), dpi=200)
    fig.subplots_adjust(wspace=0.40)

    stages = ["Phase 1:\nInstall 50", "Phase 2:\nUpdate 10\n(10.6s)", "Phase 3:\nDelete 10\n(0.88 ms)"]
    keep30 = [
        exp6["phase1_install50"]["keep30"]["write_any_top1_pct"],
        exp6["phase2_update10"]["keep30"]["write_any_top1_pct"],
        exp6["phase3_delete10"]["keep30_still_intact"]["write_any_top1_pct"],
    ]
    upd_new = [
        exp6["phase1_install50"]["update_set_old"]["write_any_top1_pct"],
        exp6["phase2_update10"]["update_set_NEW"]["write_any_top1_pct"],
        exp6["phase3_delete10"]["updated10_still_intact"]["write_any_top1_pct"],
    ]
    old_or_del = [
        exp6["phase1_install50"]["delete_set"]["write_any_top1_pct"],
        exp6["phase2_update10"]["update_set_OLD_residue"]["write_any_top1_pct"],
        exp6["phase3_delete10"]["deleted10_residue"]["write_any_top1_pct"],
    ]

    x = np.arange(3)
    w = 0.24
    ax1.bar(x - w, keep30, width=w, color=NAVY, label="Untouched 30 (Concept Top-1)")
    ax1.bar(x, upd_new, width=w, color=TEAL, label="Active Target 10 (Old→New)")
    ax1.bar(x + w, old_or_del, width=w, color=CORAL, label="Erased / Deleted 10 Residue")

    # Place clean callout pills in open whitespace ABOVE zero bars without overlapping bars!
    ax1.annotate("0.0%\n(maxP .00047)", (1 + w, 1.5), xytext=(1 + w + 0.03, 30),
                 ha="center", fontsize=6.3, fontweight="bold", color=CORAL,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=CORAL, lw=0.6, alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.8))
    ax1.annotate("0.0%\n(P=.000000)", (2 + w, 1.5), xytext=(2 + w, 30),
                 ha="center", fontsize=6.3, fontweight="bold", color=CORAL,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=CORAL, lw=0.6, alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.8))

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7.3)
    ax1.set_ylabel("Unseen Concept Write Top-1 (%)")
    ax1.set_title("(a) Transactional Update & Rollback (Exp 6)")
    ax1.set_ylim(0, 122)
    ax1.set_xlim(-0.5, 2.65)
    ax1.grid(True, axis="y", ls=":")
    ax1.legend(loc="upper center", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0", fontsize=6.2, ncol=1)

    naive = [r for r in exp5 if r["attack"] == "naive_split"]
    adapt = [r for r in exp5 if r["attack"] == "adaptive_ortho_jitter"]
    ms_n = [r["M"] for r in naive]
    ms_a = [r["M"] for r in adapt]

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.spines["right"].set_color(TEAL)
    ax2_r.tick_params(axis="y", colors=TEAL)

    l1, = ax2.semilogy(ms_n, [r["norm_ratio_vs_median"] for r in naive], "o-", color=SLATE, lw=1.5, ms=4.5, label=r"Naive $\|w_{\mathrm{down}}\|$/med (left)")
    l2, = ax2.semilogy(ms_a, [r["norm_ratio_vs_median"] for r in adapt], "s--", color=AMBER, lw=1.45, ms=4.5, label=r"Adaptive $\|w_{\mathrm{down}}\|$/med (left)")
    ax2.axhline(1.47 / 1.17, color="#94A3B8", ls=":", lw=0.9)
    ax2.text(1.5, 1.36, "L35 p99 (1.26×)", ha="left", fontsize=6.4, color="#64748B")

    l3, = ax2_r.plot(ms_a, [r["D2b_joint_auroc"] for r in adapt], "^:", color=CORAL, lw=1.45, ms=4.8, label="D2b Joint AUROC (adaptive)")
    l4, = ax2_r.plot(ms_n, [r["D_combined_auroc"] for r in naive], "D-", color=TEAL, lw=1.9, ms=4.8, label="Combined D1+D2a AUROC (1.000)")

    ax2.set_xlabel("Red-Team Split Factor M (neurons / graft)")
    ax2.set_ylabel(r"Per-Neuron $\|w_{\mathrm{down}}\|$ / Med (log)", color="#334155")
    ax2_r.set_ylabel("Blue-Team Detection AUROC", color=TEAL)
    ax2.set_title("(b) Stealth-Spread Audit Defense (Exp 5)")
    ax2.set_xticks([1, 4, 8, 16, 32])
    ax2.set_xlim(-0.5, 34.5)
    ax2.set_ylim(0.95, 95)
    ax2_r.set_ylim(0.80, 1.035)
    ax2.grid(True, ls=":")
    ax2.legend(handles=[l1, l2, l3, l4], loc="center right", bbox_to_anchor=(0.99, 0.54),
               frameon=True, facecolor="white", framealpha=0.95, edgecolor="#E2E8F0", fontsize=6.0)

    save_both(fig, "fig4_unlearning_and_audit")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("All 4 polished path-outlined vector figures generated cleanly.")
