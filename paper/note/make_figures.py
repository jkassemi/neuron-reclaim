import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/james/work/neuron-reclaim")
OUT = ROOT / "paper/note/figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.2,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
})

wg = json.loads((ROOT / "weight-graft-20261125/results_weight_graft.json").read_text())
pp = json.loads((ROOT / "readwrite-graft-v3-20261125/payload_patch_diag.json").read_text())
v3 = json.loads((ROOT / "readwrite-graft-v3-20261125/results_v3_dual_domain.json").read_text())

# === FIGURE 1 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"wspace": 0.30})

grid = wg["grid"]
scale_pts = sorted([r for r in grid if r["theta"] == 0], key=lambda r: r["gamma"])
angle_pts = sorted([r for r in grid if r["gamma"] == 1.0], key=lambda r: r["theta"])
combo_pts = sorted([r for r in grid if r["theta"] == 35 and r["gamma"] < 1.0], key=lambda r: r["gamma"])

ax1.plot(
    [max(r["adv_kl_mean"], 1e-6) for r in scale_pts],
    [100 * r["beh_ret"] for r in scale_pts],
    "o-", color="#1b6ca8", lw=1.6, ms=4.5, label=r"Behavior ($\theta=0^\circ$, $\gamma$)",
)
ax1.plot(
    [max(r["adv_kl_mean"], 1e-6) for r in scale_pts],
    [100 * r["tw_ret"] for r in scale_pts],
    "o--", color="#1b6ca8", lw=1.0, ms=3.3, alpha=0.6, label=r"Twist ($\theta=0^\circ$, $\gamma$)",
)
ax1.plot(
    [max(r["adv_kl_mean"], 1e-6) for r in angle_pts[:-1]],
    [100 * r["beh_ret"] for r in angle_pts[:-1]],
    "s-", color="#c94c24", lw=1.5, ms=4.2, label=r"Rotation ($\gamma=1.0$, $\theta$)",
)
ax1.scatter(
    [max(r["adv_kl_mean"], 1e-6) for r in combo_pts],
    [100 * r["beh_ret"] for r in combo_pts],
    marker="D", s=30, color="#2d8a4e", zorder=5, label=r"Private addr ($\theta=35^\circ$, $\gamma<1$)",
)

ann_map = {
    (0, 1.00): (r"$\gamma=1.0$", (-12, 6), "#1b6ca8"),
    (0, 0.50): (r"$\gamma=0.50$ (4,503×)", (-10, 6), "#1b6ca8"),
    (0, 0.25): (r"$\gamma=0.25$", (-4, -13), "#1b6ca8"),
    (35, 0.50): (r"$35^\circ, 0.5$", (-16, -13), "#2d8a4e"),
    (50, 1.00): (r"$\theta=50^\circ$", (-34, -4), "#c94c24"),
}
for r in grid:
    key = (r["theta"], round(r["gamma"], 2))
    if key in ann_map:
        txt, xyoff, col = ann_map[key]
        ax1.annotate(
            txt,
            xy=(max(r["adv_kl_mean"], 1e-6), 100 * r["beh_ret"]),
            xytext=xyoff, textcoords="offset points",
            fontsize=6.5, color=col,
        )

ax1.set_xscale("log")
ax1.set_xlim(5e-7, 6.0)
ax1.set_ylim(25, 114)
ax1.axhline(100, color="#aaaaaa", lw=0.6, ls=":")
ax1.set_xlabel("Adversarial Near-Neighbor KL (nats, log)")
ax1.set_ylabel("Retention vs. Naive Graft (%)")
ax1.set_title("(a) RMSNorm × Tied-Head Pareto Frontier")
ax1.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#dddddd", handlelength=1.6, fontsize=6.6)
ax1.grid(True, which="major", alpha=0.22, lw=0.5)

recs = [r for r in pp["records"] if abs(r["alpha"] - 1.5) < 1e-6]
layers = sorted(set(r["layer"] for r in recs))
def get_series(cand, idx):
    return [next(r["probs"][idx] for r in recs if r["layer"] == l and r["cand"] == cand) * 100 for l in layers]

ax2.plot(layers, get_series("C1_posmatch_adj_on_noun", 4), "x--", color="#888888", lw=1.2, ms=4.8, label=r"C1: adj-on-noun ($\approx 0\%$)")
ax2.plot(layers, get_series("C2b_superpose_Icelandic", 4), "o-", color="#1b6ca8", lw=1.6, ms=4.3, label=r"C2b: superpose adj $\to$ Reyk.")
ax2.plot(layers, get_series("C2b_superpose_Icelandic", 3), "s-", color="#2d8a4e", lw=1.4, ms=4.0, label=r"C2b: superpose adj $\to$ lang.")
ax2.plot(layers, get_series("C2a_superpose_Iceland", 4), "^-", color="#c94c24", lw=1.4, ms=4.3, label=r"C2a: superpose noun $\to$ Reyk.")
ax2.plot(layers, get_series("C3_def_from_Iceland", 4), "d:", color="#7b4fa2", lw=1.3, ms=4.0, label=r"C3: def. endpoint $\to$ Reyk.")

ax2.set_xticks(layers)
ax2.set_xticklabels([f"L{l}" for l in layers])
ax2.set_ylim(-0.8, 22.5)
ax2.set_xlabel("Injection Layer (Single-Layer Patch, α=1.5)")
ax2.set_ylabel("Zero-Shot Multi-Hop Prob (%)")
ax2.set_title("(b) Causal Payload Patching Across Depth")
ax2.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#dddddd", handlelength=1.6, fontsize=6.6)
ax2.grid(True, alpha=0.22, lw=0.5)

fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.17, wspace=0.30)
fig.savefig(OUT / "fig1_pareto_and_patching.svg", bbox_inches="tight")
fig.savefig(OUT / "fig1_pareto_and_patching.pdf", bbox_inches="tight")
plt.close(fig)

# === FIGURE 2 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"wspace": 0.56})

cats = ["G1 Alone\n(Ice. Flavor)", "G2 Stacked\n(Jpn. Sound)", "G1 Re-Check\n(Both Active)"]
write_mean = [
    v3["ev1_alone"]["write_mean_p"],
    v3["ev2_stacked"]["write_mean_p"],
    v3["ev1_after_stacking"]["write_mean_p"],
]
trap_max = [
    max(v3["ev1_alone"]["trap_write_max"], 1e-6),
    max(v3["ev2_stacked"]["trap_write_max"], 1e-6),
    max(v3["ev1_after_stacking"]["trap_write_max"], 1e-6),
]
xtalk = [
    1e-6,
    max(v3["crosstalk"]["max_g2_on_ice"], 1e-6),
    max(v3["crosstalk"]["max_g1_on_jpn"], 1e-6),
]

x = np.arange(len(cats))
w = 0.25
b1 = ax1.bar(x - w, write_mean, width=w, color="#1b6ca8", label="Target Paraphrases (mean)")
b2 = ax1.bar(x, trap_max, width=w, color="#c94c24", label="Wrong-Country Traps (max)")
b3 = ax1.bar(x + w, xtalk, width=w, color="#666666", label="Cross-Concept Bleed (max)")

for bar, val in zip(b1, write_mean):
    ax1.text(bar.get_x() + bar.get_width()/2, val * 1.25, f"{100*val:.0f}%", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#1b6ca8")
for bar, val in zip(b2, trap_max):
    ax1.text(bar.get_x() + bar.get_width()/2, val * 1.45, f"{val:.0e}", ha="center", va="bottom", fontsize=6.0, color="#c94c24")
for idx_b, (bar, val) in enumerate(zip(b3, xtalk)):
    lbl = "—" if idx_b == 0 else (f"≤1e-6" if val <= 1e-6 else f"{val:.0e}")
    ax1.text(bar.get_x() + bar.get_width()/2, val * 1.45, lbl, ha="center", va="bottom", fontsize=6.0, color="#444444")

ax1.set_yscale("log")
ax1.set_ylim(4e-7, 250.0)
ax1.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
ax1.set_xticks(x)
ax1.set_xticklabels(cats, fontsize=7.2)
ax1.set_ylabel("Next-Token Probability (log)")
ax1.set_title("(a) Write Specificity & Stacking Isolation")
ax1.legend(loc="upper center", frameon=True, framealpha=0.95, edgecolor="#dddddd", fontsize=6.1, ncol=1)
ax1.grid(True, axis="y", alpha=0.22, lw=0.5)

labels_r = [
    "G1: Reykjavik\n(Capital)",
    "G1: Icelandic\n(Language)",
    "G1: Iceland\n(6 Orig. med.)",
    "G2: Tokyo\n(Capital)",
    "G2: Japanese\n(Language)",
    "G2: Japan\n(6 Orig. med.)",
]
g_ranks = [
    v3["ev1_after_stacking"]["multihop"][0]["rank"],
    v3["ev1_after_stacking"]["multihop"][1]["rank"],
    int(np.median(v3["ev1_after_stacking"]["fact_ranks"])),
    v3["ev2_stacked"]["multihop"][0]["rank"],
    v3["ev2_stacked"]["multihop"][1]["rank"],
    int(np.median(v3["ev2_stacked"]["fact_ranks"])),
]
y = np.arange(len(labels_r))[::-1]
colors = ["#1b6ca8", "#1b6ca8", "#1b6ca8", "#2d8a4e", "#2d8a4e", "#2d8a4e"]
ax2.barh(y, g_ranks, height=0.52, color=colors, alpha=0.88)
for yi, rk, col in zip(y, g_ranks, colors):
    ax2.text(rk + 0.20, yi, f"Rank #{rk}", va="center", ha="left", fontsize=7.2, fontweight="bold", color=col)

ax2.scatter([3, 2, 4.5], [y[0], y[1], y[2]], marker="|", s=130, lw=2.0, color="#c94c24", zorder=6, label='Literal "Icelandic flavor"')
ax2.set_yticks(y)
ax2.set_yticklabels(labels_r, fontsize=7.1)
ax2.set_xlim(0, 9.6)
ax2.set_xlabel("Predicted Token Rank (Lower is Better)")
ax2.set_title("(b) Stacked Multi-Hop & Origin Ranks")
ax2.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#dddddd", fontsize=6.5)
ax2.grid(True, axis="x", alpha=0.22, lw=0.5)

fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.17, wspace=0.56)
fig.savefig(OUT / "fig2_stacking_and_multihop.svg", bbox_inches="tight")
fig.savefig(OUT / "fig2_stacking_and_multihop.pdf", bbox_inches="tight")
plt.close(fig)
print("Regenerated clean fig1 and fig2.")
