"""(C) E2 figure: all-pole recovery of the prior-free recipe by stage (Adam, + L-BFGS polish,
+ single-slot restarts), `damp8_ms`, seeds 10-29. Style as experiments/C/e1_audit/figures.py.

    python experiments/C/theory/e2_figures.py      # writes results/figures/fig_e2_stages.png (+ .pdf)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STAGE_C = ["#a9cbf2", "#5b9eec", "#1f5fae"]                 # one hue, light -> dark = later stage
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
CONDS = [("p4_ms16_T20", "p4\ndt = 20/1024"), ("p6_ms16_T80", "p6\ndt = 20/1024"),
         ("p6_ms16_T80_N4096", "p6\ndt = 20/4096")]
VARIANT, SEEDS, TOL = "damp8_ms", range(10, 30), 0.1


def rates(cond):
    ok = lambda pe: pe is not None and max(pe) < TOL
    name = lambda s: f"{cond}_{VARIANT}_s{s}.json"
    adam = [json.loads((RES / "e2_train" / name(s)).read_text())["pole_error"] for s in SEEDS]
    pol = [json.loads((RES / "e2_polish" / name(s)).read_text())["after"]["pole_error"] for s in SEEDS]
    res = [json.loads((RES / "e2_rescue" / name(s)).read_text())["after"]["pole_error"] for s in SEEDS]
    return [np.mean([ok(p) for p in st]) for st in (adam, pol, res)]


def main():
    r = np.array([rates(c) for c, _ in CONDS])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x, w = np.arange(len(CONDS)), 0.26
    labels = ["Adam (multi-start)", "+ L-BFGS polish", "+ single-slot restarts"]
    for i in range(3):
        xs = x + (i - 1) * w
        ax.bar(xs, 100 * r[:, i], width=w - 0.03, color=STAGE_C[i], label=labels[i])
        for xi, v in zip(xs, r[:, i]):
            ax.text(xi, 100 * v + 2, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [l for _, l in CONDS], fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("runs recovering every pole (%)")
    ax.set_title("Prior-free slots (8, damping-bounded), 20 runs per bar", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", ncol=3, handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 118)
    ax.grid(axis="x", visible=False)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_e2_stages.{ext}", dpi=220)
    print(r)


if __name__ == "__main__":
    main()
