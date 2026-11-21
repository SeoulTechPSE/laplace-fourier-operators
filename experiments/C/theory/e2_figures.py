"""(C) E2 / E4 figures, style as experiments/C/e1_audit/figures.py.

  fig_e2_stages  all-pole recovery of the prior-free recipe by stage (Adam, + L-BFGS polish,
                 + single-slot restarts), `damp8_ms`, seeds 10-29
  fig_e4_noise   RMS relative pole error of single runs vs output noise eta, with the
                 minimal-form Cramer-Rao bound (from results/e2_noise.json)

    python experiments/C/theory/e2_figures.py      # writes results/figures/fig_e2_stages.png, fig_e4_noise.png (+ .pdf)
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


POLES = {"p4": ["-0.4+3i", "-1", "-0.2"], "p6": ["-0.5+1.5i", "-0.3+4i", "-1", "-2"]}
# categorical slots in fixed order; a pole keeps its color in both panels (-1)
POLE_C = {"-0.4+3i": "#2a78d6", "-0.5+1.5i": "#2a78d6", "-0.3+4i": "#e87ba4", "-1": "#1baf7a",
          "-0.2": "#eb6834", "-2": "#eda100"}
NUDGE = {("p4", "-1"): -7, ("p4", "-0.2"): 6}                # label offsets (points) where end values coincide


def noise_figure():
    rows = json.loads((RES / "e2_noise.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, system in zip(axes, ("p4", "p6")):
        g = sorted((r for r in rows if r["system"] == system and r["eta"] > 0), key=lambda r: r["eta"])
        eta = np.array([r["eta"] for r in g])
        single, crb = np.array([r["single_rms"] for r in g]), np.array([r["crb"] for r in g])
        ax.axhline(TOL, color=INK2, lw=0.8, ls=":")
        for j, lab in enumerate(POLES[system]):
            c = POLE_C[lab]
            ax.plot(eta, single[:, j], color=c, lw=2, marker="o", ms=4.5, mec="white", mew=1.0)
            ax.plot(eta, crb[:, j], color=c, lw=1.2, ls="--")
            ax.annotate(lab, (eta[-1], single[-1, j]), xytext=(5, NUDGE.get((system, lab), 0)), textcoords="offset points", fontsize=7.5,
                        color=INK, va="center")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(7e-4, 0.8)
        ax.set_xlabel("output noise η (relative L2 per sample)")
        ax.set_title(system, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("RMS relative pole error")
    fig.suptitle("solid: single runs (20 per point)   dashed: Cramér–Rao bound   dotted: recovery threshold 0.1",
                 fontsize=8, color=INK2, x=0.08, ha="left", y=1.0)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_e4_noise.{ext}", dpi=220)


def main():
    noise_figure()
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
