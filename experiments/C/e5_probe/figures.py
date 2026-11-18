"""(C) E5: figures for the project C paper (`experiments/C/paper/c_paper.qmd`, Section 6).

    python experiments/C/e5_probe/figures.py      # writes figures/*.png and *.pdf

fig1_gate    data gate: low-band linear residual vs equivalent-pole shift, one point per set, coloured by whether
             the runs rule v1 trusts there are correct (results/e5_gate.json)
fig2_agree   probe agreement vs pole error of every v1-trusted run, coloured by the data gate (results/e5_v4.json)

Colours and style: the E1 figures (`experiments/C/e1_audit/figures.py`), categorical slots in fixed order.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS, FIG = HERE / "results", HERE / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # palette slots 1-4 (E1 figures)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LOW_RESID, SHIFT, AGREE, CORRECT = 0.1, 0.1, 0.1, 0.1

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
SYSTEM = {"pend": "pendulum", "c0": "undamped Duffing", "lz": "Lorenz"}


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def fig1_gate():
    g = json.loads((RESULTS / "e5_gate.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    groups = {"all correct": ([], C[0], "o"), "some wrong": ([], C[1], "s"), "none trusted": ([], INK2, "^")}
    for tag, v in g.items():
        errs = [x["err"] for x in v["v1_trusted"]]
        k = "none trusted" if not errs else "some wrong" if max(errs) >= CORRECT else "all correct"
        groups[k][0].append((v["r_low"], v["shift"], tag))
    for lab, (pts, col, mk) in groups.items():
        if pts:
            x, y, _ = zip(*pts)
            ax.scatter(x, y, s=34, color=col, marker=mk, label=lab, edgecolor="white", linewidth=0.8, zorder=3)
    for lab, (pts, _, _) in groups.items():
        for x, y, tag in pts:
            if tag in ("mixed", "moderate", "lz_wide", "lz_r10_wide", "c0_medium", "wide", "lz_r10_orig"):
                ax.annotate(tag, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    ax.axvline(LOW_RESID, color=INK2, linewidth=1, linestyle="--")
    ax.axhline(SHIFT, color=INK2, linewidth=1, linestyle="--")
    ax.fill_between([5e-4, LOW_RESID], 5e-4, SHIFT, color=GRID, alpha=0.5, zorder=0, linewidth=0)
    ax.text(6e-4, 6.5e-4, "gate passes", color=INK2, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-4, 1.5)
    ax.set_ylim(5e-4, 1.5)
    ax.set_xlabel("low-amplitude fifth: linear-fit residual")
    ax.set_ylabel("equivalent-pole shift, all vs low")
    ax.set_title("v3 data gate, one point per set (18 sets)", fontsize=9, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.3, title="runs that v1 trusts", title_fontsize=7.5)
    _save(fig, "fig1_gate")


def fig2_agree():
    v = json.loads((RESULTS / "e5_v4.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    pts = {True: [], False: []}
    for tag, s in v.items():
        for r in s["rows"]:
            if r["v1"]:
                pts[s["gate_v3"]].append((max(r["agree"], 1e-3), r["err"]))
    for gate, col, mk, lab in ((True, C[0], "o", "data gate passes"), (False, C[1], "s", "data gate fails")):
        x, y = np.array(pts[gate]).T
        ax.scatter(x, y, s=26, color=col, marker=mk, label=f"{lab} ({len(x)})", edgecolor="white", linewidth=0.8,
                   zorder=3)
    ax.axvline(AGREE, color=INK2, linewidth=1, linestyle="--")
    ax.axhline(CORRECT, color=INK2, linewidth=1, linestyle="--")
    ax.text(8e3, CORRECT * 1.12, "pole error 10 %", ha="right", color=INK2, fontsize=7.5, va="bottom")
    ax.text(AGREE * 1.12, 1.1e-3, "agreement 0.1", color=INK2, fontsize=7.5, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e4)
    ax.set_ylim(1e-3, 10)
    ax.set_xlabel("probe agreement |μ_LTI − μ_J| / |μ_J|")
    ax.set_ylabel("Jacobian pole error")
    ax.set_title("every run rule v1 trusts (18 sets)", fontsize=9, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.3)
    _save(fig, "fig2_agree")


if __name__ == "__main__":
    fig1_gate()
    fig2_agree()
    print("written:", sorted(p.name for p in FIG.glob("*.png")))
