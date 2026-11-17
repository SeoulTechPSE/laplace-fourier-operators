"""(C) E1: figures for the E1 report (``E1_report.md``).

    python experiments/C/e1_audit/figures.py      # writes figures/*.png and *.pdf

fig1_travel     pole travel / travel bound, `stable` init, baseline vs pole lr x10 (needs checkpoints)
fig2_recovery   fraction of seeds recovering each true pole (error < 0.1), per condition
fig3_profile    p6 profiles of Re mu (results/identifiability.json)

Colours: categorical slots 1-4 of the dataviz reference palette, in fixed order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e1_audit as E  # noqa: E402
import per_pole as PP  # noqa: E402

RESULTS, FIG = HERE / "results", HERE / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # palette slots 1-4
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _runs(setting, pattern):
    return [json.loads(f.read_text()) for f in sorted((RESULTS / setting).glob(pattern))]


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


# ---- fig 1 ----------------------------------------------------------------------

def _travel_fraction(r):
    ck = E.RUNS / f"{r['setting']}_{r['system']}_{r['model']}_{r['init']}_{r['protocol']}_s{r['seed']}.pt"
    sd = torch.load(ck, map_location="cpu")
    mu1 = (sd["pr.mu"] if r["setting"] == "layer"
           else torch.cat([v.flatten() for k, v in sd.items() if k.endswith(".mu")])).flatten().numpy()
    return np.abs(mu1 - PP._initial_mu(r)) / PP.travel_budget(r["protocol"])


def fig1_travel():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    bins = np.linspace(0, 1.25, 26)
    for ax, proto, title in zip(axes, ("common", "common+plr10"),
                                ("published protocol (bound 4.0)", "pole lr × 10 (bound 40)")):
        for i, (setting, label) in enumerate((("layer", "single layer"), ("deep", "deep ALNO/LFNO"))):
            fr = np.concatenate([_travel_fraction(r) for r in _runs(setting, f"*_stable_{proto}_s*.json")
                                 if not r.get("diverged") and r["model"] in ("tied", "alno", "lfno")])
            w = np.ones_like(fr) / fr.size
            ax.hist(fr, bins=bins, weights=w, histtype="step", linewidth=2, color=C[i], label=label)
        ax.axvline(1.0, color=INK2, linewidth=1, linestyle="--")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("pole travel |μ_final − μ_init| / travel bound")
    for ax in axes:
        ax.text(1.02, 0.6, "bound", color=INK2, fontsize=8, va="top")
    axes[0].set_ylabel("fraction of poles")
    axes[0].legend(loc="upper left")
    _save(fig, "fig1_travel")


# ---- fig 2 ----------------------------------------------------------------------

POLES = [("p2", (-0.5, 2.0)), ("p4", (-0.4, 3.0)), ("p4", (-1.0, 0.0)), ("p4", (-0.2, 0.0)),
         ("p6", (-0.5, 1.5)), ("p6", (-0.3, 4.0)), ("p6", (-1.0, 0.0)), ("p6", (-2.0, 0.0))]


def _recovery(rows, setting, models, init, protocol):
    out = []
    for system, z in POLES:
        rs = [r for r in rows if r["setting"] == setting and r["system"] == system and r["model"] in models
              and r["init"] == init and r["protocol"] == protocol and tuple(r["true_pole"]) == z]
        n = sum(r["seeds"] for r in rs)
        out.append(sum(r["recovered"] * r["seeds"] for r in rs) / n if n else np.nan)
    return np.array(out)


def fig2_recovery():
    rows = json.loads((RESULTS / "per_pole.json").read_text())
    for r in rows:
        r["true_pole"] = tuple(r["true_pole"])
    panels = [
        ("single tied layer (LNO v1), 10 seeds", "layer", ("tied",),
         [("stable", "common", "published"), ("stable", "common+plr10", "pole lr × 10"),
          ("band", "common+plr10", "band init + pole lr × 10"), ("warm", "common", "warm start")]),
        ("deep ALNO + LFNO, closest pooled pole (optimistic upper bound), 5 seeds each", "deep", ("alno", "lfno"),
         [("stable", "common", "published"), ("stable", "common+plr10", "pole lr × 10")]),
    ]
    labels = [f"{s}\n{z[0]:+g}{z[1]:+g}i" if z[1] else f"{s}\n{z[0]:+g}" for s, z in POLES]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
    x = np.arange(len(POLES))
    for ax, (title, setting, models, conds) in zip(axes, panels):
        w = 0.8 / len(conds)
        for i, (init, proto, lab) in enumerate(conds):
            y = _recovery(rows, setting, models, init, proto)
            ax.bar(x - 0.4 + w * (i + 0.5), y, width=w - 0.03, color=C[i], label=lab)
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("recovered")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.legend(ncol=len(conds), fontsize=7.5, loc="upper left", handlelength=1.2, columnspacing=1.0)
        ax.axvspan(1.5, 3.5, color=GRID, alpha=0.35, zorder=0, linewidth=0)
        ax.axvspan(5.5, 7.5, color=GRID, alpha=0.35, zorder=0, linewidth=0)
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_xlabel("true pole (shaded: real poles)")
    _save(fig, "fig2_recovery")


# ---- fig 3 ----------------------------------------------------------------------

def fig3_profile():
    prof = json.loads((RESULTS / "identifiability.json").read_text())["profile"]
    order = [(-0.3, 4.0), (-0.5, 1.5), (-1.0, 0.0), (-2.0, 0.0)]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for i, z in enumerate(order):
        rs = sorted([r for r in prof if r["system"] == "p6" and tuple(r["pole"]) == z], key=lambda r: r["factor"])
        c = [r["factor"] for r in rs]
        e = [r["e"] for r in rs]
        lab = f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
        ax.plot(c, e, color=C[i], linewidth=2, marker="o", markersize=4, label=lab)
    floor = min(r["e"] for r in prof if r["system"] == "p6" and r["factor"] == 1.0)
    ax.axhline(floor, color=INK2, linewidth=1, linestyle="--")
    ax.text(3.0, floor * 0.92, "representation floor", color=INK2, fontsize=7.5, ha="right", va="top")
    ax.axhline(8e-3, color=INK2, linewidth=1, linestyle=":")
    ax.text(3.0, 8e-3 * 1.15, "test error of good-fit E1 runs", color=INK2, fontsize=7.5, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.6, None)
    ax.set_xlabel("Re μ fixed at c × truth")
    ax.set_ylabel("best RMS relative L2 error")
    ax.set_title("p6, tied layer: profile of each pole's decay rate", fontsize=9, color=INK, loc="left")
    ax.legend(title="true pole", fontsize=7.5, title_fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, "fig3_profile")


if __name__ == "__main__":
    fig1_travel()
    fig2_recovery()
    fig3_profile()
    print("written:", sorted(p.name for p in FIG.glob("*.png")))
