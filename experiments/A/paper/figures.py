"""Figures for the (A) working papers (`a1_paper.qmd`, `a2_paper.qmd`), from the committed result files only.

    python experiments/A/paper/figures.py      # writes experiments/A/paper/fig_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LAYER = HERE.parent / "layer" / "results"
DEEP = HERE.parent / "deep" / "results"
# categorical order (fixed): WPO, TFNO, FNO, control; truth in neutral ink
C = {"wpo": "#2a6fdb", "tfno": "#e07b28", "fno": "#8e5fc2", "ctrl": "#3fa37c", "truth": "#444444"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "legend.frameon": False})


def load(p):
    return json.loads(Path(p).read_text())


def fig_linear():
    a3 = load(LAYER / "a3_twostage.json")["runs"]
    a4 = load(LAYER / "a4_beam.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    k = np.linspace(0, 16, 200)
    for i, (system, truth) in enumerate((("diff", lambda k: -0.1 * k ** 2), ("adr", lambda k: -0.05 * k ** 2 - 0.2))):
        r = next(r for r in a3 if r["system"] == system and r["data_seed"] == 0)
        ks = np.array([m["k"] for m in r["modes"]])
        re = np.array([m["mu"][0] for m in r["modes"]])
        kept = np.isin(ks, r["stage2"]["kept"])
        ax[i].plot(k, truth(k), color=C["truth"], lw=1, label="truth")
        ax[i].plot(ks[kept], re[kept], "o", ms=4, color=C["wpo"], label="stage 1, kept")
        ax[i].plot(ks[~kept], re[~kept], "o", ms=4, mfc="none", color=C["wpo"], label="stage 1, cut")
        ax[i].set(title=f"{system}: Re μ(k)", xlabel="k")
    ax[0].legend(fontsize=7)
    r = a4[0]
    ks = np.array([m["k"] for m in r["modes"]])
    for j in range(2):
        im = np.array([m["mu"][j][1] for m in r["modes"]])
        ok = ks >= 4
        ax[2].plot(ks[ok], im[ok], "o", ms=4, color=C["wpo"], label="stage 1, kept (a5)" if j == 0 else None)
        ax[2].plot(ks[~ok], im[~ok], "o", ms=4, mfc="none", color=C["wpo"], label="stage 1, cut" if j == 0 else None)
    ax[2].plot(k, 0.04 * k ** 2, color=C["truth"], lw=1, label="±c k²")
    ax[2].plot(k, -0.04 * k ** 2, color=C["truth"], lw=1)
    ax[2].set(title="beam: Im μ(k), seed 0", xlabel="k", ylim=(-12, 12))
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_linear.png", dpi=200)


def fig_data_eff():
    d = load(DEEP / "a14_data_eff.json")
    rows = d["runs"] + d["n200"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for kind, label in (("wpo", "WPO"), ("tfno", "TFNO (matched)")):
        ns = sorted({r["n"] for r in rows})
        v = [[r["test"] for r in rows if r["model"] == kind and r["n"] == n] for n in ns]
        m = np.array([np.mean(x) for x in v])
        lo, hi = m - np.array([min(x) for x in v]), np.array([max(x) for x in v]) - m
        ax.errorbar(ns, m, yerr=[lo, hi], marker="o", ms=4, capsize=2, color=C[kind], label=label)
    ax.set(xscale="log", yscale="log", xlabel="training samples N", ylabel="test relative L2", title="Burgers: data efficiency")
    ax.set_xticks([50, 100, 200, 400], ["50", "100", "200", "400"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.02, 0.03, 0.04, 0.06], ["0.02", "0.03", "0.04", "0.06"])
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fig_data_eff.png", dpi=200)


def fig_probe():
    a13 = load(DEEP / "a13_baselines.json")
    fig, ax = plt.subplots(figsize=(4.3, 2.6))
    ax = [ax]
    k = np.linspace(0, 16, 200)
    ax[0].axvspan(0.5, 8.5, color="#f2f2f2", lw=0)
    ax[0].plot(k, -0.1 * k ** 2, color=C["truth"], lw=1, label="−ν k²")
    for r, key, label in ((a13["wpo_a11"][0], "wpo", "WPO"), (next(x for x in a13["final"] if x["model"] == "tfno"), "tfno", "TFNO"),
                          (next(x for x in a13["final"] if x["model"] == "fno_big"), "fno", "FNO 18×")):
        ks = np.array([m["k"] for m in r["probe"]["modes"]])
        re = np.array([m["mu"][0] for m in r["probe"]["modes"]])
        ax[0].plot(ks, re, "o-", ms=3, lw=1, color=C[key], label=label)
    ax[0].axvline(10, color="#999999", lw=0.8, ls="--")
    ax[0].text(10.2, -27, "resolution cut", fontsize=7, color="#666666")
    ax[0].set(title="Burgers probe, seed 0 (shaded: forced band)", xlabel="k", ylabel="Re μ̂(k)", ylim=(-30, 2))
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_probe.png", dpi=200)


def fig_probe_ch():
    a16 = load(DEEP / "a16_cahn_hilliard.json")["runs"]
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    ax = [None, ax]
    lam = lambda k: 0.1 * k ** 2 * (1 - 0.04 * k ** 2)
    k = np.linspace(0, 8, 200)
    ax[1].axhline(0, color="#bbbbbb", lw=0.8)
    ax[1].plot(k, lam(k), color=C["truth"], lw=1, label="λ(k)")
    for (kind, key, label, mk), dx in zip((("wpo", "wpo", "WPO", "o"), ("wpo_stable", "ctrl", "WPO stable (control)", "s"),
                                            ("tfno", "tfno", "TFNO", "^")), (-0.15, 0.0, 0.15)):
        r = next(x for x in a16 if x["model"] == kind and x["seed"] == 0)
        ks = np.array([m["k"] for m in r["probe"]["modes"]])[:8]
        re = np.array([m["mu"][0] for m in r["probe"]["modes"]])[:8]
        ax[1].plot(ks + dx, re, mk, ms=4, color=C[key], label=label)
    ax[1].set(title="Cahn–Hilliard probe, seed 0 (offset in k)", xlabel="k", ylabel="Re μ̂(k)", ylim=(-6, 1.2))
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_probe_ch.png", dpi=200)


def fig_transfer():
    a34 = load(DEEP / "a34_domain.json")
    a35 = load(DEEP / "a35_tfno_4pi.json")
    series = (([r for r in a34 if r["model"] == "wpo_aug"], "wpo", "-", "WPO-2D, trained on 2π"),
              ([r for r in a34 if r["model"] == "tfno_aug"], "tfno", "-", "TFNO, trained on 2π"),
              ([r for r in a35 if r["model"] == "tfno4_m32"], "tfno", "--", "TFNO, trained on 4π (modes × 2)"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    t = np.arange(1, 22)
    ax[0].plot(t, a34[0]["L_true"], color=C["truth"], lw=2.5, alpha=0.35, label="truth")
    w = np.arange(1, 21)
    for rs, key, ls, label in series:
        L = np.array([r["L"] for r in rs])
        e = np.array([r["roll_per_window"] for r in rs])
        ax[0].plot(t, L.mean(0), ls, color=C[key], label=label)
        ax[1].fill_between(w, e.min(0), e.max(0), color=C[key], alpha=0.15, lw=0)
        ax[1].plot(w, e.mean(0), ls, color=C[key], label=label)
    ax[0].set(xscale="log", yscale="log", xlabel="t", ylabel="L(t) = 2π / ⟨k⟩", title="4π box: length scale (mean of 3 seeds)")
    ax[0].set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax[0].set_yticks([1.5, 2, 3, 4, 6, 8], ["1.5", "2", "3", "4", "6", "8"])
    for a in (ax[0].xaxis, ax[0].yaxis):
        a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].set(xlabel="window", ylabel="relative L2", title="4π box: rollout error (range over seeds)", ylim=(0, 1.6),
              xticks=[1, 5, 10, 15, 20])
    fig.tight_layout()
    fig.savefig(HERE / "fig_transfer.png", dpi=200)


if __name__ == "__main__":
    fig_linear()
    fig_data_eff()
    fig_probe()
    fig_probe_ch()
    fig_transfer()
