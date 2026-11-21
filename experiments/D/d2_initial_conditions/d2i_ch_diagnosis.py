"""(D2) Why the TFNO rolls out better on forced 1D Cahn–Hilliard with initial fields (d2g, d2h).

d2g / d2h: the TFNO's 10-window rollout error is about half the IC-path WPO's at 6000 and at 12000 steps (the budget
is not the cause). Its one-window error from true starts (`id`) is also about half. This script decomposes the
rollout error per window to see where the gap comes from.

**Decomposition.** Window k of a rollout starts from the model's own state s^_k instead of the truth s_k. With the
model M, the true solution operator S (the solver, run from s^_k with the window's fine-grid forcing) and
Y = S(s_k) the truth, over the frames of the window:
- `e`: |M(s^_k) - Y| (the rollout error d2g reports);
- `local_own`: |M(s^_k) - S(s^_k)| (the model's error on its own start);
- `inherited`: |S(s^_k) - Y| (the start error carried by the true dynamics);
- `local_true`: |M(s_k) - Y| (the model's error on the true start).
Each is a relative L2 over the window's frames 1..w, divided by |Y|, then averaged over the 50 trajectories. By the
triangle inequality e <= local_own + inherited.
Sensitivity (windows 2..10): with d = |s^_k - s_k|, `amp_true` = |S(s^_k) - Y|_end / d and `amp_model` =
|M(s^_k) - M(s_k)|_end / d, both at the window's end state. If one model amplifies its start error more than the
true dynamics do, its rollout error grows faster than its local error.
Spectrum: the end-state squared error of the `id` set (one window, true starts) split into the bands k <= 4 (the
unstable band is k < 5), 5..8, 9..16, 17..32.

**Setup.** d2g as it stands (6000 steps, `wpo_ic` and `tfno_chan`, 3 seeds, `roll10` set: 50 trajectories,
10 windows, forcing drawn per window). The forcing of `roll10` is regenerated with d2g's generator (same seed) to
run the solver from model states. Check before any fit: re-solving each window from the true start reproduces the
truth (reported as `resolve_err`).

**Predictions** (written before running).

1. Window 10: `inherited` > `local_own` for both models (mean over seeds): by then most of the error is carried.
2. The TFNO / WPO ratio of `local_true`, averaged over windows 1..10, lies in [0.4, 0.7] (the one-window gap).
3. Neither model amplifies its start error much more or less than the truth: `amp_model / amp_true`, averaged over
   windows 2..10 and seeds, lies in [0.67, 1.5] for both models.
4. `wpo_ic`: at least 50 % of the `id` end-state squared error is at k <= 8.

    python experiments/D/d2_initial_conditions/d2i_ch_diagnosis.py      # results/d2i_ch_diagnosis.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2g_ch_rollout as G  # noqa: E402

R, E, M, ch = G.R, G.E, G.M, G.ch
KINDS, SEEDS = ("wpo_ic", "tfno_chan"), (0, 1, 2)
BANDS = ((0, 4), (5, 8), (9, 16), (17, 32))
OUT = HERE / "results" / "d2i_ch_diagnosis.json"


def roll10_forcing():
    """d2g's `long_set` with its seed, keeping the fine forcing spectrum and the initial spectrum."""
    rng = np.random.default_rng(23)
    parts = [G.forcing(rng, 50, M.T1, M.NT1) for _ in range(G.N10)]
    fk = np.concatenate([parts[0][0]] + [p[0][..., 1:] for p in parts[1:]], -1)
    return fk, parts[0][1], E.initial(rng, 50, (1, 8), (0.03, 1.0))


def true_window(fk, dtf, i, s):
    """The solver over window i from the states s (n, NX): (n, NX, w + 1)."""
    w = M.NT1 - 1
    seg = fk[..., i * w * M.FINE:(i + 1) * w * M.FINE + 1]
    uk = ch.solve(seg, dtf, G.MOB, G.KAPPA, M.NX, stride=M.FINE, u0k=np.fft.rfft(s.numpy(), axis=1))
    return torch.tensor(np.fft.irfft(uk, n=M.NX, axis=1))


def nr(a, y):
    return float((a.flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean())


def diagnose(m, sets, fk, dtf, device):
    w = M.NT1 - 1
    f, u0, y = sets["roll10"]
    p = E.rollout(m, f, u0, device, G.N10)
    rows = []
    for i in range(G.N10):
        sl = slice(i * w, (i + 1) * w + 1)
        fi, Y = f[..., sl], y[..., sl]
        s_true, s_hat = y[..., i * w], p[..., i * w]
        M_hat, M_true = p[..., sl], R.predict(m, fi, s_true, device)
        S_hat = true_window(fk, dtf, i, s_hat)
        r = dict(e=nr(M_hat[..., 1:] - Y[..., 1:], Y[..., 1:]),
                 local_own=nr(M_hat[..., 1:] - S_hat[..., 1:], Y[..., 1:]),
                 inherited=nr(S_hat[..., 1:] - Y[..., 1:], Y[..., 1:]),
                 local_true=nr(M_true[..., 1:] - Y[..., 1:], Y[..., 1:]))
        if i > 0:
            d = (s_hat - s_true).norm(dim=1)
            r["amp_true"] = float(((S_hat[..., -1] - Y[..., -1]).norm(dim=1) / d).mean())
            r["amp_model"] = float(((M_hat[..., -1] - M_true[..., -1]).norm(dim=1) / d).mean())
        rows.append(r)
    fi, u0i, yi = sets["id"]
    err = np.abs(np.fft.rfft((R.predict(m, fi, u0i, device) - yi)[..., -1].numpy(), axis=1)) ** 2
    tot = err.sum()
    spec = [float(err[:, a:b + 1].sum() / tot) for a, b in BANDS]
    return rows, spec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = G.make_sets()
    fk, dtf, u0k = roll10_forcing()
    _, u0, y = sets["roll10"]
    assert np.allclose(np.fft.irfft(u0k, n=M.NX, axis=1), u0.numpy())
    resolve = max(nr(true_window(fk, dtf, i, y[..., i * (M.NT1 - 1)]) - y[..., i * (M.NT1 - 1):(i + 1) * (M.NT1 - 1) + 1],
                     y[..., i * (M.NT1 - 1):(i + 1) * (M.NT1 - 1) + 1]) for i in range(G.N10))
    print("resolve_err", resolve, flush=True)
    t = np.linspace(0, M.T1, M.NT1)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for kind in KINDS:
        for seed in SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            m = G.build(kind, seed, t, device)
            R.train(m, dict(train=sets["train"]), device, 30.0 if kind.startswith("wpo") else 1.0)
            m.eval()
            rows, spec = diagnose(m, sets, fk, dtf, device)
            r = dict(model=kind, seed=seed, resolve_err=resolve, windows=rows, id_spectrum=spec,
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print(kind, seed, "e", [round(q["e"], 3) for q in rows], "spec", np.round(spec, 3), flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
            del m
            torch.cuda.empty_cache()
    q = lambda kind, key: np.mean([[w.get(key, np.nan) for w in r["windows"]] for r in out if r["model"] == kind], 0)
    lines = ["# (D2) Forced Cahn–Hilliard: rollout error decomposition (d2g setup, 6000 steps)", "",
             f"Re-solving each window from the true start reproduces the truth to {resolve:.1e}. Means over 3 seeds.", ""]
    for kind in KINDS:
        lines += [f"**{kind}**", "", "| window | " + " | ".join(str(i + 1) for i in range(G.N10)) + " |",
                  "|---|" + "---|" * G.N10]
        for key in ("e", "local_own", "inherited", "local_true", "amp_true", "amp_model"):
            lines.append(f"| {key} | " + " | ".join("—" if np.isnan(v) else f"{v:.3f}" for v in q(kind, key)) + " |")
        lines.append("")
    lt = {k: np.mean(q(k, "local_true")) for k in KINDS}
    lines.append(f"TFNO / WPO ratio of local_true (mean over windows): {lt['tfno_chan'] / lt['wpo_ic']:.2f}.")
    for kind in KINDS:
        ratio = np.nanmean(q(kind, "amp_model")[1:] / q(kind, "amp_true")[1:])
        lines.append(f"{kind}: amp_model / amp_true (windows 2..10) {ratio:.2f}.")
    lines += ["", "End-state squared error of the `id` set by band (fraction of the total, mean over seeds):", "",
              "| model | " + " | ".join(f"k {a}..{b}" for a, b in BANDS) + " |", "|---|" + "---|" * len(BANDS)]
    for kind in KINDS:
        sp = np.mean([r["id_spectrum"] for r in out if r["model"] == kind], 0)
        lines.append(f"| {kind} | " + " | ".join(f"{v:.3f}" for v in sp) + " |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
