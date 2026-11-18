"""(C) E5, eleventh pass, part A: why are trained LNO Jacobians non-causal? (diagnostic, no training)

Hypothesis. LNO v1 computes y = steady (periodic, FFT) + transient. For one pole the periodic steady response
wraps the tail of the impulse response around the window, and the exact causal response is
y = y_periodic - C e^{mu t} with a coefficient C fixed by the whole input. The textbook layer (`PoleResidueTS1d`,
the tied layer) attaches that coefficient to the same pole, so the wrap cancels and the layer is causal (up to
discretisation). The original LNO code contracts the residue's output channel again with the pole's input channel
(`"bix,ioxz->boz"`, quirk 1 in `lfo.models.lno`), so each coefficient multiplies another pole's exponential and the
wrap is not cancelled.

Test, for every trained LNO checkpoint of the E5 sets:

- `orig`: non-causal share of the linear map of the LNO's pole-residue layer `conv0` (all channels, by autograd);
- `textbook`: the same weights wired as `PoleResidueTS1d` (coefficient on its own pole);
- `full`: the full model's Jacobian share from the probe results.

Prediction (written before running): `textbook` < 0.05 in every run; `orig` > 0.1 in most runs; the full-model
share tracks `orig`.

    python experiments/C/e5_probe/e5_causal.py [set ...]      # writes results/e5_causal.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e5_gate as G  # noqa: E402
import e5_probe as P  # noqa: E402
import e5_train as TR  # noqa: E402

from lfo.models.hybrid import PoleResidueTS1d  # noqa: E402

RL = P.RL
LNO_VARIANTS = ("base", "lno_plr10", "lno_stable_plr10")
OUT = P.RESULTS / "e5_causal.json"


def textbook_of(conv0):
    """PoleResidueTS1d with the LNO layer's poles and residues (and its sign: both use -Hw / n)."""
    w, _, k = conv0.weights_pole.shape
    tb = PoleResidueTS1d(w, w, k, conv0.t.cpu())
    with torch.no_grad():
        tb.mu.copy_(conv0.weights_pole)
        tb.beta.copy_(conv0.weights_residue)
    return tb.to(conv0.weights_pole.device)


@torch.no_grad()
def noncausal_share(layer, width, n, device, chunk=512):
    """Share of the layer's (out channel, t_out; in channel, t_in) Jacobian with t_out < t_in. The layer is linear
    with no bias, so its Jacobian columns are its responses to unit impulses (computed in chunks)."""
    up = tot = 0.0
    z = torch.arange(n, device=device)
    for i in range(width):
        for x0 in range(0, n, chunk):
            xs = torch.arange(x0, min(x0 + chunk, n), device=device)
            e = torch.zeros(xs.numel(), width, n, device=device)
            e[torch.arange(xs.numel()), i, xs] = 1.0
            a = layer(e).pow(2)                                                  # (k, o, z)
            mask = (z[None, :] < xs[:, None])[:, None, :]                        # t_out < t_in
            up += float((a * mask).sum())
            tot += float(a.sum())
    return (up / tot) ** 0.5


def check_equivalence(model, conv0, device):
    """With the transient's channel indices matched, both forms must agree on a channel-diagonal pole set."""
    tb = textbook_of(conv0)
    x = torch.randn(2, conv0.weights_pole.shape[0], conv0.t.numel(), device=device)
    with torch.no_grad():
        st_o = torch.real(torch.fft.ifft(conv0.output_PR(
            (torch.fft.fftfreq(conv0.t.numel(), float(conv0.t[1] - conv0.t[0])) * 2 * np.pi * 1j).to(device)
            [:, None, None, None], torch.fft.fft(x))[0], n=x.size(-1)))
        st_t = torch.fft.irfft(tb.steady_coeffs(x), n=x.size(-1))
    return float((st_o - st_t).norm() / st_o.norm())                       # steady parts must be identical


def evaluate(tag, device):
    pr = json.loads((P.RESULTS / f"e5_probe_{tag}.json").read_text())
    _, grids = TR.data_and_grid(tag)
    case = RL.CASES[TR.case_of(tag)]
    rows = []
    for v in LNO_VARIANTS:
        for r in pr["variants"][v]:
            model, _, _ = P.load_variant(v, r["seed"], case, grids, device, tag)
            for q in model.parameters():
                q.requires_grad_(False)
            c0 = model.conv0
            w, n = c0.weights_pole.shape[0], c0.t.numel()
            tb = textbook_of(c0)
            rows.append(dict(variant=v, seed=r["seed"], full=r["jac"]["noncausal"],
                             orig=noncausal_share(c0, w, n, device), textbook=noncausal_share(tb, w, n, device),
                             steady_diff=check_equivalence(model, c0, device),
                             unstable=float((c0.weights_pole.real > 0).float().mean())))
            print(tag, v, r["seed"], {k: round(x, 3) for k, x in rows[-1].items() if isinstance(x, float)}, flush=True)
    return rows


def summarize(res):
    lines = ["# E5 eleventh pass, part A: non-causality of the LNO pole-residue layer", "",
             "Non-causal share (‖upper-triangular part‖ / ‖J‖ over all channel pairs) of the trained LNO's pole-residue "
             "layer as coded (`orig`) and with the same weights in the textbook wiring (`textbook`, each transient "
             "coefficient on its own pole); `full`: the whole model's small-signal Jacobian (probe results). "
             "`unstable`: share of stored poles with Re μ > 0.", "",
             "| set | variant | full: median (range) | orig: median (range) | textbook: median (range) | unstable |",
             "|---|---|---|---|---|---|"]
    rng = lambda xs: f"{np.median(xs):.3f} ({min(xs):.3f}–{max(xs):.3f})"
    allr = []
    for tag, rows in res.items():
        for v in LNO_VARIANTS:
            rs = [r for r in rows if r["variant"] == v]
            allr += rs
            lines.append(f"| `{tag}` | {v} | {rng([r['full'] for r in rs])} | {rng([r['orig'] for r in rs])} | "
                         f"{rng([r['textbook'] for r in rs])} | {np.mean([r['unstable'] for r in rs]):.2f} |")
    f, o, t = (np.array([r[k] for r in allr]) for k in ("full", "orig", "textbook"))
    lines += ["", f"All {len(allr)} LNO runs: textbook < 0.05 in {int((t < 0.05).sum())}, orig > 0.1 in "
              f"{int((o > 0.1).sum())}; Spearman correlation of full vs orig {spearman(f, o):.2f}, full vs textbook "
              f"{spearman(f, t):.2f}. Max steady-term difference between the two wirings: "
              f"{max(r['steady_diff'] for r in allr):.1e} (identical by construction)."]
    (P.RESULTS / "e5_causal.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for tag in tags:
        res[tag] = evaluate(tag, device)
    res = {t: res[t] for t in G.SETS if t in res}
    OUT.write_text(json.dumps(res, indent=1))
    summarize(res)


if __name__ == "__main__":
    main(sys.argv[1:] or G.SETS)
