"""(C) E3, deep setting, first pass: the bounded-slot design in full ALNO / LFNO stacks.

Models as in the E1 deep setting (``e1_audit.deep_config``: LFNO paper budget, ≈ 5.9k
parameters; ALNO / LFNO with their published steady branches and 16 modes), trained on
the E2 rich data (p4, multisine16, T = 20, dt = 20/1024) with E1's best deep protocol
(``common`` + pole lr x 10, 1000 epochs). Variants:

  plr10    E1 setting: ``stable`` init, all slots complex and free
  bounded  in every layer and channel pair, half of the slots (the last k // 2) are real
           (log grid on [-5, -0.05], random offset per slot) and the others complex with
           |Im mu| >= 2 pi / T and |Re mu| < 0.01 + 0.5 |Im mu| (as ``damp8`` in e2_train.py);
           all poles keep decay >= 0.01 (DELTA);
           after Adam, an L-BFGS polish (e2_polish.py). No multi-start, no restarts
           (hundreds of slots).

Pole metrics (folded, true poles of p4):
  pooled    E1: Hungarian match of each true pole to all learned poles of all layers
            (optimistic; with the bounded design the real slots alone cover [-5, -0.05])
  weighted  the same over slots carrying >= 2 % of their layer's summed impulse-response
            weight |beta| ||e^{mu t}||_2

    python experiments/C/theory/e3_deep.py --kind alno lfno --variant plr10 bounded --seeds 0 1 2 3 4
    python experiments/C/theory/e3_deep.py --summarize        # writes results/e3_deep.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.parametrize  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_polish as P  # noqa: E402

T, E = P.T, P.E
from lfo.metrics import pole_error  # noqa: E402
from lfo.training import count_params, seed_everything  # noqa: E402

COND = "p4_ms16_T20"
OUT = T.R.RESULTS / "e3_deep"
C_DAMP, WEIGHT, DELTA, DEAD = 0.5, 0.02, 0.01, 0.5


class DeepBounded(torch.nn.Module):
    """Per (in, out) pair: slots with mask 0 real, Re = -(delta + |Re~|); others complex with
    Im = w_min + |Im~| and Re = -(delta + c Im sigmoid(Re~)). Every pole keeps decay >= delta.
    (A free real Re let LFNO real slots turn unstable in the first epoch: loss 600, then no fit.
    With delta = 0, LFNO real slots went to mu -> 0 (|mu| ~ 1e-12 by epoch 74) and the steady
    response beta / (i omega - mu) at omega = 0 overflowed: NaN in 5/5 seeds.)"""

    def __init__(self, mask, w_min, c, delta):
        super().__init__()
        self.register_buffer("mask", mask)
        self.w_min, self.c, self.delta = w_min, c, delta

    def forward(self, mu):
        im = self.w_min + mu.imag.abs()
        re_c = -self.c * im * torch.sigmoid(mu.real)
        re = self.mask * re_c - (1 - self.mask) * mu.real.abs() - self.delta
        return torch.complex(re, self.mask * im)


def bound(model, t, rng):
    w_min = 2 * np.pi / (float(t[-1]) + float(t[1] - t[0]))
    for pr in model.pr:
        k = pr.mu.shape[-1]
        kr = k // 2
        mask = torch.ones(k)
        mask[k - kr:] = 0
        with torch.no_grad():
            mu0 = pr.mu.clone()
            im = mu0.imag.abs().clamp_min(w_min)
            q = ((mu0.real.abs() - DELTA).clamp_min(0) / (C_DAMP * im)).clamp(0.02, 0.98)
            raw = torch.complex(torch.log(q / (1 - q)), im - w_min)
            u = rng.uniform(size=mu0[..., k - kr:].shape)
            re = -np.exp(np.log(0.05) + (np.arange(kr) + u) / kr * np.log(100.0))
            raw[..., k - kr:] = torch.tensor(re + DELTA, dtype=torch.float32).to(torch.cfloat)
            pr.mu.copy_(raw)
        torch.nn.utils.parametrize.register_parametrization(pr, "mu", DeepBounded(mask, w_min, C_DAMP, DELTA))


def layer_poles(model, T_end):
    """(mu, beta, weight share within its layer) over all layers, flattened."""
    out = []
    for pr in model.pr:
        mu, beta = (a.detach().cpu().numpy().ravel() for a in (pr.mu, pr.beta))
        re = np.minimum(mu.real, -1e-6)
        w = np.abs(beta) * np.sqrt((1 - np.exp(2 * re * T_end)) / (-2 * re))
        out.append((mu, beta, w / w.sum()))
    return [np.concatenate(z) for z in zip(*out)]


def run(kind, variant, seed, device):
    data, t, mu_true, beta_true = T.setup(COND, seed)
    seed_everything(seed)
    cfg = E.deep_config(kind, t)
    model = E._deep_model(kind, cfg["width"], cfg["k"], t, "stable")
    if variant == "bounded":
        bound(model, t, np.random.default_rng(seed))
    model.to(device)
    t0 = time.time()
    hist = E.train(model, data, E.PROTOCOLS["common"], device, 10.0)
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    test_adam = E.evaluate(model, xte, yte)
    info = P.polish(model, data, device) if variant == "bounded" else {}
    test = E.evaluate(model, xte, yte)
    mu, beta, share = layer_poles(model, float(t[-1]) + float(t[1] - t[0]))
    mt, _ = E.true_folded(mu_true, beta_true)
    finite = np.isfinite(mu) & np.isfinite(beta)
    mf, _ = E.fold(mu[finite], beta[finite])
    big = finite & (share >= WEIGHT)
    mw, _ = E.fold(mu[big], beta[big])
    res = dict(kind=kind, variant=variant, seed=seed, cond=COND, config=cfg, params=count_params(model),
               seconds=round(time.time() - t0, 1), test_rel_l2=test, test_rel_l2_adam=test_adam, history=hist,
               polish=info, oracle_rel_l2=E.oracle_error(data, t, mu_true, beta_true),
               true_poles=[[float(z.real), float(z.imag)] for z in mt],
               n_poles=int(mu.size), n_weighted=int(big.sum()),
               unstable_fraction=float(np.mean(mu[finite].real > 0)),
               pole_error_pooled=pole_error(mf, mt, reduce="none").tolist(),
               pole_error_weighted=pole_error(mw, mt, reduce="none").tolist() if mw.size else None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{kind}_{variant}_s{seed}.json").write_text(json.dumps(res, indent=1))
    T.RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), T.RUNS / f"e3_deep_{kind}_{variant}_s{seed}.pt")
    r3 = lambda pe: pe and [round(x, 3) for x in pe]
    print(f"{kind} {variant} s{seed}: params {res['params']} test {test_adam:.2e} -> {test:.2e}, pooled "
          f"{r3(res['pole_error_pooled'])}, weighted {r3(res['pole_error_weighted'])} ({res['n_weighted']} of "
          f"{res['n_poles']} slots) ({res['seconds']:.0f}s)", flush=True)


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    poles = [lab(z) for z in rs[0]["true_poles"]]
    lines = ["# E3 deep, first pass: bounded slots in full ALNO / LFNO stacks", "",
             f"p4, multisine16, T = 20, dt = 20/1024; E1 deep budget; `common` + pole lr × 10. Per true pole: "
             f"fraction of seeds with relative error < {T.RECOVERED}, pooled over all slots (E1 metric) / over slots "
             f"with ≥ {WEIGHT:.0%} of their layer's impulse-response weight. Tied-layer oracle "
             f"{rs[0]['oracle_rel_l2']:.1e}.", "",
             "| model | variant | params | seeds | NaN / dead | test rel. L2 (Adam → final) | unstable poles | "
             "slots (weighted / all) | " + " | ".join(f"{p}: pooled / weighted" for p in poles) + " |",
             "|---|---|---|---|---|---|---|---|" + "---|" * len(poles)]
    for kind in ("alno", "lfno"):
        for variant in ("plr10", "bounded"):
            g = [r for r in rs if r["kind"] == kind and r["variant"] == variant]
            if not g:
                continue
            pp = np.array([r["pole_error_pooled"] for r in g])
            pw = np.array([r["pole_error_weighted"] if r["pole_error_weighted"] else [np.inf] * len(poles) for r in g])
            ta, tf = (np.array([r[k] for r in g]) for k in ("test_rel_l2_adam", "test_rel_l2"))
            fit = np.isfinite(tf) & (tf < DEAD)        # trained runs; NaN and dead runs count as not recovered
            pp[~fit], pw[~fit] = np.inf, np.inf
            cells = [f"{np.mean(pp[:, j] < T.RECOVERED):.0%} / {np.mean(pw[:, j] < T.RECOVERED):.0%}"
                     for j in range(len(poles))]
            ms = lambda a: f"{a[fit].mean():.1e} ± {a[fit].std():.1e}" if fit.any() else "–"
            lines.append(f"| {kind.upper()} | {variant} | {g[0]['params']} | {len(g)} | {int(np.sum(~np.isfinite(tf)))} / "
                         f"{int(np.sum(np.isfinite(tf) & (tf >= DEAD)))} | {ms(ta)} → {ms(tf)} | "
                         f"{np.mean([r['unstable_fraction'] for r, f in zip(g, fit) if f]):.0%} | "
                         f"{np.mean([r['n_weighted'] for r, f in zip(g, fit) if f]):.0f} / {g[0]['n_poles']} | "
                         + " | ".join(cells) + " |")
    lines += ["", f"NaN: training diverged to NaN (a slot pole reached mu = 0, where the steady response "
              f"beta / (i omega - mu) at omega = 0 is singular). Dead: test rel. L2 >= {DEAD} (output stays near 0). "
              "Test error, unstable poles and weighted slots: mean over the other runs.", ""]
    (T.R.RESULTS / "e3_deep.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", nargs="+", default=["alno", "lfno"])
    ap.add_argument("--variant", nargs="+", default=["plr10", "bounded"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in a.kind:
        for v in a.variant:
            for s in a.seeds:
                run(k, v, s, device)


if __name__ == "__main__":
    main()
