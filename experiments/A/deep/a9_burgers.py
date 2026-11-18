"""(A) First deep validation: the deep wavenumber pole operator on forced viscous Burgers, against FNO, with a linear
probe of the learned operator.

Plan: `docs/proposal_A_wavenumber_pole_operator.qmd`, Section 4.4 (deep model, post-hoc linear probe) and RQ2
(time extrapolation). The linear recipe is settled (`experiments/A/layer`, a3-a8); this is the first step beyond one
linear layer.

**System.** u_t + u u_x = nu u_xx + f on periodic x in [0, 2 pi), zero initial state, nu = 0.1
(`lfo.data.burgers`: pseudo-spectral, 2/3 dealiasing, integrating-factor RK4 on a 32x finer time grid). Linearised
at u = 0: lambda(k) = -nu k^2.

**Data.**

- n_x = 64, T = 5, n_t = 201. Forcing: 4 random spatial modes, k in 1..8, amplitudes ~ s / k, with E1-family time
  profiles (as in `experiments/A/layer/a1_dispersion.py`). s is per sample, log-uniform in [0.03, 1].
- Measured before any training: the part of u not explained by the linear (diffusion) response is 0.5 % / 6.5 % /
  39 % at the 10th / 50th / 90th percentile, and up to 91 %. So the data mix near-linear and strongly nonlinear
  samples.
- Splits 200 / 50 / 100. Extrapolation set: 100 samples on T = 10 (n_t = 401), same distribution.

**Models** (3 seeds each, relative L2 loss, Adam lr 1e-3 with cosine decay, 4000 steps, batch 25):

- WPO: `lfo.models.WavenumberPoleOperator1d`, width 24, 4 layers, 6 poles per layer. Layer options: modes 22
  (= the dealiased band), Re cap 0.5 with leak 0.05, k_ref 16. float64.
- FNO: `neuralop.models.FNO` on the (x, t) grid, n_modes (12, 12), 4 layers, domain padding 0.25 in t (t is not
  periodic). Its width is the one whose parameter count is closest to the WPO's (within 10 %). float32.

**Metrics.**

1. Test relative L2 on T = 5.
2. Time extrapolation: relative L2 on the T = 10 set, over t in (5, 10]. The WPO runs unchanged on the longer grid
   (FOH recursion). The FNO is evaluated on the longer grid as is, which is what a user of a trained FNO would do.
3. Linear probe at u = 0.
   - Probe inputs: one spatial mode k = 1..16 each (cos(k x + phi)), 8 inputs per k, E1-family time profiles.
     Amplitude eps / k, eps = 0.01 (s = 0.03 is the smallest training scale).
   - Odd part of the operator: g = (G(eps u) - G(-eps u)) / 2, which removes the bias and the quadratic term.
   - From the pairs (u, g), the per-mode identification of `experiments/A/layer/a2_twostage.py` (one pole per k,
     grid + Nelder-Mead), the resolution and observability cuts of a5, then nu_hat by weighted least squares
     (Re mu on {1, k^2}).
   - Reported: nu_hat, the dispersion error over the kept modes, and the per-mode residual (how far the odd part is
     from LTI).
   - The same probe applied to the solver itself checks the method.
4. Reported, not predicted: the first-layer WPO poles closest to -nu k^2.

**Method check, measured before training** (a 30-step smoke run): the solver's probe gives nu_hat = 0.10046
(0.5 %), a dispersion error of 1.8 % over the kept modes k = 2..9 (k = 1 fails the observability cut, k >= 10 the
resolution cut). This is the floor for the models' probes.

**Predictions** (written before running).

1. Accuracy: the WPO's mean test error is <= 1.5x the FNO's.
2. Time extrapolation: the WPO's error over (5, 10] is <= 2x its own T = 5 test error in every seed. The FNO's is
   > 2x its own in every seed.
3. Probe: in at least 2 of 3 seeds, the WPO's probe gives nu_hat within 5 % and a dispersion error <= 10 %. In
   every seed, the FNO's probe dispersion error is larger than the WPO's (a probe with no kept mode counts as
   larger).

    python experiments/A/deep/a9_burgers.py      # results/a9_burgers.md / .json; checkpoints in runs/A/a9
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "layer"))
import a1_dispersion as A  # noqa: E402
import a2_twostage as B  # noqa: E402
from lfo.data import burgers  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

NU, NX, T1, T2, NT1, NT2, FINE = 0.1, 64, 5.0, 10.0, 201, 401, 32
SPLIT, N_EXT = (200, 50, 100), 100
S_RANGE, BAND = (0.03, 1.0), 8
STEPS, LR, BATCH, SEEDS = 4000, 1e-3, 25, (0, 1, 2)
WPO_KW = dict(width=24, n_layers=4, n_poles=6, modes=22, re_max=0.5, re_leak=0.05, k_ref=16.0)
FNO_MODES = (12, 12)
EPS, N_PER_K, KS = 0.01, 8, np.arange(1, 17)
CUT, OBS = 0.25, 2.0
OUT = HERE / "results" / "a9_burgers.json"
CKPT = ROOT / "runs" / "A" / "a9"


def forcing(rng, n, T, n_t):
    """Fine-grid forcing spectrum (n, K, (n_t - 1) * FINE + 1)."""
    tf = np.linspace(0, T, (n_t - 1) * FINE + 1)
    fk = np.zeros((n, NX // 2 + 1, tf.size), complex)
    s = np.exp(rng.uniform(np.log(S_RANGE[0]), np.log(S_RANGE[1]), size=n))
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        km = rng.integers(1, BAND + 1, size=n)
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * NX / 2 * s
        fk[np.arange(n), km] += amp[:, None] * prof
    return fk, tf[1] - tf[0]


def fields(fk, dtf):
    uk = burgers.solve(fk, dtf, NU, NX, stride=FINE)
    f = np.fft.irfft(fk[..., ::FINE], n=NX, axis=1)
    u = np.fft.irfft(uk, n=NX, axis=1)
    return torch.tensor(f), torch.tensor(u)


def make_data():
    rng = np.random.default_rng(0)
    f, u = fields(*forcing(rng, sum(SPLIT), T1, NT1))
    s = np.cumsum([0, *SPLIT])
    data = {k: (f[a:b], u[a:b]) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}
    data["ext"] = fields(*forcing(np.random.default_rng(1), N_EXT, T2, NT2))
    return data


def probe_inputs():
    """Single-mode probe forcings: fine-grid spectrum (len(KS) * N_PER_K, K, n_tf)."""
    rng = np.random.default_rng(2)
    n = len(KS) * N_PER_K
    tf = np.linspace(0, T1, (NT1 - 1) * FINE + 1)
    fk = np.zeros((n, NX // 2 + 1, tf.size), complex)
    prof = A.profiles(tf, rng, n)
    ks = np.repeat(KS, N_PER_K)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))
    fk[np.arange(n), ks] = (EPS / ks * NX / 2 * phase)[:, None] * prof
    return fk, tf[1] - tf[0]


class FNOWrap(torch.nn.Module):
    """neuralop FNO on the (x, t) grid; (batch, n_x, n_t) in and out."""

    def __init__(self, width):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=FNO_MODES, in_channels=1, out_channels=1, hidden_channels=width, n_layers=4,
                       domain_padding=[0.0, 0.25])

    def forward(self, f):
        return self.net(f.float()[:, None])[:, 0].to(f.dtype)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(kind, seed, t, device):
    torch.manual_seed(seed)
    if kind == "wpo":
        return WavenumberPoleOperator1d(t, **WPO_KW).to(device)
    target = n_params(WavenumberPoleOperator1d(t, **WPO_KW))
    width = min(range(4, 40), key=lambda w: abs(n_params(FNOWrap(w)) - target))
    torch.manual_seed(seed)
    return FNOWrap(width).to(device)


def batched(model, x, bs=10):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs]) for i in range(0, x.shape[0], bs)])


def train(model, data, device):
    xtr, ytr = data["train"]
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(0)
    for step in range(STEPS):
        idx = torch.randint(0, xtr.shape[0], (BATCH,), generator=g)
        opt.zero_grad()
        loss = A.rel(model(xtr[idx].to(device)), ytr[idx].to(device))
        loss.backward()
        opt.step()
        sch.step()
    return float(loss)


def probe(G, fk, dtf, device):
    """Per-mode poles of the odd part of G (a callable on fine-grid spectra -> sampled (f, g) fields)."""
    f_plus, g_plus = G(fk)
    _, g_minus = G(-fk)
    pdata = {"train": (f_plus, 0.5 * (g_plus - g_minus))}
    dt = T1 / (NT1 - 1)
    modes = B.per_mode(pdata, dt, device)
    kept = [m for m in modes if abs(complex(*m["mu"])) * dt <= CUT and max(map(abs, m["mu"])) * T1 >= OBS]
    if not kept:
        return dict(modes=modes, kept=[], nu_hat=float("nan"), offset=float("nan"), disp_err=float("inf"),
                    resid_median=float("nan"))
    ks = np.array([m["k"] for m in kept], float)
    mu = np.array([complex(*m["mu"]) for m in kept])
    w = 1 / np.array([max(m["resid"], 1e-6) for m in kept]) ** 2
    V = np.stack([ks ** 0, ks ** 2], 1)
    sw = np.sqrt(w)
    a, *_ = np.linalg.lstsq(V * sw[:, None], mu.real * sw, rcond=None)
    true = -NU * ks ** 2
    return dict(modes=modes, kept=[int(k) for k in ks], nu_hat=float(-a[1]), offset=float(a[0]),
                disp_err=float(np.linalg.norm(mu - true) / np.linalg.norm(true)),
                resid_median=float(np.median([m["resid"] for m in kept])))


def model_probe(model, fk, dtf, device):
    def G(spec):
        f = torch.tensor(np.fft.irfft(spec[..., ::FINE], n=NX, axis=1))
        return f, batched(model, f.to(device)).cpu().to(torch.float64)
    return probe(G, fk, dtf, device)


def first_layer_poles(model, device):
    k = torch.tensor(KS, dtype=torch.float64, device=device)
    with torch.no_grad():
        mu = model.K[0].mu(k).cpu().numpy()                            # (N, len(KS))
    lam = -NU * KS ** 2
    best = np.abs(mu - lam).argmin(0)
    near = mu[best, np.arange(len(KS))]
    return dict(rel_err=[float(e) for e in np.abs(near - lam) / np.abs(lam)],
                poles_k1_k4_k8=[[[float(z.real), float(z.imag)] for z in mu[:, j]] for j in (0, 3, 7)])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = make_data()
    t = np.linspace(0, T1, NT1)
    pfk, pdt = probe_inputs()
    solver = probe(lambda spec: fields(spec, pdt), pfk, pdt, device)
    print("solver probe", f"nu_hat {solver['nu_hat']:.5f}", f"disp {solver['disp_err']:.2e}", "kept", solver["kept"],
          f"data {time.time() - t0:.0f} s", flush=True)
    out = dict(solver_probe=solver, runs=[])
    n_ext = NT1 - 1                                                       # index of t = 5 on the T = 10 grid
    for kind in ("wpo", "fno"):
        for seed in SEEDS:
            t1 = time.time()
            model = build(kind, seed, t, device)
            last = train(model, data, device)
            torch.save(model.state_dict(), CKPT / f"{kind}_{seed}.pt")
            model.eval()
            xte, yte = data["test"]
            test = float(A.rel(batched(model, xte.to(device)).cpu().to(torch.float64), yte))
            xe, ye = data["ext"]
            pe = batched(model, xe.to(device)).cpu().to(torch.float64)
            ext_late = float(A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:]))
            ext_early = float(A.rel(pe[..., :n_ext + 1], ye[..., :n_ext + 1]))
            pr = model_probe(model, pfk, pdt, device)
            r = dict(model=kind, seed=seed, params=n_params(model), train_loss=last, test=test, ext_early=ext_early,
                     ext_late=ext_late, probe=pr, seconds=round(time.time() - t1, 1))
            if kind == "wpo":
                r["first_layer"] = first_layer_poles(model, device)
            out["runs"].append(r)
            print(kind, seed, f"params {r['params']}", f"test {test:.3e}", f"ext (0,5] {ext_early:.3e} (5,10] {ext_late:.3e}",
                  f"probe nu {pr['nu_hat']:.5f} disp {pr['disp_err']:.2e} resid {pr['resid_median']:.2e} kept {pr['kept']}",
                  f"{r['seconds']} s", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1, default=float))
            del model
            torch.cuda.empty_cache()
    summarize(out)


def summarize(out):
    s = out["solver_probe"]
    lines = ["# (A) Deep wavenumber pole operator on forced Burgers (ν = 0.1), against FNO", "",
             f"Solver probe (method check): ν̂ = {s['nu_hat']:.5f}, dispersion error {s['disp_err']:.1e}, kept k = "
             f"{s['kept'][0]}–{s['kept'][-1]}.", "",
             "Probe: kept k listed per run in the JSON; no kept mode → ν̂ = nan, dispersion error inf.", "",
             "| model | seed | params | test (T = 5) | T = 10: (0, 5] / (5, 10] | probe ν̂ | probe dispersion error | "
             "probe residual (median) |", "|---|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        p = r["probe"]
        lines.append(f"| {r['model']} | {r['seed']} | {r['params']} | {r['test']:.3e} | {r['ext_early']:.3e} / "
                     f"{r['ext_late']:.3e} | {p['nu_hat']:.5f} | {p['disp_err']:.1e} | {p['resid_median']:.1e} |")
    lines += ["", "WPO first layer: relative error of the pole closest to −ν k² at k = 1, 4, 8, 16:", ""]
    for r in out["runs"]:
        if "first_layer" in r:
            e = r["first_layer"]["rel_err"]
            lines.append(f"- seed {r['seed']}: {e[0]:.2f}, {e[3]:.2f}, {e[7]:.2f}, {e[15]:.2f}")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
