"""(A) Pole layer, fourth validation: the Euler-Bernoulli beam, two poles per wavenumber (N = 2).

u_tt = -c^2 u_xxxx + f on periodic x in [0, 2 pi), zero initial state. The output is the velocity v = u_t, so each
mode has the transfer function s / (s^2 + c^2 k^4) = (1/2) [1 / (s - i c k^2) + 1 / (s + i c k^2)]: poles
+-i c k^2 on the imaginary axis (undamped), residues 1/2. The displacement would need residues 1 / (2 i c k^2),
which polynomial residues cannot express (see `lfo.models.wavenumber`). c = 0.04: frequencies up to 10.2 rad/s at
k = 16.

Data as in `a1_dispersion.py`: n_x = 64, T = 5, n_t = 201; 4 forced spatial modes in k = 1..16 with E1-family time
profiles; truth on a 32x finer time grid; data seeds 0-4.

Recipe (the `a3_twostage.py` recipe, extended to two poles per mode):

1. **Per mode, two poles.**
   - Grid for one pole (as in a2).
   - With it fixed, grid for a second pole; both residues by least squares.
   - Nelder-Mead on all four real parameters.
   - The poles are labelled by the sign of Im mu, with no truth used.
2. **Polynomials.** Weighted least squares per label on the modes with max |mu| dt <= 0.25. Re mu on
   {1, k^2, k^4}, Im mu on {k, k^2, k^3}, residue on {1, k^2}. No degree selection: a3 showed that the fine-tune
   removes extra terms.
3. **Fine-tune.** `WavenumberPoleLayer1d(2, im_powers=(1, 2, 3), k_ref=16)` with L-BFGS, 200 iterations.

Predictions (written before running):

1. Stage 1: for k >= 3, both poles of every mode within 2 % of +-i c k^2, in every data seed. The modes k = 1, 2
   (c k^2 T = 0.2 and 0.8 rad over the window: less than a period) are reported, not predicted.
2. Fine-tuned: c (the k^2 coefficient of Im mu of the +i pole) within 1 % in at least 4 of 5 seeds;
   |Re mu| <= 0.01 |Im mu| for k = 3..16; test error <= 1.2x the oracle floor in at least 4 of 5 seeds.

    python experiments/A/layer/a4_beam.py      # results/a4_beam.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a1_dispersion as A  # noqa: E402
import a2_twostage as B  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d, foh_kernel, foh_response  # noqa: E402

C_BEAM, CUT, K_REF = 0.04, 0.25, 16.0
KS = np.arange(1, 17)
OUT = HERE / "results" / "a4_beam.json"


def true_poles(k):
    w = C_BEAM * k ** 2
    return 1j * w, -1j * w


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    n = sum(A.SPLIT)
    tf = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    K = A.N_X // 2 + 1
    fk = np.zeros((n, K, tf.size), complex)
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        km = rng.integers(1, 17, size=n)
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * A.N_X / 2
        fk[np.arange(n), km] += amp[:, None] * prof
    kk = np.arange(K)
    p1, p2 = true_poles(kk)
    dtf = tf[1] - tf[0]
    vk = 0.5 * (A.foh_np(p1, fk, dtf) + A.foh_np(p2, fk, dtf))
    f = np.fft.irfft(fk[..., ::A.FINE], n=A.N_X, axis=1)
    v = np.fft.irfft(vk[..., ::A.FINE], n=A.N_X, axis=1)
    s = np.cumsum([0, *A.SPLIT])
    t = np.linspace(0, A.T_END, A.N_T)
    return {k: (torch.tensor(f[a:b]), torch.tensor(v[a:b])) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}, t


def unit_responses(mu, fk, dt):
    g, c = foh_kernel(mu, fk.shape[-1], dt)
    return foh_response(g[:, None], c[:, None], fk[None])          # (P, B, n_t)


def lsq(X, uk):
    """Residues b (P,) minimising || uk - sum_p b_p X_p ||, and the relative residual."""
    A_ = X.reshape(X.shape[0], -1).T
    b = torch.linalg.lstsq(A_, uk.reshape(-1, 1)).solution[:, 0]
    r = (uk.reshape(-1) - A_ @ b).abs().pow(2).sum().sqrt() / uk.abs().pow(2).sum().sqrt()
    return b, float(r)


def two_poles(fk, uk, dt, device):
    grid = [complex(re, im) for re in B.GRID_RE for im in B.GRID_IM]
    best1 = (np.inf, None)
    for i in range(0, len(grid), 161):
        mu = torch.tensor(grid[i:i + 161], dtype=torch.complex128, device=device)
        r, _ = B.residual(mu, fk, uk, dt)
        j = int(r.argmin())
        if float(r[j]) < best1[0]:
            best1 = (float(r[j]), complex(mu[j]))
    x1 = unit_responses(torch.tensor([best1[1]], dtype=torch.complex128, device=device), fk, dt)[0]
    best2 = (np.inf, None)
    for i in range(0, len(grid), 161):
        mu = torch.tensor(grid[i:i + 161], dtype=torch.complex128, device=device)
        X2 = unit_responses(mu, fk, dt)                                # (G, B, n)
        # 2x2 normal equations per candidate
        a11 = (x1.abs() ** 2).sum()
        a12 = (x1.conj()[None] * X2).sum((1, 2))
        a22 = (X2.abs() ** 2).sum((1, 2))
        y1 = (x1.conj() * uk).sum()
        y2 = (X2.conj() * uk[None]).sum((1, 2))
        det = a11 * a22 - a12 * a12.conj()
        b1 = (a22 * y1 - a12 * y2) / det
        b2 = (a11 * y2 - a12.conj() * y1) / det
        res = (uk[None] - b1[:, None, None] * x1[None] - b2[:, None, None] * X2).abs().pow(2).sum((1, 2))
        j = int(res.argmin())
        if float(res[j]) < best2[0]:
            best2 = (float(res[j]), complex(mu[j]))

    def f(p):
        mu = torch.tensor([complex(p[0], p[1]), complex(p[2], p[3])], dtype=torch.complex128, device=device)
        return lsq(unit_responses(mu, fk, dt), uk)[1]

    z0 = [best1[1].real, best1[1].imag, best2[1].real, best2[1].imag]
    opt = minimize(f, z0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=8000))
    mu = [complex(opt.x[0], opt.x[1]), complex(opt.x[2], opt.x[3])]
    b, r = lsq(unit_responses(torch.tensor(mu, dtype=torch.complex128, device=device), fk, dt), uk)
    order = np.argsort([-m.imag for m in mu])                         # label 0: Im > 0 first
    return [mu[i] for i in order], [complex(b[i]) for i in order], r


def wls(ks, y, w, powers):
    V = np.stack([ks ** p for p in powers], 1)
    sw = np.sqrt(w)
    return np.linalg.lstsq(V * sw[:, None], y * sw, rcond=None)[0]


def fit_beam(data, t, kept, device):
    """Stages 2 and 3 on the stage-1 modes `kept`: weighted polynomials, layer init, L-BFGS. Returns a dict."""
    dt = float(t[1] - t[0])
    ks = np.array([m["k"] for m in kept], float)
    w = 1 / np.array([max(m["resid"], 1e-6) for m in kept]) ** 2
    model = WavenumberPoleLayer1d(2, t, mu_degree=(3, 3), beta_degree=2, im_powers=(1, 2, 3), modes=A.MODES,
                                  re_max=A.RE_MAX, k_ref=K_REF).to(device)
    with torch.no_grad():
        for i in range(2):
            mu_i = np.array([complex(*m["mu"][i]) for m in kept])
            b_i = np.array([complex(*m["b"][i]) for m in kept])
            a = wls(ks, mu_i.real, w, [0, 2, 4]) * K_REF ** np.array([0, 2, 4])
            c = wls(ks, mu_i.imag, w, [1, 2, 3]) * K_REF ** np.array([1, 2, 3])
            br = wls(ks, b_i.real, w, [0, 2]) * K_REF ** np.array([0, 2])
            bi = wls(ks, b_i.imag, w, [0, 2]) * K_REF ** np.array([0, 2])
            model.mu_re[i] = torch.tensor(a)
            model.mu_im[i] = torch.tensor(c)
            model.beta[i, :, 0], model.beta[i, :, 1] = torch.tensor(br), torch.tensor(bi)
    c_init = float(model.mu_im[0, 1]) / K_REF ** 2
    (xtr, ytr), (xte, yte) = ((x.to(device) for x in data[k]) for k in ("train", "test"))
    with torch.no_grad():
        test0 = float(A.rel(model(xte), yte))
    opt = torch.optim.LBFGS(model.parameters(), max_iter=200, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50)

    def closure():
        opt.zero_grad()
        loss = A.rel(model(xtr), ytr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        test = float(A.rel(model(xte), yte))
        mu, _ = model.poles(torch.tensor(KS, dtype=torch.float64, device=device))
        mu = mu.cpu().numpy()
        # oracle: true poles and residues in the same layer
        orc = WavenumberPoleLayer1d(2, t, mu_degree=(1, 3), beta_degree=1, im_powers=(1, 2, 3),
                                    modes=A.MODES).to(device)
        orc.mu_re.zero_(), orc.mu_im.zero_(), orc.beta.zero_()
        orc.mu_im[0, 1], orc.mu_im[1, 1] = C_BEAM, -C_BEAM
        orc.beta[:, 0, 0] = 0.5
        oracle = float(A.rel(orc(xte), yte))
    lab = 0 if mu[0, -1].imag > 0 else 1
    c_hat = float(model.mu_im[lab, 1]) / K_REF ** 2
    k3 = KS >= 3
    re_ratio = float(np.max(np.abs(mu[:, k3].real) / np.abs(mu[:, k3].imag)))
    true = np.stack([true_poles(KS)[0], true_poles(KS)[1]])
    mu_sorted = np.stack([mu[lab], mu[1 - lab]])
    return dict(kept=[int(k) for k in ks], c_init=c_init, c=c_hat, re_ratio=re_ratio, test_init=test0, test=test,
                oracle=oracle, disp_err=float(np.linalg.norm(mu_sorted - true) / np.linalg.norm(true)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    floors = {}
    for ds in range(5):
        t0 = time.time()
        data, t = make_data(ds)
        dt = float(t[1] - t[0])
        modes = []
        for k in KS:
            fk, uk = (a.to(device) for a in B.mode_series(data, int(k)))
            mu, b, r = two_poles(fk, uk, dt, device)
            modes.append(dict(k=int(k), mu=[[m.real, m.imag] for m in mu], b=[[x.real, x.imag] for x in b], resid=r))
        errs = []
        for m in modes:
            tp = true_poles(m["k"])
            errs.append(max(abs(complex(*m["mu"][i]) - tp[i]) / abs(tp[i]) for i in range(2)))
        kept = [m for m in modes if max(abs(complex(*z)) for z in m["mu"]) * dt <= CUT]
        fit = fit_beam(data, t, kept, device)
        c_init, c_hat, re_ratio, test0, test, floors[ds], dispe, ks = (fit[k] for k in (
            "c_init", "c", "re_ratio", "test_init", "test", "oracle", "disp_err", "kept"))
        r = dict(data_seed=ds, modes=modes, mode_err=errs, kept=[int(k) for k in ks], c_init=c_init, c=c_hat,
                 re_ratio=re_ratio, test_init=test0, test=test, oracle=floors[ds],
                 disp_err=dispe, seconds=round(time.time() - t0, 1))
        out.append(r)
        print(ds, "stage1 err k>=3 max", f"{max(errs[2:]):.2e}", "k=1,2", [f"{e:.2e}" for e in errs[:2]],
              f"c init {c_init:.5f} -> {c_hat:.5f}", f"re/im {re_ratio:.1e}", f"test {test0:.2e} -> {test:.2e} (oracle "
              f"{floors[ds]:.2e})", f"disp {r['disp_err']:.1e}", flush=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out):
    lines = ["# (A) Euler–Bernoulli beam (velocity output): two poles per wavenumber", "",
             f"c = {C_BEAM}; poles ±i c k²; truth on a 32× finer time grid; recipe of `a3_twostage.py` with two poles per "
             "mode. Data seeds 0–4.", "",
             "| seed | stage 1 max error, k ≥ 3 | k = 1, 2 | c: stage 2 → tuned | max Re/Im (k ≥ 3) | test: init → tuned (oracle) | dispersion error |",
             "|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['data_seed']} | {max(r['mode_err'][2:]):.1e} | {r['mode_err'][0]:.1e}, {r['mode_err'][1]:.1e} | "
                     f"{r['c_init']:.5f} → {r['c']:.5f} | {r['re_ratio']:.1e} | {r['test_init']:.2e} → {r['test']:.2e} "
                     f"({r['oracle']:.2e}) | {r['disp_err']:.1e} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
