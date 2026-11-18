"""(A) Pole layer, sixth validation: 2D diffusion and 2D advection-diffusion-reaction with the 2D layer.

Layer: `lfo.models.wavenumber.WavenumberPoleLayer2d` (isotropic Re mu in |k|^2, advection c . k in Im mu, FOH time
axis). Recipe: the 1D recipe of a3 + a5, applied per wavevector.

**Systems** (periodic (x, y) in [0, 2 pi)^2, zero initial state, u_t = L u + f, output u):

- `diff2d`: lambda(k) = -D |k|^2, D = 0.1;
- `adr2d`: lambda(k) = -D |k|^2 - i v . k - k_r, D = 0.05, v = (0.4, -0.3), k_r = 0.2 (anisotropic through v).

**Data.**

- Grid 32 x 32, T = 5, n_t = 201. Splits 200 / 50 / 100.
- Forcing: 6 wavevectors per sample, drawn from the half plane H of the band |k_x|, k_y <= 8
  (k_y > 0, or k_y = 0 and k_x > 0; 144 wavevectors). Amplitudes ~ 1 / |k|; time profiles from the E1 families,
  as in `a1_dispersion.py`.
- Truth per excitation: exact FOH response on a 32x finer time grid, in numpy (independent of the layer), then
  sampled. Modes on the row k_y = 0 get their conjugate partner at -k_x, so the fields are real.

**Recipe.**

1. Per wavevector in H: one pole, grid + Nelder-Mead, residue by least squares (as `a2_twostage.per_mode`).
2. Stage 2 keeps modes with |mu_hat| dt <= 0.25 (resolution) and max(|Re mu_hat|, |Im mu_hat|) T >= 2
   (observability). Weighted least squares (1 / residual^2): Re mu on {1, |k|^2} or {1, |k|^2, |k|^4} by BIC;
   Im mu on {k_x, k_y}; beta on {1, |k|^2}.
3. Fine-tune: `WavenumberPoleLayer2d(1, modes=9, re_max=0.5, k_ref=8)`, L-BFGS 200 iterations, full batch, float64.

**Metrics.** Test relative L2 against the oracle floor (true lambda in the same layer); dispersion error over H as
|| mu - lambda || / || lambda ||; recovered D, v, k_r.

**Predictions** (written before running).

1. Stage 1: every kept mode within 2 % of lambda(k), in every data seed and both systems.
2. Stage 2: BIC drops |k|^4 in at least 4 of 5 seeds per system; D, v_x, v_y within 2 % (k_r within 0.01 absolute).
3. Fine-tuned: in every seed and both systems, test error <= 1.2x the oracle floor, dispersion error <= 1 %, D and
   v within 1 % and k_r within 0.002 absolute.

    python experiments/A/layer/a6_diffusion2d.py      # results/a6_diffusion2d.md / .json
"""
from __future__ import annotations

import json
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
from lfo.models.wavenumber import WavenumberPoleLayer2d  # noqa: E402

N, BANDMAX, M_FORCE = 32, 8, 6
CUT, OBS, K_REF, MODES = 0.25, 2.0, 8.0, BANDMAX + 1
SYSTEMS = {"diff2d": dict(D=0.1, v=(0.0, 0.0), kr=0.0), "adr2d": dict(D=0.05, v=(0.4, -0.3), kr=0.2)}
H = [(kx, ky) for ky in range(0, BANDMAX + 1) for kx in range(-BANDMAX, BANDMAX + 1) if ky > 0 or kx > 0]
OUT = HERE / "results" / "a6_diffusion2d.json"


def lam(kx, ky, D, v, kr):
    kx, ky = np.asarray(kx, float), np.asarray(ky, float)
    return -D * (kx ** 2 + ky ** 2) - 1j * (v[0] * kx + v[1] * ky) - kr


def make_data(system, seed=0):
    p = SYSTEMS[system]
    rng = np.random.default_rng(seed)
    n = sum(A.SPLIT)
    tf = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    Fk = np.zeros((n, N, N // 2 + 1, A.N_T), complex)
    Uk = np.zeros_like(Fk)
    idx = rng.integers(len(H), size=(n, M_FORCE))
    for m in range(M_FORCE):
        prof = A.profiles(tf, rng, n)                                       # (n, n_tf)
        k = np.array([H[i] for i in idx[:, m]])                             # (n, 2)
        kabs = np.hypot(k[:, 0], k[:, 1])
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / kabs * N * N / 2
        fk = amp[:, None] * prof                                            # (n, n_tf)
        lm = lam(k[:, 0], k[:, 1], **p)                                     # (n,)
        uk = A.foh_np(lm, fk[None], tf[1] - tf[0])[0]                      # (n, n_tf): one mode per sample
        fs, us = fk[:, ::A.FINE], uk[:, ::A.FINE]
        rows = np.arange(n)
        Fk[rows, k[:, 0] % N, k[:, 1]] += fs
        Uk[rows, k[:, 0] % N, k[:, 1]] += us
        row0 = k[:, 1] == 0                                                 # conjugate partner on the k_y = 0 row
        Fk[rows[row0], (-k[row0, 0]) % N, 0] += fs[row0].conj()
        Uk[rows[row0], (-k[row0, 0]) % N, 0] += us[row0].conj()
    f = np.fft.irfft2(Fk, s=(N, N), axes=(1, 2))
    u = np.fft.irfft2(Uk, s=(N, N), axes=(1, 2))
    s = np.cumsum([0, *A.SPLIT])
    t = np.linspace(0, A.T_END, A.N_T)
    return {k: (torch.tensor(f[a:b]), torch.tensor(u[a:b])) for k, a, b in zip(("train", "vali", "test"), s[:-1], s[1:])}, t


def per_mode(data, dt, device):
    f, u = data["train"]
    Fk, Uk = torch.fft.rfft2(f, dim=(1, 2)), torch.fft.rfft2(u, dim=(1, 2))
    out = []
    for kx, ky in H:
        fk, uk = Fk[:, kx % N, ky], Uk[:, kx % N, ky]
        keep = fk.abs().amax(1) > 1e-8 * Fk.abs().max()
        fk, uk = fk[keep].to(device), uk[keep].to(device)
        if fk.shape[0] == 0:
            continue
        best = (np.inf, None)
        for re in B.GRID_RE:
            mu = torch.tensor(re + 1j * B.GRID_IM, dtype=torch.complex128, device=device)
            r, _ = B.residual(mu, fk, uk, dt)
            i = int(r.argmin())
            if float(r[i]) < best[0]:
                best = (float(r[i]), complex(mu[i]))
        fn = lambda q: float(B.residual(torch.tensor([complex(q[0], q[1])], dtype=torch.complex128, device=device),
                                        fk, uk, dt)[0][0])
        opt = minimize(fn, [best[1].real, best[1].imag], method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
        mu = complex(opt.x[0], opt.x[1])
        r, b = B.residual(torch.tensor([mu], dtype=torch.complex128, device=device), fk, uk, dt)
        out.append(dict(k=[kx, ky], mu=[mu.real, mu.imag], b=[float(b[0].real), float(b[0].imag)], resid=float(r[0]),
                        n_samples=int(fk.shape[0])))
    return out


def wls(V, y, w):
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(V * sw[:, None], y * sw, rcond=None)
    rss = float((w * (V @ c - y) ** 2).sum() / w.mean())
    n = y.size
    return c, n * np.log(max(rss / n, 1e-300)) + V.shape[1] * np.log(n)


def stage2(modes, dt):
    kept = [m for m in modes if abs(complex(*m["mu"])) * dt <= CUT and max(map(abs, m["mu"])) * A.T_END >= OBS]
    k = np.array([m["k"] for m in kept], float)
    k2 = (k ** 2).sum(1)
    mu = np.array([complex(*m["mu"]) for m in kept])
    b = np.array([complex(*m["b"]) for m in kept])
    w = 1 / np.array([max(m["resid"], 1e-6) for m in kept]) ** 2
    re_opts = {"a0,a1": np.stack([k2 ** 0, k2], 1), "a0,a1,a2": np.stack([k2 ** 0, k2, k2 ** 2], 1)}
    re_fit = {key: wls(V, mu.real, w) for key, V in re_opts.items()}
    re_key = min(re_fit, key=lambda q: re_fit[q][1])
    a = list(re_fit[re_key][0]) + [0.0] * (3 - len(re_fit[re_key][0]))
    adv, _ = wls(k, mu.imag, w)
    Vb = np.stack([k2 ** 0, k2], 1)
    br, _ = wls(Vb, b.real, w)
    bi, _ = wls(Vb, b.imag, w)
    return dict(kept=[m["k"] for m in kept], re_degree=re_key, a=a, adv=list(adv), beta=[[br[0], bi[0]], [br[1], bi[1]]],
                D=-a[1], v=[-adv[0], -adv[1]], kr=-a[0])


def layer(t, device):
    return WavenumberPoleLayer2d(1, t, re_degree=3, beta_degree=2, modes=MODES, re_max=A.RE_MAX, k_ref=K_REF).to(device)


@torch.no_grad()
def oracle(system, data, t, device):
    p = SYSTEMS[system]
    m = layer(t, device)
    m.mu_re.zero_(), m.beta.zero_()
    m.mu_re[0, 0], m.mu_re[0, 1] = -p["kr"], -p["D"] * K_REF ** 2
    m.adv[0] = -torch.tensor(p["v"], dtype=torch.float64) * K_REF
    m.beta[0, 0, 0] = 1.0
    xte, yte = (a.to(device) for a in data["test"])
    return float(A.rel(m(xte), yte))


def finetune(data, t, s2, device):
    model = layer(t, device)
    a, adv, b = s2["a"], s2["adv"], s2["beta"]
    with torch.no_grad():                                   # coefficients of (k / k_ref)^p: multiply by k_ref^p
        model.mu_re.copy_(torch.tensor([[a[0], a[1] * K_REF ** 2, a[2] * K_REF ** 4]], dtype=torch.float64))
        model.adv.copy_(torch.tensor([[adv[0] * K_REF, adv[1] * K_REF]], dtype=torch.float64))
        model.beta.copy_(torch.tensor([[[b[0][0], b[0][1]], [b[1][0] * K_REF ** 2, b[1][1] * K_REF ** 2]]],
                                      dtype=torch.float64))
    (xtr, ytr), (xte, yte) = ((x.to(device) for x in data[k]) for k in ("train", "test"))
    with torch.no_grad():
        test0 = float(A.rel(model(xte), yte))
    opt = torch.optim.LBFGS(model.parameters(), max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50)

    def closure():
        opt.zero_grad()
        loss = A.rel(model(xtr), ytr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        test = float(A.rel(model(xte), yte))
        kk = torch.tensor(H, dtype=torch.float64, device=device)
        mu = model.poles(kk[:, 0], kk[:, 1])[0][0].cpu().numpy()
        ar = model.mu_re.detach().cpu().numpy()[0]
        av = model.adv.detach().cpu().numpy()[0]
    tuned = dict(D=float(-ar[1] / K_REF ** 2), v=[float(-av[0] / K_REF), float(-av[1] / K_REF)], kr=float(-ar[0]),
                 a2=float(ar[2] / K_REF ** 4))
    return test0, test, mu, tuned


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for system, p in SYSTEMS.items():
        for ds in range(5):
            t0 = time.time()
            data, t = make_data(system, seed=ds)
            dt = float(t[1] - t[0])
            floor = oracle(system, data, t, device)
            modes = per_mode(data, dt, device)
            err1 = {tuple(m["k"]): abs(complex(*m["mu"]) - lam(*m["k"], **p)) / abs(lam(*m["k"], **p)) for m in modes}
            s2 = stage2(modes, dt)
            test0, test, mu_ft, tuned = finetune(data, t, s2, device)
            true = lam(np.array(H)[:, 0], np.array(H)[:, 1], **p)
            r = dict(system=system, data_seed=ds, modes=modes, n_modes=len(modes), n_kept=len(s2["kept"]),
                     kept_err_max=max(err1[tuple(k)] for k in s2["kept"]), stage2=s2, test_init=test0, test=test,
                     oracle=floor, tuned=tuned, disp_err=float(np.linalg.norm(mu_ft - true) / np.linalg.norm(true)),
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print(system, ds, f"modes {len(modes)} kept {len(s2['kept'])} err max {r['kept_err_max']:.2e}",
                  s2["re_degree"], f"D {s2['D']:.5f} v {s2['v'][0]:.5f},{s2['v'][1]:.5f} kr {s2['kr']:.5f}",
                  f"test {test0:.2e} -> {test:.2e} ({floor:.2e})", f"disp {r['disp_err']:.2e}", tuned,
                  f"{r['seconds']} s", flush=True)
            del data
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out):
    lines = ["# (A) 2D diffusion and 2D ADR with the 2D wavenumber pole layer", "",
             f"32 × 32 grid, 6 forced wavevectors per sample from |k_x|, k_y ≤ {BANDMAX} (half plane, 144). Stage 2: "
             f"|μ̂| Δt ≤ {CUT}, max(|Re μ̂|, |Im μ̂|) T ≥ {OBS:g}, weights 1 / residual². Fine-tune: k_ref = {K_REF:g}, "
             "L-BFGS 200 iterations. Data seeds 0–4.", "",
             "| system | seed | modes kept / identified | stage 1 max error (kept) | Re degree | stage 2: D; v; k_r | "
             "test: init → tuned (oracle) | dispersion error | tuned: D; v; k_r |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        s2, tu = r["stage2"], r["tuned"]
        lines.append(f"| {r['system']} | {r['data_seed']} | {r['n_kept']} / {r['n_modes']} | {r['kept_err_max']:.1e} | "
                     f"{s2['re_degree']} | {s2['D']:.5f}; ({s2['v'][0]:.4f}, {s2['v'][1]:.4f}); {s2['kr']:.5f} | "
                     f"{r['test_init']:.2e} → {r['test']:.2e} ({r['oracle']:.2e}) | {r['disp_err']:.1e} | "
                     f"{tu['D']:.5f}; ({tu['v'][0]:.5f}, {tu['v'][1]:.5f}); {tu['kr']:.5f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
