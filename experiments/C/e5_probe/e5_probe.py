"""(C) E5 / RQ4, first pass: linear-probe poles of the trained Duffing LNO against its layer poles and the
linearised system.

Plan: `docs/proposal_C_identifiability.qmd`, Section 5 (diagnostic for nonlinear models) and E5.

- **Linearised truth.** Damped Duffing (LNO data `1D_Duffing_c05`: x'' + 0.5 x' + x + k3 x^3 = f). The poles of the
  linearisation at x = 0 are estimated from the data alone: a first-order-hold fit of one complex pair to the 4
  smallest-amplitude training samples (|f| <= 0.2). Theory: -0.25 +- i sqrt(1 - 0.0625) = -0.25 +- 0.9682i.
- **Models.** The original 1-layer LNO (width 4, 16 poles per channel pair, sin activation; 4 x 4 x 16 = 256 layer
  poles), trained on the original split (`experiments/foundation/reproduce_lno.py --case duffing_c05 --tag e5`).
- **Layer poles.** All 256 stored mu (the exponents of the layer's transient term): the closest to the linearised
  pole, and how many lie within 10 %.
- **Linear probe.** Probe inputs u_j: 16 smooth random signals (8 sinusoids, 0.2-8 rad/s, max |u| = 1). At scale eps,
  the odd part g = (G(eps u) - G(-eps u)) / (2 eps) of the trained operator G (removes the bias and even terms; the
  Duffing system is odd). One complex pair (and a pair + one real pole) fitted to (u_j, g_j) with the first-order-hold
  VarPro of `e4_profile` = the probe poles; relative fit residual = how well a causal LTI model describes G.
- **Jacobian probe.** J = dG/dx at x = 0 (n x n, autograd): energy above the diagonal (output before input:
  non-causal), relative residual of a Toeplitz (shift-invariant) fit of its causal part, and the poles of its
  average impulse response h_d = mean of the d-th subdiagonal (matrix pencil, order 2, on h_1 ... h_799).

    python experiments/C/e5_probe/e5_probe.py          # writes results/e5_probe.md / .json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "theory"))
sys.path.insert(0, str(ROOT / "experiments" / "foundation"))
sys.path.insert(0, str(ROOT / "src"))
import e4_profile as EP  # noqa: E402
import reproduce_lno as RL  # noqa: E402

CASE = "duffing_c05"
SEEDS = (0, 1, 2, 3, 4)
EPS = (0.01, 0.1, 0.3, 1.0, 3.0)
N_PROBE = 16
CKPT = ROOT / "runs" / "foundation" / "ood" / "e5" / CASE / "lno"
RESULTS = HERE / "results"
STARTS = ([complex(-0.3, 1.0)], [complex(-1.0, 3.0)], [complex(-0.1, 0.5)])


def linearised(f, u, t, device):
    amp = np.abs(f).max(1)
    sel = np.argsort(amp)[:4]
    vp = EP.VarPro({"train": (torch.tensor(f[sel]), torch.tensor(u[sel]))}, t, device, "foh")
    th, re_ = EP.pack([complex(-0.3, 1.0)])
    th, _ = vp.fit(th, re_)
    return EP.unpack_theta(th, re_)[0], float(amp[sel].max())


def probe_inputs(t, rng):
    out = []
    for _ in range(N_PROBE):
        w = rng.uniform(0.2, 8.0, 8)
        ph = rng.uniform(0, 2 * math.pi, 8)
        a = rng.uniform(0.5, 1.0, 8)
        x = (a[:, None] * np.sin(np.outer(w, t) + ph[:, None])).sum(0)
        out.append(x / np.abs(x).max())
    return np.array(out)


def fit_poles(u, g, t, device):
    """Best of a few starts for one pair, then one pair + one real pole; (poles, relative residual) each."""
    vp = EP.VarPro({"train": (torch.tensor(u), torch.tensor(g))}, t, device, "foh")
    res = {}
    for name, extra in (("pair", []), ("pair+real", [complex(-2.0, 0.0)])):
        best = None
        for s in STARTS:
            th, re_ = EP.pack(s + extra)
            th, S = vp.fit(th, re_)
            if best is None or S < best[1]:
                best = (th, S, re_)
        th, S, re_ = best
        with torch.no_grad():
            pred = vp.pred(torch.tensor(th, device=device), re_)
        rel = float((vp.y - pred).norm() / vp.y.norm())
        res[name] = (EP.unpack_theta(th, re_), rel)
    return res


def pencil_poles(h, dt, M=2):
    L = len(h) // 2
    Y = np.array([h[i:i + L + 1] for i in range(len(h) - L)])
    V = np.linalg.svd(Y, full_matrices=False)[2][:M].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:]).astype(complex)
    return [u for u in np.log(z) / dt if u.imag >= 0]


def jacobian_probe(model, n, dt, device):
    for q in model.parameters():
        q.requires_grad_(False)
    f = lambda x: model(x[None, :, None])[0, :, 0]
    J = torch.func.jacrev(f)(torch.zeros(n, device=device)).detach().double().cpu().numpy()   # J[t_out, t_in]
    noncausal = float(np.linalg.norm(np.triu(J, 1)) / np.linalg.norm(J))
    h = np.array([np.diag(J, -d).mean() for d in range(n)])
    Tz = sum(np.diag(np.full(n - d, h[d]), -d) for d in range(n))
    toeplitz = float(np.linalg.norm(np.tril(J) - Tz) / np.linalg.norm(np.tril(J)))
    return dict(noncausal=noncausal, toeplitz_resid=toeplitz,
                h_poles=[[u.real, u.imag] for u in pencil_poles(h[1:800], dt)])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case = RL.CASES[CASE]
    split, grids = RL.load(case, None, "ood")
    xtr, ytr = split["train"]
    f, u = (a.double().reshape(a.shape[0], -1).numpy() for a in (xtr, ytr))
    t = np.asarray(grids["t"], float).ravel()
    mu_lin, amp_max = linearised(f, u, t, device)
    print(f"linearised pole {mu_lin:.4f} (from |f| <= {amp_max:.2f})", flush=True)
    P = probe_inputs(t, np.random.default_rng(0))
    rows = []
    for s in SEEDS:
        model, _ = RL.build("lno", case, grids)
        model.load_state_dict(torch.load(CKPT / f"seed{s}.pt", map_location="cpu"))
        model = model.to(device).eval()
        test = json.loads((RL.RESULTS / "ood" / "e5" / f"{CASE}_lno_seed{s}.json").read_text())["test_rel_l2"]
        mu = model.conv0.weights_pole.detach().cpu().numpy().ravel()
        mu = np.where(mu.imag < 0, mu.conj(), mu)                       # fold to Im >= 0
        d = np.abs(mu - mu_lin) / abs(mu_lin)
        layer = dict(closest=[float(mu[d.argmin()].real), float(mu[d.argmin()].imag)], closest_err=float(d.min()),
                     within10=int((d < 0.1).sum()), n=int(mu.size),
                     stable=int((mu.real < 0).sum()))
        probes = {}
        for eps in EPS:
            with torch.no_grad():
                xp = torch.tensor(P, dtype=torch.float32, device=device)[..., None]
                g = ((model(eps * xp) - model(-eps * xp)) / (2 * eps))[..., 0].double().cpu().numpy()
            fp = fit_poles(P, g, t, device)
            probes[eps] = {k: dict(poles=[[z.real, z.imag] for z in v[0]], rel=v[1]) for k, v in fp.items()}
            pr = fp["pair"][0][0]
            print(f"seed {s} eps {eps:g}: probe pair {pr:.4f} (rel resid {fp['pair'][1]:.1e}); "
                  f"pair+real {[f'{z:.3f}' for z in fp['pair+real'][0]]} ({fp['pair+real'][1]:.1e})", flush=True)
        jp = jacobian_probe(model, len(t), float(t[1] - t[0]), device)
        print(f"seed {s}: Jacobian non-causal {jp['noncausal']:.2f}, Toeplitz residual {jp['toeplitz_resid']:.2f}, "
              f"impulse-response pole {[complex(*z) for z in jp['h_poles']]}", flush=True)
        rows.append(dict(seed=s, test=test, layer=layer, jac=jp, probes={str(k): v for k, v in probes.items()}))
        print(f"seed {s}: test {test:.3f}; layer closest {complex(*layer['closest']):.3f} "
              f"({layer['closest_err']:.2f}), within 10 %: {layer['within10']}/{layer['n']}", flush=True)
    res = dict(case=CASE, mu_lin=[mu_lin.real, mu_lin.imag], amp_max=amp_max, eps=list(EPS), rows=rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e5_probe.json").write_text(json.dumps(res, indent=1))
    summarize(res)


def summarize(res):
    mu_lin = complex(*res["mu_lin"])
    err = lambda z: abs(complex(*z) - mu_lin) / abs(mu_lin)
    lines = ["# E5, first pass: linear-probe poles of the Duffing LNO", "",
             f"Damped Duffing (`1D_Duffing_c05`), original split, original 1-layer LNO (256 layer poles), 5 seeds. "
             f"Linearised pole from the 4 smallest-amplitude training samples (|f| ≤ {res['amp_max']:.2f}): "
             f"**{mu_lin.real:.4f} ± {abs(mu_lin.imag):.4f}i** (theory −0.25 ± 0.9682i). Probe: odd part of the trained "
             f"operator at scale ε on {N_PROBE} smooth random inputs, one complex pair fitted (first-order hold); "
             "error = relative distance to the linearised pole; residual = relative fit residual of the pair.", "",
             "| seed | test rel. L2 | layer: closest pole (error), within 10 % | "
             + " | ".join(f"probe ε = {e:g}: pole (error), residual" for e in res["eps"]) + " |",
             "|---|---|---|" + "---|" * len(res["eps"])]
    for r in res["rows"]:
        L = r["layer"]
        cells = []
        for e in res["eps"]:
            p = r["probes"][str(e)]["pair"]
            z = complex(*p["poles"][0])
            cells.append(f"{z.real:.3f}{abs(z.imag):+.3f}i ({err(p['poles'][0]):.2f}), {p['rel']:.1e}")
        lines.append(f"| {r['seed']} | {r['test']:.3f} | {L['closest'][0]:.3f}{L['closest'][1]:+.3f}i "
                     f"({L['closest_err']:.2f}), {L['within10']}/{L['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Jacobian at x = 0", "",
              "Non-causal = share of ‖J‖ above the diagonal (output before input); Toeplitz residual = relative residual "
              "of the best shift-invariant fit of the causal part; impulse-response pole = order-2 matrix pencil on the "
              "average subdiagonals h_1 … h_799 (truth pencil on the linearised impulse response: exact).", "",
              "| seed | non-causal | Toeplitz residual | impulse-response pole (error) | layer poles within 10 % |",
              "|---|---|---|---|---|"]
    for r in res["rows"]:
        j = r["jac"]
        z = complex(*j["h_poles"][0]) if j["h_poles"] else None
        lines.append(f"| {r['seed']} | {j['noncausal']:.2f} | {j['toeplitz_resid']:.2f} | "
                     + (f"{z.real:.3f}{z.imag:+.3f}i ({abs(z - mu_lin) / abs(mu_lin):.2f})" if z is not None else "–")
                     + f" | {r['layer']['within10']}/{r['layer']['n']} |")
    lines += ["", "Pair + one real pole at each scale (poles; residual):", ""]
    for r in res["rows"]:
        lines.append(f"- seed {r['seed']}: " + "; ".join(
            f"ε {e:g}: " + ", ".join(f"{complex(*z):.3f}" for z in r["probes"][str(e)]["pair+real"]["poles"])
            + f" ({r['probes'][str(e)]['pair+real']['rel']:.1e})" for e in res["eps"]))
    (RESULTS / "e5_probe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
