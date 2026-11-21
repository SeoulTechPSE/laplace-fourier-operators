"""(C) E5 / RQ4, first pass: linear-probe poles of the trained Duffing LNO against its layer poles and the
linearized system.

Plan: `docs/proposal_C_identifiability.qmd`, Section 5 (diagnostic for nonlinear models) and E5.

- **Linearized truth.** Damped Duffing (LNO data `1D_Duffing_c05`: x'' + 0.5 x' + x + k3 x^3 = f). The poles of the
  linearization at x = 0 are estimated from the data alone: a first-order-hold fit of one complex pair to the 4
  smallest-amplitude training samples (|f| <= 0.2). Theory: -0.25 +- i sqrt(1 - 0.0625) = -0.25 +- 0.9682i.
- **Models.** The original 1-layer LNO (width 4, 16 poles per channel pair, sin activation; 4 x 4 x 16 = 256 layer
  poles), trained on the original split (`experiments/foundation/reproduce_lno.py --case duffing_c05 --tag e5`).
- **Layer poles.** All 256 stored mu (the exponents of the layer's transient term): the closest to the linearized
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "foundation"))
sys.path.insert(0, str(ROOT / "src"))
import e4_profile as EP  # noqa: E402
import reproduce_lno as RL  # noqa: E402
import e5_train as TR  # noqa: E402

CASE = "duffing_c05"
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ("base", "lno_plr10", "lno_stable_plr10", "tied4_stable_plr10")
MU_TRUE = complex(-0.25, math.sqrt(1 - 0.0625))
MU_TRUE_C0 = complex(0.0, 1.0)              # undamped Duffing (eighth pass): poles on the imaginary axis
MU_TRUE_LZ = complex(-0.9287, 4.1476)       # forced Lorenz rho = 5 at C+ (ninth pass): dominant pair (+ -11.81)
MU_TRUE_LZ10 = complex(-0.5955, 6.1742)     # forced Lorenz rho = 10 at C+ (tenth pass): dominant pair (+ -12.48)
TRUST_D, TRUST_TEST = 0.1, 0.5         # RQ4 trust rule: valid probe, d(probe, layer) < TRUST_D, test < TRUST_TEST
GATE = 0.1                             # rule v2 (fixed before the medium / moderate runs): the data's own best linear
                                       # pair must leave < GATE of the output, else no linear pole is read as physical
EPS = (0.01, 0.1, 0.3, 1.0, 3.0)
N_PROBE = 16
CKPT = ROOT / "runs" / "foundation" / "ood" / "e5" / CASE / "lno"
RESULTS = HERE / "results"
STARTS = ([complex(-0.3, 1.0)], [complex(-1.0, 3.0)], [complex(-0.1, 0.5)])
STARTS_R10 = STARTS + ([complex(-0.5, 6.0)], [complex(-1.0, 8.0)])     # tenth pass (rho = 10): higher frequencies


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


def fit_poles(u, g, t, device, starts=STARTS):
    """Best of a few starts for one pair, then one pair + one real pole; (poles, relative residual) each."""
    vp = EP.VarPro({"train": (torch.tensor(u), torch.tensor(g))}, t, device, "foh")
    res = {}
    for name, extra in (("pair", []), ("pair+real", [complex(-2.0, 0.0)])):
        best = None
        for s in starts:
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
    J = torch.func.jacrev(f, chunk_size=128)(torch.zeros(n, device=device)).detach().double().cpu().numpy()   # J[t_out, t_in]
    noncausal = float(np.linalg.norm(np.triu(J, 1)) / np.linalg.norm(J))
    h = np.array([np.diag(J, -d).mean() for d in range(n)])
    Tz = sum(np.diag(np.full(n - d, h[d]), -d) for d in range(n))
    toeplitz = float(np.linalg.norm(np.tril(J) - Tz) / np.linalg.norm(np.tril(J)))
    return dict(noncausal=noncausal, toeplitz_resid=toeplitz,
                h_poles=[[u.real, u.imag] for u in pencil_poles(h[1:800], dt)])


def equivalent_linear(f, u, t, device, center=False, start=None):
    """Best linear pair (first-order hold, per-sample relative weights) for the training data by amplitude band.
    center (Lorenz): fit u - mean(u) to f - mean(f) within each band, which removes the zero-input response."""
    amp = np.abs(f).max(1)
    out = {}
    for lab, sel in (("all", amp > 0), ("A <= 1", amp <= 1), ("1 < A <= 3", (amp > 1) & (amp <= 3)), ("A > 3", amp > 3)):
        if sel.sum() < 4 or (lab != "all" and sel.all()):
            continue
        fs, us = f[sel], u[sel]
        if center:
            fs, us = fs - fs.mean(0), us - us.mean(0)
        vp = EP.VarPro({"train": (torch.tensor(fs), torch.tensor(us))}, t, device, "foh")
        th, re_ = EP.pack([start or (complex(-1.0, 4.0) if center else complex(-0.3, 1.0))])
        th, _ = vp.fit(th, re_)
        with torch.no_grad():
            pred = vp.pred(torch.tensor(th, device=device), re_)
        z = EP.unpack_theta(th, re_)[0]
        out[lab] = dict(pole=[z.real, z.imag], rel=float((vp.y - pred).norm() / vp.y.norm()), n=int(sel.sum()))
    return out


def load_variant(variant, seed, case, grids, device, data_tag="orig"):
    """(model, test error, stored poles folded to Im >= 0). base on the LNO data = first pass (reproduce_lno, tag e5)."""
    if variant == "base" and data_tag == "orig":
        model, _ = RL.build("lno", case, grids)
        model.load_state_dict(torch.load(CKPT / f"seed{seed}.pt", map_location="cpu"))
        test = json.loads((RL.RESULTS / "ood" / "e5" / f"{CASE}_lno_seed{seed}.json").read_text())["test_rel_l2"]
    else:
        model = TR.build(variant, grids, seed, data_tag)
        name = TR.tag(variant, data_tag)
        model.load_state_dict(torch.load(TR.CKPT / f"{name}_seed{seed}.pt", map_location="cpu"))
        test = json.loads((TR.RESULTS / f"{name}_seed{seed}.json").read_text())["test_rel_l2"]
    model = model.to(device).eval()
    return model, test, TR.poles_of(model)


def main(data_tag="orig", only=None):
    """only: probe just these variants and merge them into the existing results (eleventh pass)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case = RL.CASES[TR.case_of(data_tag)]
    split, grids = TR.data_and_grid(data_tag)
    xtr, ytr = split["train"]
    f, u = (a.double().reshape(a.shape[0], -1).numpy() for a in (xtr, ytr))
    t = np.asarray(grids["t"], float).ravel()
    mu_lin, amp_max = linearised(f, u, t, device)
    if data_tag != "orig":                                  # simulated sets: the exact small-signal pole
        mu_lin = MU_TRUE_C0 if data_tag.startswith("c0_") else MU_TRUE_LZ10 if data_tag.startswith("lz_r10") else MU_TRUE_LZ if data_tag.startswith("lz_") else MU_TRUE
    print(f"linearized pole {mu_lin:.4f} (from |f| <= {amp_max:.2f})", flush=True)
    P = probe_inputs(t, np.random.default_rng(0))
    out_p = RESULTS / f"e5_probe{suffix(data_tag)}.json"
    out = json.loads(out_p.read_text())["variants"] if only else {}
    for v in only or VARIANTS + (("tied4o_stable_plr10",) if data_tag.startswith("lz_r10") else ()):
        rows = []
        for s in SEEDS:
            model, test, mu = load_variant(v, s, case, grids, device, data_tag)
            d = np.abs(mu - mu_lin) / abs(mu_lin)
            layer = dict(closest=[float(mu[d.argmin()].real), float(mu[d.argmin()].imag)], closest_err=float(d.min()),
                         within10=int((d < 0.1).sum()), n=int(mu.size), stable=int((mu.real < 0).sum()))
            probes = {}
            for eps in EPS:
                with torch.no_grad():
                    xp = torch.tensor(P, dtype=torch.float32, device=device)[..., None]
                    g = ((model(eps * xp) - model(-eps * xp)) / (2 * eps))[..., 0].double().cpu().numpy()
                fp = fit_poles(P, g, t, device, STARTS_R10 if data_tag.startswith("lz_r10") else STARTS)
                probes[str(eps)] = {k: dict(poles=[[z.real, z.imag] for z in w[0]], rel=w[1]) for k, w in fp.items()}
            jp = jacobian_probe(model, len(t), float(t[1] - t[0]), device)
            # RQ4 indicator: distance from the Jacobian impulse-response pole to the nearest stored pole, and whether
            # the probe is valid (J causal and shift-invariant within 10 %)
            ir = complex(*jp["h_poles"][0]) if jp["h_poles"] else None
            jp["d_probe_layer"] = float(np.min(np.abs(mu - ir)) / abs(ir)) if ir is not None else None
            jp["valid"] = bool(jp["noncausal"] < 0.1 and jp["toeplitz_resid"] < 0.1)
            jp["trusted"] = bool(jp["valid"] and jp["d_probe_layer"] is not None and jp["d_probe_layer"] < TRUST_D
                                 and test < TRUST_TEST)
            rows.append(dict(seed=s, test=test, layer=layer, jac=jp, probes=probes))
            pr = complex(*probes[str(EPS[0])]["pair"]["poles"][0])
            print(f"{v} s{s}: test {test:.3f}; layer within 10 % {layer['within10']}/{layer['n']} "
                  f"(closest {complex(*layer['closest']):.3f}); LTI probe {pr:.3f} "
                  f"(resid {probes[str(EPS[0])]['pair']['rel']:.2f}); J non-causal {jp['noncausal']:.2f}, "
                  f"Toeplitz {jp['toeplitz_resid']:.2f}, IR pole {[complex(*z) for z in jp['h_poles']]}", flush=True)
        out[v] = rows
    res = dict(case=TR.case_of(data_tag), data=data_tag, mu_lin=[mu_lin.real, mu_lin.imag], amp_max=amp_max, eps=list(EPS), variants=out,
               equivalent=equivalent_linear(f, u, t, device, center=data_tag.startswith("lz_"),
                                                start=complex(-0.6, 6.2) if data_tag.startswith("lz_r10") else None))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"e5_probe{suffix(data_tag)}.json").write_text(json.dumps(res, indent=1))
    summarize(res)


def trusted(r):
    j = r["jac"]
    return bool(j["valid"] and j["d_probe_layer"] is not None and j["d_probe_layer"] < TRUST_D and r["test"] < TRUST_TEST)


def gate(res):
    """Rule v2 linearity gate: the data's best linear pair (all training samples) fits within GATE."""
    return res["equivalent"]["all"]["rel"] < GATE


def suffix(data_tag):
    return "" if data_tag == "orig" else f"_{data_tag}"


def summarize(res):
    mu_lin = complex(*res["mu_lin"])
    err = lambda z: abs(complex(*z) - mu_lin) / abs(mu_lin)
    fmt = lambda z: f"{z[0]:.3f}{z[1]:+.3f}i ({err(z):.2f})"
    data = res.get("data", "orig")
    lines = ["# E5: linear-probe poles of Duffing models" + ("" if data == "orig" else f", `{data}` data set"), "",
             (f"Damped Duffing (`1D_Duffing_c05`), original split, 5 seeds per variant. Linearized pole from the 4 "
              f"smallest-amplitude training samples (|f| ≤ {res['amp_max']:.2f}): **{mu_lin.real:.4f} ± "
              f"{abs(mu_lin.imag):.4f}i** (theory −0.25 ± 0.9682i).") if data == "orig" else
             (f"Driven pendulum x'' + 0.5 x' + sin x = f, " + ("the original LNO data (`1D_Pendulum_c05`)" if data == "pend_orig"
              else f"`e5_data.py` set `{data}` (diverse E1 waveforms)") + ", pendulum LNO architecture (grid input, 8 "
              "modes), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.")
             if data.startswith("pend") else
             (f"Undamped Duffing x'' + x + x³ = f, " + ("the original LNO data (`1D_Duffing_c0`)" if data == "c0_orig"
              else f"`e5_data.py` set `{data}` (diverse E1 waveforms)") + ", 5 seeds per variant. Reference: the exact "
              "small-signal pole **± i** (on the imaginary axis).")
             if data.startswith("c0_") else
             (f"Forced Lorenz ρ = 10 (z' = xy − 8/3 z − f, output x), " + ("the original LNO data (`1D_Lorenz_rho10`, from "
              "x(0) = 1, y(0) = z(0) = 0, inputs A sin 5t)" if data == "lz_r10_orig" else f"`e5_data.py` set `{data}` (diverse "
              "E1 waveforms, started at the equilibrium C+)") + ", Lorenz ρ = 10 LNO architecture (grid input, tanh, 84 "
              "modes), 5 seeds per variant; the tied layer has a learned output offset started at the training mean. "
              "Reference: the dominant pole of the linearization at C+, **−0.5955 ± 6.1742i** (third pole −12.48). "
              "Equivalent poles from centered data.")
             if data.startswith("lz_r10") else
             (f"Forced Lorenz ρ = 5 (z' = xy − 8/3 z − f, output x), " + ("the original LNO data (`1D_Lorenz_rho5`, from "
              "x(0) = 1, y(0) = z(0) = 0, inputs A sin 5t)" if data == "lz_orig" else f"`e5_data.py` set `{data}` (diverse "
              "E1 waveforms, started at the equilibrium C+)") + ", Lorenz LNO architecture (grid input, tanh, 16 modes), 5 "
              "seeds per variant; the tied layer has a learned output offset. Reference: the dominant pole of the "
              "linearization at C+, **−0.9287 ± 4.1476i** (third pole −11.81). Equivalent poles from centered data.")
             if data.startswith("lz_") else
             (f"Simulated damped Duffing x'' + 0.5 x' + x + x³ = f, `e5_data.py` set `{data}` (diverse E1 waveforms), 5 "
              f"seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**."),
             "Errors in brackets: relative distance to it.", "",
             "- **layer**: stored poles within 10 % / all, and the closest.",
             f"- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on {N_PROBE} smooth random inputs, one complex pair "
             "fitted (first-order hold); pole at ε = " + f"{res['eps'][0]:g}" + ", and the relative fit residual "
             "over all ε = " + ", ".join(f"{e:g}" for e in res["eps"]) + " (how well a causal LTI model describes G).",
             "- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a "
             "shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse "
             "response (mean subdiagonals h_1 … h_799).", "",
             "Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; "
             "`lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots "
             "(linear), stable init + pole lr × 10 (`e5_train.py`)" + ("; `tied4o_stable_plr10` the same with no weight decay on "
             "the output offset (tenth-pass addendum)" if "tied4o_stable_plr10" in res["variants"] else "")
             + ("; `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass)."
                if "lnotb_stable_plr10" in res["variants"] else "")
             + ("; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass)."
                if "lnots_stable_plr10" in res["variants"] else "."), "",
             "Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by "
             "amplitude band: " + "; ".join(f"{k} ({v['n']}): {v['pole'][0]:.3f}{v['pole'][1]:+.3f}i, residual "
                                            f"{v['rel']:.2f}" for k, v in res["equivalent"].items()) + ".", "",
             "RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); "
             "valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and "
             f"`mixed` runs): valid ∧ d < {TRUST_D:g} ∧ test rel. L2 < {TRUST_TEST:g}. Rule v2 (fixed before the "
             f"`medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < {GATE:g}.", "",
             "| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | "
             "J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v, rows in res["variants"].items():
        for r in rows:
            L, j = r["layer"], r["jac"]
            pr = r["probes"][str(res["eps"][0])]["pair"]
            rel = [r["probes"][str(e)]["pair"]["rel"] for e in res["eps"]]
            lines.append(f"| {v} | {r['seed']} | {r['test']:.3f} | {L['within10']}/{L['n']}, {fmt(L['closest'])} | "
                         f"{fmt(pr['poles'][0])}, {min(rel):.2f}–{max(rel):.2f} | {j['noncausal']:.2f}, "
                         f"{j['toeplitz_resid']:.2f} | " + (fmt(j["h_poles"][0]) if j["h_poles"] else "–")
                         + f" | {'yes' if j['valid'] else 'no'}, "
                         + (f"{j['d_probe_layer']:.2f}" if j["d_probe_layer"] is not None else "–")
                         + f" | {'**yes**' if trusted(r) else 'no'} |")
    tr = [(v, r) for v, rows in res["variants"].items() for r in rows if trusted(r)]
    g = res["equivalent"]["all"]
    eq = complex(*g["pole"])
    lines += ["", f"Trusted runs (rule v1): {len(tr)} of {sum(len(x) for x in res['variants'].values())}. "
              + ("Their J impulse-response pole vs the reference (and vs the data's equivalent pole): "
                 + ", ".join(f"{v} s{r['seed']} {err(r['jac']['h_poles'][0]):.2f} "
                             f"({abs(complex(*r['jac']['h_poles'][0]) - eq) / abs(eq):.2f})" for v, r in tr) if tr else ""),
              "", f"Rule v2 linearity gate: the data's best linear pair leaves {g['rel']:.3f} of the output "
              f"(gate {GATE:g}): " + ("**passed**, v2 trusts the same runs as v1." if gate(res) else
                                      "**failed**, v2 reads no linear pole as physical on this set.")]
    (RESULTS / f"e5_probe{suffix(res.get('data', 'orig'))}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "orig", a[a.index("--only") + 1:] if "--only" in a else None)
