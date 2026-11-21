"""(C) E4: pole errors under output noise, compared with the Cramér–Rao bound.

Noise model (``e2_train.make_splits``): white Gaussian on the train / validation outputs,
per sample with relative L2 norm eta, i.e. std s = eta / sqrt(n_t) on the per-sample
normalized residual (y_hat - y) / ||y||, which is the residual analyzed in phase 1. For the
minimal-form tied layer at the truth (float64) the Fisher information is J^T J / s^2, with J
the Jacobian of the normalized residual over the 200 training samples. For each pole the
trace bound E|mu_hat - mu|^2 >= CRB(Re mu) + CRB(Im mu) gives the relative error bound

    rel_CRB = sqrt(CRB(Re mu) + CRB(Im mu)) / |mu|        (unbiased estimators)

Compared with the RMS relative error of (a) single runs after polish + restarts and
(b) the truth-free consensus pole of each group of 5 runs (one noise draw per group;
e2_floor.py). Runs and groups at eta = 0 are the reference. Two restart rules: the ratio rule
(e2_rescue.py; consensus over runs certified by the ratio rule) and the noise-aware rule
(e2_rescue.py --rule noise; consensus over runs certified by the chi-square rule).

    python experiments/C/theory/e2_noise.py      # writes results/e2_noise.md (+ .json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_train as T  # noqa: E402

R, E = T.R, T.E
ID = R.ID
ETAS = (0.0, 1e-3, 1e-2, 1e-1, 3e-1)
SYSTEMS = {s: (base, {eta: f"{base}_n{tag}" for eta, tag in ((1e-3, "1e-3"), (1e-2, "1e-2"), (1e-1, "1e-1"), (3e-1, "3e-1"))})
           for s, base in (("p4", "p4_ms16_T20"), ("p6", "p6_ms16_T80_N4096"))}
VARIANT = "damp8_ms"
RULES = (("ratio rule", "e2_rescue", "rescue", "ratio"), ("noise-aware rule", "e2_rescue_na", "rescue_na", "noise"))


def crb_unit(cond):
    """sqrt(CRB_Re + CRB_Im) / |mu| per folded true pole, for s = 1 (scale by eta / sqrt(n_t))."""
    system, family, Tw, K, N = T.unpack(cond)
    R.DT = 20.0 / N
    t, x, y, m, b = R.make(system, family, Tw, K, seed=0)
    x, y = x[:200], y[:200]
    P = ID.Params(m, b)
    v0 = torch.tensor(P.vector())
    J = torch.autograd.functional.jacobian(lambda v: ID.residual(P, v, x, y, t).reshape(-1), v0,
                                           vectorize=True, strategy="forward-mode")
    C = torch.linalg.inv(J.T @ J)
    out = []
    for j, z in enumerate(m):
        idx = [P.names.index((j, part)) for part in ("Re mu", "Im mu") if (j, part) in P.names]
        out.append(float(np.sqrt(sum(float(C[k, k]) for k in idx)) / abs(z)))
    return dict(poles=[[float(z.real), float(z.imag)] for z in m], unit=out, n_t=int(t.numel()))


def run_errors(cond, src="e2_rescue"):
    """Signed-free relative pole errors per run (after restarts), matched as in pole_error."""
    rs = []
    for s in range(10, 30):
        f = T.R.RESULTS / src / f"{T.run_name(cond, VARIANT, s)}.json"
        if f.exists():
            rs.append(json.loads(f.read_text())["after"]["pole_error"])
    return np.array(rs)


def consensus_errors(cond, true, stage="rescue", rule="ratio"):
    floor = json.loads((T.R.RESULTS / "e2_floor.json").read_text())
    rows = [a for a in floor if a.get("stage") == stage and a.get("rule") == rule and a.get("cond") == cond
            and a.get("variant") == VARIANT]
    if not rows:
        return None
    errs = []
    for g in rows[0]["groups"]:
        cons = np.array([complex(*z) for z in g["consensus"]])
        errs.append([np.min(np.abs(cons - z)) / abs(z) if cons.size else np.inf for z in true])
    return np.array(errs)


def main():
    out = []
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    lines = ["# E4: pole error under output noise vs the Cramér–Rao bound", "",
             "Tied layer, prior-free `damp8_ms` + polish + restarts, seeds 10-29 (4 noise draws, one per group "
             "of 5). Per pole: RMS relative error of single runs / of the group consensus (truth-free, "
             "e2_floor.py; ∞ = pole missing from the consensus) / relative CRB (trace bound, unbiased "
             "estimators). Recovered = single runs with relative error < 0.1. η = 0: no noise (bias / "
             "optimization only).", ""]
    for system, (base, noisy) in SYSTEMS.items():
        cr = crb_unit(base)
        true = [complex(*z) for z in cr["poles"]]
        for rname, src, stage, rule in RULES:
            lines += [f"## {system} (n_t = {cr['n_t']}), {rname}", "",
                      "| η | " + " | ".join(f"{lab(z)}: single / consensus / CRB, recovered" for z in cr["poles"]) + " |",
                      "|---|" + "---|" * len(true)]
            for eta in ETAS:
                cond = base if eta == 0 else noisy[eta]
                # no noise: the noise-aware rule reduces to the ratio rule
                key = ("e2_rescue", "rescue", "ratio") if eta == 0 else (src, stage, rule)
                pe = run_errors(cond, key[0])
                ce = consensus_errors(cond, true, *key[1:])
                crb = [u * eta / np.sqrt(cr["n_t"]) for u in cr["unit"]]
                cells = []
                for j in range(len(true)):
                    if pe.size == 0:
                        cells.append("–")
                        continue
                    single = float(np.sqrt(np.mean(pe[:, j] ** 2)))
                    cons = (float(np.sqrt(np.mean(ce[:, j] ** 2))) if ce is not None and np.isfinite(ce[:, j]).all()
                            else (f"∞ ({int(np.sum(~np.isfinite(ce[:, j])))}/{len(ce)} missing)" if ce is not None else "–"))
                    cs = f"{cons:.1e}" if isinstance(cons, float) else cons
                    cells.append(f"{single:.1e} / {cs} / {crb[j]:.1e}, {np.mean(pe[:, j] < T.RECOVERED):.0%}")
                lines.append(f"| {eta:g} | " + " | ".join(cells) + " |")
                out.append(dict(system=system, rule=rname, eta=eta, cond=cond, crb=crb, runs=int(len(pe)),
                                single_rms=None if pe.size == 0 else np.sqrt(np.mean(pe ** 2, 0)).tolist(),
                                consensus=None if ce is None else ce.tolist()))
            lines.append("")
    (T.R.RESULTS / "e2_noise.json").write_text(json.dumps(out, indent=1, default=float))
    (T.R.RESULTS / "e2_noise.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
