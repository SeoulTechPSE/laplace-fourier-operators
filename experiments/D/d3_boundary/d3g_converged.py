"""(D) d3e and d3f again, with the fit stopped by convergence instead of a fixed iteration count.

Both d3e (Robin coefficient by a profile) and d3f (variable coefficient, learned basis) missed their pre-registered
criteria in 2 and 4 of 5 seeds. Post hoc, more L-BFGS iterations brought the checked seeds to the oracle. This
pass fixes the rule in advance and reruns all seeds, so the claim no longer rests on two post hoc refits.

**Rule** (`converge`). After each script's own fit (unchanged, same seeds and starts), continue L-BFGS on the
selected model in chunks of 300 iterations (all parameters free; for Robin, beta released). Stop when a chunk lowers
the training loss by less than a relative 1e-4, or after 20 chunks (6000 iterations). Iterations used are reported.

**Predictions** (written before running).

1. Robin (d3e criteria), every seed: beta within 2 %, D within 1 %, test error <= 1.2x the oracle, both channel
   laws within 2 % on m <= 8.
2. Variable coefficient (d3f criteria), every seed: test error <= 1.2x the oracle, lambda_1..8 within 1 %,
   |<v_m, v_hat_m>| >= 0.999 for m <= 8.
3. Every run stops by the convergence test, not the cap.

    python experiments/D/d3_boundary/d3g_converged.py      # results/d3g_converged.md / .json
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
import d3d_robin as R  # noqa: E402
import d3e_robin_profile as P  # noqa: E402
import d3f_variable as V  # noqa: E402

A = R.A
CHUNK, MAX_CHUNKS, TOL = 300, 20, 1e-4
OUT = HERE / "results" / "d3g_converged.json"


def converge(m, loss_fn, params):
    """L-BFGS in chunks until the relative drop of the training loss in a chunk is < TOL. Returns (iters, capped)."""
    prev = float(loss_fn().detach())
    for c in range(MAX_CHUNKS):
        opt = torch.optim.LBFGS(params, max_iter=CHUNK, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = loss_fn()
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        now = float(loss_fn().detach())
        if not np.isfinite(now) or prev - now < TOL * prev:
            return (c + 1) * CHUNK, False
        prev = now
    return MAX_CHUNKS * CHUNK, True


def robin(seed, t, device):
    data = R.make_data(seed)
    floor = R.oracle(data, t, device)
    tr = [z.to(device) for z in data["train"]]
    fte, bte, yte = (z.to(device) for z in data["test"])
    m, _, bg = P.fit(data, t, device)
    m.rho.requires_grad_(True)
    iters, capped = converge(m, lambda: A.rel(m(tr[0], tr[1]), tr[2]), [p for p in m.parameters() if p.requires_grad])
    with torch.no_grad():
        test = float(A.rel(m(fte, bte), yte))
        k = R.roots(m.beta)
        lw = m.law(k).cpu().numpy()
        k = k.cpu().numpy()
    tl = R.true_laws(k)
    return dict(case="robin", seed=seed, ratio=test / floor, beta=float(m.beta), beta_grid=bg,
                D=float(-m.core.mu_re[0, 1] / m.kref ** 2),
                law_err8=[float(np.linalg.norm(lw[j, :8] - tl[j, :8]) / np.linalg.norm(tl[j, :8])) for j in range(2)],
                iters=iters, capped=capped)


def variable(seed, t, device):
    _, lam, Vm = V.operator()
    data = V.make_data(seed)
    tr, te = ([z.to(device) for z in data[s]] for s in ("train", "test"))
    orc = V.VarLayer(t, "diag").to(device)
    with torch.no_grad():
        orc.rho.copy_(torch.tensor(np.log(-lam)))
        orc.Qfix.copy_(torch.tensor(Vm))
    floor = V.val(orc, te)
    m = V.fit("mix", data, t, device)
    iters, capped = converge(m, lambda: A.rel(m(tr[0]), tr[1]), list(m.parameters()))
    with torch.no_grad():
        lh, Q = m.lam().cpu().numpy(), m.Q().cpu().numpy()
    o = np.argsort(-lh)
    lh, Q = lh[o], Q[:, o]
    return dict(case="variable", seed=seed, ratio=V.val(m, te) / floor,
                lam_err8=[float(abs(lh[i] / lam[i] - 1)) for i in range(V.N_CLAIM)],
                vec_dot8=[float(abs(Q[:, i] @ Vm[:, i])) for i in range(V.N_CLAIM)], iters=iters, capped=capped)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_r, t_v = np.linspace(0, R.T, R.NT), np.linspace(0, A.T_END, A.N_T)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for case, fn, t in (("variable", variable, t_v), ("robin", robin, t_r)):
        for seed in range(5):
            if any(r["case"] == case and r["seed"] == seed for r in out):
                continue
            t0 = time.time()
            r = fn(seed, t, device)
            r["seconds"] = round(time.time() - t0, 1)
            out.append(r)
            print({q: (round(v, 6) if isinstance(v, float) else v) for q, v in r.items()}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) d3e and d3f with the fit stopped by convergence", "",
             "| case | seed | test / oracle | key estimates | iterations after the original fit (capped) |",
             "|---|---|---|---|---|"]
    for r in out:
        if r["case"] == "robin":
            est = (f"β̂ {r['beta']:.4f}, D̂ {r['D']:.5f}, laws {r['law_err8'][0]:.1e}, {r['law_err8'][1]:.1e}")
        else:
            est = f"max λ error m ≤ 8 {max(r['lam_err8']):.1e}, min overlap {min(r['vec_dot8']):.5f}"
        lines.append(f"| {r['case']} | {r['seed']} | {r['ratio']:.3f} | {est} | {r['iters']} ({r['capped']}) |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
