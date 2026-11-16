"""(C) E2: second-order polish after Adam. Does L-BFGS reach the representation floor, and
does reaching it recover the poles that phase 1 calls identifiable?

On the fine grid (p6_ms16_T80_N4096) Adam stops at test 4.2e-4 against an oracle floor of
5.6e-5, so the phase-1 prediction (-1 identifiable at N = 4096) was not tested at the
floor. Here every saved Adam checkpoint (``runs/C/e2/<run>.pt``) is polished with
L-BFGS-B (scipy, full training batch, data loss only: no weight decay, no penalty;
float32 model, float64 optimiser state; at most MAXITER iterations, restarted after a
line-search stall while a restart still gains > 1e-4 of the starting loss) and re-scored.
Before and after are the same run, so the comparison is paired.

    python experiments/C/theory/e2_polish.py --cond p6_ms16_T80_N4096 --variant mixed_off_ms --seeds 10 11
    python experiments/C/theory/e2_polish.py --summarize     # writes results/e2_polish.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_train as T  # noqa: E402

E = T.E
from lfo.metrics import pole_error  # noqa: E402
from lfo.training import relative_l2_loss  # noqa: E402

OUT = T.R.RESULTS / "e2_polish"
MAXITER, HISTORY, RESTARTS = 3000, 50, 10
K = 5                                                     # best-of-K groups of consecutive seeds
SEL, NEW, OLD = range(10, 60), range(10, 30), range(0, 20)   # mixed_off_ms: seeds 0-9 fed the bootstrap
CASES = [("p4_ms16_T20", "mixed_off_ms", SEL), ("p6_ms16_T80", "mixed_off_ms", SEL),
         ("p6_ms16_T80_N4096", "mixed_off_ms", NEW),
         ("p4_ms16_T20", "bound8_ms", OLD), ("p6_ms16_T80", "bound8_ms", OLD),
         ("p4_ms16_T20", "damp8_ms", NEW), ("p6_ms16_T80", "damp8_ms", NEW),
         ("p4_ms16_T20", "damp8w_ms", NEW), ("p6_ms16_T80", "damp8w_ms", NEW),
         ("p6_ms16_T80_N4096", "damp8_ms", NEW)] + [
    (c, "damp8_ms", NEW) for c in ("p4_ms16_T20_alno", "p4_ms16_T20_lfno", "p6_ms16_T80_N4096_alno",
                                   "p6_ms16_T80_N4096_lfno", "p4_ms16_T20_n1e-3", "p4_ms16_T20_n1e-2",
                                   "p6_ms16_T80_N4096_n1e-3", "p6_ms16_T80_N4096_n1e-2",
                                   "p4_ms16_T20_n1e-1", "p4_ms16_T20_n3e-1", "p6_ms16_T80_N4096_n1e-1",
                                   "p6_ms16_T80_N4096_n3e-1", "p4_ms16_T20_alno_pt03", "p4_ms16_T20_lfno_pt03",
                                   "p6_ms16_T80_N4096_alno_pt003", "p6_ms16_T80_N4096_alno_pt03",
                                   "p6_ms16_T80_N4096_lfno_pt003", "p6_ms16_T80_N4096_lfno_pt03", *["p4_ms16_T20_foh", "p6_ms16_T80_foh", "p6_ms16_T80_N4096_foh", "p6_ms16_T80_N4096_n1e-2_foh", "p6_ms16_T80_N4096_n1e-1_foh", "p4_e1_T20", "p4_e1_T20_foh", "p6_e1_T20", "p6_e1_T20_foh", "p4_e1_T20_n1e-1", "p4_e1_T20_n3e-1", "p6_e1_T20_n1e-2", "p6_e1_T20_n3e-2", "p6_e1_T20_n1e-1", "p4_ms16_T20_n1e-2_u1e-2", "p4_ms16_T20_n1e-2_u3e-2", "p4_ms16_T20_n1e-2_u1e-1", "p6_e1_T20_n1e-2_u1e-2", "p6_e1_T20_n1e-2_u3e-2"])]


def score(model, data, mu_true, beta_true, device):
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    xva, yva = (a.to(device)[..., None] for a in data["vali"])
    active = T.keep_mask(model).cpu().numpy()
    mu_l, beta_l = (a.detach().cpu().numpy()[active] for a in model.pole_sets())
    mt, _ = E.true_folded(mu_true, beta_true)
    mf, _ = E.fold(mu_l, beta_l)
    ok = np.isfinite(mu_l).all()
    return dict(test=E.evaluate(model, xte, yte), vali=E.evaluate(model, xva, yva),
                pole_error=pole_error(mf, mt, reduce="none").tolist() if ok else None,
                learned_poles=[[float(z.real), float(z.imag)] for z in mu_l])


def polish(model, data, device, maxiter=MAXITER):
    xtr, ytr = (a.to(device)[..., None] for a in data["train"])
    ps = [p for p in model.parameters() if p.requires_grad]
    real = [torch.view_as_real(p) if p.is_complex() else p for p in ps]
    sizes = [r.numel() for r in real]

    def put(x):
        with torch.no_grad():
            for r, chunk in zip(real, np.split(x, np.cumsum(sizes)[:-1])):
                r.copy_(torch.tensor(chunk, dtype=r.dtype, device=device).reshape(r.shape))

    scale = [1.0]

    def f(x):                                             # objective = loss / (loss at the start)
        put(x)
        for p in ps:
            p.grad = None
        loss = relative_l2_loss(model(xtr), ytr)
        loss.backward()
        g = np.concatenate([(torch.view_as_real(p.grad) if p.is_complex() else p.grad).detach().cpu().double().numpy().ravel()
                            for p in ps])
        return float(loss) / scale[0], g / scale[0]

    model.train()
    x0 = np.concatenate([r.detach().cpu().double().numpy().ravel() for r in real])
    scale[0] = loss0 = f(x0)[0]
    x, fx, nit, nfev, msgs = x0, 1.0, 0, 0, []
    for _ in range(RESTARTS):                              # restart after a line-search stall (float32 noise)
        res = minimize(f, x, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter - nit, maxfun=2 * maxiter, maxcor=HISTORY, ftol=0.0, gtol=0.0))
        nit, nfev = nit + int(res.nit), nfev + int(res.nfev)
        msgs.append(str(res.message))
        gain = fx - res.fun
        if res.fun < fx:
            x, fx = res.x, float(res.fun)
        if nit >= maxiter or gain < 1e-4:
            break
    put(x)
    return dict(train_before=loss0, train_after=fx * loss0, nit=nit, nfev=nfev, message=" | ".join(msgs))


def run(cond, variant, seed, device):
    name = T.run_name(cond, variant, seed)
    ck = T.RUNS / f"{name}.pt"
    src = T.OUT / f"{name}.json"
    if not (ck.exists() and src.exists()):
        print(f"skip {name}: no checkpoint", flush=True)
        return
    system, family, Tw, Kk, N = T.unpack(cond)
    data, t, mu_true, beta_true = T.setup(cond, seed)
    model, _ = T.build_for(cond, variant, seed, t, mu_true, beta_true)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.to(device)
    t0 = time.time()
    before = score(model, data, mu_true, beta_true, device)
    info = polish(model, data, device)
    after = score(model, data, mu_true, beta_true, device)
    res = dict(cond=cond, variant=variant, seed=seed, system=system, n_per_20=N, seconds=round(time.time() - t0, 1),
               oracle_rel_l2=json.loads(src.read_text())["oracle_rel_l2"],
               true_poles=[[float(z.real), float(z.imag)] for z in E.true_folded(mu_true, beta_true)[0]],
               before=before, after=after, **info)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), T.RUNS / f"{name}_lbfgs.pt")
    r3 = lambda pe: pe and [round(x, 3) for x in pe]
    print(f"{name}: test {before['test']:.2e} -> {after['test']:.2e} (oracle {res['oracle_rel_l2']:.1e}), "
          f"pole_err {r3(before['pole_error'])} -> {r3(after['pole_error'])}, nit {info['nit']} ({res['seconds']:.0f}s)",
          flush=True)


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    ok = lambda s: s["pole_error"] is not None and max(s["pole_error"]) < T.RECOVERED
    lines = ["# E2: L-BFGS polish after Adam", "",
             f"Each Adam checkpoint polished with L-BFGS-B (full batch, data loss, ≤ {MAXITER} iterations); "
             "before → after on the same runs. 'all' = every true pole within relative error "
             f"{T.RECOVERED}. Best of {K}: lowest validation loss in disjoint groups of {K} consecutive seeds "
             "(mixed_off_ms: seeds 10-59 as in e2_select; fine grid and damp8*: 10-29; bound8: 0-19, as trained). ρ = Spearman(validation loss, mean pole error) over runs.", "",
             "| condition | variant | runs | test rel. L2 | oracle | train/oracle ratio | all (single) | "
             f"best of {K}: all | ρ | per pole recovered |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for cond, variant, seeds in CASES:
        g = sorted((r for r in rs if r["cond"] == cond and r["variant"] == variant and r["seed"] in seeds),
                   key=lambda r: r["seed"])
        g = [r for r in g if r["before"]["pole_error"] is not None and r["after"]["pole_error"] is not None]
        if not g:
            continue
        row = dict(cond=cond, variant=variant, runs=len(g))
        cells = {}
        for when in ("before", "after"):
            te = np.array([r[when]["test"] for r in g])
            pe = np.array([r[when]["pole_error"] for r in g])
            single = np.mean([ok(r[when]) for r in g])
            by_seed = {r["seed"]: r for r in g}
            groups = []
            for g0 in range(seeds.start, seeds.stop, K):
                grp = [by_seed[s] for s in range(g0, g0 + K) if s in by_seed]
                if len(grp) == K:
                    groups.append(ok(min(grp, key=lambda r: r[when]["vali"])[when]))
            rho = spearmanr([r[when]["vali"] for r in g], pe.mean(1)).statistic
            row[when] = dict(test_mean=float(te.mean()), test_std=float(te.std()), single=float(single),
                             best_of_k=[int(sum(groups)), len(groups)], rho=float(rho),
                             per_pole=(pe < T.RECOVERED).mean(0).tolist())
            cells[when] = row[when]
        b, a = cells["before"], cells["after"]
        ratio = np.median([r["train_after"] for r in g]) / g[0]["oracle_rel_l2"]
        pp = ", ".join(f"{lab(z)} {x:.0%} → {y:.0%}" for z, x, y in zip(g[0]["true_poles"], b["per_pole"], a["per_pole"]))
        lines.append(f"| {cond} | {variant} | {len(g)} | {b['test_mean']:.1e} ± {b['test_std']:.1e} → "
                     f"{a['test_mean']:.1e} ± {a['test_std']:.1e} | {g[0]['oracle_rel_l2']:.1e} | {ratio:.1f} | "
                     f"{b['single']:.0%} → {a['single']:.0%} | {b['best_of_k'][0]}/{b['best_of_k'][1]} → "
                     f"{a['best_of_k'][0]}/{a['best_of_k'][1]} | {b['rho']:+.2f} → {a['rho']:+.2f} | {pp} |")
        out.append(row)
    lines += ["", "train/oracle ratio: median final training loss after the polish / test oracle error "
              "(the oracle on the training set is of the same size).", "",
              "## p6: does reaching the floor certify the poles?", "",
              "All polished p6 multisine16 T = 80 runs (seeds 10-29, every variant above), binned by final "
              "training loss / oracle error; 'all' = every true pole recovered.", "",
              "| grid | train/oracle < 1.5 | 1.5–3 | ≥ 3 |", "|---|---|---|---|"]
    cert = []
    for grid, cond in (("dt = 20/1024", "p6_ms16_T80"), ("dt = 20/4096", "p6_ms16_T80_N4096")):
        g = [r for r in rs if r["cond"] == cond and 10 <= r["seed"] < 30 and r["after"]["pole_error"] is not None]
        q = [r["train_after"] / r["oracle_rel_l2"] for r in g]
        cells = []
        for lo, hi in ((0, 1.5), (1.5, 3), (3, np.inf)):
            b = [r for r, x in zip(g, q) if lo <= x < hi]
            cells.append(f"{sum(ok(r['after']) for r in b)}/{len(b)}")
            cert.append(dict(grid=grid, lo=lo, hi=hi, ok=sum(ok(r["after"]) for r in b), n=len(b)))
        lines.append(f"| {grid} | " + " | ".join(cells) + " |")
    lines.append("")
    out.append(dict(certificate=cert))
    (T.R.RESULTS / "e2_polish.json").write_text(json.dumps(out, indent=1))
    (T.R.RESULTS / "e2_polish.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+")
    ap.add_argument("--variant", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(60)))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for v in a.variant:
            for s in a.seeds:
                run(c, v, s, device)


if __name__ == "__main__":
    main()
