"""(C) E2: truth-free restarts for polished runs that stop above the floor.

After the L-BFGS polish (``e2_polish.py``) the remaining failures are local minima: real
slots coalesced at one decay, or a damped spare complex slot holding a fast decay. Here
each polished run is given single-slot moves, without using the truth:

  candidates  every slot j is moved, with residue 0 (so the output is unchanged at first),
              to each point of a grid: real slots to Re = -geomspace(0.05, 5, 8); complex
              slots to Im = geomspace(0.3, 5, 8), Re = -0.1 Im (inside every slot bound)
  screen      each candidate is polished for SCREEN_ITERS L-BFGS iterations
  accept      the best candidate, if it lowers the training loss by >= 1 %; it is then
              polished in full (as in e2_polish.py), and the next round starts
  stop        after ROUNDS rounds or when no candidate is accepted

Every run is treated the same, whether or not it is at the floor; only training loss is used.

Noise-aware rule (``--rule noise``, E4): with output noise the training loss of every run sits at
the noise level, and a 1 % fall is out of reach. The noise variance per normalised output sample,
sigma^2, is then estimated from the training outputs alone (6th-order differences: the band-limited
signal cancels, white noise does not; ``noise_var``), and the loss is measured as the sum of
squared normalised residuals S. A run is in the noise regime when S <= 2 x the noise
expectation n_s n_t sigma^2. There, the screen picks the candidate with the lowest S and accepts it
if S falls by >= chi2_4(0.99) sigma^2 (a likelihood-ratio test for the 4 real parameters a move
changes). Outside the noise regime the ratio rule above applies unchanged.

Partial-tie starts (E3, H3; ``--start``):
  lbfgs    the polished run (default)
  reset    as lbfgs, and every candidate also sets the steady residual R_res to 0; one extra
           candidate resets R_res only (the residual can hold part of a slow decay and so
           pin a wrong pole)
  release  the tied-layer run on the same data and seed after its own restarts (``_rescue.pt``
           of the tied condition), loaded with R_res = 0 (delta schedule 0 -> delta), then a
           full polish, then the restarts. Truth-free: the tied run used no truth. With
           ``--rule noise`` the tied start is the noise-aware run (``_rescue_na.pt``), for the
           noisy partial-tie conditions (``*_pt03_n*``, which have no Adam runs of their own)

    python experiments/C/theory/e2_rescue.py --cond p6_ms16_T80_N4096 --variant damp8_ms --seeds 10 11
    python experiments/C/theory/e2_rescue.py --cond p6_ms16_T80_N4096_n1e-1 --variant damp8_ms --rule noise
    python experiments/C/theory/e2_rescue.py --cond p6_ms16_T80_N4096_alno_pt03 --variant damp8_ms --start release
    python experiments/C/theory/e2_rescue.py --summarize      # writes results/e2_rescue{,_rr,_rel,_na,_na_rel}.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_polish as P  # noqa: E402

T, E = P.T, P.E
OUT = T.R.RESULTS / "e2_rescue"
OUT_NA = T.R.RESULTS / "e2_rescue_na"
STARTS = {"lbfgs": "", "reset": "_rr", "release": "_rel"}         # output suffix per start


def tag(rule="ratio", start="lbfgs"):
    """Suffix of the output directory (e2_rescue<tag>) and checkpoint (<name>_rescue<tag>.pt)."""
    return ("_na" if rule == "noise" else "") + STARTS[start]
ROUNDS, SCREEN_ITERS, ACCEPT = 3, 300, 0.99
NOISE_REGIME, CHI2_MOVE = 2.0, 13.28                     # chi2_4(0.99)
DIFF = 6                                                 # difference order of the noise estimator
REAL_GRID = -np.geomspace(0.05, 5.0, 8)
IMAG_GRID = np.geomspace(0.3, 5.0, 8)


def train_loss(model, data, device):
    xtr, ytr = (a.to(device)[..., None] for a in data["train"])
    with torch.no_grad():
        model.eval()
        return float(P.relative_l2_loss(model(xtr), ytr))


def noise_var(y):
    """sigma^2: noise variance per element of the normalised outputs y_i / ||y_i||, estimated from
    DIFF-th differences (white noise: variance x C(2 DIFF, DIFF); a smooth signal: ~ (omega dt)^(2 DIFF))."""
    y = y.double()
    d = torch.diff(y, n=DIFF, dim=1)
    v = d.pow(2).mean(1) / math.comb(2 * DIFF, DIFF)
    return float((v / y.pow(2).sum(1)).mean())


def sse(model, data, device):
    """S = sum_i ||y_hat_i - y_i||^2 / ||y_i||^2 over the training set (float64)."""
    xtr, ytr = (a.to(device)[..., None] for a in data["train"])
    with torch.no_grad():
        model.eval()
        r = (model(xtr) - ytr).double()[..., 0]
        return float((r.pow(2).sum(1) / ytr.double()[..., 0].pow(2).sum(1)).sum())


def set_slot(model, j, mu_new):
    """Write pole mu_new into slot j through the raw parameter of its parametrisation; residue 0."""
    par = model.pr.parametrizations.mu[0]
    raw = model.pr.parametrizations.mu.original
    real = float(par.mask.flatten()[j]) == 0
    with torch.no_grad():
        if real:
            v = complex(mu_new.real, 0.0)
        elif isinstance(par, T.DampFloor):
            im = abs(mu_new.imag)
            q = float(np.clip(abs(mu_new.real) / (par.c * im), 0.02, 0.98))
            v = complex(np.log(q / (1 - q)), im - par.w_min)
        elif isinstance(par, T.ImagFloor):
            v = complex(mu_new.real, abs(mu_new.imag) - par.w_min)
        else:                                               # ImagMask
            v = complex(mu_new.real, mu_new.imag)
        raw[..., j] = v
        model.pr.beta[..., j] = 0


def candidates(model, reset=False):
    """(slot, new pole); slot None = no pole move (only with reset: R_res to 0 alone)."""
    if reset:
        yield None, None
    mask = model.pr.parametrizations.mu[0].mask.flatten().cpu().numpy()
    for j, m in enumerate(mask):
        for z in (REAL_GRID if m == 0 else -0.1 * IMAG_GRID + 1j * IMAG_GRID):
            yield j, complex(z)


def reset_residual(model):
    """Partial tie: steady residual R_res to 0 (raw weight of its Bounded parametrisation)."""
    with torch.no_grad():
        model.spec.parametrizations.weight.original.zero_()


def tied_base(cond):
    """Tied condition with the same data as a partial-tie condition (drops ``_alno_pt..`` /
    ``_lfno_pt..``, keeps a noise suffix)."""
    out = re.sub(r"_(alno|lfno)_pt\d+", "", cond)
    if out == cond:
        raise ValueError(cond)
    return out


def run(cond, variant, seed, device, rule="ratio", start="lbfgs"):
    name = T.run_name(cond, variant, seed)
    tied_name = T.run_name(tied_base(cond), variant, seed) if start == "release" else None
    ck = (T.RUNS / f"{tied_name}_rescue{tag(rule)}.pt" if start == "release"
          else T.RUNS / f"{name}_lbfgs.pt")
    if not ck.exists():
        print(f"skip {name}: no checkpoint {ck.name}", flush=True)
        return
    system, family, Tw, Kk, N = T.unpack(cond)
    data, t, mu_true, beta_true = T.setup(cond, seed)
    model, _ = T.build_for(cond, variant, seed, t, mu_true, beta_true)
    if start == "release":                                   # tied state; R_res stays at its init 0
        missing, unexpected = model.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
        assert not unexpected and all(k.startswith("spec.") for k in missing), (missing, unexpected)
    else:
        model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.to(device)
    t0 = time.time()
    start_info = {}
    if start == "release":
        start_info = dict(tied=P.score(model, data, mu_true, beta_true, device),
                          polish=P.polish(model, data, device))
    reset = start == "reset"
    before = P.score(model, data, mu_true, beta_true, device)
    loss = train_loss(model, data, device)
    sig2 = noise_var(data["train"][1])
    n_s, n_t = data["train"][1].shape
    S = sse(model, data, device)
    noisy = rule == "noise" and S <= NOISE_REGIME * n_s * n_t * sig2
    # criterion to minimise and the acceptance test of a move
    crit = (lambda: sse(model, data, device)) if noisy else (lambda: train_loss(model, data, device))
    accept = (lambda new, old: old - new >= CHI2_MOVE * sig2) if noisy else (lambda new, old: new <= ACCEPT * old)
    loss0, S0, moves = loss, S, []
    cur = crit()
    for rnd in range(ROUNDS):
        state = {k: v.clone() for k, v in model.state_dict().items()}
        best = (cur, None, None)
        for j, z in candidates(model, reset):
            model.load_state_dict(state)
            if reset:
                reset_residual(model)
            if j is not None:
                set_slot(model, j, z)
            P.polish(model, data, device, maxiter=SCREEN_ITERS)
            cj = crit()
            if cj < best[0]:
                best = (cj, j, z, {k: v.clone() for k, v in model.state_dict().items()})
        if len(best) == 3 or not accept(best[0], cur):
            model.load_state_dict(state)
            break
        model.load_state_dict(best[3])
        P.polish(model, data, device)
        new = crit()
        moves.append(dict(round=rnd + 1, slot=None if best[1] is None else int(best[1]),
                          to=None if best[2] is None else [best[2].real, best[2].imag], crit_before=cur,
                          crit_after=new, loss_after=train_loss(model, data, device)))
        cur = new
    loss = train_loss(model, data, device)
    after = P.score(model, data, mu_true, beta_true, device)
    res = dict(cond=cond, variant=variant, seed=seed, system=system, n_per_20=N, seconds=round(time.time() - t0, 1),
               oracle_rel_l2=json.loads((P.OUT / f"{tied_name or name}.json").read_text())["oracle_rel_l2"],
               true_poles=[[float(z.real), float(z.imag)] for z in E.true_folded(mu_true, beta_true)[0]],
               train_before=loss0, train_after=loss, moves=moves, before=before, after=after, rule=rule,
               noise_regime=bool(noisy), sigma2=sig2, eta_hat=float(np.sqrt(n_t * sig2)), sse_before=S0,
               sse_after=sse(model, data, device), start=start, **start_info)
    out = T.R.RESULTS / f"e2_rescue{tag(rule, start)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), T.RUNS / f"{name}_rescue{tag(rule, start)}.pt")
    r3 = lambda pe: pe and [round(x, 3) for x in pe]
    print(f"{name}: train/oracle {loss0 / res['oracle_rel_l2']:.2f} -> {loss / res['oracle_rel_l2']:.2f}, "
          f"moves {len(moves)}, pole_err {r3(before['pole_error'])} -> {r3(after['pole_error'])} ({res['seconds']:.0f}s)",
          flush=True)


CASES = [("p4_ms16_T20", "mixed_off_ms"), ("p4_ms16_T20", "damp8_ms"),
         ("p6_ms16_T80", "mixed_off_ms"), ("p6_ms16_T80", "damp8_ms"),
         ("p6_ms16_T80_N4096", "mixed_off_ms"), ("p6_ms16_T80_N4096", "damp8_ms"),
         ("p4_ms16_T20_alno", "damp8_ms"), ("p4_ms16_T20_lfno", "damp8_ms"),
         ("p6_ms16_T80_N4096_alno", "damp8_ms"), ("p6_ms16_T80_N4096_lfno", "damp8_ms"),
         ("p4_ms16_T20_n1e-3", "damp8_ms"), ("p4_ms16_T20_n1e-2", "damp8_ms"),
         ("p6_ms16_T80_N4096_n1e-3", "damp8_ms"), ("p6_ms16_T80_N4096_n1e-2", "damp8_ms"),
         ("p4_ms16_T20_n1e-1", "damp8_ms"), ("p4_ms16_T20_n3e-1", "damp8_ms"),
         ("p6_ms16_T80_N4096_n1e-1", "damp8_ms"), ("p6_ms16_T80_N4096_n3e-1", "damp8_ms"),
         ("p4_ms16_T20_alno_pt03", "damp8_ms"), ("p4_ms16_T20_lfno_pt03", "damp8_ms"),
         ("p6_ms16_T80_N4096_alno_pt003", "damp8_ms"), ("p6_ms16_T80_N4096_alno_pt03", "damp8_ms"),
         ("p6_ms16_T80_N4096_lfno_pt003", "damp8_ms"), ("p6_ms16_T80_N4096_lfno_pt03", "damp8_ms"),
         *((f"p6_ms16_T80_N4096_{k}_pt03_n{e}", "damp8_ms") for k in ("alno", "lfno") for e in ("1e-2", "1e-1")),
         *((c, "damp8_ms") for c in ['p4_ms16_T20_foh', 'p6_ms16_T80_foh', 'p6_ms16_T80_N4096_foh', 'p6_ms16_T80_N4096_n1e-2_foh', 'p6_ms16_T80_N4096_n1e-1_foh'])]


def summarize():
    table(OUT, "e2_rescue", "accept if training loss falls ≥ 1 %")
    table(T.R.RESULTS / "e2_rescue_rr", "e2_rescue_rr", "partial tie, every candidate also resets R_res to 0 "
          "(plus a reset-only candidate), accept if training loss falls ≥ 1 %")
    table(T.R.RESULTS / "e2_rescue_rel", "e2_rescue_rel", "partial tie released from the restarted tied run "
          "(R_res = 0, then a full polish; before = after that polish), accept if training loss falls ≥ 1 %")
    table(T.R.RESULTS / "e2_rescue_na_rel", "e2_rescue_na_rel", "partial tie released from the noise-aware tied run, "
          f"then noise-aware restarts (accept if S falls ≥ {CHI2_MOVE} σ̂² in the noise regime)")
    table(OUT_NA, "e2_rescue_na", f"noise-aware rule: in the noise regime (S ≤ {NOISE_REGIME:g} n_s n_t σ̂²) "
          f"minimise S and accept if S falls ≥ {CHI2_MOVE} σ̂², otherwise accept if training loss falls ≥ 1 %")


def table(src, stem, rule):
    rs = [json.loads(f.read_text()) for f in sorted(src.glob("*.json"))]
    if not rs:
        return
    na = src.name.startswith("e2_rescue_na")
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    ok = lambda s: max(s["pole_error"]) < T.RECOVERED
    lines = ["# E2: truth-free single-slot restarts after the polish" + {"e2_rescue_na": ", noise-aware rule (E4)", "e2_rescue_rr": ", partial tie with residual reset (E3)",
                                                               "e2_rescue_rel": ", partial tie released from the tied run (E3)",
                                                               "e2_rescue_na_rel": ", partial tie released from the tied run, under noise (E3 / E4)"}.get(stem, ""), "",
             f"Every polished run (seeds 10-29) gets up to {ROUNDS} rounds of single-slot moves (residue 0, "
             f"{SCREEN_ITERS}-iteration screen, {rule}, then full polish). "
             "Before = after the polish (e2_polish.py). Best of 5: lowest validation loss in groups of 5 "
             "consecutive seeds." + (" η̂: noise level estimated from the training outputs (median over runs); "
                                     "noise regime: runs that used the noise-aware test." if na else ""), "",
             "| condition | variant | runs | " + ("η̂ | noise regime | " if na else "") + "runs moved | train/oracle (median) | test rel. L2 (median) | all (single) | "
             "best of 5: all | per pole recovered | s per run |",
             "|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if na else "")]
    out = []
    for cond, variant in CASES:
        g = sorted((r for r in rs if r["cond"] == cond and r["variant"] == variant), key=lambda r: r["seed"])
        if not g:
            continue
        by = {r["seed"]: r for r in g}
        row = dict(cond=cond, variant=variant, runs=len(g), moved=sum(bool(r["moves"]) for r in g))
        for when, tr in (("before", "train_before"), ("after", "train_after")):
            groups = [ok(min((by[s] for s in range(g0, g0 + 5)), key=lambda r: r[when]["vali"])[when])
                      for g0 in range(10, 30, 5) if all(s in by for s in range(g0, g0 + 5))]
            row[when] = dict(ratio=float(np.median([r[tr] / r["oracle_rel_l2"] for r in g])),
                             test=float(np.median([r[when]["test"] for r in g])),
                             single=float(np.mean([ok(r[when]) for r in g])), best=[int(sum(groups)), len(groups)],
                             per_pole=(np.array([r[when]["pole_error"] for r in g]) < T.RECOVERED).mean(0).tolist())
        b, a = row["before"], row["after"]
        pp = ", ".join(f"{lab(z)} {x:.0%} → {y:.0%}" for z, x, y in zip(g[0]["true_poles"], b["per_pole"], a["per_pole"]))
        extra = (f"{np.median([r['eta_hat'] for r in g]):.2g} | {sum(r['noise_regime'] for r in g)} | " if na else "")
        lines.append(f"| {cond} | {variant} | {len(g)} | {extra}{row['moved']} | {b['ratio']:.2f} → {a['ratio']:.2f} | "
                     f"{b['test']:.1e} → {a['test']:.1e} | {b['single']:.0%} → {a['single']:.0%} | "
                     f"{b['best'][0]}/{b['best'][1]} → {a['best'][0]}/{a['best'][1]} | {pp} | "
                     f"{np.median([r['seconds'] for r in g]):.0f} |")
        out.append(row)
    lines.append("")
    (T.R.RESULTS / f"{stem}.json").write_text(json.dumps(out, indent=1))
    (T.R.RESULTS / f"{stem}.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+")
    ap.add_argument("--variant", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 30)))
    ap.add_argument("--rule", choices=("ratio", "noise"), default="ratio")
    ap.add_argument("--start", choices=tuple(STARTS), default="lbfgs")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for v in a.variant:
            for s in a.seeds:
                run(c, v, s, device, a.rule, a.start)


if __name__ == "__main__":
    main()
