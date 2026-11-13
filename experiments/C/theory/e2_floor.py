"""(C) E2: a floor certificate without the truth.

``e2_polish.py`` showed (p6, dt = 20/4096) that every run within 1.5 x the oracle floor
recovers all poles. The oracle needs the true poles. Here the floor is estimated from
the runs themselves, and a second, truth-free check asks whether the runs at the
estimated floor agree on their poles:

  floor estimate  lowest final training loss among the K runs of a group
  certified       runs with training loss < CERT x floor estimate
  effective poles folded learned poles; slots with < 2 % of the summed impulse-response
                  norm |beta| ||e^{mu t}||_2 dropped; poles within 10 % merged (weighted mean)
  consensus       the poles present (within 10 %) in >= 2/3 of the certified runs (>= 2
                  certified runs needed): the poles the data identify, as far as the runs tell

Noise-aware rule (E4; conditions with output noise): when the floor estimate is at the noise
level (S_min <= 2 n_s n_t sigma^2, with S the sum of squared normalised training residuals and
sigma^2 estimated from the outputs, ``e2_rescue.noise_var``), the 1.5 x ratio certifies every
run. A run is then certified when S - S_min <= chi2_p(0.99) sigma^2 (likelihood-ratio test,
p = real parameters of the layer): the runs the data cannot tell apart from the best one.

The truth is used only to score the consensus: per true pole, whether it is in the
consensus (within 0.1); false = consensus poles that match no true pole.

    python experiments/C/theory/e2_floor.py               # polished runs and rescued runs -> results/e2_floor.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_polish as P  # noqa: E402
import e2_rescue as RS  # noqa: E402

T, E = P.T, P.E
CERT, K, AGREE, MERGE, WEIGHT = 1.5, 5, 2 / 3, 0.1, 0.02
SEEDS = range(10, 30)
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
         ("p6_ms16_T80_N4096_lfno_pt003", "damp8_ms"), ("p6_ms16_T80_N4096_lfno_pt03", "damp8_ms")]
STAGES = {"polish": (P.OUT, "_lbfgs"), "rescue": (T.R.RESULTS / "e2_rescue", "_rescue"),
          "rescue_na": (RS.OUT_NA, "_rescue_na")}
NOISY = lambda cond: T.extras(cond)["noise"] > 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def effective(mu, beta, T_end):
    mu, beta = E.fold(mu, beta)
    re = np.minimum(mu.real, -1e-6)
    w = np.abs(beta) * np.sqrt((1 - np.exp(2 * re * T_end)) / (-2 * re))
    keep = w >= WEIGHT * w.sum()
    mu, w = mu[keep], w[keep]
    clusters = []                                          # [weighted sum, weight]
    for i in np.argsort(-w):
        for c in clusters:
            z = c[0] / c[1]
            if abs(mu[i] - z) < MERGE * abs(z):
                c[0] += w[i] * mu[i]
                c[1] += w[i]
                break
        else:
            clusters.append([w[i] * mu[i], w[i]])
    return np.array([c[0] / c[1] for c in clusters])


def consensus(sets):
    """Poles present (within MERGE) in >= AGREE of the runs; representative = mean of the nearest pole per run."""
    need = int(np.ceil(AGREE * len(sets)))
    cands = []
    for z in np.concatenate(sets):
        near = [e[np.argmin(np.abs(e - z))] for e in sets if np.min(np.abs(e - z)) < MERGE * abs(z)]
        if len(near) >= need:
            cands.append((len(near), np.mean(near)))
    out = []
    for n, z in sorted(cands, key=lambda c: -c[0]):
        if all(abs(z - y) >= MERGE * abs(y) for y in out):
            out.append(z)
    return np.array(out)


def load(stage, cond, variant):
    out, suffix = STAGES[stage]
    system, family, Tw, Kk, N = T.unpack(cond)
    rs, data = [], None
    for s in SEEDS:
        name = T.run_name(cond, variant, s)
        f, ck = out / f"{name}.json", T.RUNS / f"{name}{suffix}.pt"
        if not (f.exists() and ck.exists()):
            continue
        r = json.loads(f.read_text())
        mu_true, beta_true = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
        t = np.arange(int(round(Tw / (20.0 / N)))) * (20.0 / N)
        model, _ = T.build_for(cond, variant, s, t, mu_true, beta_true)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        mu, beta = (a.detach().numpy() for a in model.pole_sets())
        extra = {}
        if NOISY(cond):
            if data is None or s // T.NOISE_GROUP != data[0]:
                data = (s // T.NOISE_GROUP, T.setup(cond, s)[0])
            y = data[1]["train"][1]
            extra = dict(S=RS.sse(model.to(DEVICE), data[1], DEVICE), sig2=RS.noise_var(y), n_el=int(y.numel()),
                         p=sum(2 * q.numel() if q.is_complex() else q.numel() for q in model.parameters()))
        rs.append(dict(seed=s, train=r["train_after"], oracle=r["oracle_rel_l2"],
                       ok=max(r["after"]["pole_error"]) < T.RECOVERED, eff=effective(mu, beta, Tw), **extra))
    return rs


def verdict(group, true, rule="ratio"):
    floor = min(r["train"] for r in group)
    best = min(group, key=lambda r: r["train"])
    cert = [r for r in group if r["train"] < CERT * floor]
    if rule == "noise":
        s_min, sig2 = min(r["S"] for r in group), np.mean([r["sig2"] for r in group])
        if s_min <= RS.NOISE_REGIME * group[0]["n_el"] * sig2:
            cert = [r for r in group if r["S"] - s_min <= chi2.ppf(0.99, group[0]["p"]) * sig2]
    cons = consensus([r["eff"] for r in cert]) if len(cert) >= 2 else np.array([])
    hit = lambda z, pts: pts.size > 0 and np.min(np.abs(pts - z)) < T.RECOVERED * abs(z)
    return dict(floor=floor, oracle=best["oracle"], n_cert=len(cert), cert_ok=sum(r["ok"] for r in cert),
                consensus=[[float(z.real), float(z.imag)] for z in cons],
                identified=[bool(hit(z, cons)) for z in true],               # per true pole
                false=int(sum(not hit(z, true) for z in cons)))


def analyse(stage, cond, variant, rule="ratio"):
    rs = load(stage, cond, variant)
    if len(rs) < K:
        return None
    by = {r["seed"]: r for r in rs}
    system = T.unpack(cond)[0]
    true = E.true_folded(*(np.array(a, dtype=complex) for a in E.SYSTEMS[system]))[0]
    groups = [verdict([by[s] for s in range(g, g + K)], true, rule) for g in range(SEEDS.start, SEEDS.stop, K)
              if all(s in by for s in range(g, g + K))]
    pooled = verdict(rs, true, "ratio")                      # pooled runs span 4 noise draws: ratio rule only
    return dict(stage=stage, rule=rule, cond=cond, variant=variant, runs=len(rs), groups=groups, pooled=pooled,
                true_poles=[[float(z.real), float(z.imag)] for z in true],
                single_ok=float(np.mean([r["ok"] for r in rs])))


def main():
    jobs = [(st, c, v, "ratio") for st in ("polish", "rescue") for c, v in CASES]
    jobs += [(st, c, v, "noise") for st in ("rescue", "rescue_na") for c, v in CASES if NOISY(c)]
    out = [a for st, c, v, rule in jobs if (a := analyse(st, c, v, rule))]
    (T.R.RESULTS / "e2_floor.json").write_text(json.dumps(out, indent=1, default=float))
    lab = lambda z: f"{z[0]:+.2f}{z[1]:+.2f}i" if z[1] else f"{z[0]:+.2f}"
    lines = ["# E2: floor certificate without the truth", "",
             f"Groups of {K} consecutive seeds (10-29). Floor estimate = lowest training loss in the group; "
             f"certified = training loss < {CERT} x estimate. Consensus = effective poles present (within "
             f"{MERGE:.0%}) in ≥ 2/3 of the certified runs (needs ≥ 2 certified); slots below {WEIGHT:.0%} "
             "impulse-response weight dropped, slots within 10 % merged. Per true pole: groups whose consensus "
             "contains it (within 0.1); false = consensus poles matching no true pole (truth used only for "
             "these two scores). Pooled: the same over all 20 runs (ratio rule). Rule `noise`: certified = "
             "S − S_min ≤ χ²_p(0.99) σ̂² when the floor is at the noise level (see the docstring); stage "
             "`rescue_na` = restarts with the noise-aware acceptance (e2_rescue.py --rule noise).", ""]
    for system in ("p4", "p6"):
        rows = [a for a in out if a["cond"].startswith(system)]
        if not rows:
            continue
        labels = [lab(z) for z in rows[0]["true_poles"]]
        lines += [f"## {system}", "",
                  "| stage | rule | condition | variant | single: all | estimate / oracle | certified: all recovered | "
                  + " | ".join(f"{l} identified" for l in labels) + " | false | pooled consensus |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(labels) + "---|---|"]
        for a in rows:
            g = a["groups"]
            ratio = np.median([x["floor"] / x["oracle"] for x in g])
            cert = f"{sum(x['cert_ok'] for x in g)}/{sum(x['n_cert'] for x in g)}"
            per = [f"{sum(x['identified'][j] for x in g)}/{len(g)}" for j in range(len(labels))]
            p = a["pooled"]
            lines.append(f"| {a['stage']} | {a['rule']} | {a['cond']} | {a['variant']} | {a['single_ok']:.0%} | {ratio:.2f} | {cert} | "
                         + " | ".join(per) + f" | {sum(x['false'] for x in g)} | "
                         + (", ".join(lab(z) for z in p["consensus"]) or "–") + f" ({p['n_cert']} cert.) |")
        lines.append("")
    lines.append("")
    (T.R.RESULTS / "e2_floor.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
