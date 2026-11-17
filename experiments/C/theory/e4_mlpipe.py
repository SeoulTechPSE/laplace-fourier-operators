"""(C) E4, input noise: the frequency-domain EIV ML through the whole pipeline, on development and fresh groups.

The twentieth pass refitted the `foh_eivo` reports under EIVML and found the EIV bias gone without a gate.
Here EIVML runs the whole pipeline (`e4_profile.py --hold foh_ml`): every run refitted and reduced by backward
elimination with plain chi2 thresholds (no bootstrap calibration), the certificate, the consensus, the order test
against the best run's full model, and profile intervals, all under the EIVML cost (noise units, sigma^2 = 1). A
reported pole within 10 % of another reported pole, or above Nyquist, is unresolved. The rule was fixed before the
fresh groups were trained (p4 multisine eta_u = 0.3 and p6 E1 0.1: groups 50-65; p6 E1 0.05: groups 30-45).

This script scores `foh_ml` against `foh_eivo` per condition, development groups and fresh groups apart.

Finding: up to eta_u = 0.05 `foh_ml` works without calibration (p6 E1 0.05 fresh: 15/16, 1 false; `foh_eivo`
13/16, 3); at 0.1 (p4) and 0.3 it is worse, because undamped poles on DFT-grid frequencies remove lines from
the EIVML cost (see EP.EIVMLT, whose finite-record variance degenerates in turn). README "E4, twenty-first pass".

    python experiments/C/theory/e4_mlpipe.py      # writes results/e4_mlpipe.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e4_profile as EP  # noqa: E402

T = EP.T
FRESH = {("p4_ms16_T20_n1e-2_u3e-1", g) for g in (50, 55, 60, 65)} \
    | {("p6_e1_T20_n1e-2_u1e-1", g) for g in (50, 55, 60, 65)} \
    | {("p6_e1_T20_n1e-2_u5e-2", g) for g in (30, 35, 40, 45)}
HOLDS = ("foh_ml", "foh_eivo")


def tally(hold):
    """{(cond, fresh): [groups, identified, true poles, false, unresolved, order ok, seconds]}."""
    out = {}
    for f in sorted(EP.out_dir(hold).glob("*.json")):
        r = json.loads(f.read_text())
        if r["cond"] not in EP.EIV_CONDS:
            continue
        per, false = EP.score(r)
        t = out.setdefault((r["cond"], (r["cond"], r["group"]) in FRESH), [0, 0, 0, 0, 0, 0, 0.0])
        t[0] += 1
        t[1] += sum(bool(x and x["closed"]) for x in per.values())
        t[2] += len(per)
        t[3] += false
        t[4] += len(r.get("unresolved", []))
        t[5] += bool(r["order_ok"])
        t[6] += r["seconds"]
    return out


def groups(hold, cond):
    """Per-group report lines (poles, unresolved marked) for one condition."""
    rows = []
    for f in sorted(EP.out_dir(hold).glob(f"{cond}_g*.json"), key=lambda p: int(p.stem.split("_g")[-1])):
        r = json.loads(f.read_text())
        un = [complex(*z) for z in r.get("unresolved", [])]
        lab = lambda z: (f"{z.real:.3g}{z.imag:+.3g}i" if z.imag else f"{z.real:.3g}") + (" (u)" if z in un else "")
        per, false = EP.score(r)
        rows.append(f"| {cond} | {r['group']}{' (fresh)' if (cond, r['group']) in FRESH else ''} | {r['n_cert']} | "
                    f"{r['source']} | {r['order_ok']} | {', '.join(lab(complex(*z)) for z in r['report'])} | "
                    f"{sum(bool(x and x['closed']) for x in per.values())}/{len(per)} | {false} |")
    return rows


def main():
    tl = {h: tally(h) for h in HOLDS}
    keys = sorted(set(tl["foh_ml"]) | set(tl["foh_eivo"]))
    lines = ["# E4: EIV ML through the whole pipeline (`--hold foh_ml`) against the calibrated pipeline (`foh_eivo`)", "",
             "Identified = within 10 % of a true pole with a closed 95 % profile interval; false = other reported "
             "poles with a closed interval; unresolved (`foh_ml` only) = duplicates within 10 % or poles above Nyquist, "
             "neither identified nor false. Fresh groups were trained after the `foh_ml` rule was fixed.", "",
             "| condition | groups | `foh_ml`: identified, false, unresolved, order test passed | "
             "`foh_eivo`: identified, false, order test passed | seconds per group (ml / eivo) |",
             "|---|---|---|---|---|"]
    for k in keys:
        a, b = tl["foh_ml"].get(k), tl["foh_eivo"].get(k)
        ca = f"{a[1]}/{a[2]}, {a[3]}, {a[4]}, {a[5]}/{a[0]}" if a else "–"
        cb = f"{b[1]}/{b[2]}, {b[3]}, {b[5]}/{b[0]}" if b else "–"
        sec = f"{a[6] / a[0]:.0f}" if a else "–"
        sec += f" / {b[6] / b[0]:.0f}" if b else " / –"
        lines.append(f"| {k[0]}{' (fresh)' if k[1] else ''} | {(a or b)[0]} | {ca} | {cb} | {sec} |")
    lines += ["", "## `foh_ml` reports per group", "",
              "| condition | group | certified runs | report from | order test passed | reported poles ((u) = unresolved) "
              "| identified | false |", "|---|---|---|---|---|---|---|---|"]
    for c in sorted({k[0] for k in tl["foh_ml"]}):
        lines += groups("foh_ml", c)
    (T.R.RESULTS / "e4_mlpipe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
