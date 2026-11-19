"""(C) E5, fourteenth pass: rule v5 = v4 and agreement of the Jacobian pole with the data's low-amplitude pole.

Pre-registered in `results/e5_v5_prereg.md`. v4 (`e5_v4.py`) checks that the model is self-consistent; v5 adds
|mu_J - mu_low| / |mu_low| < 0.1, with mu_low the equivalent linear pole of the lowest-amplitude fifth of the
training samples (the v3 gate's quantity, `results/e5_gate.json`, from the data alone).

Reads `results/e5_v4.json`, `results/e5_probe_<set>.json` and `results/e5_gate.json`; no model is loaded.

    python experiments/C/e5_probe/e5_v5.py      # writes results/e5_v5.md / .json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA_AGREE, CORRECT = 0.1, 0.1
NEW = "lnots2_stable_plr10"


def fold(z):
    return z if z.imag >= 0 else z.conjugate()


def main():
    v4 = json.loads((RES / "e5_v4.json").read_text())
    gate = json.loads((RES / "e5_gate.json").read_text())
    rows = []
    for s, d in v4.items():
        pr = json.loads((RES / f"e5_probe_{s}.json").read_text())["variants"]
        mu_low = complex(*gate[s]["mu_low"])
        for r in d["rows"]:
            p = next(x for x in pr[r["variant"]] if x["seed"] == r["seed"])
            hp = p["jac"].get("h_poles") or []
            dj = abs(fold(complex(*hp[0])) - fold(mu_low)) / abs(mu_low) if hp else float("inf")
            rows.append(dict(set=s, variant=r["variant"], seed=r["seed"], err=r["err"], v4=bool(r["v4"]), data_agree=dj,
                             v5=bool(r["v4"]) and dj < DATA_AGREE))
    (RES / "e5_v5.json").write_text(json.dumps(rows, indent=1))

    def count(rs, rule):
        t = [r for r in rs if r[rule]]
        return len(t), sum(r["err"] >= CORRECT for r in t)

    old = [r for r in rows if r["variant"] != NEW]
    new = [r for r in rows if r["variant"] == NEW]
    lines = ["# E5: rule v5 (v4 + Jacobian pole within 10 % of the data's low-amplitude pole)", "",
             "| runs | n | v4 trusted / false | v5 trusted / false | v4-correct kept by v5 |", "|---|---|---|---|---|"]
    for name, rs in (("design (passes 1–13)", old), (f"prospective (`{NEW}`)", new)):
        a, b = count(rs, "v4"), count(rs, "v5")
        good = [r for r in rs if r["v4"] and r["err"] < CORRECT]
        kept = sum(r["v5"] for r in good)
        lines.append(f"| {name} | {len(rs)} | {a[0]} / {a[1]} | {b[0]} / {b[1]} | {kept} / {len(good)} |")
    lines += ["", "v4-trusted runs that v5 drops or that are false (pole error ≥ 10 %):", "",
              "| set | variant | seed | J pole error | |μ_J − μ_low| / |μ_low| | v5 |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["v4"] and (not r["v5"] or r["err"] >= CORRECT):
            lines.append(f"| `{r['set']}` | {r['variant']} | {r['seed']} | {r['err']:.3f} | {r['data_agree']:.3f} | "
                         f"{'trusted' if r['v5'] else 'dropped'} |")
    (RES / "e5_v5.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
