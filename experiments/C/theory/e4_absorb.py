"""(C) E4, input noise: can the calibrated reduction path tell a merged close pair from EIV bias?

`e4_profile.py --hold foh_eivu` flags a kept pole as unresolved when a calibration-only drop
(chi2_dp(0.99) < dS <= kappa chi2_dp(0.99)) removes a neighbouring pole (|z_removed - z_kept| <=
|z_kept|). This script re-reads the stored `foh_eivu` steps (no refits) and compares that rule
with a second truth-free rule:

  neighbour  as foh_eivu (flags stored in the group JSON)
  absorb     a kept pole that moves by more than MERGE |z| in a calibration-only drop
             (it absorbed what the dropped pole carried)

Per condition: groups with a flag in the report or in a certified run, and the slow poles
(|z| < 3) that moved under the absorb rule, as before -> after.

    python experiments/C/theory/e4_absorb.py      # writes results/e4_absorb.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e4_profile as EP  # noqa: E402

SRC = EP.T.R.RESULTS / "e4_profile_foh_eivu"
MERGE = EP.F.MERGE
SLOW = 3.0


def C(zs):
    return [complex(*z) for z in zs]


def absorbed(steps, start):
    """(after, before) for kept poles that move by > MERGE |z| in an accepted calibration-only drop."""
    out, before = [], C(start)
    for st in steps:
        after = C(st["poles"])
        if st.get("removed") and st.get("move") == "drop" and st["dS"] > st["chi2_threshold"]:
            for z in after:
                b = min(before, key=lambda w: abs(w - z))
                if abs(b - z) > MERGE * abs(z):
                    out.append((z, b))
        if st.get("removed"):
            before = after
    return out


def main():
    rs = [json.loads(f.read_text()) for f in sorted(SRC.glob("*.json"))]
    lines = ["# E4: unresolved flags from the calibrated reduction path", "",
             "Stored `foh_eivu` steps, no refits. neighbour = the `foh_eivu` rule (a calibration-only drop of a "
             "pole within |z_kept| of a kept one); absorb = a kept pole moves by > "
             f"{MERGE:g} |z| in a calibration-only drop. Groups with a flag in the report or a certified run; "
             f"slow poles (|z| < {SLOW:g}) moved under absorb, before → after.", "",
             "| condition | groups | neighbour: groups flagged | absorb: groups flagged | slow poles moved (absorb) |",
             "|---|---|---|---|---|"]
    for cond in dict.fromkeys(r["cond"] for r in rs):
        g = [r for r in rs if r["cond"] == cond]
        n_nb = sum(bool(r.get("unresolved")) for r in g)
        n_ab, moved = 0, set()
        for r in g:
            hits = absorbed(r["calibrated_steps"], r["report_before"])
            for x in r["runs"]:
                if x["certified"] and "steps" in x:
                    hits += absorbed(x["steps"], x["poles_before"])
            n_ab += bool(hits)
            moved |= {f"{b.real:.2f} → {z.real:.2f}" for z, b in hits if abs(z) < SLOW and abs(b) < SLOW and not z.imag}
        lines.append(f"| {cond} | {len(g)} | {n_nb} | {n_ab} | {', '.join(sorted(moved)) or '–'} |")
    (EP.T.R.RESULTS / "e4_absorb.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
