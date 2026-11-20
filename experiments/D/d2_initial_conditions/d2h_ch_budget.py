"""(D2) Is d2g's 2x TFNO edge on Cahn–Hilliard the training budget? d2g again at 12000 steps.

d2g (6000 steps, d2c's developed-start protocol) found the TFNO 2x better than `wpo_ic` both in distribution
(0.013 against 0.026) and over 4 chained windows (0.024 against 0.050). A16 (no initial state, same system,
12000 steps) had the two at parity. d2g's write-up named the halved budget as a possible cause without checking it.

**Change.** Only the step count: d2b's `STEPS` = 12000 (cosine schedule over the full run). Data, models, seeds,
evaluation sets and metrics are d2g's (`d2g_ch_rollout.py`). Models: `wpo_ic` and `tfno_chan` (the pair of d2g's
prediction 4); `wpo_chan` is not rerun.

**Predictions** (written before running; d2g's 6000-step means for reference).

1. Both models improve in distribution: mean `id` error at 12000 steps below the 6000-step mean (wpo_ic 0.026,
   tfno_chan 0.013).
2. The budget explains part of the gap but not all of it: the ratio of means roll4(tfno_chan) / roll4(wpo_ic) is
   above d2g's 0.48 and below 0.9.
3. The unseen-mode finding survives the budget: the TFNO's mean ratio high / id is at least 2x `wpo_ic`'s.

    python experiments/D/d2_initial_conditions/d2h_ch_budget.py      # results/d2h_ch_budget.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2b_burgers_rollout as R  # noqa: E402
import d2g_ch_rollout as G  # noqa: E402

R.STEPS = 12000
G.KINDS = ("wpo_ic", "tfno_chan")
G.OUT = HERE / "results" / "d2h_ch_budget.json"


def main():
    G.main()
    new = json.loads(G.OUT.read_text())
    old = json.loads((HERE / "results" / "d2g_ch_rollout.json").read_text())
    mean = lambda rs, kind, q: np.mean([r[q] for r in rs if r["model"] == kind])
    lines = G.OUT.with_suffix(".md").read_text().splitlines()
    lines[0] = "# (D2) d2g at 12000 training steps (Cahn–Hilliard with initial fields)"
    lines += ["", "6000 (d2g) against 12000 steps, means over seeds:", "",
              "| model | id 6k / 12k | high / id 6k / 12k | roll4 6k / 12k |", "|---|---|---|---|"]
    for kind in G.KINDS:
        hr = lambda rs: np.mean([r["high"] / r["id"] for r in rs if r["model"] == kind])
        lines.append(f"| {kind} | {mean(old, kind, 'id'):.4f} / {mean(new, kind, 'id'):.4f} | {hr(old):.2f} / "
                     f"{hr(new):.2f} | {mean(old, kind, 'roll4'):.4f} / {mean(new, kind, 'roll4'):.4f} |")
    r6 = mean(old, "tfno_chan", "roll4") / mean(old, "wpo_ic", "roll4")
    r12 = mean(new, "tfno_chan", "roll4") / mean(new, "wpo_ic", "roll4")
    lines += ["", f"roll4 ratio tfno_chan / wpo_ic: {r6:.2f} (6k), {r12:.2f} (12k)."]
    G.OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
