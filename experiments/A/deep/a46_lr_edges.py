"""(A) a45 follow-up: extend the learning-rate sweep where a45 selected an edge of its grid.

a45 (lr 1e-3, 3e-3, 1e-2; pole lr x1 / x30 for ALNO and LFNO) selected the upper edge, lr 1e-2, for F:alno,
F:lfno, P:s4d, P:alno and P:lfno, and the lower edge, 1e-3, for P:lno. A reviewer can read an edge selection as
an under-tuned baseline. This pass adds grid points beyond each edge, at a45's protocol (sweep 4000 steps with
seed 0, selection by validation over the old and new points together; final 12000 steps, seeds 0-4):

- upper edge: lr 3e-2 and 1e-1 (ALNO / LFNO with pole lr x1 and x30). For S4D the kernel parameters keep
  lr min(lr, 1e-3), as in a45;
- P:lno: lr 3e-4 and 1e-4.

If the selection does not change, a45's five runs stand and are copied; otherwise five new final runs replace them.
F:s4d (selected 3e-3, inside the grid) and F:lno (a36's protocol) are not rerun. Results in
`results/a46_lr_edges.json` / `.md`, in a45's format; a45's files are not changed.

**Predictions** (written before running; a45 values in parentheses; WPO from a42: F test 0.0239, unseen / test
1.26 [0.92, 1.61]; P nRMSE 0.0450):

1. P:s4d: the mean nRMSE after the extended sweep is below a45's (0.0513); its validation was still falling
   at the edge (0.134, 0.084, 0.067).
2. P:s4d: it stays at or above the WPO's 0.0450 (mean).
3. F:alno and F:lfno: mean test stays >= 1.5x the WPO's (>= 0.036; a45 0.077 and 0.094).
4. F:alno and F:lfno: mean unseen / test stays > 3 (a45 114 and 9.2).
5. P:lno: nRMSE stays > 0.5 at every lr (a45 0.878): the original layer does not train on PDEBench.
6. P:alno and P:lfno: mean nRMSE stays above the WPO's 0.0450 (a45 0.088 and 0.275).

    python experiments/A/deep/a46_lr_edges.py [F:alno F:lfno P:s4d P:alno P:lfno P:lno] [--summary]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a45_baselines as X45  # noqa: E402

A45 = json.loads(X45.OUT.read_text())
X45.OUT = HERE / "results" / "a46_lr_edges.json"
UP = (3e-2, 1e-1)
NEW = {"alno": [dict(lr=lr, plr=p) for lr in UP for p in (1.0, 30.0)],
       "lfno": [dict(lr=lr, plr=p) for lr in UP for p in (1.0, 30.0)],
       "s4d": [dict(lr=lr) for lr in UP],
       "lno": [dict(lr=lr) for lr in (3e-4, 1e-4)]}
for k, cs in NEW.items():
    X45.SWEEP[k] = X45.SWEEP[k] + cs
JOBS = ["F:alno", "F:lfno", "P:s4d", "P:alno", "P:lfno", "P:lno"]


def seed_file():
    """Copy a45's budgets and sweep once, and the runs not reswept (F:s4d, F:lno); the other a45 runs are copied
    per job only if its selection stands."""
    def init(o):
        if not o["sweep"]:
            o["budget"].update(A45["budget"])
            o["sweep"].extend(A45["sweep"])
            o["runs"].extend(r for r in A45["runs"] if (r["part"], r["model"]) in (("F", "s4d"), ("F", "lno")))
    X45.locked(init)


def copy_runs_if_unchanged(part, kind):
    def f(o):
        cfg = X45.best_cfg(part, kind, o)
        have = {r["seed"] for r in o["runs"] if r["part"] == part and r["model"] == kind}
        old = [r for r in A45["runs"] if r["part"] == part and r["model"] == kind and r["cfg"] == cfg]
        o["runs"].extend(r for r in old if r["seed"] not in have)
    X45.locked(f)


def sweep_only(part, kind, device):
    """Run the new sweep points (run_f / run_p skip the ones present), then stop before the final runs."""
    seeds = X45.SEEDS
    X45.SEEDS = ()
    try:
        (X45.run_f if part == "F" else X45.run_p)(kind, device)
    finally:
        X45.SEEDS = seeds


def summarize():
    out = X45.locked(lambda o: None)
    print("selection (a45 -> a46):")
    for j in JOBS:
        p, k = j.split(":")
        old = X45.best_cfg(p, k, A45)
        new = X45.best_cfg(p, k, out)
        print(f"  {j}: {old} -> {new}", "(unchanged)" if old == new else "(changed)")
    X45.summarize()
    md = X45.OUT.with_suffix(".md")
    md.write_text(md.read_text().replace("# (A) a45:", "# (A) a46 (a45 with the lr sweep extended at its edges):", 1))


def main():
    jobs = [a for a in sys.argv[1:] if a in JOBS]
    if "--summary" not in sys.argv:
        device = torch.device("cuda")
        seed_file()
        for j in jobs or JOBS:
            part, kind = j.split(":")
            sweep_only(part, kind, device)
            copy_runs_if_unchanged(part, kind)
            (X45.run_f if part == "F" else X45.run_p)(kind, device)   # final runs only if the selection changed
    summarize()


if __name__ == "__main__":
    main()
