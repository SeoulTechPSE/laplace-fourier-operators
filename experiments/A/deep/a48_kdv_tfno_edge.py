"""(A) a47 follow-up: extend the TFNO's lr sweep on forced KdV-Burgers beyond the selected edge.

a47 swept the real-budget TFNO over lr 1e-3, 3e-3, 1e-2 and selected the upper edge (validation 0.0379, 0.0316,
0.0292). As a46 did for a45, this adds lr 3e-2 and 1e-1 at a47's sweep protocol (4000 steps, seed 0) and selects
over all five points. If the selection does not change, a47's five TFNO runs stand and are copied; otherwise five
new final runs (12000 steps, seeds 0-4) replace them. The other models are copied unchanged. Results in
`results/a48_kdv_tfno_edge.json` / `.md`; a47's files are not changed.

**Predictions** (written before running; a47 TFNO: test 0.0338 [0.0283, 0.0393], unseen / test 19.6, probe outside
the band 0.87; WPO: test 0.0245, unseen / test 2.10 [1.08, 3.12]):

1. The selection does not change (as for every edge in a46).
2. Whatever is selected, the TFNO's mean test error stays above the WPO's 0.0245.
3. The TFNO's mean unseen / test stays > 5 and its mean probe error outside the band >= 0.35.

    python experiments/A/deep/a48_kdv_tfno_edge.py [--summary]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a47_kdv_burgers as K  # noqa: E402

A47 = json.loads(K.OUT.read_text())
K.OUT = HERE / "results" / "a48_kdv_tfno_edge.json"
K.SWEEP["tfno_r"] = K.SWEEP["tfno_r"] + [dict(lr=3e-2), dict(lr=1e-1)]


def best(out):
    return min((r for r in out["sweep"] if r["model"] == "tfno_r"), key=lambda r: r["vali"])["cfg"]


def main():
    if "--summary" not in sys.argv:
        def init(o):
            if not o["sweep"]:
                o["check"] = A47["check"]
                o["budget"].update(A47["budget"])
                o["sweep"].extend(A47["sweep"])
                o["runs"].extend(r for r in A47["runs"] if r["model"] != "tfno_r")
        K.locked(init)
        seeds, K.SEEDS = K.SEEDS, ()
        K.run("tfno_r", torch.device("cuda"))                       # new sweep points only
        K.SEEDS = seeds
        cfg = best(K.locked(lambda o: None))

        def copy(o):
            if not any(r["model"] == "tfno_r" for r in o["runs"]):
                o["runs"].extend(r for r in A47["runs"] if r["model"] == "tfno_r" and r["cfg"] == cfg)
        K.locked(copy)
        K.run("tfno_r", torch.device("cuda"))                       # final runs only if the selection changed
    out = K.locked(lambda o: None)
    print("tfno_r selection (a47 -> a48):", best(A47), "->", best(out))
    K.summarize()
    md = K.OUT.with_suffix(".md")
    md.write_text(md.read_text().replace("# (A) a47:", "# (A) a48 (a47 with the TFNO's lr sweep extended):", 1))


if __name__ == "__main__":
    main()
