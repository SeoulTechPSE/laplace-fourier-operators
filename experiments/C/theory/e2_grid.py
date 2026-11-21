"""(C) E2 phase 1b: grid resolution. Does a finer time grid lower the representation floor
and make p6's fast real poles practically identifiable?

Same analysis as ``e2_richness.py`` (tied layer in minimal form, float64, at the truth;
ratio = profiled error of a 10 % change in Re mu / floor), with dt varied at fixed window T:
dt = 20 / N for N in 256 ... 8192 (E1 / E2 use N = 1024). The outputs are exact
(``lfo.data.lti.response``), so only the layer's discretization changes with dt.

    python experiments/C/theory/e2_grid.py      # writes results/e2_grid.md (+ .json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_richness as R  # noqa: E402

N_PER_20 = (256, 512, 1024, 2048, 4096, 8192)
CONFIGS = [("p6", "e1", 20.0, 0), ("p6", "multisine", 20.0, 16), ("p6", "multisine", 80.0, 16),
           ("p4", "e1", 20.0, 0), ("p4", "multisine", 20.0, 16)]


def main():
    out = []
    for cfg in CONFIGS:
        for N in N_PER_20:
            R.DT = 20.0 / N
            r = R.local(*cfg)
            r["dt"], r["n_per_20"] = R.DT, N
            out.append(r)
            print(f"{cfg} N={N}: n_t {r['n_t']} floor {r['floor']:.2e} cond {r['cond']:.1e} ratios "
                  + " ".join(f"{p['Re mu']['ratio']:.1f}" for p in r["poles"]), flush=True)
            (R.RESULTS / "e2_grid.json").write_text(json.dumps(out, indent=1))
    write_md(out)


def write_md(out):
    fmt = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    lines = ["# E2 phase 1b: grid resolution (tied layer, at the truth)", "",
             "dt = 20 / N at fixed T. ratio = profiled error of a 10 % change in Re μ / floor; "
             f"**bold** ≥ {R.RATIO_OK:g}. Also shown: the profiled error itself (s, absolute).", ""]
    for system in ("p4", "p6"):
        rs = [r for r in out if r["system"] == system]
        poles = [fmt(p["pole"]) for p in rs[0]["poles"]]
        lines += [f"## {system}", "",
                  "| family | K | T | N per 20 s | n_t | floor | cond(F) | " +
                  " | ".join(f"{p}: ratio (s)" for p in poles) + " |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(poles)]
        for r in rs:
            cells = []
            for p in r["poles"]:
                q, sv = p["Re mu"]["ratio"], p["Re mu"]["profiled_e"]
                cells.append((f"**{q:.1f}**" if q >= R.RATIO_OK else f"{q:.1f}") + f" ({sv:.1e})")
            lines.append(f"| {r['family']} | {r['K'] or '–'} | {r['T']:g} | {r['n_per_20']} | {r['n_t']} | "
                         f"{r['floor']:.1e} | {r['cond']:.1e} | " + " | ".join(cells) + " |")
        lines.append("")
    (R.RESULTS / "e2_grid.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
