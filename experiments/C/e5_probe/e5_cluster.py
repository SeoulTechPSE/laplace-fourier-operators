"""(C) E5, eleventh pass, follow-up: right Jacobian pole but no stored pole within 10 %. (diagnostic, no training)

In 13 runs on gated sets (11 of them textbook-wired LNOs) the Jacobian impulse-response pole is within 10 % of the
small-signal pole, but the nearest stored pole is 12-20 % from it (d >= 0.1), so rule v4 declines.

Hypothesis. A one-layer LNO is feed-forward. In the textbook wiring its small-signal impulse response is a sum of
exponentials at the stored poles, with gains from the lifting, the channel mixing and the projection, plus a
direct term. The order-2 pencil summarises that sum by one pair. If several stored poles straddle the
small-signal pole, the summary can be right while no single stored pole is.

Test, for every textbook-wired LNO on a gated set (d >= 0.1 runs against d < 0.1 runs):

- pencil poles of the average impulse response h_1 ... h_799 at orders 2, 4 and 6;
- a ridge least-squares fit of h with the stored poles fixed (real and imaginary parts of e^{mu t} as basis);
  its relative residual, and the share of the fitted oscillatory energy carried by stored poles within 25 % of
  the small-signal pole ("cluster") and by the single largest of them.

Prediction (written before running): in the d >= 0.1 runs the stored-pole fit leaves < 5 % of h, at least 2
stored poles within 25 % of the reference each carry >= 10 % of the energy, and together they carry >= 50 %; in
the d < 0.1 runs one stored pole within 10 % carries the largest share.

    python experiments/C/e5_probe/e5_cluster.py        # writes results/e5_cluster.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e5_probe as P  # noqa: E402
import e5_train as TR  # noqa: E402

RL = P.RL
VARIANT = "lnotb_stable_plr10"
NEAR, SHARE, RIDGE = 0.25, 0.10, 1e-6
OUT = P.RESULTS / "e5_cluster.json"


def impulse_response(model, n, device):
    f = lambda x: model(x[None, :, None])[0, :, 0]
    J = torch.func.jacrev(f, chunk_size=128)(torch.zeros(n, device=device)).detach().double().cpu().numpy()
    return np.array([np.diag(J, -d).mean() for d in range(n)])


def stored_fit(h, t, mu):
    """Ridge LS of h(t) on Re / Im e^{mu_k t}; returns (relative residual, energy share per stored pole)."""
    E = np.exp(np.outer(t, mu))                                          # (n, K)
    A = np.concatenate([E.real, E.imag], 1)
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-300
    coef = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ h)
    res = float(np.linalg.norm(A @ coef - h) / np.linalg.norm(h))
    K = mu.size
    parts = np.array([np.linalg.norm(A[:, [k, K + k]] @ coef[[k, K + k]]) ** 2 for k in range(K)])
    return res, parts / parts.sum()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = json.loads((P.RESULTS / "e5_gate.json").read_text())
    v4 = json.loads((P.RESULTS / "e5_v4.json").read_text())
    out = []
    for tag, g in v4.items():
        if not gate[tag]["v3"] or tag.startswith("lz_r10"):              # rho = 10: 1344 poles > 799 samples
            continue
        pr = json.loads((P.RESULTS / f"e5_probe_{tag}.json").read_text())
        ref = complex(*pr["mu_lin"])
        _, grids = TR.data_and_grid(tag)
        case = RL.CASES[TR.case_of(tag)]
        t = np.asarray(grids["t"], float).ravel()
        for r in g["rows"]:
            if r["variant"] != VARIANT or not r["valid"] or r["err"] >= 0.1:
                continue
            model, _, mu = P.load_variant(VARIANT, r["seed"], case, grids, device, tag)
            for q in model.parameters():
                q.requires_grad_(False)
            h = impulse_response(model, t.size, device)
            hh, tt = h[1:800], t[1:800]
            pen = {M: [complex(z) for z in P.pencil_poles(hh, float(t[1] - t[0]), M)] for M in (2, 4, 6)}
            res, share = stored_fit(hh, tt, mu)
            dist = np.abs(mu - ref) / abs(ref)
            near = dist < NEAR
            top = int(np.argmax(np.where(near, share, -1)))
            out.append(dict(set=tag, seed=r["seed"], d=r["d"], err=r["err"], fit_resid=res,
                            cluster_share=float(share[near].sum()), n_near_big=int((near & (share >= SHARE)).sum()),
                            top_share=float(share[top]), top_dist=float(dist[top]),
                            top_pole=[mu[top].real, mu[top].imag], closest_dist=float(dist.min()),
                            pencil={str(M): [[z.real, z.imag] for z in v] for M, v in pen.items()}))
            x = out[-1]
            print(tag, r["seed"], f"d {x['d']:.2f} fit {res:.3f} cluster {x['cluster_share']:.2f} "
                  f"n_big {x['n_near_big']} top {x['top_share']:.2f}@{x['top_dist']:.2f}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    summarize(out)


def summarize(out):
    lines = ["# E5: right Jacobian pole, no stored pole within 10 % (textbook-wired LNO, gated sets, ρ = 5 and oscillators)",
             "", f"Stored-pole ridge fit of the Jacobian's average impulse response; cluster = stored poles within "
             f"{NEAR:.0%} of the small-signal pole; big = share ≥ {SHARE:.0%}.", "",
             "| set | seed | d | J pole error | fit residual | cluster share | big poles in cluster | largest: share @ distance | closest stored pole | order-4 pencil poles |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(out, key=lambda x: (x["d"] < 0.1, x["set"], x["seed"])):
        p4 = ", ".join(f"{a:.2f}{b:+.2f}i" for a, b in x["pencil"]["4"])
        lines.append(f"| `{x['set']}` | {x['seed']} | {x['d']:.2f} | {x['err']:.2f} | {x['fit_resid']:.3f} | "
                     f"{x['cluster_share']:.2f} | {x['n_near_big']} | {x['top_share']:.2f} @ {x['top_dist']:.2f} | "
                     f"{x['closest_dist']:.2f} | {p4} |")
    (P.RESULTS / "e5_cluster.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
