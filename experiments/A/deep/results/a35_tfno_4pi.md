# (A) 2D Cahn–Hilliard coarsening on the 4 pi box: TFNO trained there against the transferred WPO-2D

Truth on 4 pi: L(21) = 3.452, n = 0.260. Ranges over 3 seeds.

| model | params | window-end error 4 pi | rollout, window 1 / 5 / 20 | L(21) / truth | n | max abs u(21) |
|---|---|---|---|---|---|---|
| tfno4 | 15473 | 0.0306–0.0380 | 0.238 / 0.555 / 0.831 | 1.031–1.042 | 0.270–0.274 | 1.04–1.06 |
| tfno4_m32 | 19109 | 0.0186–0.0227 | 0.153 / 0.383 / 0.627 | 1.020–1.034 | 0.265–0.273 | 1.05–1.06 |
| wpo_aug (2 pi, transferred; a34) | 18737 | 0.0354–0.0406 | 0.238 / 0.577 / 0.849 | 1.038–1.078 | 0.272–0.286 | 1.07–1.09 |
