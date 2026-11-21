# (A) 2D Cahn–Hilliard coarsening: transfer from the 2 pi box to a 4 pi box (same spacing)

Truth on 4 pi: L(1) = 1.509, L(21) = 3.452, n = 0.260 (a31, 2 pi box: L(21) = 3.522, n = 0.271). Ranges over 3 seeds.

| model | window-end error 2 pi | window-end error 4 pi | ratio | rollout 4 pi, window 1 / 5 / 20 | L(21) / truth | n | max abs u(21) |
|---|---|---|---|---|---|---|---|
| wpo_aug | 0.0346–0.0393 | 0.0354–0.0406 | 1.02–1.03 | 0.238 / 0.577 / 0.849 | 1.038–1.078 | 0.272–0.286 | 1.07–1.09 |
| tfno_aug | 0.0181–0.0197 | 0.6605–0.7139 | 34.97–37.83 | 1.125 / 1.344 / 1.41 | 2.154–2.194 | 0.306–0.348 | 1.03–1.03 |
