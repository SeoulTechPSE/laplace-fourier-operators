# (D2) Deep rollout on forced Burgers with initial fields: 4 chained windows

| model | params | window test (3 seeds) | rollout error over [0, 20] | rollout / test |
|---|---|---|---|---|
| wpo_ic | 61130 | 0.0249–0.0260 (mean 0.0254) | 0.0511–0.0568 (mean 0.0541) | 2.05–2.18 (mean 2.13) |
| wpo_chan | 59553 | 0.0259–0.0288 (mean 0.0273) | 0.0581–0.0618 (mean 0.0603) | 2.02–2.35 (mean 2.21) |
| tfno_chan | 59113 | 0.0243–0.0259 (mean 0.0250) | 0.0603–0.0624 (mean 0.0611) | 2.34–2.51 (mean 2.44) |
