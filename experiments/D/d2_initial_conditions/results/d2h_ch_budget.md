# (D2) d2g at 12000 training steps (Cahn–Hilliard with initial fields)

| model | params | id | amp (ratio to id) | high (ratio to id) | roll4 | roll10 [0, 50] |
|---|---|---|---|---|---|---|
| wpo_ic | 61130 | 0.0185 | 0.0920 (4.68–5.15) | 0.0423 (1.96–2.51) | 0.0446 | 0.0486 |
| tfno_chan | 59113 | 0.0097 | 0.0621 (5.56–6.91) | 0.1308 (10.62–15.48) | 0.0207 | 0.0303 |

Per-window error on roll10 (mean over seeds):

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| wpo_ic | 0.020 | 0.030 | 0.031 | 0.036 | 0.038 | 0.046 | 0.045 | 0.046 | 0.052 | 0.056 |
| tfno_chan | 0.010 | 0.016 | 0.017 | 0.020 | 0.023 | 0.030 | 0.031 | 0.030 | 0.033 | 0.034 |

6000 (d2g) against 12000 steps, means over seeds:

| model | id 6k / 12k | high / id 6k / 12k | roll4 6k / 12k |
|---|---|---|---|
| wpo_ic | 0.0255 / 0.0185 | 2.35 / 2.29 | 0.0498 / 0.0446 |
| tfno_chan | 0.0126 / 0.0097 | 10.46 / 13.47 | 0.0240 / 0.0207 |

roll4 ratio tfno_chan / wpo_ic: 0.48 (6k), 0.46 (12k).
