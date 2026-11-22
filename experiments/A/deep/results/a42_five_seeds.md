# (A) Five seeds with 95 % t-intervals for the headline claims

Mean ± standard deviation [95 % t-interval] over seeds 0–4 (seeds 0–2 from a36 / a37).

## F: the form claim, forced Burgers (a36 setup)

| model | n | test | (5, 10] / test | probe outside the band | unseen / test |
|---|---|---|---|---|---|
| wpo | 5 | 0.0239 ± 0.0006 [0.0232, 0.0245] | 1.85 ± 0.08 [1.76, 1.95] | 0.098 ± 0.033 [0.058, 0.139] | 1.26 ± 0.28 [0.92, 1.61] |
| wpo_index | 5 | 0.0230 ± 0.0008 [0.0219, 0.0240] | 1.75 ± 0.06 [1.67, 1.83] | 0.980 ± 0.026 [0.948, 1.012] | 180.52 ± 59.65 [106.47, 254.56] |
| tfno_r | 5 | 0.0287 ± 0.0020 [0.0262, 0.0311] | 10.04 ± 0.67 [9.20, 10.88] | 0.874 ± 0.076 [0.779, 0.969] | 30.50 ± 9.52 [18.69, 42.32] |

## P: PDEBench 1D Burgers (a37 setup)

| model | n | nRMSE 256 | probe ν' (truth 0.1257) | seeds with ν' within 5 % | dispersion error |
|---|---|---|---|---|---|
| wpo | 5 | 0.0450 ± 0.0008 [0.0440, 0.0459] | 0.1014 ± 0.0179 [0.0792, 0.1235] | 1 of 5 | 0.246 ± 0.116 [0.103, 0.390] |
| tfno_r | 5 | 0.0254 ± 0.0006 [0.0247, 0.0261] | 0.0402 ± 0.0084 [0.0297, 0.0507] | 0 of 5 | 0.691 ± 0.078 [0.594, 0.787] |

Ratio of mean nRMSE, WPO / TFNO: 1.77.
