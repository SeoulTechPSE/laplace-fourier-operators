# (A) The WPO in float32: cost and accuracy

## Cost, measured alone on the GPU (NVIDIA GeForce RTX 4090, torch 2.6.0+cu124)

| setup | model | real params | dtype | train step (ms) | ratio to tfno_r | ratio to wpo (float64) | inference (ms / sample) | peak train memory (MB) |
|---|---|---|---|---|---|---|---|---|
| F | wpo | 59529 | float64 | 128.8 | 4.56 | 1.00 | 1.668 | 2197 |
| F | wpo32 | 59529 | float32 | 28.2 | 1.00 | 0.22 | 0.417 | 1112 |
| F | tfno_r | 54649 | float32 | 28.2 | 1.00 | 0.22 | 0.460 | 1277 |
| P | wpo | 61130 | float64 | 78.2 | 4.13 | 1.00 | 1.060 | 1570 |
| P | wpo32 | 61130 | float32 | 24.9 | 1.31 | 0.32 | 0.258 | 815 |
| P | tfno_r | 64745 | float32 | 19.0 | 1.00 | 0.24 | 0.292 | 975 |

## Cost with the fixed layer (post hoc) (NVIDIA GeForce RTX 4090, torch 2.6.0+cu124)

| setup | model | real params | dtype | train step (ms) | ratio to tfno_r | ratio to wpo (float64) | inference (ms / sample) | peak train memory (MB) |
|---|---|---|---|---|---|---|---|---|
| F | wpo | 59529 | float64 | 152.3 | 4.96 | 1.00 | 2.044 | 2197 |
| F | wpo32 | 59529 | float32 | 32.1 | 1.05 | 0.21 | 0.494 | 1112 |
| F | tfno_r | 54649 | float32 | 30.7 | 1.00 | 0.20 | 0.527 | 1277 |
| P | wpo | 61130 | float64 | 88.2 | 4.46 | 1.00 | 1.228 | 1570 |
| P | wpo32 | 61130 | float32 | 26.0 | 1.31 | 0.29 | 0.296 | 815 |
| P | tfno_r | 64745 | float32 | 19.8 | 1.00 | 0.22 | 0.329 | 975 |

## F: forced Burgers (a42 setup)

Mean ± standard deviation [95 % t-interval] over seeds 0–4; float64 rows from a42.

| model | n | test | (5, 10] / test | probe outside the band | unseen / test |
|---|---|---|---|---|---|
| wpo | 5 | 0.0239 ± 0.0006 [0.0232, 0.0245] | 1.85 ± 0.08 [1.76, 1.95] | 0.098 ± 0.033 [0.058, 0.139] | 1.26 ± 0.28 [0.92, 1.61] |
| wpo_index | 5 | 0.0230 ± 0.0008 [0.0219, 0.0240] | 1.75 ± 0.06 [1.67, 1.83] | 0.980 ± 0.026 [0.948, 1.012] | 180.52 ± 59.65 [106.47, 254.56] |
| tfno_r | 5 | 0.0287 ± 0.0020 [0.0262, 0.0311] | 10.04 ± 0.67 [9.20, 10.88] | 0.874 ± 0.076 [0.779, 0.969] | 30.50 ± 9.52 [18.69, 42.32] |
| wpo32 | 5 | 0.0238 ± 0.0004 [0.0233, 0.0243] | 1.83 ± 0.16 [1.63, 2.02] | 0.121 ± 0.074 [0.028, 0.213] | 1.33 ± 0.27 [1.00, 1.67] |

## P: PDEBench 1D Burgers (a42 setup)

| model | n | nRMSE 256 | nRMSE 1024 / 256 | probe ν' (truth 0.1257) | dispersion error |
|---|---|---|---|---|---|
| wpo | 5 | 0.0450 ± 0.0008 [0.0440, 0.0459] | 1.00 ± 0.00 [1.00, 1.00] | 0.1014 ± 0.0179 [0.0792, 0.1235] | 0.246 ± 0.116 [0.103, 0.390] |
| tfno_r | 5 | 0.0254 ± 0.0006 [0.0247, 0.0261] | 1.00 ± 0.00 [1.00, 1.00] | 0.0402 ± 0.0084 [0.0297, 0.0507] | 0.691 ± 0.078 [0.594, 0.787] |
| wpo32 (pre-registered; cuFFT defect) | 5 | 0.7741 ± 0.3382 [0.3542, 1.1940] | 1.00 ± 0.00 [1.00, 1.00] | nan ± nan [nan, nan] | 0.868 ± 0.188 [0.635, 1.102] |
| wpo32 rerun (fixed layer, post hoc) | 5 | 0.0459 ± 0.0016 [0.0439, 0.0479] | 1.00 ± 0.00 [1.00, 1.00] | 0.0912 ± 0.0127 [0.0755, 0.1069] | 0.304 ± 0.107 [0.171, 0.436] |
