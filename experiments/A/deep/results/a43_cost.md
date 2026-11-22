# (A) Cost per model, measured alone on the GPU

NVIDIA GeForce RTX 4090, torch 2.6.0+cu124. Median of 5 × 20 steps after 10 warm-up steps; training batch 25, inference batch 50.

| setup | model | real params | dtype | train step (ms) | ratio to tfno_r | inference (ms / sample) | peak train memory (MB) |
|---|---|---|---|---|---|---|---|
| F | wpo | 59529 | float64 | 131.2 | 4.66 | 1.667 | 2197 |
| F | wpo_index | 60465 | float64 | 128.1 | 4.55 | 1.664 | 2197 |
| F | tfno_r | 54649 | float32 | 28.2 | 1.00 | 0.461 | 1277 |
| F | lno | 55834 | float32 | 27.2 | 0.97 | 0.354 | 2238 |
| P | wpo | 61130 | float64 | 78.7 | 4.36 | 1.067 | 1570 |
| P | tfno_r | 64745 | float32 | 18.0 | 1.00 | 0.292 | 975 |
| P | wpo_mlp | 63382 | float64 | 98.2 | 5.44 | 1.319 | 1931 |
| P | fno_x | 62257 | float32 | 19.1 | 1.06 | 0.138 | 822 |
