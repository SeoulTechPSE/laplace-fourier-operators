# E2: truth-free single-slot restarts after the polish, partial tie with residual reset (E3)

Every polished run (seeds 10-29) gets up to 3 rounds of single-slot moves (residue 0, 300-iteration screen, partial tie, every candidate also resets R_res to 0 (plus a reset-only candidate), accept if training loss falls ≥ 1 %, then full polish). Before = after the polish (e2_polish.py). Best of 5: lowest validation loss in groups of 5 consecutive seeds.

| condition | variant | runs | runs moved | train/oracle (median) | test rel. L2 (median) | all (single) | best of 5: all | per pole recovered | s per run |
|---|---|---|---|---|---|---|---|---|---|
| p4_ms16_T20_alno_pt03 | damp8_ms | 12 | 7 | 0.75 → 0.75 | 4.7e-04 → 4.7e-04 | 100% → 100% | 2/2 → 2/2 | -0.4+3i 100% → 100%, -1 100% → 100%, -0.2 100% → 100% | 2089 |
| p4_ms16_T20_lfno_pt03 | damp8_ms | 9 | 9 | 0.80 → 0.74 | 5.0e-04 → 4.7e-04 | 89% → 100% | 1/1 → 1/1 | -0.4+3i 100% → 100%, -1 89% → 100%, -0.2 89% → 100% | 2896 |
| p6_ms16_T80_N4096_alno_pt003 | damp8_ms | 20 | 20 | 4.76 → 1.55 | 2.7e-04 → 8.7e-05 | 5% → 10% | 0/4 → 2/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 60% → 75%, -2 5% → 15% | 2490 |
| p6_ms16_T80_N4096_alno_pt03 | damp8_ms | 20 | 20 | 3.12 → 1.25 | 1.8e-04 → 7.3e-05 | 10% → 15% | 2/4 → 3/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 85% → 95%, -2 10% → 15% | 2562 |
| p6_ms16_T80_N4096_lfno_pt003 | damp8_ms | 20 | 19 | 3.06 → 1.66 | 1.7e-04 → 9.3e-05 | 20% → 30% | 0/4 → 3/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 75% → 75%, -2 20% → 35% | 2565 |
| p6_ms16_T80_N4096_lfno_pt03 | damp8_ms | 20 | 20 | 4.77 → 1.50 | 2.7e-04 → 8.7e-05 | 10% → 15% | 2/4 → 2/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 75% → 80%, -2 10% → 15% | 2636 |
