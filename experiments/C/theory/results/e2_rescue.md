# E2: truth-free single-slot restarts after the polish

Every polished run (seeds 10-29) gets up to 3 rounds of single-slot moves (residue 0, 300-iteration screen, accept if training loss falls ≥ 1 %, then full polish). Before = after the polish (e2_polish.py). Best of 5: lowest validation loss in groups of 5 consecutive seeds.

| condition | variant | runs | runs moved | train/oracle (median) | test rel. L2 (median) | all (single) | best of 5: all | per pole recovered | s per run |
|---|---|---|---|---|---|---|---|---|---|
| p4_ms16_T20 | mixed_off_ms | 20 | 16 | 0.90 → 0.86 | 5.7e-04 → 5.5e-04 | 65% → 95% | 3/4 → 4/4 | -0.4+3i 100% → 100%, -1 65% → 95%, -0.2 95% → 100% | 279 |
| p4_ms16_T20 | damp8_ms | 20 | 16 | 1.11 → 0.86 | 6.9e-04 → 5.4e-04 | 85% → 90% | 4/4 → 4/4 | -0.4+3i 100% → 100%, -1 90% → 100%, -0.2 90% → 90% | 365 |
| p6_ms16_T80 | mixed_off_ms | 20 | 19 | 1.08 → 0.73 | 4.3e-04 → 2.9e-04 | 40% → 35% | 2/4 → 0/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 80% → 95%, -2 40% → 40% | 380 |
| p6_ms16_T80 | damp8_ms | 20 | 20 | 0.93 → 0.72 | 3.7e-04 → 2.9e-04 | 35% → 80% | 2/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 60% → 90%, -2 45% → 85% | 445 |
| p6_ms16_T80_N4096 | mixed_off_ms | 20 | 19 | 5.03 → 0.92 | 2.8e-04 → 5.2e-05 | 45% → 80% | 4/4 → 3/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 90% → 100%, -2 45% → 80% | 286 |
| p6_ms16_T80_N4096 | damp8_ms | 20 | 20 | 3.50 → 0.93 | 2.0e-04 → 5.2e-05 | 50% → 90% | 3/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 65% → 90%, -2 50% → 100% | 386 |
