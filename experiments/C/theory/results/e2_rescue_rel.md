# E2: truth-free single-slot restarts after the polish, partial tie released from the tied run (E3)

Every polished run (seeds 10-29) gets up to 3 rounds of single-slot moves (residue 0, 300-iteration screen, partial tie released from the restarted tied run (R_res = 0, then a full polish; before = after that polish), accept if training loss falls ≥ 1 %, then full polish). Before = after the polish (e2_polish.py). Best of 5: lowest validation loss in groups of 5 consecutive seeds.

| condition | variant | runs | runs moved | train/oracle (median) | test rel. L2 (median) | all (single) | best of 5: all | per pole recovered | s per run |
|---|---|---|---|---|---|---|---|---|---|
| p4_ms16_T20_alno_pt03 | damp8_ms | 20 | 0 | 0.73 → 0.73 | 4.7e-04 → 4.7e-04 | 90% → 90% | 4/4 → 4/4 | -0.4+3i 100% → 100%, -1 100% → 100%, -0.2 90% → 90% | 1213 |
| p4_ms16_T20_lfno_pt03 | damp8_ms | 20 | 2 | 0.74 → 0.74 | 4.7e-04 → 4.7e-04 | 90% → 90% | 4/4 → 4/4 | -0.4+3i 100% → 100%, -1 100% → 100%, -0.2 90% → 90% | 1127 |
| p6_ms16_T80_N4096_alno_pt003 | damp8_ms | 20 | 11 | 0.93 → 0.91 | 5.2e-05 → 5.2e-05 | 90% → 90% | 4/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 90% → 90%, -2 100% → 100% | 1181 |
| p6_ms16_T80_N4096_alno_pt03 | damp8_ms | 20 | 15 | 0.88 → 0.85 | 4.9e-05 → 4.9e-05 | 90% → 90% | 4/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 90% → 90%, -2 100% → 100% | 1952 |
| p6_ms16_T80_N4096_lfno_pt003 | damp8_ms | 20 | 5 | 0.93 → 0.92 | 5.2e-05 → 5.2e-05 | 90% → 90% | 4/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 90% → 90%, -2 100% → 100% | 796 |
| p6_ms16_T80_N4096_lfno_pt03 | damp8_ms | 20 | 14 | 0.87 → 0.85 | 5.0e-05 → 4.9e-05 | 90% → 90% | 4/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 90% → 90%, -2 100% → 100% | 2332 |
