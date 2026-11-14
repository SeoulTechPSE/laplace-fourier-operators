# E2: truth-free single-slot restarts after the polish, partial tie released from the tied run, under noise (E3 / E4)

Every polished run (seeds 10-29) gets up to 3 rounds of single-slot moves (residue 0, 300-iteration screen, partial tie released from the noise-aware tied run, then noise-aware restarts (accept if S falls ≥ 13.28 σ̂² in the noise regime), then full polish). Before = after the polish (e2_polish.py). Best of 5: lowest validation loss in groups of 5 consecutive seeds. η̂: noise level estimated from the training outputs (median over runs); noise regime: runs that used the noise-aware test.

| condition | variant | runs | η̂ | noise regime | runs moved | train/oracle (median) | test rel. L2 (median) | all (single) | best of 5: all | per pole recovered | s per run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p6_ms16_T80_N4096_alno_pt03_n1e-2 | damp8_ms | 20 | 0.01 | 20 | 18 | 180.05 → 180.05 | 6.3e-05 → 1.0e-04 | 100% → 100% | 4/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 100% → 100%, -2 100% → 100% | 2016 |
| p6_ms16_T80_N4096_alno_pt03_n1e-1 | damp8_ms | 20 | 0.099 | 20 | 19 | 1791.65 → 1791.58 | 2.7e-04 → 8.8e-04 | 45% → 25% | 3/4 → 0/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 100% → 85%, -2 45% → 30% | 1779 |
| p6_ms16_T80_N4096_lfno_pt03_n1e-2 | damp8_ms | 20 | 0.01 | 20 | 19 | 180.05 → 180.05 | 6.1e-05 → 8.0e-05 | 100% → 100% | 4/4 → 4/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 100% → 100%, -2 100% → 100% | 1984 |
| p6_ms16_T80_N4096_lfno_pt03_n1e-1 | damp8_ms | 20 | 0.099 | 20 | 20 | 1791.65 → 1791.61 | 2.5e-04 → 6.3e-04 | 45% → 25% | 3/4 → 0/4 | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 100% → 90%, -2 45% → 35% | 1430 |
