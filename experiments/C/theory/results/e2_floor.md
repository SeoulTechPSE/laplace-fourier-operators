# E2: floor certificate without the truth

Groups of 5 consecutive seeds (10-29). Floor estimate = lowest training loss in the group; certified = training loss < 1.5 x estimate. Consensus = effective poles present (within 10%) in ≥ 2/3 of the certified runs (needs ≥ 2 certified); slots below 2% impulse-response weight dropped, slots within 10 % merged. Per true pole: groups whose consensus contains it (within 0.1); false = consensus poles matching no true pole (truth used only for these two scores). Pooled: the same over all 20 runs (ratio rule). Rule `noise`: certified = S − S_min ≤ χ²_p(0.99) σ̂² when the floor is at the noise level (see the docstring); stage `rescue_na` = restarts with the noise-aware acceptance (e2_rescue.py --rule noise); `rescue_rr` / `rescue_rel` = partial-tie restarts with residual reset / released from the tied run (e2_rescue.py --start reset / release).

## p4

| stage | rule | condition | variant | single: all | estimate / oracle | certified: all recovered | -0.40+3.00i identified | -1.00 identified | -0.20 identified | false | pooled consensus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| polish | ratio | p4_ms16_T20 | mixed_off_ms | 65% | 0.87 | 13/18 | 4/4 | 3/4 | 4/4 | 3 | -0.40+3.00i, -0.20+0.00i, -0.93+0.01i, -1.06 (18 cert.) |
| polish | ratio | p4_ms16_T20 | damp8_ms | 85% | 0.89 | 12/15 | 4/4 | 4/4 | 4/4 | 3 | -0.39+3.00i, -1.06, -0.20 (13 cert.) |
| polish | ratio | p4_ms16_T20_alno | damp8_ms | 75% | 0.90 | 15/20 | 4/4 | 3/4 | 4/4 | 0 | -0.41+3.01i, -0.20, -1.03 (20 cert.) |
| polish | ratio | p4_ms16_T20_lfno | damp8_ms | 50% | 0.96 | 8/12 | 4/4 | 3/4 | 3/4 | 2 | -0.40+3.00i, -0.97, -0.20 (11 cert.) |
| polish | ratio | p4_ms16_T20_n1e-3 | damp8_ms | 80% | 1.97 | 16/18 | 4/4 | 4/4 | 4/4 | 3 | -0.40+2.99i, -1.01, -0.20 (18 cert.) |
| polish | ratio | p4_ms16_T20_n1e-2 | damp8_ms | 65% | 17.01 | 13/20 | 4/4 | 3/4 | 4/4 | 2 | -0.40+2.99i, -1.07, -0.20, -0.92 (20 cert.) |
| polish | ratio | p4_ms16_T20_n1e-1 | damp8_ms | 45% | 168.92 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.38+2.98i, -0.20, -1.12, -0.97 (20 cert.) |
| polish | ratio | p4_ms16_T20_n3e-1 | damp8_ms | 45% | 488.02 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.36+2.99i, -1.03, -0.19 (20 cert.) |
| polish | ratio | p4_ms16_T20_alno_pt03 | damp8_ms | 85% | 0.74 | 17/20 | 4/4 | 4/4 | 4/4 | 1 | -0.41+3.00i, -0.95, -0.20 (20 cert.) |
| polish | ratio | p4_ms16_T20_lfno_pt03 | damp8_ms | 80% | 0.78 | 15/17 | 4/4 | 4/4 | 4/4 | 0 | -0.42+3.01i, -0.20, -0.95, -1.05 (17 cert.) |
| polish | ratio | p4_ms16_T20_foh | damp8_ms | 70% | 28.90 | 4/5 | 1/4 | 1/4 | 0/4 | 0 | – (1 cert.) |
| polish | ratio | p4_e1_T20 | damp8_ms | 45% | 1.04 | 8/12 | 4/4 | 2/4 | 4/4 | 6 | -0.40+3.01i, -0.20, -0.98 (10 cert.) |
| polish | ratio | p4_e1_T20_foh | damp8_ms | 55% | 7.83 | 4/6 | 1/4 | 1/4 | 0/4 | 2 | – (1 cert.) |
| polish | ratio | p4_e1_T20_n1e-1 | damp8_ms | 40% | 99.16 | 8/20 | 4/4 | 0/4 | 2/4 | 0 | -0.41+3.00i, -0.20 (20 cert.) |
| polish | ratio | p4_e1_T20_n3e-1 | damp8_ms | 30% | 286.31 | 6/20 | 4/4 | 0/4 | 1/4 | 3 | -0.41+2.95i (20 cert.) |
| polish | ratio | p4_ms16_T20_n1e-2_u1e-2 | damp8_ms | 65% | 17.36 | 13/20 | 4/4 | 4/4 | 3/4 | 5 | -0.38+2.99i, -0.20, -1.07, -0.96 (20 cert.) |
| polish | ratio | p4_ms16_T20_n1e-2_u1e-1 | damp8_ms | 75% | 38.09 | 15/20 | 4/4 | 2/4 | 4/4 | 3 | -0.39+3.00i, -1.03, -0.20 (20 cert.) |
| rescue | ratio | p4_ms16_T20 | mixed_off_ms | 95% | 0.86 | 19/20 | 4/4 | 4/4 | 4/4 | 0 | -0.40+3.00i, -0.20+0.00i, -1.02+0.00i (20 cert.) |
| rescue | ratio | p4_ms16_T20 | damp8_ms | 90% | 0.85 | 18/20 | 4/4 | 4/4 | 3/4 | 0 | -0.41+3.00i, -1.00, -0.20 (20 cert.) |
| rescue | ratio | p4_ms16_T20_alno | damp8_ms | 75% | 0.90 | 15/20 | 4/4 | 3/4 | 4/4 | 0 | -0.41+3.01i, -0.20, -1.03 (20 cert.) |
| rescue | ratio | p4_ms16_T20_lfno | damp8_ms | 65% | 0.91 | 11/15 | 4/4 | 3/4 | 3/4 | 1 | -0.40+3.00i, -0.97, -0.20 (15 cert.) |
| rescue | ratio | p4_ms16_T20_n1e-3 | damp8_ms | 100% | 1.96 | 20/20 | 4/4 | 4/4 | 4/4 | 0 | -0.41+3.00i, -0.20, -0.99 (20 cert.) |
| rescue | ratio | p4_ms16_T20_n1e-2 | damp8_ms | 70% | 17.01 | 14/20 | 4/4 | 4/4 | 4/4 | 2 | -0.40+2.99i, -1.07, -0.20, -0.93 (20 cert.) |
| rescue | ratio | p4_ms16_T20_n1e-1 | damp8_ms | 45% | 168.92 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.38+2.98i, -0.20, -1.12, -0.97 (20 cert.) |
| rescue | ratio | p4_ms16_T20_n3e-1 | damp8_ms | 45% | 488.02 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.36+2.99i, -1.03, -0.19 (20 cert.) |
| rescue | ratio | p4_ms16_T20_alno_pt03 | damp8_ms | 95% | 0.74 | 19/20 | 4/4 | 4/4 | 4/4 | 0 | -0.41+3.00i, -0.20, -0.97 (20 cert.) |
| rescue | ratio | p4_ms16_T20_lfno_pt03 | damp8_ms | 85% | 0.77 | 17/20 | 4/4 | 4/4 | 4/4 | 0 | -0.42+3.01i, -0.20, -0.99 (20 cert.) |
| rescue | ratio | p4_ms16_T20_foh | damp8_ms | 95% | 0.61 | 6/6 | 1/4 | 1/4 | 1/4 | 0 | -0.20, -0.40+3.00i, -1.00 (4 cert.) |
| rescue | ratio | p4_e1_T20 | damp8_ms | 90% | 1.02 | 18/19 | 4/4 | 4/4 | 4/4 | 0 | -0.40+3.00i, -0.20, -0.99 (19 cert.) |
| rescue | ratio | p4_e1_T20_foh | damp8_ms | 85% | 3.52 | 4/4 | 0/4 | 0/4 | 0/4 | 0 | -0.40+3.00i, -0.20, -0.98 (2 cert.) |
| rescue | ratio | p4_e1_T20_n1e-1 | damp8_ms | 40% | 99.16 | 8/20 | 4/4 | 0/4 | 2/4 | 0 | -0.41+3.00i, -0.20 (20 cert.) |
| rescue | ratio | p4_e1_T20_n3e-1 | damp8_ms | 30% | 286.31 | 6/20 | 4/4 | 0/4 | 1/4 | 3 | -0.41+2.95i (20 cert.) |
| rescue | ratio | p4_ms16_T20_n1e-2_u1e-2 | damp8_ms | 65% | 17.36 | 13/20 | 4/4 | 4/4 | 3/4 | 5 | -0.38+2.99i, -0.20, -1.07, -0.96 (20 cert.) |
| rescue | ratio | p4_ms16_T20_n1e-2_u1e-1 | damp8_ms | 75% | 38.09 | 15/20 | 4/4 | 2/4 | 4/4 | 3 | -0.39+3.00i, -1.03, -0.20 (20 cert.) |
| rescue_rr | ratio | p4_ms16_T20_alno_pt03 | damp8_ms | 100% | 0.73 | 10/10 | 2/2 | 2/2 | 2/2 | 0 | -0.41+3.00i, -0.20, -0.99 (12 cert.) |
| rescue_rr | ratio | p4_ms16_T20_lfno_pt03 | damp8_ms | 100% | 0.74 | 5/5 | 1/1 | 1/1 | 1/1 | 0 | -0.43+3.01i, -0.20, -1.00 (9 cert.) |
| rescue_rel | ratio | p4_ms16_T20_alno_pt03 | damp8_ms | 90% | 0.73 | 18/20 | 4/4 | 4/4 | 3/4 | 0 | -0.41+3.00i, -1.00, -0.20 (20 cert.) |
| rescue_rel | ratio | p4_ms16_T20_lfno_pt03 | damp8_ms | 90% | 0.74 | 18/20 | 4/4 | 4/4 | 3/4 | 0 | -0.41+3.00i, -1.00, -0.20 (20 cert.) |
| rescue | noise | p4_ms16_T20_n1e-3 | damp8_ms | 100% | 1.96 | 4/4 | 0/4 | 0/4 | 0/4 | 0 | -0.41+3.00i, -0.20, -0.99 (20 cert.) |
| rescue | noise | p4_ms16_T20_n1e-2 | damp8_ms | 70% | 17.01 | 9/10 | 3/4 | 3/4 | 3/4 | 0 | -0.40+2.99i, -1.07, -0.20, -0.93 (20 cert.) |
| rescue | noise | p4_ms16_T20_n1e-1 | damp8_ms | 45% | 168.92 | 6/10 | 3/4 | 3/4 | 3/4 | 6 | -0.38+2.98i, -0.20, -1.12, -0.97 (20 cert.) |
| rescue | noise | p4_ms16_T20_n3e-1 | damp8_ms | 45% | 488.02 | 8/14 | 4/4 | 4/4 | 3/4 | 7 | -0.36+2.99i, -1.03, -0.19 (20 cert.) |
| rescue | noise | p4_e1_T20_n1e-1 | damp8_ms | 40% | 99.16 | 6/10 | 4/4 | 2/4 | 3/4 | 2 | -0.41+3.00i, -0.20 (20 cert.) |
| rescue | noise | p4_e1_T20_n3e-1 | damp8_ms | 30% | 286.31 | 4/8 | 2/4 | 1/4 | 1/4 | 3 | -0.41+2.95i (20 cert.) |
| rescue | noise | p4_ms16_T20_n1e-2_u1e-2 | damp8_ms | 65% | 17.36 | 5/5 | 1/4 | 1/4 | 1/4 | 0 | -0.38+2.99i, -0.20, -1.07, -0.96 (20 cert.) |
| rescue | noise | p4_ms16_T20_n1e-2_u1e-1 | damp8_ms | 75% | 38.09 | 15/20 | 4/4 | 2/4 | 4/4 | 3 | -0.39+3.00i, -1.03, -0.20 (20 cert.) |
| rescue_na | noise | p4_ms16_T20_n1e-3 | damp8_ms | 100% | 1.96 | 5/5 | 1/4 | 1/4 | 1/4 | 0 | -0.40+3.00i, -0.20, -1.00 (20 cert.) |
| rescue_na | noise | p4_ms16_T20_n1e-2 | damp8_ms | 100% | 17.01 | 20/20 | 4/4 | 4/4 | 4/4 | 0 | -0.41+3.00i, -0.20, -1.01 (20 cert.) |
| rescue_na | noise | p4_ms16_T20_n1e-1 | damp8_ms | 65% | 168.91 | 13/20 | 4/4 | 2/4 | 4/4 | 1 | -0.40+3.00i, -0.21, -1.06 (20 cert.) |
| rescue_na | noise | p4_ms16_T20_n3e-1 | damp8_ms | 55% | 488.01 | 11/20 | 4/4 | 3/4 | 2/4 | 2 | -0.40+3.00i, -0.99, -0.20 (20 cert.) |
| rescue_na | noise | p4_e1_T20_n1e-1 | damp8_ms | 55% | 99.15 | 11/20 | 4/4 | 1/4 | 3/4 | 0 | -0.40+3.00i, -0.20 (20 cert.) |
| rescue_na | noise | p4_e1_T20_n3e-1 | damp8_ms | 55% | 286.30 | 11/20 | 4/4 | 2/4 | 2/4 | 2 | -0.41+3.00i, -0.20 (20 cert.) |
| rescue_na | noise | p4_ms16_T20_n1e-2_u1e-2 | damp8_ms | 95% | 17.36 | 19/20 | 4/4 | 4/4 | 4/4 | 0 | -0.40+3.00i, -0.20, -0.97 (20 cert.) |
| rescue_na | noise | p4_ms16_T20_n1e-2_u1e-1 | damp8_ms | 75% | 38.09 | 15/20 | 4/4 | 2/4 | 4/4 | 3 | -0.39+3.00i, -1.03, -0.20 (20 cert.) |

## p6

| stage | rule | condition | variant | single: all | estimate / oracle | certified: all recovered | -0.50+1.50i identified | -0.30+4.00i identified | -1.00 identified | -2.00 identified | false | pooled consensus |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| polish | ratio | p6_ms16_T80 | mixed_off_ms | 40% | 0.75 | 6/12 | 4/4 | 4/4 | 4/4 | 1/4 | 3 | -0.50+1.50i, -0.30+4.00i, -0.96, -2.15 (11 cert.) |
| polish | ratio | p6_ms16_T80 | damp8_ms | 35% | 0.76 | 6/15 | 4/4 | 4/4 | 4/4 | 3/4 | 1 | -0.30+4.00i, -0.50+1.49i, -0.96 (14 cert.) |
| polish | ratio | p6_ms16_T80_N4096 | mixed_off_ms | 45% | 1.11 | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 | -0.51+1.50i, -0.30+4.00i, -1.02, -2.03, -1.39 (3 cert.) |
| polish | ratio | p6_ms16_T80_N4096 | damp8_ms | 50% | 1.68 | 6/8 | 3/4 | 3/4 | 3/4 | 2/4 | 1 | -0.50+1.49i, -0.30+4.00i, -0.97, -1.91, -2.19 (5 cert.) |
| polish | ratio | p6_ms16_T80_N4096_alno | damp8_ms | 15% | 1.33 | 2/8 | 3/4 | 3/4 | 0/4 | 0/4 | 2 | -0.30+4.00i, -0.51+1.49i (4 cert.) |
| polish | ratio | p6_ms16_T80_N4096_lfno | damp8_ms | 15% | 1.89 | 2/6 | 1/4 | 1/4 | 1/4 | 0/4 | 1 | -0.30+4.00i, -0.48+1.50i, -1.72, -0.97 (2 cert.) |
| polish | ratio | p6_ms16_T80_N4096_n1e-3 | damp8_ms | 55% | 18.06 | 11/16 | 4/4 | 4/4 | 4/4 | 2/4 | 0 | -0.30+4.00i, -0.50+1.49i, -0.96, -1.96 (16 cert.) |
| polish | ratio | p6_ms16_T80_N4096_n1e-2 | damp8_ms | 50% | 180.07 | 10/20 | 4/4 | 4/4 | 2/4 | 0/4 | 1 | -0.30+4.00i, -0.49+1.49i, -0.96 (20 cert.) |
| polish | ratio | p6_ms16_T80_N4096_n1e-1 | damp8_ms | 5% | 1791.69 | 1/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.47+1.48i, -0.30+4.00i, -0.68 (20 cert.) |
| polish | ratio | p6_ms16_T80_N4096_n3e-1 | damp8_ms | 0% | 5174.26 | 0/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.46+1.50i, -0.30+4.00i, -0.67, -0.55+1.68i (20 cert.) |
| polish | ratio | p6_ms16_T80_N4096_alno_pt003 | damp8_ms | 5% | 2.01 | 1/8 | 3/4 | 3/4 | 3/4 | 0/4 | 2 | -0.30+4.00i, -0.50+1.50i, -0.97, -2.54 (6 cert.) |
| polish | ratio | p6_ms16_T80_N4096_alno_pt03 | damp8_ms | 10% | 2.12 | 2/10 | 3/4 | 3/4 | 3/4 | 0/4 | 2 | – (1 cert.) |
| polish | ratio | p6_ms16_T80_N4096_lfno_pt003 | damp8_ms | 20% | 2.12 | 2/11 | 3/4 | 3/4 | 3/4 | 0/4 | 1 | -0.29+3.99i, -0.51+1.49i, -0.93 (8 cert.) |
| polish | ratio | p6_ms16_T80_N4096_lfno_pt03 | damp8_ms | 10% | 2.79 | 2/11 | 3/4 | 3/4 | 3/4 | 0/4 | 2 | – (1 cert.) |
| polish | ratio | p6_ms16_T80_foh | damp8_ms | 40% | 271.09 | 2/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 | -0.48+1.50i, -0.30+4.00i, -0.99, -1.89 (2 cert.) |
| polish | ratio | p6_ms16_T80_N4096_foh | damp8_ms | 30% | 233.95 | 3/5 | 1/4 | 1/4 | 1/4 | 1/4 | 0 | -0.50+1.50i, -0.30+4.00i, -1.10, -1.87 (2 cert.) |
| polish | ratio | p6_ms16_T80_N4096_n1e-2_foh | damp8_ms | 40% | 34037.40 | 8/20 | 4/4 | 4/4 | 2/4 | 0/4 | 1 | -0.30+4.00i, -0.49+1.48i (20 cert.) |
| polish | ratio | p6_ms16_T80_N4096_n1e-1_foh | damp8_ms | 5% | 338670.79 | 1/20 | 4/4 | 4/4 | 0/4 | 0/4 | 4 | -0.48+1.48i, -0.30+4.00i, -0.65 (20 cert.) |
| polish | ratio | p6_e1_T20 | damp8_ms | 55% | 1.00 | 11/19 | 4/4 | 4/4 | 3/4 | 2/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.05, -1.89 (19 cert.) |
| polish | ratio | p6_e1_T20_foh | damp8_ms | 55% | 83.45 | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 | – (1 cert.) |
| polish | ratio | p6_e1_T20_n1e-2 | damp8_ms | 10% | 14.83 | 2/20 | 4/4 | 4/4 | 2/4 | 0/4 | 3 | -0.50+1.50i, -0.30+4.00i, -1.07 (20 cert.) |
| polish | ratio | p6_e1_T20_n3e-2 | damp8_ms | 5% | 44.35 | 1/20 | 4/4 | 4/4 | 3/4 | 0/4 | 4 | -0.30+4.00i, -0.48+1.50i, -1.04, -0.92, -0.82 (20 cert.) |
| polish | ratio | p6_e1_T20_n1e-1 | damp8_ms | 0% | 147.12 | 0/20 | 4/4 | 4/4 | 1/4 | 0/4 | 5 | -0.30+4.00i, -0.47+1.49i, -0.76 (20 cert.) |
| polish | ratio | p6_e1_T20_n1e-2_u1e-2 | damp8_ms | 20% | 15.08 | 4/20 | 4/4 | 4/4 | 3/4 | 1/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.05 (20 cert.) |
| polish | ratio | p6_e1_T20_n1e-2_u3e-2 | damp8_ms | 10% | 16.87 | 2/20 | 4/4 | 4/4 | 3/4 | 0/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.01, -1.59 (20 cert.) |
| rescue | ratio | p6_ms16_T80 | mixed_off_ms | 35% | 0.69 | 7/20 | 4/4 | 4/4 | 4/4 | 1/4 | 0 | -0.50+1.50i, -0.30+4.00i, -0.98 (20 cert.) |
| rescue | ratio | p6_ms16_T80 | damp8_ms | 80% | 0.67 | 16/20 | 4/4 | 4/4 | 4/4 | 3/4 | 2 | -0.50+1.50i, -0.30+4.00i, -0.98, -2.00, -1.08 (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096 | mixed_off_ms | 80% | 0.91 | 16/20 | 4/4 | 4/4 | 4/4 | 4/4 | 1 | -0.50+1.50i, -0.30+4.00i, -1.00, -2.03+0.01i (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096 | damp8_ms | 90% | 0.91 | 18/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.30+4.00i, -0.50+1.50i, -1.99, -1.02 (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_alno | damp8_ms | 15% | 1.33 | 2/8 | 3/4 | 3/4 | 0/4 | 0/4 | 2 | -0.30+4.00i, -0.51+1.49i (4 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_lfno | damp8_ms | 15% | 1.17 | 1/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 | -0.50+1.52i, -0.30+4.00i, -0.98, -1.81 (4 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_n1e-3 | damp8_ms | 85% | 18.04 | 17/20 | 4/4 | 4/4 | 4/4 | 2/4 | 0 | -0.30+4.00i, -0.50+1.50i, -0.99, -1.96 (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_n1e-2 | damp8_ms | 70% | 180.06 | 14/20 | 4/4 | 4/4 | 4/4 | 3/4 | 1 | -0.30+4.00i, -0.49+1.49i, -0.97, -2.02 (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_n1e-1 | damp8_ms | 5% | 1791.69 | 1/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.47+1.48i, -0.30+4.00i, -0.68 (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_n3e-1 | damp8_ms | 0% | 5174.26 | 0/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.46+1.50i, -0.30+4.00i, -0.67, -0.55+1.68i (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_alno_pt003 | damp8_ms | 10% | 1.97 | 1/11 | 3/4 | 3/4 | 3/4 | 0/4 | 1 | -0.30+4.00i, -0.50+1.49i, -0.96 (8 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_alno_pt03 | damp8_ms | 30% | 1.08 | 6/15 | 4/4 | 4/4 | 4/4 | 0/4 | 2 | -0.30+4.00i, -0.50+1.50i, -1.03, -1.75, -1.55 (14 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_lfno_pt003 | damp8_ms | 20% | 1.95 | 1/10 | 3/4 | 3/4 | 3/4 | 0/4 | 2 | -0.29+3.99i, -0.51+1.49i, -0.93, -2.54 (9 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_lfno_pt03 | damp8_ms | 15% | 1.67 | 3/7 | 3/4 | 3/4 | 3/4 | 0/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.01, -1.81 (6 cert.) |
| rescue | ratio | p6_ms16_T80_foh | damp8_ms | 100% | 6.09 | 5/5 | 1/4 | 1/4 | 1/4 | 1/4 | 0 | -0.50+1.50i, -0.30+4.00i, -1.00, -2.00 (2 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_foh | damp8_ms | 100% | 12.92 | 7/7 | 3/4 | 3/4 | 3/4 | 3/4 | 0 | – (1 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_n1e-2_foh | damp8_ms | 50% | 34034.89 | 10/20 | 4/4 | 4/4 | 3/4 | 2/4 | 1 | -0.30+4.00i, -0.49+1.48i, -1.04 (20 cert.) |
| rescue | ratio | p6_ms16_T80_N4096_n1e-1_foh | damp8_ms | 5% | 338670.79 | 1/20 | 4/4 | 4/4 | 0/4 | 0/4 | 4 | -0.48+1.48i, -0.30+4.00i, -0.65 (20 cert.) |
| rescue | ratio | p6_e1_T20 | damp8_ms | 65% | 1.00 | 13/20 | 4/4 | 4/4 | 4/4 | 2/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.01, -1.88 (20 cert.) |
| rescue | ratio | p6_e1_T20_foh | damp8_ms | 80% | 6.18 | 5/5 | 1/4 | 1/4 | 1/4 | 1/4 | 0 | – (1 cert.) |
| rescue | ratio | p6_e1_T20_n1e-2 | damp8_ms | 10% | 14.83 | 2/20 | 4/4 | 4/4 | 2/4 | 0/4 | 3 | -0.50+1.50i, -0.30+4.00i, -1.07 (20 cert.) |
| rescue | ratio | p6_e1_T20_n3e-2 | damp8_ms | 5% | 44.35 | 1/20 | 4/4 | 4/4 | 3/4 | 0/4 | 4 | -0.30+4.00i, -0.48+1.50i, -1.04, -0.92, -0.82 (20 cert.) |
| rescue | ratio | p6_e1_T20_n1e-1 | damp8_ms | 0% | 147.12 | 0/20 | 4/4 | 4/4 | 1/4 | 0/4 | 5 | -0.30+4.00i, -0.47+1.49i, -0.76 (20 cert.) |
| rescue | ratio | p6_e1_T20_n1e-2_u1e-2 | damp8_ms | 20% | 15.08 | 4/20 | 4/4 | 4/4 | 3/4 | 1/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.05 (20 cert.) |
| rescue | ratio | p6_e1_T20_n1e-2_u3e-2 | damp8_ms | 10% | 16.87 | 2/20 | 4/4 | 4/4 | 3/4 | 0/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.01, -1.59 (20 cert.) |
| rescue_rr | ratio | p6_ms16_T80_N4096_alno_pt003 | damp8_ms | 10% | 1.22 | 2/13 | 3/4 | 3/4 | 2/4 | 0/4 | 4 | -0.50+1.50i, -0.30+4.00i, -1.01, -1.92, -1.22 (2 cert.) |
| rescue_rr | ratio | p6_ms16_T80_N4096_alno_pt03 | damp8_ms | 15% | 0.91 | 3/12 | 4/4 | 4/4 | 4/4 | 2/4 | 2 | -0.30+4.00i, -0.50+1.50i, -1.02, -1.78 (11 cert.) |
| rescue_rr | ratio | p6_ms16_T80_N4096_lfno_pt003 | damp8_ms | 30% | 1.10 | 6/9 | 3/4 | 3/4 | 3/4 | 2/4 | 0 | -0.50+1.50i, -0.30+4.00i, -0.99, -1.99 (9 cert.) |
| rescue_rr | ratio | p6_ms16_T80_N4096_lfno_pt03 | damp8_ms | 15% | 1.02 | 3/12 | 4/4 | 4/4 | 4/4 | 0/4 | 2 | -0.50+1.49i, -0.30+4.00i, -1.00, -1.72 (10 cert.) |
| rescue_rel | ratio | p6_ms16_T80_N4096_alno_pt003 | damp8_ms | 90% | 0.89 | 18/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.30+4.00i, -0.50+1.50i, -1.99, -1.02 (20 cert.) |
| rescue_rel | ratio | p6_ms16_T80_N4096_alno_pt03 | damp8_ms | 90% | 0.83 | 18/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.30+4.00i, -0.50+1.50i, -1.99, -1.02 (20 cert.) |
| rescue_rel | ratio | p6_ms16_T80_N4096_lfno_pt003 | damp8_ms | 90% | 0.91 | 18/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.30+4.00i, -0.50+1.50i, -1.99, -1.02 (20 cert.) |
| rescue_rel | ratio | p6_ms16_T80_N4096_lfno_pt03 | damp8_ms | 90% | 0.83 | 18/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.30+4.00i, -0.50+1.50i, -1.99, -1.02 (20 cert.) |
| rescue | noise | p6_ms16_T80_N4096_n1e-3 | damp8_ms | 85% | 18.04 | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 | -0.30+4.00i, -0.50+1.50i, -0.99, -1.96 (20 cert.) |
| rescue | noise | p6_ms16_T80_N4096_n1e-2 | damp8_ms | 70% | 180.06 | 6/6 | 2/4 | 2/4 | 2/4 | 2/4 | 0 | -0.30+4.00i, -0.49+1.49i, -0.97, -2.02 (20 cert.) |
| rescue | noise | p6_ms16_T80_N4096_n1e-1 | damp8_ms | 5% | 1791.69 | 0/5 | 1/4 | 1/4 | 1/4 | 0/4 | 1 | -0.47+1.48i, -0.30+4.00i, -0.68 (20 cert.) |
| rescue | noise | p6_ms16_T80_N4096_n3e-1 | damp8_ms | 0% | 5174.26 | 0/5 | 1/4 | 1/4 | 0/4 | 0/4 | 0 | -0.46+1.50i, -0.30+4.00i, -0.67, -0.55+1.68i (20 cert.) |
| rescue | noise | p6_ms16_T80_N4096_n1e-2_foh | damp8_ms | 50% | 34034.89 | 2/5 | 1/4 | 1/4 | 1/4 | 1/4 | 0 | -0.30+4.00i, -0.49+1.48i, -1.04 (20 cert.) |
| rescue | noise | p6_ms16_T80_N4096_n1e-1_foh | damp8_ms | 5% | 338670.79 | 0/6 | 2/4 | 2/4 | 1/4 | 0/4 | 3 | -0.48+1.48i, -0.30+4.00i, -0.65 (20 cert.) |
| rescue | noise | p6_e1_T20_n1e-2 | damp8_ms | 10% | 14.83 | 2/13 | 4/4 | 4/4 | 3/4 | 0/4 | 5 | -0.50+1.50i, -0.30+4.00i, -1.07 (20 cert.) |
| rescue | noise | p6_e1_T20_n3e-2 | damp8_ms | 5% | 44.35 | 1/8 | 3/4 | 3/4 | 2/4 | 0/4 | 3 | -0.30+4.00i, -0.48+1.50i, -1.04, -0.92, -0.82 (20 cert.) |
| rescue | noise | p6_e1_T20_n1e-1 | damp8_ms | 0% | 147.12 | 0/16 | 4/4 | 4/4 | 1/4 | 0/4 | 8 | -0.30+4.00i, -0.47+1.49i, -0.76 (20 cert.) |
| rescue | noise | p6_e1_T20_n1e-2_u1e-2 | damp8_ms | 20% | 15.08 | 3/12 | 4/4 | 4/4 | 4/4 | 0/4 | 5 | -0.50+1.50i, -0.30+4.00i, -1.05 (20 cert.) |
| rescue | noise | p6_e1_T20_n1e-2_u3e-2 | damp8_ms | 10% | 16.87 | 1/12 | 4/4 | 4/4 | 4/4 | 0/4 | 7 | -0.50+1.50i, -0.30+4.00i, -1.01, -1.59 (20 cert.) |
| rescue_na | noise | p6_ms16_T80_N4096_n1e-3 | damp8_ms | 85% | 18.04 | 6/6 | 2/4 | 2/4 | 2/4 | 2/4 | 0 | -0.50+1.50i, -0.30+4.00i, -1.00, -2.01 (20 cert.) |
| rescue_na | noise | p6_ms16_T80_N4096_n1e-2 | damp8_ms | 100% | 180.05 | 19/19 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.50+1.50i, -0.30+4.00i, -1.00, -1.97 (20 cert.) |
| rescue_na | noise | p6_ms16_T80_N4096_n1e-1 | damp8_ms | 45% | 1791.65 | 9/20 | 4/4 | 4/4 | 4/4 | 0/4 | 1 | -0.30+4.00i, -0.50+1.50i, -1.00 (20 cert.) |
| rescue_na | noise | p6_ms16_T80_N4096_n3e-1 | damp8_ms | 15% | 5174.07 | 3/20 | 4/4 | 4/4 | 0/4 | 0/4 | 3 | -0.47+1.51i, -0.30+4.00i (20 cert.) |
| rescue_na | noise | p6_ms16_T80_N4096_n1e-2_foh | damp8_ms | 95% | 34033.39 | 19/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.30+4.00i, -0.50+1.50i, -1.00, -1.99 (20 cert.) |
| rescue_na | noise | p6_ms16_T80_N4096_n1e-1_foh | damp8_ms | 20% | 338662.02 | 4/20 | 4/4 | 4/4 | 3/4 | 0/4 | 1 | -0.49+1.50i, -0.30+4.00i, -0.98 (20 cert.) |
| rescue_na | noise | p6_e1_T20_n1e-2 | damp8_ms | 35% | 14.83 | 7/19 | 4/4 | 4/4 | 4/4 | 1/4 | 6 | -0.30+4.00i, -0.50+1.50i, -1.02, -1.77 (20 cert.) |
| rescue_na | noise | p6_e1_T20_n3e-2 | damp8_ms | 60% | 44.35 | 12/20 | 4/4 | 4/4 | 3/4 | 2/4 | 1 | -0.50+1.50i, -0.30+4.00i, -1.01, -1.83 (20 cert.) |
| rescue_na | noise | p6_e1_T20_n1e-1 | damp8_ms | 50% | 147.11 | 10/20 | 4/4 | 4/4 | 3/4 | 1/4 | 3 | -0.30+4.00i, -0.50+1.50i, -0.96 (20 cert.) |
| rescue_na | noise | p6_e1_T20_n1e-2_u1e-2 | damp8_ms | 40% | 15.08 | 8/20 | 4/4 | 4/4 | 4/4 | 1/4 | 5 | -0.50+1.50i, -0.30+4.00i, -1.03, -1.70 (20 cert.) |
| rescue_na | noise | p6_e1_T20_n1e-2_u3e-2 | damp8_ms | 50% | 16.87 | 10/20 | 4/4 | 4/4 | 2/4 | 3/4 | 2 | -0.50+1.50i, -0.30+4.00i, -1.04, -1.88 (20 cert.) |
| rescue_na_rel | noise | p6_ms16_T80_N4096_alno_pt03_n1e-2 | damp8_ms | 100% | 180.05 | 20/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.50+1.50i, -0.30+4.00i, -0.99, -1.99 (20 cert.) |
| rescue_na_rel | noise | p6_ms16_T80_N4096_alno_pt03_n1e-1 | damp8_ms | 25% | 1791.57 | 5/20 | 4/4 | 4/4 | 3/4 | 0/4 | 3 | -0.50+1.50i, -0.30+4.00i, -0.94 (20 cert.) |
| rescue_na_rel | noise | p6_ms16_T80_N4096_lfno_pt03_n1e-2 | damp8_ms | 100% | 180.05 | 20/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.50+1.50i, -0.30+4.00i, -1.98, -1.00 (20 cert.) |
| rescue_na_rel | noise | p6_ms16_T80_N4096_lfno_pt03_n1e-1 | damp8_ms | 25% | 1791.61 | 5/20 | 4/4 | 4/4 | 4/4 | 0/4 | 1 | -0.50+1.51i, -0.30+4.00i, -0.97 (20 cert.) |

