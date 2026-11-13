# E2: floor certificate without the truth

Groups of 5 consecutive seeds (10-29). Floor estimate = lowest training loss in the group; certified = training loss < 1.5 x estimate. Consensus = effective poles present (within 10%) in ≥ 2/3 of the certified runs (needs ≥ 2 certified); slots below 2% impulse-response weight dropped, slots within 10 % merged. Per true pole: groups whose consensus contains it (within 0.1); false = consensus poles matching no true pole (truth used only for these two scores). Pooled: the same over all 20 runs.

## p4

| stage | condition | variant | single: all | estimate / oracle | certified: all recovered | -0.40+3.00i identified | -1.00 identified | -0.20 identified | false | pooled consensus |
|---|---|---|---|---|---|---|---|---|---|---|
| polish | p4_ms16_T20 | mixed_off_ms | 65% | 0.87 | 13/18 | 4/4 | 3/4 | 4/4 | 3 | -0.40+3.00i, -0.20+0.00i, -0.93+0.01i, -1.06 (18 cert.) |
| polish | p4_ms16_T20 | damp8_ms | 85% | 0.89 | 12/15 | 4/4 | 4/4 | 4/4 | 3 | -0.39+3.00i, -1.06, -0.20 (13 cert.) |
| polish | p4_ms16_T20_alno | damp8_ms | 75% | 0.90 | 15/20 | 4/4 | 3/4 | 4/4 | 0 | -0.41+3.01i, -0.20, -1.03 (20 cert.) |
| polish | p4_ms16_T20_lfno | damp8_ms | 50% | 0.96 | 8/12 | 4/4 | 3/4 | 3/4 | 2 | -0.40+3.00i, -0.97, -0.20 (11 cert.) |
| polish | p4_ms16_T20_n1e-3 | damp8_ms | 80% | 1.97 | 16/18 | 4/4 | 4/4 | 4/4 | 3 | -0.40+2.99i, -1.01, -0.20 (18 cert.) |
| polish | p4_ms16_T20_n1e-2 | damp8_ms | 65% | 17.01 | 13/20 | 4/4 | 3/4 | 4/4 | 2 | -0.40+2.99i, -1.07, -0.20, -0.92 (20 cert.) |
| polish | p4_ms16_T20_n1e-1 | damp8_ms | 45% | 168.92 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.38+2.98i, -0.20, -1.12, -0.97 (20 cert.) |
| polish | p4_ms16_T20_n3e-1 | damp8_ms | 45% | 488.02 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.36+2.99i, -1.03, -0.19 (20 cert.) |
| rescue | p4_ms16_T20 | mixed_off_ms | 95% | 0.86 | 19/20 | 4/4 | 4/4 | 4/4 | 0 | -0.40+3.00i, -0.20+0.00i, -1.02+0.00i (20 cert.) |
| rescue | p4_ms16_T20 | damp8_ms | 90% | 0.85 | 18/20 | 4/4 | 4/4 | 3/4 | 0 | -0.41+3.00i, -1.00, -0.20 (20 cert.) |
| rescue | p4_ms16_T20_alno | damp8_ms | 75% | 0.90 | 15/20 | 4/4 | 3/4 | 4/4 | 0 | -0.41+3.01i, -0.20, -1.03 (20 cert.) |
| rescue | p4_ms16_T20_lfno | damp8_ms | 65% | 0.91 | 11/15 | 4/4 | 3/4 | 3/4 | 1 | -0.40+3.00i, -0.97, -0.20 (15 cert.) |
| rescue | p4_ms16_T20_n1e-3 | damp8_ms | 100% | 1.96 | 20/20 | 4/4 | 4/4 | 4/4 | 0 | -0.41+3.00i, -0.20, -0.99 (20 cert.) |
| rescue | p4_ms16_T20_n1e-2 | damp8_ms | 70% | 17.01 | 14/20 | 4/4 | 4/4 | 4/4 | 2 | -0.40+2.99i, -1.07, -0.20, -0.93 (20 cert.) |
| rescue | p4_ms16_T20_n1e-1 | damp8_ms | 45% | 168.92 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.38+2.98i, -0.20, -1.12, -0.97 (20 cert.) |
| rescue | p4_ms16_T20_n3e-1 | damp8_ms | 45% | 488.02 | 9/20 | 4/4 | 2/4 | 3/4 | 2 | -0.36+2.99i, -1.03, -0.19 (20 cert.) |

## p6

| stage | condition | variant | single: all | estimate / oracle | certified: all recovered | -0.50+1.50i identified | -0.30+4.00i identified | -1.00 identified | -2.00 identified | false | pooled consensus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| polish | p6_ms16_T80 | mixed_off_ms | 40% | 0.75 | 6/12 | 4/4 | 4/4 | 4/4 | 1/4 | 3 | -0.50+1.50i, -0.30+4.00i, -0.96, -2.15 (11 cert.) |
| polish | p6_ms16_T80 | damp8_ms | 35% | 0.76 | 6/15 | 4/4 | 4/4 | 4/4 | 3/4 | 1 | -0.30+4.00i, -0.50+1.49i, -0.96 (14 cert.) |
| polish | p6_ms16_T80_N4096 | mixed_off_ms | 45% | 1.11 | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 | -0.51+1.50i, -0.30+4.00i, -1.02, -2.03, -1.39 (3 cert.) |
| polish | p6_ms16_T80_N4096 | damp8_ms | 50% | 1.68 | 6/8 | 3/4 | 3/4 | 3/4 | 2/4 | 1 | -0.50+1.49i, -0.30+4.00i, -0.97, -1.91, -2.19 (5 cert.) |
| polish | p6_ms16_T80_N4096_alno | damp8_ms | 15% | 1.33 | 2/8 | 3/4 | 3/4 | 0/4 | 0/4 | 2 | -0.30+4.00i, -0.51+1.49i (4 cert.) |
| polish | p6_ms16_T80_N4096_lfno | damp8_ms | 15% | 1.89 | 2/6 | 1/4 | 1/4 | 1/4 | 0/4 | 1 | -0.30+4.00i, -0.48+1.50i, -1.72, -0.97 (2 cert.) |
| polish | p6_ms16_T80_N4096_n1e-3 | damp8_ms | 55% | 18.06 | 11/16 | 4/4 | 4/4 | 4/4 | 2/4 | 0 | -0.30+4.00i, -0.50+1.49i, -0.96, -1.96 (16 cert.) |
| polish | p6_ms16_T80_N4096_n1e-2 | damp8_ms | 50% | 180.07 | 10/20 | 4/4 | 4/4 | 2/4 | 0/4 | 1 | -0.30+4.00i, -0.49+1.49i, -0.96 (20 cert.) |
| polish | p6_ms16_T80_N4096_n1e-1 | damp8_ms | 5% | 1791.69 | 1/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.47+1.48i, -0.30+4.00i, -0.68 (20 cert.) |
| polish | p6_ms16_T80_N4096_n3e-1 | damp8_ms | 0% | 5174.26 | 0/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.46+1.50i, -0.30+4.00i, -0.67, -0.55+1.68i (20 cert.) |
| rescue | p6_ms16_T80 | mixed_off_ms | 35% | 0.69 | 7/20 | 4/4 | 4/4 | 4/4 | 1/4 | 0 | -0.50+1.50i, -0.30+4.00i, -0.98 (20 cert.) |
| rescue | p6_ms16_T80 | damp8_ms | 80% | 0.67 | 16/20 | 4/4 | 4/4 | 4/4 | 3/4 | 2 | -0.50+1.50i, -0.30+4.00i, -0.98, -2.00, -1.08 (20 cert.) |
| rescue | p6_ms16_T80_N4096 | mixed_off_ms | 80% | 0.91 | 16/20 | 4/4 | 4/4 | 4/4 | 4/4 | 1 | -0.50+1.50i, -0.30+4.00i, -1.00, -2.03+0.01i (20 cert.) |
| rescue | p6_ms16_T80_N4096 | damp8_ms | 90% | 0.91 | 18/20 | 4/4 | 4/4 | 4/4 | 4/4 | 0 | -0.30+4.00i, -0.50+1.50i, -1.99, -1.02 (20 cert.) |
| rescue | p6_ms16_T80_N4096_alno | damp8_ms | 15% | 1.33 | 2/8 | 3/4 | 3/4 | 0/4 | 0/4 | 2 | -0.30+4.00i, -0.51+1.49i (4 cert.) |
| rescue | p6_ms16_T80_N4096_lfno | damp8_ms | 15% | 1.17 | 1/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 | -0.50+1.52i, -0.30+4.00i, -0.98, -1.81 (4 cert.) |
| rescue | p6_ms16_T80_N4096_n1e-3 | damp8_ms | 85% | 18.04 | 17/20 | 4/4 | 4/4 | 4/4 | 2/4 | 0 | -0.30+4.00i, -0.50+1.50i, -0.99, -1.96 (20 cert.) |
| rescue | p6_ms16_T80_N4096_n1e-2 | damp8_ms | 70% | 180.06 | 14/20 | 4/4 | 4/4 | 4/4 | 3/4 | 1 | -0.30+4.00i, -0.49+1.49i, -0.97, -2.02 (20 cert.) |
| rescue | p6_ms16_T80_N4096_n1e-1 | damp8_ms | 5% | 1791.69 | 1/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.47+1.48i, -0.30+4.00i, -0.68 (20 cert.) |
| rescue | p6_ms16_T80_N4096_n3e-1 | damp8_ms | 0% | 5174.26 | 0/20 | 4/4 | 4/4 | 0/4 | 0/4 | 5 | -0.46+1.50i, -0.30+4.00i, -0.67, -0.55+1.68i (20 cert.) |

