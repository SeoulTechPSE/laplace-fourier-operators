# E5 fourteenth pass: rule v5, a model-independent check against the data (written 2026-11-20, before any v5 number)

**Motivation** (thirteenth pass). With 4 shared poles, v4 made its first 2 false trusts in 270 later runs: J poles
10.4 % (`lz_small` seed 4) and 25 % (`lz_r10_small` seed 2) off the reference. In both, the two probes agreed with
each other and with a stored pole. v4 checks that the model is self-consistent, and a constrained model can be
consistently wrong.

**Rule v5** = v4 ∧ |μ_J − μ_low| / |μ_low| < 0.1. μ_low is the equivalent linear pole of the lowest-amplitude fifth of
the training samples: the v3 gate's quantity, computed from the data alone (`results/e5_gate.json`). μ_J is the
Jacobian impulse-response pole (`e5_probe`). So v5 also asks the model's small-signal pole to agree with a pole fitted
directly to the low-amplitude data.

**Design data and a prospective test.**

- (a) The 645 existing runs. v5 was designed after seeing the two v4 false trusts, so this part is post hoc by
  construction. It is reported to show what v5 costs.
- (b) New: `lnots2_stable_plr10`, the shared-pole LNO with 2 shared poles, on all 18 sets, 5 seeds (90 runs). Two
  poles cover the one pair of Duffing and the pendulum, but not the real pole plus pair of forced Lorenz. These
  constrained models should be most prone to being consistently wrong. Probed and judged like the thirteenth pass.

**Predictions.**

1. (a) v5 removes the 25 % false trust (`lz_r10_small` `lnots4` seed 2) and keeps at least 90 % of v4's correct
   trusts (≥ 157 of 174).
2. (b) v5 makes no false trust among the `lnots2` runs.
3. (b) v4 makes at least one false trust among the `lnots2` runs. If it makes none, prediction 2 is recorded as
   uninformative.

    python experiments/C/e5_probe/e5_train.py --variant lnots2_stable_plr10 --data <set>      # each of the 18 sets
    python experiments/C/e5_probe/e5_probe.py <set> --only lnots2_stable_plr10
    python experiments/C/e5_probe/e5_v4.py && python experiments/C/e5_probe/e5_v5.py
