# E5 thirteenth pass: few shared poles against a split dominant mode (written 2026-11-19, before training)

**Motivation** (twelfth pass). Sharing the poles across channel pairs (`lnots_stable_plr10`, 16 shared poles; ρ = 10:
84) damped the stored modes close to the truth: median damping ratio 0.84 against 0.70. But the dominant mode is
still split between two shared poles on either side of the true frequency, so the nearest stored pole stays 10–15 %
from the Jacobian pole. v4 trusted 32 runs, against 37 for `lnotb_stable_plr10`. A split needs at least two spare
poles near one mode. The README's next step was therefore "fewer poles (one per mode)".

**Variant** `lnots4_stable_plr10`. It is `lnots_stable_plr10` with N_FEW = 4 shared poles on every set. That covers
the small-signal modes of every system: one complex pair for Duffing and the pendulum, and at most one real pole plus
one pair for forced Lorenz, with one spare pole. Residues stay per channel pair. Everything else is unchanged: stable
initialisation, pole lr × 10, `common` protocol. 5 seeds × 18 sets, 90 runs. Probed with `e5_probe.py --only`; rule
v4 unchanged.

*Budget.* The layer has fewer parameters than the baselines: 865–869 against 1057–3489 for `lnotb`. This pass asks
about pole placement, not accuracy at equal budget, as in the twelfth pass.

**Predictions.** Baselines are given in brackets: `lnots` (16 shared) / `lnotb`.

1. v4 makes no false trust among these runs and trusts at least 40 (32 / 37).
2. Among runs with a correct J pole on the gated sets (ρ = 10 excluded), at most 15 % have d ≥ 0.1 (31 % / 26 %).
3. On `lz_small` + `lz_wide`, v4 trusts at least 3 of 10 (2 / 0).
4. Fewer poles cost some fit. On at least 12 of 18 sets, the median test error is within 2× of
   `lnotb_stable_plr10`'s (`lnots`: 16 / 18).
5. Reported: the share-weighted damping ratio of the stored modes (`e5_cluster.py`), against 0.84 / 0.70.

    python experiments/C/e5_probe/e5_train.py --variant lnots4_stable_plr10 --data <set>      # each of the 18 sets
    python experiments/C/e5_probe/e5_probe.py <set> --only lnots4_stable_plr10
    python experiments/C/e5_probe/e5_v4.py && python experiments/C/e5_probe/e5_cluster.py lnots4_stable_plr10
