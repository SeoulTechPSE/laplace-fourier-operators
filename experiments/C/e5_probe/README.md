# (C) E5 / RQ4: linear-probe poles of a nonlinear operator

Plan: `docs/proposal_C_identifiability.qmd`, Section 5 ("Diagnostic for nonlinear models") and
experiment set E5.

**Question.** After training, how trustworthy is the pole interpretation of a nonlinear model?
The plan: probe the trained operator with small perturbations, extract poles from the measured
response, and compare them with the stored layer poles and with the poles of the linearisation.

## Setup

| item | choice |
|---|---|
| **system** | Damped Duffing from the LNO data (`1D_Duffing_c05`): x″ + 0.5 x′ + x + k₃x³ = f; T = 20.48, dt = 0.01, n = 2048 |
| **training data** | the original split: 200 inputs A sin 5t with A = 0.05 … 10 (one waveform) |
| **linearised poles (reference)** | estimated from the data alone: a first-order-hold fit of one complex pair to the 4 smallest-amplitude samples (A ≤ 0.2) gives **−0.2500 ± 0.9684i**; theory −c/2 ± i√(k − c²/4) = −0.25 ± 0.9682i |
| **nonlinearity** | the response shape changes by 2 % at A = 1, 15 % at A = 3 and 82 % at A = 10, relative to A = 0.1 |
| **model** | the original 1-layer LNO: lift to 4 channels, 4 × 4 × 16 = 256 poles, sin activation, 1,821 parameters; `experiments/foundation/reproduce_lno.py --case duffing_c05 --tag e5`, 5 seeds; test rel. L2 0.13 / 0.13 / 0.13 / 0.22 / 0.37 |

**Probes.**

1. **Layer poles.** All 256 stored μ: the one closest to the linearised pole, and how many lie
   within 10 %.
2. **Causal LTI probe.** Take the odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs,
   at ε = 0.01 … 3. Fit one complex pair, then a pair plus a real pole, by FOH VarPro.
3. **Jacobian probe.** J = ∂G/∂x at x = 0 (2048 × 2048, by autograd). Measure:
   - the share of ‖J‖ above the diagonal (non-causal);
   - the residual of a shift-invariant (Toeplitz) fit of its causal part;
   - the poles of its average impulse response (mean subdiagonals, order-2 matrix pencil).

```
python experiments/foundation/reproduce_lno.py --case duffing_c05 --model lno --seeds 0 1 2 3 4 --tag e5   # ~1 min each
python experiments/C/e5_probe/e5_train.py                                                             # 3 variants x 5 seeds, ~2 min each
python experiments/C/e5_probe/e5_probe.py                                                             # results/e5_probe.md
python experiments/C/e5_probe/e5_oppoint.py                                                           # results/e5_oppoint.md
python experiments/C/e5_probe/e5_data.py                                                              # runs/C/e5_data/{small,mixed}.npz
python experiments/C/e5_probe/e5_train.py --data small                                                # and --data mixed; 20 runs each
python experiments/C/e5_probe/e5_probe.py small                                                       # results/e5_probe_small.md (and mixed)
python experiments/C/e5_probe/e5_data.py medium moderate                                              # fifth pass; then --data medium | moderate, e5_probe.py medium | moderate
```

## First pass (2026-11-17)

| seed | test rel. L2 | layer poles within 10 % (closest, error) | causal LTI probe residual (all ε) | Jacobian: non-causal, Toeplitz residual | impulse-response pole (error) |
|---|---|---|---|---|---|
| 0 | 0.13 | 0/256 (0.005 + 0.076i, 0.93) | 0.86–0.87 | 0.88, 0.59 | −0.169 + 0.968i (0.08) |
| 1 | 0.13 | 0/256 (0.010 + 0.087i, 0.92) | 0.83–0.84 | 0.79, 0.62 | −0.246 + 1.050i (0.08) |
| 2 | 0.13 | 0/256 (0.028 + 0.092i, 0.92) | 0.82 | 0.84, 0.58 | −0.150 + 1.072i (0.14) |
| 3 | 0.22 | 0/256 (0.028 + 0.086i, 0.93) | 0.98 | 0.90, 0.95 | −0.189 + 0.757i (0.22) |
| 4 | 0.37 | 0/256 (0.007 + 0.110i, 0.90) | 0.97–0.98 | 0.76, 0.97 | −0.159 + 0.987i (0.09) |

**Findings.**

1. **The stored poles carry no physics.** All 256 layer poles of every seed stay where they were
   initialised: |μ| ≈ 0.1. The original code initialises Re μ and Im μ uniform in [0, 1/16), and
   200–237 of the 256 poles are still in the right half-plane. None lies within 10 % of the linearised pole. This is E1's finding 1 (the
   learning-rate schedule bounds the pole travel) in a nonlinear model at its published
   configuration.
2. **The trained operator is not a causal LTI map, even for small signals.**
   - 76–90 % of its small-signal Jacobian lies above the diagonal: the output responds before
     the input.
   - Its causal part is far from shift-invariant (Toeplitz residual 0.58–0.97).
   - A causal LTI probe therefore leaves 80–98 % of the response unexplained at every scale,
     and its poles are meaningless.
   - The steady branch is periodic (circulant), and the lifting, the 1 × 1 channel mixing and
     the activation make the map time-varying.
3. **Its average impulse response still carries the oscillation.** The order-2 pencil on the
   mean subdiagonals of J gives −0.15 … −0.25 ± 0.76 … 1.07i, within 10 % of the linearised
   pole in 3 of 5 seeds (frequency within 11 % in 4 of 5; damping underestimated). The model
   represents the dynamics through its steady branch and channel mixing, not through its
   poles.

**Reading for RQ4.**

- **The plan's indicator works.** The linear-probe poles and the stored poles disagree
  completely here: the probe finds the physical pole roughly, and the layer does not contain
  it. That disagreement is the signal that the stored poles must not be interpreted.
- **The probe needs two preconditions, checked first.** These are causality (the non-causal
  share of J) and shift invariance (the Toeplitz residual). When they fail, as here, a causal
  LTI fit is not a valid probe. The average impulse response then gives only a rough estimate,
  with an error of 8–22 %.

## Second pass (2026-11-17): models whose poles move

`e5_train.py` trains three more variants, 5 seeds each, with the E1 protocol `common`:

| variant | model | init | pole lr |
|---|---|---|---|
| `lno_plr10` | original LNO | original | × 10 |
| `lno_stable_plr10` | original LNO | E1 "stable" | × 10 |
| `tied4_stable_plr10` | E1 tied single layer, 4 slots (linear) | E1 "stable" | × 10 |

`e5_probe.py` probes all four variants, 20 runs (`results/e5_probe.md`).

**RQ4 indicator** (defined after a first look at these runs, so the 20 runs are development data for it).

- The probe is valid if J is causal and shift-invariant: the non-causal share and the Toeplitz
  residual are both below 0.1.
- d is the relative distance from J's impulse-response pole to the nearest stored pole.

**Equivalent linear pole of the training data.** This is the best linear pair, fitted with
per-sample relative weights:

| amplitude band | samples | pole | residual |
|---|---|---|---|
| A ≤ 1 | 20 | −0.250 + 0.971i | 0.01 |
| 1 < A ≤ 3 | 40 | −0.251 + 0.997i | 0.04 |
| A > 3 | 140 | −0.298 + 1.199i | 0.28 |
| all | 200 | −0.304 + 1.137i | 0.31 |

The system is hardening, so the equivalent frequency rises with the amplitude.

| variant | test rel. L2 | layer poles near the linearised pole (≤ 10 %) | J valid | d (probe ↔ layer) | probe pole → linearised |
|---|---|---|---|---|---|
| `base` | 0.13–0.37 | 0/5 runs (best 0.90–0.93) | 0/5 (non-causal 0.76–0.90) | 0.89–0.92 | 0.08–0.22 |
| `lno_plr10` | 0.15–0.30 | 0/5 (best 0.32–0.68) | 0/5 (0.52–0.65) | 0.21–0.69 | 0.07–0.98 |
| `lno_stable_plr10` | 0.33–0.50 | **3/5 (best 0.01–0.05)** | 0/5 (0.21–0.76) | 0.02–0.49 | 0.03–7.4 |
| `tied4_stable_plr10` | 0.26–0.27 (3 seeds); 0.90–0.93 (2) | 1/5 (good fits: 0.08–0.12) | **5/5** (0.01–0.02) | **0.00–0.06** (good fits) | 0.12–0.14 (good fits) |

**Findings.**

1. **Only the linear tied layer gives a valid probe.**
   - Its Jacobian is causal and shift-invariant to 1–2 %.
   - Probe and stored poles agree (d ≤ 0.06 in the good fits).
   - Every LNO variant fails the validity check (non-causal share 0.21–0.90), including the
     ones whose poles moved.
2. **Agreement is not correctness.**
   - The tied layer's agreed pole, −0.27 + 1.09 … 1.11i, is 8–14 % from the small-signal pole.
     It lies between that pole and the training set's equivalent linear pole (−0.30 + 1.14i).
     A linear model trained over all amplitudes learns an amplitude-averaged pole, not the
     linearisation at x = 0.
   - The two failed tied fits (test 0.90–0.93) also agree with their probes (d 0.03–0.18) while
     being far off. Agreement must be read together with the fit error.
3. **A movable LNO can contain the physical pole, but nothing points to it.**
   - With the stable init and pole lr × 10, the layer holds a pole within 5 % of the linearised
     one in 3/5 seeds (best 0.01).
   - Picking it out of 256 needs the truth: in all three of those seeds the invalid probe lands
     elsewhere (0.40–1.49 from the linearised pole).
   - The model is also less accurate (test 0.33–0.50 against 0.13–0.37).

**Reading for RQ4.** A stored pole is trustworthy only if three conditions hold:

- (i) the probe is valid, i.e. J is causal and shift-invariant;
- (ii) probe and layer agree;
- (iii) the model fits well.

Even then, it is the equivalent linearisation over the training amplitudes, not the
small-signal pole. Under this rule the 3 good tied fits are trusted, and they lie within 8–14 %
of the linearised pole. No LNO run is trusted. No run is trusted wrongly, because the two
failed tied fits are excluded by (iii).

## Third pass (2026-11-17): equivalent-linear (secant) poles along the amplitude

`e5_oppoint.py` → `results/e5_oppoint.md`.

**Why a secant probe.** At an operating point A sin 5t the Duffing linearisation is
time-varying (stiffness k + 3k₃x₀(t)²), so it has no poles. The comparable quantity is the
*equivalent pole*: the best linear pair (FOH VarPro) for one input–response pair at amplitude A.

**Method.**

- It is computed for the data's stored response and for each model's own response to the same
  input.
- The training waveform (in sample) and the test waveform (decaying envelope, out of sample) are
  both used.
- Amplitudes are A = 0.1 … 10.

Im of the equivalent pole, training waveform (models: median over 5 seeds):

| source | A = 0.1 | 0.5 | 1 | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|---|---|
| data | 0.968 | 0.970 | 0.975 | 0.995 | 1.024 | 1.104 | 1.204 | 1.425 |
| `base` (layer poles at their init) | 0.975 | 0.957 | 0.968 | 0.993 | 1.028 | 1.110 | 1.203 | 1.358 |
| `lno_stable_plr10` | 0.927 | 0.979 | 0.989 | 1.005 | 1.027 | 1.103 | 1.213 | 1.351 |
| `tied4_stable_plr10` (linear) | 1.116 | 1.116 | 1.116 | 1.116 | 1.116 | 1.116 | 1.116 | 1.116 |

Median relative error of the model's equivalent frequency against the data's:

| variant | training waveform, A ≤ 7 | A = 10 | test waveform, A ≤ 7 | A = 8.9 |
|---|---|---|---|---|
| `base` | 0.1–1.5 % (max over seeds 3.8 %) | 4.7 % | 0.6–1.2 % (max 4.3 %) | 3.9 % |
| `lno_plr10` | 0.4–3.6 % | 6.9 % | 1.0–2.2 % | 3.7 % |
| `lno_stable_plr10` | 0.1–4.3 % | 5.2 % | 0.6–5.2 % | 6.2 % |
| `tied4_stable_plr10` | 1.1–15 % | 22 % | 1.4–15 % | 17 % |

**Findings.**

1. **The LNO reproduces the amplitude-dependent dynamics, and a secant probe reads it out.**
   - The original LNO's equivalent pole follows the data's hardening drift (0.968 → 1.20 up to
     A = 7) within 0.1–1.5 % (median).
   - It does so on the out-of-sample test waveform too (0.6–1.2 %).
   - Its 256 stored poles have not moved from the initialisation.
   - The dynamics, including the amplitude dependence, are carried by the nonlinearity and the
     steady branch.
2. **The linear tied layer cannot follow the drift.** Its equivalent pole is fixed at 1.116,
     the compromise found in the second pass, and it is 15 % off at small amplitudes.
3. **The secant probe is robust to the fit error.** The base model's output error at these
   samples is 5–30 %, but its equivalent frequency is within 1.5 %.

**Reading for RQ4.**

- **The pole interpretation of a nonlinear operator should come from probes, not from its
  stored poles.** For the published LNO, the stored poles are uninformative (first pass), while
  its secant probes reproduce the system's amplitude-dependent equivalent pole to about 1 %.
- **The two probes answer different questions.**
  - The Jacobian probe at x = 0 needs a causal, shift-invariant Jacobian to be valid.
  - The secant probe needs only input–response pairs from the model. It describes the operating
    regime of those inputs.

## Fourth pass (2026-11-17): the trust rule on new data

**Data** (`e5_data.py`; the npz files are kept in `runs/C/e5_data/`).

- The LNO data follow x″ + 0.5x′ + x + k₃x³ = f with **k₃ = 1**. A `solve_ivp` simulation
  reproduces stored training samples to 2.5e-4 (A = 0.05), 4.5e-3 (A = 3) and 1.2e-2 (A = 10).
- Two new sets on the same grid use the four E1 input families (step, ramp, PRBS, smooth
  random), 200 / 50 / 130 samples:
  - `small`: peak |f| in [0.02, 0.1], so |x| ≤ 0.23 and k₃x²/k ≤ 0.05;
  - `mixed`: peak |f| in [0.05, 5].
- A first `small` version used |f| ≤ 0.5. It reached |x| = 0.88, where the cubic term is no
  longer small, and was replaced before any training.
- The same 4 variants were trained, 5 seeds each (`e5_train.py --data small | mixed`), 40 runs.

**Rule, fixed before these runs** (`e5_probe.py`: `TRUST_D`, `TRUST_TEST`):

- trusted = J valid (non-causal and Toeplitz shares < 0.1) ∧ d(probe, layer) < 0.1 ∧ test rel. L2 < 0.5;
- prediction: on `small`, a trusted pole is within 10 % of the small-signal pole −0.25 ± 0.9682i.

| set | variant | test rel. L2 | layer poles ≤ 10 % from the small-signal pole | J valid | trusted | trusted pole → small-signal pole |
|---|---|---|---|---|---|---|
| `small` | `base` | 0.41–0.53 | 0/5 | 0/5 | 0 | – |
| `small` | `lno_plr10` | 0.14–0.95 | 0/5 | 0/5 | 0 | – |
| `small` | `lno_stable_plr10` | **0.009–0.10** | 4/5 (up to 7 poles) | 1/5 (others: non-causal 0.13–0.21) | 1 | 0.01 |
| `small` | `tied4_stable_plr10` | 0.005–0.007 | 5/5 | 5/5 | 5 | 0.00 (all) |
| `mixed` | `base` | 0.34–0.42 | 0/5 | 1/5 | 0 | – |
| `mixed` | `lno_plr10` | 0.25–0.28 | 3/5 | 0/5 | 0 | – |
| `mixed` | `lno_stable_plr10` | 0.14–0.16 | 4/5 | 0/5 | 0 | – |
| `mixed` | `tied4_stable_plr10` | 0.43–0.44 | 0/5 | 5/5 | 2 | **0.86, 0.43** |

**Findings.**

1. **On nearly linear data the rule works, and it is conservative.**
   - All 6 trusted runs are within 1 % of the small-signal pole: 5 tied layers, plus the LNO
     with stable init and pole lr × 10.
   - Three more of those LNO runs have an impulse-response pole within 3–4 % but are not trusted,
     because their Jacobian is slightly non-causal (0.13–0.21).
   - The original LNO (original init, poles at the initialisation) fits these data poorly
     (test 0.41–0.53) and is never trusted.
2. **On strongly nonlinear data the rule fails as a small-signal statement.**
   - Two tied layers are trusted, with poles 43 % and 86 % from the small-signal pole.
   - They agree with the data's own equivalent linear pole (−0.89 + 1.75i for the whole set). So
     the rule's reading, "a trusted pole is the equivalent linearisation", holds.
   - But the best linear fit of the data leaves 44 % of the output unexplained, and the tied
     layers' test error is 0.43. The fit criterion (test < 0.5) was too lax: at that error a
     linear model's pole is not a meaningful summary.
3. **Stable init with pole lr × 10 is what makes an LNO's poles physical.** On both sets those
   LNOs hold poles within 10 % of the small-signal pole in 4/5 seeds, and they fit best
   (`small` 0.009–0.10, `mixed` 0.14–0.16). Their Jacobians are valid probes in only 1 of 10 runs,
   however (non-causal 0.05–0.40), so the rule certifies one of them.

**Reading for RQ4.**

- **The three-part rule is sound in the linear regime.** It produced no false trust and
  recovered the pole to 1 %.
- **Its fit criterion has to be stricter, or relative to the nonlinearity.** One candidate:
  require that the best linear model of the data itself fits (residual < 0.1) before any
  linear-model pole is read as physical. This candidate is set after seeing these results, so
  it would need a third data set.

## Fifth pass (2026-11-18): rule v2 with a linearity gate on the data

**Rule v2** (`e5_probe.py`: `GATE`; pre-registered in `results/e5_rule_v2_prereg.md` before training):

- gate (data): the training data's own best linear pair (FOH VarPro, all samples) leaves < 0.1 of the output;
- rule v1 (model): J valid ∧ d(probe, layer) < 0.1 ∧ test < 0.5.

Applied to the fourth pass, the gate passes on `small` (0.005) and fails on `mixed` (0.44). Since the threshold
was proposed after those results, two new sets were made (data seed 1, E1 input families). Their bands come
from a data-only scan of the gate residual, with no models:

- `medium`: peak |f| in [0.1, 0.5], gate residual **0.095** (passes, near the threshold);
- `moderate`: peak |f| in [0.2, 1], gate residual **0.20** (fails).

Same 4 variants × 5 seeds, 40 runs (`results/e5_probe_medium.md`, `results/e5_probe_moderate.md`).

| set | data equivalent pole (resid) | v1 trusted | their J pole → small-signal pole | → data equivalent pole | v2 |
|---|---|---|---|---|---|
| `medium` | −0.280 + 1.044i (0.095) | 6 (tied 5/5, stable LNO 1/5) | 0.05–0.06 | 0.02–0.03 | gate passes: same 6 trusted |
| `moderate` | −0.359 + 1.176i (0.20) | 4 (tied 4/5) | **0.15–0.17** | 0.06–0.07 | gate fails: none trusted |

**Findings.**

1. **Both predictions hold.**
   - On `medium` every trusted pole is within 10 % of the small-signal pole (5–6 %). There is no false trust.
   - On `moderate` rule v1 alone would have trusted 4 tied layers whose poles are 15–17 % off. The gate
     rejects the set, so v2 trusts nothing.
2. **What a trusted pole measures is the data's equivalent linear pole.** On both sets the trusted J poles are
   2–7 % from it, and their distance to the small-signal pole grows with the amplitude band (≤ 1 % on `small`,
   5–6 % on `medium`, 15–17 % on `moderate`). The gate bounds this bias: a residual below 0.1 kept it under
   10 % here.
3. **The LNOs with stable init and pole lr × 10 again hold physical poles.** Their J pole is 4–6 % from the
   small-signal pole on `medium` in 4/5 seeds and 8–19 % on `moderate`, with test 0.04–0.09. But their
   Jacobians are mostly slightly non-causal (0.08–0.30), and the rule certifies one of ten.
4. **The original LNO and pole lr × 10 alone** (original init) are never trusted: invalid J, J poles 11–98 % off.

**Reading for RQ4.**

- Over four simulated sets (80 runs), rule v2 trusted 12 runs, all within 6 % of the small-signal pole, and it
  abstained on the two sets where a linear summary is poor. Rule v1 alone gave 6 false trusts (2 on `mixed`,
  4 on `moderate`).
- The rule reads a stored pole as the system's equivalent linear pole over the training regime. That pole
  equals the small-signal pole only to the extent the data are linear, which is what the gate checks.
- Limits: one system (Duffing, hardening), one gate threshold, and a bias that grows towards the threshold
  (6 % at 0.095).

**Next.**

- Duffing c = 0 (undamped, poles on the axis) as a boundary case.
- Secant probes on other nonlinear LNO benchmarks (pendulum, Lorenz).
