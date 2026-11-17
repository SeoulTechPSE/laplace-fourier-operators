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
python experiments/C/e5_probe/e5_probe.py                                                             # results/e5_probe.md
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

**Next.**

- Apply the same probes to models whose poles do move: pole learning rate × 10, as in E1; the
  `damp8` recipe; a tied single layer trained on Duffing.
- Probe around nonzero operating points: amplitude-dependent linearisation along the training
  waveform.
- Duffing c = 0 (undamped, poles on the axis) as a boundary case.
