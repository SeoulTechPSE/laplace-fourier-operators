# E5 rule v2 on the driven pendulum: pre-registration (written 2026-11-17T17:56Z, before any pendulum training)

**System.** x'' + 0.5 x' + sin x = f (the LNO data `1D_Pendulum_c05`; a simulation reproduces stored samples to
2.4e-4 at A = 0.05 and 1.4e-2 at A = 10). Small-signal pole −0.25 ± 0.9682i, as for the Duffing sets. The spring
softens, where the Duffing spring hardens.

**Rule v2, unchanged** (`e5_probe.py`): gate (data best linear pair residual < 0.1) ∧ rule v1 (J valid ∧
d(J pole, stored pole) < 0.1 ∧ test < 0.5).

**Sets and models.** The pendulum LNO architecture (grid input, 8 modes), the same 4 variants, `common`
protocol, 5 seeds. Gate residuals from a data-only scan (no models):

- `pend_orig`: the original LNO data; gate 0.121 (fails), equivalent pole −0.255 + 0.924i (4.6 % off).
- `pend_medium`: peak |f| ~ U[0.1, 0.5], data seed 2; scan 0.039 (passes), equivalent pole 2 % off.
- `pend_large`: peak |f| ~ U[0.3, 1], data seed 2; scan 0.26 (fails); the pendulum goes over the top.

**Predictions.**

1. `pend_medium`: the gate passes; every run trusted by v2 has a J pole within 10 % of the small-signal pole.
2. `pend_large` and `pend_orig`: the gate fails and v2 trusts nothing. On `pend_orig` this abstention is
   expected to be conservative: the data's equivalent pole is only 4.6 % off.
3. The LNO variants take the time grid as an input channel, so their Jacobian need not be shift-invariant; few
   or no LNO runs are expected to have a valid probe. The tied layer has no grid input.
