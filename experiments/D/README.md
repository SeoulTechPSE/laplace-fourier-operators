# (D) Non-periodic inputs and parametric initial conditions

Plan: `docs/proposal_D_nonperiodic_initial_conditions.qmd`

| Work package | Months | Folder |
|---|---|---|
| D1 non-periodic input representation (RQ1) | 4–8 | `d1_inputs/` |
| D2 initial conditions and rollout (RQ2, RQ3) | 13–19 | `d2_initial_conditions/` (started 2026-11-19) |
| Non-periodic space: time-varying boundaries (plan Table 6.1) | – | `d3_boundary/` (started 2026-11-19) |

Input families (step, ramp, PRBS, non-periodic smooth signals) are in
`lfo.data.lti`. Hand-offs: D1 decides the time-axis computation for (A) by month 6;
D2 delivers the initial-condition encoder to (A) application 2 by month 17.

Working paper: `paper/d_paper.qmd` (D2, boundaries). D1 moved to Appendix A of the (A) method paper
(`experiments/A/paper/a1_paper.qmd`) as the month-8 gate prescribes (decision 2026-11-23).
