# (C) Identifiability of dual-branch operators

Plan: `docs/proposal_C_identifiability.qmd`

| Work package | Months | Folder |
|---|---|---|
| E1 audit of existing models (RQ1) | 3–5 | `e1_audit/` |
| Theory H1–H3, E2–E4 (RQ2, RQ3) | 6–12 | `theory/`, `remedies/` |
| E5 nonlinear probe diagnosis (RQ4) | 10–12 | `probe/` |

Ground truth comes from `lfo.data.lti` (exact responses, known poles).
Score poles with `lfo.metrics.pole_error`, `residue_error`, and `seed_spread`.
Hand-off to (A): the evaluation scripts from E1 by month 5.
