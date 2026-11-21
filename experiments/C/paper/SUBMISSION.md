# (C) paper: submission checklist (2026-11-22)

Draft: `c_paper.qmd` (15 pages in the working format, 20 references, 3 + 2 figures, appendix grid).

## Venue options (decision open, see `STATUS.md`)

| option | fit | format work |
|---|---|---|
| TMLR (recommended) | audits, negative results and careful empirical work are in scope; no page limit; rolling submission; double-blind | TMLR LaTeX template; anonymise; the appendix can stay |
| NeurIPS / ICML main track | wider audience; the trust rule is the method contribution | 8–9 pages: move E1 detail and the rule history to the appendix |
| ML-for-physical-sciences workshop (NeurIPS / ICLR) | the planned month-5 output; quick | 4 pages: E1 + the trust rule's result only |
| Journal (J. Comput. Phys., Neural Networks) | the LNO and PILNO venues; reviewers who know the models | journal template; no anonymisation |

## Done

- Numbers checked against the READMEs (2026-11-22, 8ed7bcf).
- Related work added; references checked against doi.org / arXiv.
- Code public: github.com/SeoulTechPSE/laplace-fourier-operators-release (for a double-blind venue use an
  anonymous mirror at submission and the public link after acceptance).

## To do before submission

- Venue template and length cut.
- Anonymisation (author, SeoulTech, repository link, "our earlier" phrasing).
- A short reproducibility statement (seeds, runs, compute: `experiments/INDEX.md`).
- Figures at the venue's column width; check fonts embedded.
- One more read of the abstract for the venue's audience.
