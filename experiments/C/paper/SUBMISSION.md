# (C) paper: submission checklist (2026-11-22)

Draft: `c_paper.qmd` (15 pages in the working format, 20 references, 3 + 2 figures, appendix grid).

## Venue: TMLR (user decision, 2026-11-23)

Options that were considered:

| option | fit | format work |
|---|---|---|
| TMLR (recommended) | audits, negative results and careful empirical work are in scope; no page limit; rolling submission; double-blind | TMLR LaTeX template; anonymize; the appendix can stay |
| NeurIPS / ICML main track | wider audience; the trust rule is the method contribution | 8–9 pages: move E1 detail and the rule history to the appendix |
| ML-for-physical-sciences workshop (NeurIPS / ICLR) | the planned month-5 output; quick | 4 pages: E1 + the trust rule's result only |
| Journal (J. Comput. Phys., Neural Networks) | the LNO and PILNO venues; reviewers who know the models | journal template; no anonymization |

## Done

- Numbers checked against the READMEs (2026-11-22, 8ed7bcf).
- Related work added; references checked against doi.org / arXiv.
- Code public: github.com/SeoulTechPSE/laplace-fourier-operators-release (for a double-blind venue use an
  anonymous mirror at submission and the public link after acceptance).

## TMLR submission build

- `python experiments/C/paper/tmlr/make_tmlr.py` → `c_paper_tmlr.qmd` / `.pdf` (generated; edit `c_paper.qmd`).
  TMLR style (anonymous), pdflatex, natbib author-year, one-paragraph abstract (`tmlr/abstract.md`),
  references before the lettered appendix. Main body 10.5 pages (TMLR: more than 12 pages lengthens review).
- TMLR submissions, reviews and supplements are public on OpenReview, so the submission carries the release's
  neutral sentence instead of the follow-up summary (same rules as `release/make_release.py`).
- Code: `python experiments/C/paper/tmlr/make_supplement.py RELEASE_DIR c_tmlr_supplement.zip` (anonymous:
  shared package, tests, the (C) experiments and the modules they import; no A / D experiments or drafts; fails
  if an identifying string remains). Upload as supplementary material; do not link the public repository.

## To do before submission (user)

- Read the PDF once, in particular the one-paragraph abstract (condensed from the working paper's).
- OpenReview: TMLR console, submit the PDF and the supplement zip; authors and conflicts entered in OpenReview
  only.
- After acceptance: `\usepackage[accepted]{tmlr}`, month / year / OpenReview link, de-anonymize, link the
  public repository.
