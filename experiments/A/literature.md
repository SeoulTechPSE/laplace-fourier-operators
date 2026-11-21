# (A) Literature re-check (2026-11-22)

The A plan (Section 2) asked for a systematic re-check of the gap "no work treats wavenumber-dependent poles, the
dispersion relation itself, as the learning target". Searched: learned dispersion relations in neural operators,
per-Fourier-mode generators / exponential integrators in operators, SSM-based operators, LNO follow-ups, domain
transfer. Details marked "read" come from the paper text; others from abstracts.

## Closest work

| work | what it learns | relation to the WPO |
|---|---|---|
| **SGNO**, Spectral Generator Neural Operator (Li, Wang, Salim; arXiv 2602.18801, 2026; read) | per layer, a diagonal generator with non-positive real part per Fourier **index** and channel, a phi_1 (exponential-integrator) update of a pointwise-MLP nonlinearity, plus a fitted per-index linear multiplier; autoregressive one-step model on APEBench (1D–3D, KdV, KS, Burgers, Kolmogorov) | the same exponential update per mode, but parameters per mode index, no function of k, no claim to recover a dispersion relation, no domain / resolution transfer, no forcing input. Does not cite LNO. |
| **LGN-KM**, Linear Generator Networks (Jamil, Kapadia; arXiv 2603.29264, NeurIPS 2026; read) | continuous-time Koopman generator L_k = S − D_k per Fourier mode in a latent space from a nonlinear encoder; D_k = softplus(d) + softplus(alpha) \|k\|^2 / k_max^2 (or \|k\|^p) — a function of the physical wavenumber, shared S | the closest in spirit: damping as a function of k, a "dispersion relation" read from the generator (2D Navier–Stokes branches, fiber beta_2 within 3 %). Unforced (no input term; forced flows decay), no domain transfer, latent (not physical) modes, accuracy below FNO in the training horizon. Does not cite LNO. |
| **ENNO**, Exponential-Nemytskii Neural Operators (Li, Li; arXiv 2610.39041, 2026; abstract) | exact linear propagation + local nonlinear function of the field (first-order Duhamel), across step sizes; local identifiability of both parts from two step sizes | the "exact linear + learned nonlinear" split of the planned hybrid (next-level step 4); identifiability in the step-size sense. |
| Semigroup-aware kernel NO (arXiv 2610.39845; abstract) | latent exponential (phi_1) integration for forced linear PDEs | forced linear case only. |
| SSMs as neural operators (Hu et al., arXiv 2409.03231); Mamba NO (arXiv 2410.02113); DeepOMamba | S4 / Mamba along time; LNO among the baselines (LNO best out of distribution on one task) | SSM baselines exist for ODE-type operators; no wavenumber structure. |
| PILNO (arXiv 2602.12706), WLNO (arXiv 2605.24658) | LNO variants (transient + FNO steady branch; wavelet–Laplace) | pole interpretability claimed for the transient branch only; no identifiability study. |
| Domain extension: reference-frequency wavefield operators (arXiv 2405.01272), transformer domain extension (arXiv 2606.14597), local operators with learned composition | larger domains by rescaling or locality | none ties spectral weights to the physical wavenumber. |

## Consequences for the (A) claims

- "Per-mode exponential poles" is **not** new (SGNO, S4D-type operators). The (A) claim must be the
  **physical-wavenumber form**: mu_n(k) smooth in k, which gives (i) identification of lambda(k), (ii) continuation
  outside the forced band, (iii) transfer to other domain sizes (`set_domain`), (iv) tolerance of unseen modes.
  a36 tests (ii) and (iv) against the per-index alternative in the same architecture.
- LGN-KM also ties damping to |k|; the differences to state are forcing (input-driven operator with an exact FOH
  time axis), physical (not latent) modes, poles with residues (several per k), parametric forms beyond |k|^p
  (complex dispersion, unstable band), and domain transfer. A direct comparison on a shared unforced benchmark
  would be needed before claiming accuracy differences.
- Pole identifiability of LNO-type models (project C) still has no direct prior work in this search.
- Cite: SGNO, LGN-KM, ENNO, the SSM operator papers, PILNO, WLNO in the A paper's related work.
