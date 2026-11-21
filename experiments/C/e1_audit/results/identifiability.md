# E1 practical identifiability (tied layer, at the truth)

Float64 re-implementation vs `PoleResidueTS1d`: rel. diff 2.3e-07. e = RMS relative L2 error over the 200 training inputs. Compare with the oracle floor (≈ 1e-3, discretization) and the E1 test errors.

## Local: error caused by a 10 % change of one parameter

fixed: all other parameters at the truth; profiled: the others re-fitted (linearized). Small profiled e = the other poles/residues compensate the change.

| system | pole | parameter | fixed e | profiled e | cond(F) |
|---|---|---|---|---|---|
| p2 | -0.5+2.0i | Re mu | 3.5e-02 | 2.5e-02 | 4.7e+01 |
| p2 | -0.5+2.0i | Im mu | 1.7e-01 | 8.4e-02 | 4.7e+01 |
| p2 | -0.5+2.0i | Re beta | 4.8e-02 | 2.5e-02 | 4.7e+01 |
| p2 | -0.5+2.0i | Im beta | 6.3e-02 | 2.2e-02 | 4.7e+01 |
| p4 | -0.4+3.0i | Re mu | 2.0e-02 | 1.0e-02 | 3.9e+04 |
| p4 | -0.4+3.0i | Im mu | 1.6e-01 | 8.4e-02 | 3.9e+04 |
| p4 | -0.4+3.0i | Re beta | 1.8e-02 | 8.9e-03 | 3.9e+04 |
| p4 | -0.4+3.0i | Im beta | 5.8e-02 | 8.9e-03 | 3.9e+04 |
| p4 | -1.0 | Re mu | 1.6e-01 | 6.3e-03 | 3.9e+04 |
| p4 | -1.0 | Re beta | 1.8e-01 | 1.2e-02 | 3.9e+04 |
| p4 | -0.2 | Re mu | 1.7e-01 | 1.3e-02 | 3.9e+04 |
| p4 | -0.2 | Re beta | 3.0e-01 | 8.0e-03 | 3.9e+04 |
| p6 | -0.5+1.5i | Re mu | 2.6e-02 | 3.4e-03 | 6.8e+06 |
| p6 | -0.5+1.5i | Im mu | 9.8e-02 | 1.2e-02 | 6.8e+06 |
| p6 | -0.5+1.5i | Re beta | 3.6e-02 | 1.3e-03 | 6.8e+06 |
| p6 | -0.5+1.5i | Im beta | 3.5e-02 | 1.3e-03 | 6.8e+06 |
| p6 | -0.3+4.0i | Re mu | 1.3e-02 | 6.4e-03 | 6.8e+06 |
| p6 | -0.3+4.0i | Im mu | 1.7e-01 | 9.0e-02 | 6.8e+06 |
| p6 | -0.3+4.0i | Re beta | 1.8e-02 | 6.7e-03 | 6.8e+06 |
| p6 | -0.3+4.0i | Im beta | 1.3e-02 | 3.3e-03 | 6.8e+06 |
| p6 | -1.0 | Re mu | 7.6e-02 | 4.9e-04 | 6.8e+06 |
| p6 | -1.0 | Re beta | 8.4e-02 | 1.1e-04 | 6.8e+06 |
| p6 | -2.0 | Re mu | 3.4e-02 | 1.6e-04 | 6.8e+06 |
| p6 | -2.0 | Re beta | 3.6e-02 | 8.5e-05 | 6.8e+06 |

## Profile of Re mu (nonlinear): minimum e with Re mu fixed at c × truth

All other parameters re-fitted with L-BFGS from the truth (the fixed pole's Im mu and residue included). Im mu column: fitted Im of the profiled pole.

| system | pole | c=0.25 | c=0.5 | c=0.75 | c=0.9 | c=1 | c=1.1 | c=1.25 | c=1.5 | c=2 | c=3 | Im mu at c=0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p2 | -0.5+2.0i | 3.6e-01 | 1.8e-01 | 7.4e-02 | 2.7e-02 | 9.7e-04 | 2.4e-02 | 5.5e-02 | 9.7e-02 | 1.6e-01 | 2.3e-01 | +2.12 |
| p4 | -0.4+3.0i | 1.0e-01 | 5.8e-02 | 2.7e-02 | 1.0e-02 | 1.3e-03 | 9.7e-03 | 2.3e-02 | 4.2e-02 | 7.2e-02 | 1.1e-01 | +3.03 |
| p4 | -0.2 | 3.0e-02 | 1.1e-02 | 2.8e-03 | 1.4e-03 | 1.3e-03 | 1.1e-02 | 2.3e-02 | 3.4e-02 | 4.0e-02 | nan | +0.01 |
| p4 | -1.0 | 1.8e-02 | 3.3e-02 | 1.7e-02 | 6.5e-03 | 1.3e-03 | 6.2e-03 | 6.5e-03 | 1.0e-02 | 1.6e-02 | 2.2e-02 | -0.10 |
| p6 | -0.3+4.0i | 8.6e-02 | 5.0e-02 | 2.1e-02 | 7.3e-03 | 8.2e-04 | 6.3e-03 | 1.5e-02 | 2.7e-02 | 4.5e-02 | 6.9e-02 | +4.00 |
| p6 | -0.5+1.5i | 5.3e-02 | 3.2e-02 | 1.4e-02 | 5.3e-03 | 8.2e-04 | 5.0e-03 | 1.3e-02 | 2.0e-02 | 3.5e-02 | 3.3e-02 | +1.45 |
| p6 | -1.0 | 3.3e-03 | 2.8e-03 | 9.3e-04 | 1.1e-03 | 8.2e-04 | 8.8e-04 | 1.1e-03 | 4.1e-03 | 1.2e-02 | 1.1e-03 | -0.15 |
| p6 | -2.0 | 3.6e-03 | 1.4e-03 | 9.1e-04 | 8.3e-04 | 8.2e-04 | 8.3e-04 | 9.0e-04 | 1.1e-03 | 1.5e-03 | 2.1e-03 | -0.00 |
