# E1 per-pole breakdown

Per true pole (Im ≥ 0): median relative error over seeds, fraction recovered (error < 0.1), median signed deviation of the matched pole (ΔRe, ΔIm = learned − true). Diverged runs excluded. Deep: closest pooled poles.

## layer

| system | model | init | protocol | true pole | seeds | median err | recovered | ΔRe | ΔIm |
|---|---|---|---|---|---|---|---|---|---|
| p2 | alno | band | common | -0.5+2.0i | 10 | 0.059 | 70% | -0.02 | -0.07 |
| p2 | alno | band | common+plr10 | -0.5+2.0i | 10 | 0.027 | 80% | -0.01 | +0.00 |
| p2 | alno | lno | common | -0.5+2.0i | 10 | 0.809 | 0% | +0.94 | -1.33 |
| p2 | alno | stable | common | -0.5+2.0i | 10 | 5.945 | 30% | -0.36 | +12.17 |
| p2 | alno | stable | common+plr10 | -0.5+2.0i | 10 | 0.009 | 90% | +0.01 | -0.01 |
| p2 | alno | warm | common | -0.5+2.0i | 10 | 0.062 | 100% | +0.04 | +0.01 |
| p2 | lfno | band | common | -0.5+2.0i | 10 | 0.066 | 60% | +0.00 | -0.06 |
| p2 | lfno | band | common+plr10 | -0.5+2.0i | 10 | 0.039 | 80% | +0.04 | +0.02 |
| p2 | lfno | band | lfno_paper | -0.5+2.0i | 10 | 0.064 | 80% | +0.04 | -0.09 |
| p2 | lfno | band | lfno_paper+plr10 | -0.5+2.0i | 10 | 0.088 | 80% | +0.03 | -0.18 |
| p2 | lfno | lno | common | -0.5+2.0i | 10 | 0.809 | 0% | +0.94 | -1.33 |
| p2 | lfno | lno | lfno_paper | -0.5+2.0i | 10 | 0.814 | 0% | +0.94 | -1.36 |
| p2 | lfno | stable | common | -0.5+2.0i | 10 | 6.117 | 10% | -1.05 | +12.53 |
| p2 | lfno | stable | common+plr10 | -0.5+2.0i | 10 | 2.195 | 40% | -0.92 | +4.42 |
| p2 | lfno | stable | lfno_paper | -0.5+2.0i | 10 | 5.896 | 10% | -0.35 | +12.14 |
| p2 | lfno | stable | lfno_paper+plr10 | -0.5+2.0i | 10 | 0.089 | 60% | -0.09 | -0.04 |
| p2 | lfno | warm | common | -0.5+2.0i | 10 | 0.059 | 100% | +0.06 | -0.09 |
| p2 | lfno | warm | lfno_paper | -0.5+2.0i | 10 | 0.056 | 100% | +0.02 | -0.10 |
| p2 | tied | band | common | -0.5+2.0i | 10 | 0.039 | 70% | +0.02 | -0.02 |
| p2 | tied | band | common+plr10 | -0.5+2.0i | 10 | 0.028 | 80% | +0.02 | +0.00 |
| p2 | tied | lno | common | -0.5+2.0i | 10 | 0.809 | 0% | +0.94 | -1.33 |
| p2 | tied | stable | common | -0.5+2.0i | 10 | 5.537 | 20% | -1.03 | +11.36 |
| p2 | tied | stable | common+plr10 | -0.5+2.0i | 10 | 0.002 | 90% | +0.00 | +0.00 |
| p2 | tied | warm | common | -0.5+2.0i | 10 | 0.004 | 100% | +0.01 | +0.00 |
| p4 | alno | band | common | -0.4+3.0i | 10 | 0.040 | 70% | -0.04 | +0.06 |
| p4 | alno | band | common | -1.0 | 10 | 0.787 | 0% | +0.57 | +0.44 |
| p4 | alno | band | common | -0.2 | 10 | 0.865 | 0% | -0.06 | +0.15 |
| p4 | alno | band | common+plr10 | -0.4+3.0i | 10 | 0.029 | 80% | +0.01 | -0.04 |
| p4 | alno | band | common+plr10 | -1.0 | 10 | 0.706 | 0% | +0.56 | +0.39 |
| p4 | alno | band | common+plr10 | -0.2 | 10 | 0.733 | 0% | -0.03 | +0.13 |
| p4 | alno | lno | common | -0.4+3.0i | 10 | 0.815 | 0% | +1.01 | -2.19 |
| p4 | alno | lno | common | -1.0 | 10 | 1.764 | 0% | +1.65 | +0.33 |
| p4 | alno | lno | common | -0.2 | 10 | 3.764 | 0% | +0.73 | +0.08 |
| p4 | alno | stable | common | -0.4+3.0i | 10 | 7.098 | 0% | -0.10 | +21.48 |
| p4 | alno | stable | common | -1.0 | 10 | 14.307 | 0% | +0.08 | +14.30 |
| p4 | alno | stable | common | -0.2 | 10 | 12.664 | 0% | -0.69 | +2.35 |
| p4 | alno | stable | common+plr10 | -0.4+3.0i | 10 | 0.389 | 40% | -0.05 | +1.08 |
| p4 | alno | stable | common+plr10 | -1.0 | 10 | 2.854 | 0% | +0.33 | +2.82 |
| p4 | alno | stable | common+plr10 | -0.2 | 10 | 0.690 | 0% | -0.07 | +0.11 |
| p4 | alno | warm | common | -0.4+3.0i | 10 | 0.021 | 100% | +0.03 | +0.01 |
| p4 | alno | warm | common | -1.0 | 10 | 0.210 | 20% | +0.21 | +0.00 |
| p4 | alno | warm | common | -0.2 | 10 | 0.045 | 90% | -0.01 | +0.00 |
| p4 | lfno | band | common | -0.4+3.0i | 10 | 0.112 | 40% | -0.01 | -0.12 |
| p4 | lfno | band | common | -1.0 | 10 | 0.968 | 0% | +0.42 | +0.88 |
| p4 | lfno | band | common | -0.2 | 10 | 1.117 | 0% | -0.05 | +0.21 |
| p4 | lfno | band | common+plr10 | -0.4+3.0i | 10 | 0.048 | 70% | -0.01 | -0.06 |
| p4 | lfno | band | common+plr10 | -1.0 | 10 | 0.819 | 0% | +0.48 | +0.57 |
| p4 | lfno | band | common+plr10 | -0.2 | 10 | 0.946 | 0% | -0.04 | +0.18 |
| p4 | lfno | band | lfno_paper | -0.4+3.0i | 10 | 0.052 | 80% | +0.03 | -0.16 |
| p4 | lfno | band | lfno_paper | -1.0 | 10 | 0.833 | 0% | +0.67 | +0.50 |
| p4 | lfno | band | lfno_paper | -0.2 | 10 | 0.970 | 0% | -0.02 | +0.19 |
| p4 | lfno | band | lfno_paper+plr10 | -0.4+3.0i | 10 | 0.129 | 20% | -0.06 | -0.38 |
| p4 | lfno | band | lfno_paper+plr10 | -1.0 | 10 | 0.932 | 0% | +0.75 | +0.47 |
| p4 | lfno | band | lfno_paper+plr10 | -0.2 | 10 | 1.014 | 0% | +0.01 | +0.20 |
| p4 | lfno | lno | common | -0.4+3.0i | 10 | 0.815 | 0% | +1.01 | -2.19 |
| p4 | lfno | lno | common | -1.0 | 10 | 1.764 | 0% | +1.65 | +0.33 |
| p4 | lfno | lno | common | -0.2 | 10 | 3.763 | 0% | +0.73 | +0.08 |
| p4 | lfno | lno | lfno_paper | -0.4+3.0i | 10 | 0.823 | 0% | +1.00 | -2.23 |
| p4 | lfno | lno | lfno_paper | -1.0 | 10 | 1.731 | 0% | +1.66 | +0.41 |
| p4 | lfno | lno | lfno_paper | -0.2 | 10 | 3.720 | 0% | +0.72 | +0.17 |
| p4 | lfno | stable | common | -0.4+3.0i | 10 | 7.139 | 0% | -0.52 | +21.60 |
| p4 | lfno | stable | common | -1.0 | 10 | 15.081 | 0% | -0.22 | +15.08 |
| p4 | lfno | stable | common | -0.2 | 10 | 21.618 | 0% | -0.72 | +4.26 |
| p4 | lfno | stable | common+plr10 | -0.4+3.0i | 10 | 1.442 | 0% | -0.88 | +4.09 |
| p4 | lfno | stable | common+plr10 | -1.0 | 10 | 3.942 | 0% | -0.99 | +3.33 |
| p4 | lfno | stable | common+plr10 | -0.2 | 10 | 0.811 | 0% | -0.07 | +0.15 |
| p4 | lfno | stable | lfno_paper | -0.4+3.0i | 10 | 7.233 | 0% | +0.24 | +21.89 |
| p4 | lfno | stable | lfno_paper | -1.0 | 10 | 14.488 | 0% | +0.67 | +14.46 |
| p4 | lfno | stable | lfno_paper | -0.2 | 10 | 6.763 | 0% | -0.24 | +1.32 |
| p4 | lfno | stable | lfno_paper+plr10 | -0.4+3.0i | 10 | 0.188 | 30% | +0.12 | +0.13 |
| p4 | lfno | stable | lfno_paper+plr10 | -1.0 | 10 | 1.705 | 0% | +0.48 | +1.70 |
| p4 | lfno | stable | lfno_paper+plr10 | -0.2 | 10 | 0.872 | 0% | -0.05 | +0.16 |
| p4 | lfno | warm | common | -0.4+3.0i | 10 | 0.036 | 100% | +0.11 | +0.01 |
| p4 | lfno | warm | common | -1.0 | 10 | 0.133 | 20% | +0.13 | +0.00 |
| p4 | lfno | warm | common | -0.2 | 10 | 0.019 | 100% | -0.00 | +0.00 |
| p4 | lfno | warm | lfno_paper | -0.4+3.0i | 10 | 0.017 | 100% | -0.01 | -0.05 |
| p4 | lfno | warm | lfno_paper | -1.0 | 10 | 0.441 | 0% | +0.35 | +0.25 |
| p4 | lfno | warm | lfno_paper | -0.2 | 10 | 0.253 | 20% | +0.03 | +0.04 |
| p4 | tied | band | common | -0.4+3.0i | 10 | 0.145 | 50% | -0.02 | +0.02 |
| p4 | tied | band | common | -1.0 | 10 | 1.014 | 0% | +0.47 | +0.87 |
| p4 | tied | band | common | -0.2 | 10 | 1.065 | 0% | -0.06 | +0.21 |
| p4 | tied | band | common+plr10 | -0.4+3.0i | 10 | 0.040 | 70% | +0.08 | -0.07 |
| p4 | tied | band | common+plr10 | -1.0 | 10 | 0.845 | 0% | +0.55 | +0.59 |
| p4 | tied | band | common+plr10 | -0.2 | 10 | 0.962 | 0% | -0.03 | +0.19 |
| p4 | tied | lno | common | -0.4+3.0i | 10 | 0.814 | 0% | +1.01 | -2.19 |
| p4 | tied | lno | common | -1.0 | 10 | 1.749 | 0% | +1.65 | +0.33 |
| p4 | tied | lno | common | -0.2 | 10 | 3.901 | 0% | +0.76 | +0.08 |
| p4 | tied | stable | common | -0.4+3.0i | 10 | 7.153 | 0% | -0.53 | +21.64 |
| p4 | tied | stable | common | -1.0 | 10 | 15.037 | 0% | -0.13 | +15.03 |
| p4 | tied | stable | common | -0.2 | 10 | 17.886 | 0% | -0.42 | +3.55 |
| p4 | tied | stable | common+plr10 | -0.4+3.0i | 10 | 0.602 | 40% | -0.96 | +1.30 |
| p4 | tied | stable | common+plr10 | -1.0 | 10 | 3.098 | 0% | +0.03 | +2.85 |
| p4 | tied | stable | common+plr10 | -0.2 | 10 | 0.983 | 0% | -0.08 | +0.18 |
| p4 | tied | warm | common | -0.4+3.0i | 10 | 0.003 | 100% | +0.01 | +0.01 |
| p4 | tied | warm | common | -1.0 | 10 | 0.000 | 70% | +0.00 | +0.00 |
| p4 | tied | warm | common | -0.2 | 10 | 0.000 | 100% | -0.00 | +0.00 |
| p6 | alno | band | common | -0.5+1.5i | 10 | 0.117 | 40% | +0.14 | +0.06 |
| p6 | alno | band | common | -0.3+4.0i | 10 | 0.037 | 80% | +0.03 | +0.09 |
| p6 | alno | band | common | -1.0 | 10 | 0.564 | 0% | +0.46 | +0.12 |
| p6 | alno | band | common | -2.0 | 10 | 0.894 | 0% | +1.49 | +0.85 |
| p6 | alno | band | common+plr10 | -0.5+1.5i | 10 | 0.085 | 50% | +0.08 | -0.02 |
| p6 | alno | band | common+plr10 | -0.3+4.0i | 10 | 0.029 | 100% | +0.05 | +0.03 |
| p6 | alno | band | common+plr10 | -1.0 | 10 | 0.419 | 10% | +0.32 | +0.08 |
| p6 | alno | band | common+plr10 | -2.0 | 10 | 0.842 | 0% | +1.51 | +0.66 |
| p6 | alno | lno | common | -0.5+1.5i | 10 | 0.825 | 0% | +1.14 | -0.59 |
| p6 | alno | lno | common | -0.3+4.0i | 10 | 0.841 | 0% | +1.01 | -3.22 |
| p6 | alno | lno | common | -1.0 | 10 | 1.585 | 0% | +1.54 | +0.21 |
| p6 | alno | lno | common | -2.0 | 10 | 1.335 | 0% | +2.64 | +0.27 |
| p6 | alno | stable | common | -0.5+1.5i | 10 | 6.219 | 0% | -0.56 | +9.77 |
| p6 | alno | stable | common | -0.3+4.0i | 10 | 3.962 | 0% | -0.30 | +15.89 |
| p6 | alno | stable | common | -1.0 | 10 | 5.113 | 0% | -0.24 | +5.03 |
| p6 | alno | stable | common | -2.0 | 10 | 8.999 | 0% | +1.03 | +17.97 |
| p6 | alno | stable | common+plr10 | -0.5+1.5i | 10 | 0.852 | 0% | +0.09 | +1.34 |
| p6 | alno | stable | common+plr10 | -0.3+4.0i | 10 | 0.260 | 30% | +0.13 | +1.01 |
| p6 | alno | stable | common+plr10 | -1.0 | 10 | 2.017 | 0% | +0.29 | +1.97 |
| p6 | alno | stable | common+plr10 | -2.0 | 10 | 2.419 | 0% | +1.42 | +4.55 |
| p6 | alno | warm | common | -0.5+1.5i | 10 | 0.057 | 100% | +0.08 | +0.01 |
| p6 | alno | warm | common | -0.3+4.0i | 10 | 0.026 | 100% | +0.01 | +0.09 |
| p6 | alno | warm | common | -1.0 | 10 | 0.098 | 50% | +0.10 | +0.00 |
| p6 | alno | warm | common | -2.0 | 10 | 0.104 | 40% | +0.21 | +0.00 |
| p6 | lfno | band | common | -0.5+1.5i | 10 | 0.133 | 30% | +0.14 | -0.06 |
| p6 | lfno | band | common | -0.3+4.0i | 10 | 0.040 | 90% | +0.08 | -0.08 |
| p6 | lfno | band | common | -1.0 | 10 | 0.520 | 0% | +0.31 | +0.23 |
| p6 | lfno | band | common | -2.0 | 10 | 0.936 | 0% | +1.56 | +0.88 |
| p6 | lfno | band | common+plr10 | -0.5+1.5i | 10 | 0.184 | 30% | +0.09 | -0.19 |
| p6 | lfno | band | common+plr10 | -0.3+4.0i | 10 | 0.027 | 100% | +0.09 | -0.02 |
| p6 | lfno | band | common+plr10 | -1.0 | 10 | 0.458 | 20% | +0.29 | +0.19 |
| p6 | lfno | band | common+plr10 | -2.0 | 10 | 0.972 | 0% | +1.41 | +0.77 |
| p6 | lfno | band | lfno_paper | -0.5+1.5i | 10 | 0.222 | 0% | +0.29 | -0.14 |
| p6 | lfno | band | lfno_paper | -0.3+4.0i | 10 | 0.605 | 10% | +0.15 | -2.41 |
| p6 | lfno | band | lfno_paper | -1.0 | 10 | 0.705 | 0% | +0.68 | +0.15 |
| p6 | lfno | band | lfno_paper | -2.0 | 10 | 0.915 | 0% | +1.74 | +0.51 |
| p6 | lfno | band | lfno_paper+plr10 | -0.5+1.5i | 10 | 0.219 | 0% | +0.19 | -0.28 |
| p6 | lfno | band | lfno_paper+plr10 | -0.3+4.0i | 10 | 0.829 | 0% | +0.16 | -3.32 |
| p6 | lfno | band | lfno_paper+plr10 | -1.0 | 10 | 0.738 | 0% | +0.73 | +0.12 |
| p6 | lfno | band | lfno_paper+plr10 | -2.0 | 10 | 0.909 | 0% | +1.80 | +0.25 |
| p6 | lfno | lno | common | -0.5+1.5i | 10 | 0.825 | 0% | +1.14 | -0.59 |
| p6 | lfno | lno | common | -0.3+4.0i | 10 | 0.841 | 0% | +1.01 | -3.22 |
| p6 | lfno | lno | common | -1.0 | 10 | 1.585 | 0% | +1.54 | +0.29 |
| p6 | lfno | lno | common | -2.0 | 10 | 1.335 | 0% | +2.64 | +0.18 |
| p6 | lfno | lno | lfno_paper | -0.5+1.5i | 10 | 0.807 | 0% | +1.13 | -0.59 |
| p6 | lfno | lno | lfno_paper | -0.3+4.0i | 10 | 0.842 | 0% | +1.01 | -3.25 |
| p6 | lfno | lno | lfno_paper | -1.0 | 10 | 1.568 | 0% | +1.52 | +0.25 |
| p6 | lfno | lno | lfno_paper | -2.0 | 10 | 1.329 | 0% | +2.62 | +0.26 |
| p6 | lfno | stable | common | -0.5+1.5i | 10 | 6.679 | 0% | -0.69 | +10.50 |
| p6 | lfno | stable | common | -0.3+4.0i | 10 | 3.984 | 0% | -0.65 | +15.97 |
| p6 | lfno | stable | common | -1.0 | 10 | 6.558 | 0% | -0.23 | +6.52 |
| p6 | lfno | stable | common | -2.0 | 10 | 9.058 | 0% | +0.93 | +18.09 |
| p6 | lfno | stable | common+plr10 | -0.5+1.5i | 10 | 0.669 | 30% | +0.03 | +1.05 |
| p6 | lfno | stable | common+plr10 | -0.3+4.0i | 10 | 0.159 | 50% | -0.12 | +0.57 |
| p6 | lfno | stable | common+plr10 | -1.0 | 10 | 0.829 | 0% | +0.18 | +0.78 |
| p6 | lfno | stable | common+plr10 | -2.0 | 10 | 1.968 | 0% | +0.97 | +3.63 |
| p6 | lfno | stable | lfno_paper | -0.5+1.5i | 10 | 5.758 | 10% | -0.00 | +9.10 |
| p6 | lfno | stable | lfno_paper | -0.3+4.0i | 10 | 4.027 | 0% | +0.04 | +16.15 |
| p6 | lfno | stable | lfno_paper | -1.0 | 10 | 4.944 | 0% | +0.22 | +4.94 |
| p6 | lfno | stable | lfno_paper | -2.0 | 10 | 9.106 | 0% | +1.69 | +18.14 |
| p6 | lfno | stable | lfno_paper+plr10 | -0.5+1.5i | 10 | 0.195 | 10% | +0.06 | -0.09 |
| p6 | lfno | stable | lfno_paper+plr10 | -0.3+4.0i | 10 | 0.059 | 80% | +0.10 | -0.08 |
| p6 | lfno | stable | lfno_paper+plr10 | -1.0 | 10 | 0.574 | 0% | +0.31 | +0.42 |
| p6 | lfno | stable | lfno_paper+plr10 | -2.0 | 10 | 1.034 | 0% | +1.59 | +1.32 |
| p6 | lfno | warm | common | -0.5+1.5i | 10 | 0.062 | 100% | +0.10 | -0.03 |
| p6 | lfno | warm | common | -0.3+4.0i | 10 | 0.034 | 90% | +0.07 | +0.07 |
| p6 | lfno | warm | common | -1.0 | 10 | 0.056 | 70% | +0.05 | +0.00 |
| p6 | lfno | warm | common | -2.0 | 10 | 0.069 | 70% | -0.01 | +0.00 |
| p6 | lfno | warm | lfno_paper | -0.5+1.5i | 10 | 0.161 | 0% | +0.18 | -0.21 |
| p6 | lfno | warm | lfno_paper | -0.3+4.0i | 10 | 0.069 | 60% | +0.07 | -0.19 |
| p6 | lfno | warm | lfno_paper | -1.0 | 10 | 0.581 | 0% | +0.57 | +0.06 |
| p6 | lfno | warm | lfno_paper | -2.0 | 10 | 0.882 | 0% | +1.72 | +0.39 |
| p6 | tied | band | common | -0.5+1.5i | 10 | 0.080 | 60% | +0.10 | +0.02 |
| p6 | tied | band | common | -0.3+4.0i | 10 | 0.016 | 90% | +0.03 | -0.02 |
| p6 | tied | band | common | -1.0 | 10 | 0.511 | 0% | +0.46 | +0.13 |
| p6 | tied | band | common | -2.0 | 10 | 0.846 | 0% | +1.43 | +0.85 |
| p6 | tied | band | common+plr10 | -0.5+1.5i | 10 | 0.087 | 60% | +0.10 | -0.07 |
| p6 | tied | band | common+plr10 | -0.3+4.0i | 10 | 0.015 | 100% | +0.05 | -0.02 |
| p6 | tied | band | common+plr10 | -1.0 | 10 | 0.423 | 0% | +0.39 | +0.16 |
| p6 | tied | band | common+plr10 | -2.0 | 10 | 0.773 | 0% | +1.14 | +0.97 |
| p6 | tied | lno | common | -0.5+1.5i | 10 | 0.824 | 0% | +1.14 | -0.59 |
| p6 | tied | lno | common | -0.3+4.0i | 10 | 0.841 | 0% | +1.01 | -3.22 |
| p6 | tied | lno | common | -1.0 | 10 | 1.615 | 0% | +1.54 | +0.21 |
| p6 | tied | lno | common | -2.0 | 10 | 1.335 | 0% | +2.64 | +0.27 |
| p6 | tied | stable | common | -0.5+1.5i | 10 | 6.144 | 0% | -1.59 | +9.57 |
| p6 | tied | stable | common | -0.3+4.0i | 10 | 4.013 | 0% | -1.12 | +16.05 |
| p6 | tied | stable | common | -1.0 | 10 | 5.158 | 0% | -0.90 | +5.07 |
| p6 | tied | stable | common | -2.0 | 10 | 9.025 | 0% | +0.58 | +18.04 |
| p6 | tied | stable | common+plr10 | -0.5+1.5i | 10 | 0.145 | 50% | +0.00 | +0.19 |
| p6 | tied | stable | common+plr10 | -0.3+4.0i | 10 | 0.030 | 80% | +0.00 | +0.08 |
| p6 | tied | stable | common+plr10 | -1.0 | 10 | 0.579 | 0% | +0.22 | +0.42 |
| p6 | tied | stable | common+plr10 | -2.0 | 10 | 1.890 | 10% | +0.90 | +3.56 |
| p6 | tied | warm | common | -0.5+1.5i | 10 | 0.036 | 100% | +0.05 | -0.00 |
| p6 | tied | warm | common | -0.3+4.0i | 10 | 0.003 | 100% | -0.00 | +0.00 |
| p6 | tied | warm | common | -1.0 | 10 | 0.049 | 90% | +0.01 | +0.00 |
| p6 | tied | warm | common | -2.0 | 10 | 0.068 | 70% | -0.09 | +0.00 |

## deep

| system | model | init | protocol | true pole | seeds | median err | recovered | ΔRe | ΔIm |
|---|---|---|---|---|---|---|---|---|---|
| p2 | alno | lno | common | -0.5+2.0i | 5 | 0.942 | 0% | +0.50 | -1.86 |
| p2 | alno | lno | common+plr10 | -0.5+2.0i | 5 | 0.642 | 0% | +0.29 | -1.29 |
| p2 | alno | stable | common | -0.5+2.0i | 5 | 0.120 | 20% | +0.24 | -0.02 |
| p2 | alno | stable | common+plr10 | -0.5+2.0i | 5 | 0.056 | 100% | +0.11 | +0.01 |
| p2 | lfno | lno | common | -0.5+2.0i | 5 | 0.961 | 0% | +0.50 | -1.91 |
| p2 | lfno | lno | lfno_paper | -0.5+2.0i | 5 | 0.958 | 0% | +0.48 | -1.92 |
| p2 | lfno | stable | common | -0.5+2.0i | 3 | 0.081 | 67% | +0.17 | +0.01 |
| p2 | lfno | stable | common+plr10 | -0.5+2.0i | 3 | 0.030 | 100% | +0.02 | -0.03 |
| p2 | lfno | stable | lfno_paper | -0.5+2.0i | 3 | 0.209 | 0% | +0.38 | -0.20 |
| p2 | lfno | stable | lfno_paper+plr10 | -0.5+2.0i | 4 | 0.166 | 0% | +0.27 | -0.12 |
| p2 | lno_v1 | lno | common | -0.5+2.0i | 5 | 0.963 | 0% | +0.49 | -1.92 |
| p2 | lno_v1 | lno | common+plr10 | -0.5+2.0i | 5 | 0.567 | 0% | +0.42 | -1.07 |
| p4 | alno | lno | common | -0.4+3.0i | 5 | 0.964 | 0% | +0.44 | -2.88 |
| p4 | alno | lno | common | -1.0 | 5 | 0.949 | 0% | +0.95 | +0.03 |
| p4 | alno | lno | common | -0.2 | 5 | 0.728 | 0% | +0.15 | +0.02 |
| p4 | alno | lno | common+plr10 | -0.4+3.0i | 5 | 0.760 | 0% | +0.34 | -2.22 |
| p4 | alno | lno | common+plr10 | -1.0 | 5 | 0.515 | 0% | +0.47 | +0.07 |
| p4 | alno | lno | common+plr10 | -0.2 | 5 | 0.087 | 80% | -0.01 | +0.01 |
| p4 | alno | stable | common | -0.4+3.0i | 5 | 0.031 | 100% | +0.09 | +0.01 |
| p4 | alno | stable | common | -1.0 | 5 | 0.606 | 0% | +0.58 | +0.25 |
| p4 | alno | stable | common | -0.2 | 5 | 0.375 | 20% | -0.06 | +0.01 |
| p4 | alno | stable | common+plr10 | -0.4+3.0i | 5 | 0.054 | 100% | +0.15 | -0.01 |
| p4 | alno | stable | common+plr10 | -1.0 | 5 | 0.485 | 0% | +0.49 | +0.24 |
| p4 | alno | stable | common+plr10 | -0.2 | 5 | 0.097 | 60% | +0.02 | +0.01 |
| p4 | lfno | lno | common | -0.4+3.0i | 5 | 0.972 | 0% | +0.42 | -2.91 |
| p4 | lfno | lno | common | -1.0 | 5 | 0.975 | 0% | +0.97 | +0.04 |
| p4 | lfno | lno | common | -0.2 | 5 | 0.882 | 0% | +0.18 | +0.02 |
| p4 | lfno | lno | lfno_paper | -0.4+3.0i | 5 | 0.971 | 0% | +0.40 | -2.91 |
| p4 | lfno | lno | lfno_paper | -1.0 | 5 | 0.971 | 0% | +0.97 | +0.06 |
| p4 | lfno | lno | lfno_paper | -0.2 | 5 | 0.882 | 0% | +0.17 | +0.03 |
| p4 | lfno | stable | common | -0.4+3.0i | 4 | 0.027 | 100% | +0.03 | +0.04 |
| p4 | lfno | stable | common | -1.0 | 4 | 0.733 | 0% | +0.68 | +0.30 |
| p4 | lfno | stable | common | -0.2 | 4 | 0.336 | 25% | -0.03 | +0.02 |
| p4 | lfno | stable | common+plr10 | -0.4+3.0i | 5 | 0.012 | 100% | +0.02 | -0.01 |
| p4 | lfno | stable | common+plr10 | -1.0 | 5 | 0.528 | 0% | +0.50 | +0.17 |
| p4 | lfno | stable | common+plr10 | -0.2 | 5 | 0.222 | 20% | +0.00 | +0.04 |
| p4 | lfno | stable | lfno_paper | -0.4+3.0i | 5 | 0.116 | 0% | +0.34 | -0.01 |
| p4 | lfno | stable | lfno_paper | -1.0 | 5 | 0.842 | 0% | +0.83 | +0.12 |
| p4 | lfno | stable | lfno_paper | -0.2 | 5 | 0.413 | 20% | +0.07 | +0.00 |
| p4 | lfno | stable | lfno_paper+plr10 | -0.4+3.0i | 5 | 0.037 | 80% | +0.11 | +0.03 |
| p4 | lfno | stable | lfno_paper+plr10 | -1.0 | 5 | 0.799 | 20% | +0.78 | +0.09 |
| p4 | lfno | stable | lfno_paper+plr10 | -0.2 | 5 | 0.196 | 20% | +0.01 | +0.04 |
| p4 | lno_v1 | lno | common | -0.4+3.0i | 5 | 0.981 | 0% | +0.39 | -2.94 |
| p4 | lno_v1 | lno | common | -1.0 | 5 | 0.965 | 0% | +0.96 | +0.02 |
| p4 | lno_v1 | lno | common | -0.2 | 5 | 0.769 | 0% | +0.15 | +0.01 |
| p4 | lno_v1 | lno | common+plr10 | -0.4+3.0i | 5 | 0.761 | 0% | +0.36 | -2.28 |
| p4 | lno_v1 | lno | common+plr10 | -1.0 | 5 | 0.403 | 0% | +0.23 | +0.18 |
| p4 | lno_v1 | lno | common+plr10 | -0.2 | 5 | 0.066 | 80% | +0.00 | +0.01 |
| p6 | alno | lno | common | -0.5+1.5i | 5 | 0.931 | 0% | +0.49 | -1.38 |
| p6 | alno | lno | common | -0.3+4.0i | 5 | 0.971 | 0% | +0.33 | -3.88 |
| p6 | alno | lno | common | -1.0 | 5 | 0.941 | 0% | +0.94 | +0.01 |
| p6 | alno | lno | common | -2.0 | 5 | 0.971 | 0% | +1.94 | +0.02 |
| p6 | alno | lno | common+plr10 | -0.5+1.5i | 5 | 0.463 | 0% | +0.42 | -0.65 |
| p6 | alno | lno | common+plr10 | -0.3+4.0i | 5 | 0.811 | 0% | +0.42 | -3.19 |
| p6 | alno | lno | common+plr10 | -1.0 | 5 | 0.536 | 0% | +0.52 | +0.10 |
| p6 | alno | lno | common+plr10 | -2.0 | 5 | 0.812 | 0% | +1.62 | +0.13 |
| p6 | alno | stable | common | -0.5+1.5i | 5 | 0.140 | 20% | +0.21 | -0.07 |
| p6 | alno | stable | common | -0.3+4.0i | 5 | 0.015 | 100% | +0.03 | -0.03 |
| p6 | alno | stable | common | -1.0 | 5 | 0.578 | 0% | +0.56 | +0.09 |
| p6 | alno | stable | common | -2.0 | 5 | 0.802 | 0% | +1.59 | +0.41 |
| p6 | alno | stable | common+plr10 | -0.5+1.5i | 5 | 0.075 | 80% | +0.07 | +0.06 |
| p6 | alno | stable | common+plr10 | -0.3+4.0i | 5 | 0.034 | 100% | +0.06 | -0.06 |
| p6 | alno | stable | common+plr10 | -1.0 | 5 | 0.404 | 0% | +0.40 | +0.10 |
| p6 | alno | stable | common+plr10 | -2.0 | 5 | 0.773 | 0% | +1.41 | +0.16 |
| p6 | lfno | lno | common | -0.5+1.5i | 5 | 0.950 | 0% | +0.50 | -1.42 |
| p6 | lfno | lno | common | -0.3+4.0i | 5 | 0.978 | 0% | +0.32 | -3.91 |
| p6 | lfno | lno | common | -1.0 | 5 | 0.975 | 0% | +0.97 | +0.03 |
| p6 | lfno | lno | common | -2.0 | 5 | 0.988 | 0% | +1.98 | +0.04 |
| p6 | lfno | lno | lfno_paper | -0.5+1.5i | 5 | 0.945 | 0% | +0.48 | -1.42 |
| p6 | lfno | lno | lfno_paper | -0.3+4.0i | 5 | 0.978 | 0% | +0.28 | -3.91 |
| p6 | lfno | lno | lfno_paper | -1.0 | 5 | 0.970 | 0% | +0.97 | +0.05 |
| p6 | lfno | lno | lfno_paper | -2.0 | 5 | 0.985 | 0% | +1.97 | +0.06 |
| p6 | lfno | stable | common | -0.5+1.5i | 3 | 0.132 | 0% | +0.17 | -0.10 |
| p6 | lfno | stable | common | -0.3+4.0i | 3 | 0.015 | 100% | +0.03 | -0.03 |
| p6 | lfno | stable | common | -1.0 | 3 | 0.656 | 0% | +0.63 | +0.09 |
| p6 | lfno | stable | common | -2.0 | 3 | 0.833 | 0% | +1.63 | +0.33 |
| p6 | lfno | stable | common+plr10 | -0.5+1.5i | 4 | 0.039 | 100% | +0.01 | -0.02 |
| p6 | lfno | stable | common+plr10 | -0.3+4.0i | 4 | 0.009 | 100% | -0.02 | -0.02 |
| p6 | lfno | stable | common+plr10 | -1.0 | 4 | 0.527 | 0% | +0.47 | +0.13 |
| p6 | lfno | stable | common+plr10 | -2.0 | 4 | 0.743 | 0% | +1.35 | +0.50 |
| p6 | lfno | stable | lfno_paper | -0.5+1.5i | 2 | 0.270 | 0% | +0.43 | -0.01 |
| p6 | lfno | stable | lfno_paper | -0.3+4.0i | 2 | 0.076 | 100% | +0.29 | -0.01 |
| p6 | lfno | stable | lfno_paper | -1.0 | 2 | 0.788 | 0% | +0.79 | +0.01 |
| p6 | lfno | stable | lfno_paper | -2.0 | 2 | 0.921 | 0% | +1.83 | +0.21 |
| p6 | lfno | stable | lfno_paper+plr10 | -0.5+1.5i | 4 | 0.204 | 25% | +0.28 | -0.08 |
| p6 | lfno | stable | lfno_paper+plr10 | -0.3+4.0i | 4 | 0.040 | 100% | +0.16 | +0.00 |
| p6 | lfno | stable | lfno_paper+plr10 | -1.0 | 4 | 0.748 | 25% | +0.74 | +0.06 |
| p6 | lfno | stable | lfno_paper+plr10 | -2.0 | 4 | 0.897 | 0% | +1.79 | +0.08 |
| p6 | lno_v1 | lno | common | -0.5+1.5i | 5 | 0.958 | 0% | +0.48 | -1.44 |
| p6 | lno_v1 | lno | common | -0.3+4.0i | 5 | 0.979 | 0% | +0.32 | -3.91 |
| p6 | lno_v1 | lno | common | -1.0 | 5 | 0.940 | 0% | +0.94 | +0.01 |
| p6 | lno_v1 | lno | common | -2.0 | 5 | 0.973 | 0% | +1.95 | +0.01 |
| p6 | lno_v1 | lno | common+plr10 | -0.5+1.5i | 5 | 0.225 | 0% | +0.35 | -0.20 |
| p6 | lno_v1 | lno | common+plr10 | -0.3+4.0i | 5 | 0.617 | 0% | +0.17 | -2.47 |
| p6 | lno_v1 | lno | common+plr10 | -1.0 | 5 | 0.445 | 20% | +0.43 | +0.05 |
| p6 | lno_v1 | lno | common+plr10 | -2.0 | 5 | 0.678 | 0% | +1.16 | +0.30 |

## pole travel

|mu_final − mu_init| over all learned poles and seeds; budget = sum(lr × steps) (Adam moves a parameter by at most ≈ lr per step). ΔIm: median signed change.

| setting | system | model | init | protocol | median travel | max | budget | frac ≥ 0.9 budget | median ΔRe | median ΔIm |
|---|---|---|---|---|---|---|---|---|---|---|
| deep | p2 | alno | stable | common | 3.88 | 3.95 | 4.00 | 72% | +0.03 | -3.88 |
| deep | p2 | alno | stable | common+plr10 | 19.00 | 32.33 | 39.96 | 0% | +0.02 | -19.00 |
| deep | p2 | lfno | stable | common | 3.85 | 3.95 | 4.00 | 63% | +0.02 | -3.85 |
| deep | p2 | lfno | stable | common+plr10 | 14.29 | 32.37 | 39.96 | 0% | +0.01 | -14.29 |
| deep | p2 | lfno | stable | lfno_paper | 3.48 | 3.56 | 3.57 | 81% | +0.15 | -3.47 |
| deep | p2 | lfno | stable | lfno_paper+plr10 | 18.94 | 29.46 | 35.71 | 0% | +0.15 | -18.94 |
| deep | p4 | alno | stable | common | 3.87 | 3.94 | 4.00 | 72% | +0.03 | -3.87 |
| deep | p4 | alno | stable | common+plr10 | 19.16 | 32.36 | 39.96 | 0% | +0.01 | -19.16 |
| deep | p4 | lfno | stable | common | 3.87 | 3.95 | 4.00 | 69% | +0.02 | -3.87 |
| deep | p4 | lfno | stable | common+plr10 | 17.98 | 32.36 | 39.96 | 0% | +0.03 | -17.97 |
| deep | p4 | lfno | stable | lfno_paper | 3.47 | 3.56 | 3.57 | 80% | +0.16 | -3.47 |
| deep | p4 | lfno | stable | lfno_paper+plr10 | 17.17 | 29.46 | 35.71 | 0% | +0.16 | -17.17 |
| deep | p6 | alno | stable | common | 3.87 | 3.96 | 4.00 | 70% | +0.03 | -3.87 |
| deep | p6 | alno | stable | common+plr10 | 19.12 | 32.33 | 39.96 | 0% | +0.02 | -19.12 |
| deep | p6 | lfno | stable | common | 3.88 | 3.95 | 4.00 | 71% | +0.02 | -3.88 |
| deep | p6 | lfno | stable | common+plr10 | 17.08 | 32.36 | 39.96 | 0% | +0.04 | -17.08 |
| deep | p6 | lfno | stable | lfno_paper | 3.48 | 3.56 | 3.57 | 82% | +0.17 | -3.47 |
| deep | p6 | lfno | stable | lfno_paper+plr10 | 17.49 | 29.46 | 35.71 | 0% | +0.15 | -17.49 |
| layer | p2 | alno | band | common | 0.62 | 2.67 | 4.00 | 0% | -0.30 | +0.01 |
| layer | p2 | alno | band | common+plr10 | 1.60 | 2.87 | 39.96 | 0% | -0.41 | -0.92 |
| layer | p2 | alno | stable | common | 3.38 | 4.00 | 4.00 | 45% | -0.53 | -3.19 |
| layer | p2 | alno | stable | common+plr10 | 19.08 | 32.78 | 39.96 | 0% | -0.44 | -19.08 |
| layer | p2 | alno | warm | common | 0.27 | 0.96 | 4.00 | 0% | +0.04 | +0.06 |
| layer | p2 | lfno | band | common | 0.70 | 2.95 | 4.00 | 0% | -0.47 | -0.06 |
| layer | p2 | lfno | band | common+plr10 | 1.80 | 3.90 | 39.96 | 0% | -0.39 | -0.83 |
| layer | p2 | lfno | band | lfno_paper | 1.07 | 2.93 | 3.57 | 0% | -0.40 | -0.93 |
| layer | p2 | lfno | band | lfno_paper+plr10 | 1.75 | 3.23 | 35.71 | 0% | -0.43 | -1.54 |
| layer | p2 | lfno | stable | common | 3.11 | 4.10 | 4.00 | 35% | -1.32 | -2.77 |
| layer | p2 | lfno | stable | common+plr10 | 15.24 | 31.64 | 39.96 | 0% | -1.53 | -15.14 |
| layer | p2 | lfno | stable | lfno_paper | 3.55 | 3.64 | 3.57 | 80% | -0.62 | -3.51 |
| layer | p2 | lfno | stable | lfno_paper+plr10 | 19.24 | 29.37 | 35.71 | 0% | -0.55 | -19.22 |
| layer | p2 | lfno | warm | common | 0.25 | 0.67 | 4.00 | 0% | +0.05 | -0.00 |
| layer | p2 | lfno | warm | lfno_paper | 0.48 | 1.32 | 3.57 | 0% | -0.14 | -0.02 |
| layer | p2 | tied | band | common | 0.90 | 2.95 | 4.00 | 0% | -0.40 | -0.08 |
| layer | p2 | tied | band | common+plr10 | 1.93 | 3.13 | 39.96 | 0% | -0.39 | -1.13 |
| layer | p2 | tied | stable | common | 3.87 | 4.77 | 4.00 | 70% | -1.33 | -3.53 |
| layer | p2 | tied | stable | common+plr10 | 18.18 | 31.41 | 39.96 | 0% | -0.45 | -17.98 |
| layer | p2 | tied | warm | common | 0.26 | 0.86 | 4.00 | 0% | -0.01 | -0.01 |
| layer | p4 | alno | band | common | 0.68 | 3.44 | 4.00 | 0% | -0.18 | -0.27 |
| layer | p4 | alno | band | common+plr10 | 1.04 | 3.95 | 39.96 | 0% | -0.17 | -0.74 |
| layer | p4 | alno | stable | common | 3.76 | 4.28 | 4.00 | 62% | -0.30 | -3.73 |
| layer | p4 | alno | stable | common+plr10 | 18.71 | 32.82 | 39.96 | 0% | -0.26 | -18.70 |
| layer | p4 | alno | warm | common | 0.24 | 2.53 | 4.00 | 0% | -0.02 | -0.00 |
| layer | p4 | lfno | band | common | 0.65 | 2.07 | 4.00 | 0% | -0.25 | -0.27 |
| layer | p4 | lfno | band | common+plr10 | 1.09 | 3.77 | 39.96 | 0% | -0.18 | -0.57 |
| layer | p4 | lfno | band | lfno_paper | 1.37 | 3.91 | 3.57 | 8% | -0.15 | -1.35 |
| layer | p4 | lfno | band | lfno_paper+plr10 | 1.46 | 4.40 | 35.71 | 0% | -0.12 | -1.42 |
| layer | p4 | lfno | stable | common | 3.25 | 3.99 | 4.00 | 42% | -0.70 | -3.13 |
| layer | p4 | lfno | stable | common+plr10 | 17.10 | 30.10 | 39.96 | 0% | -1.09 | -16.85 |
| layer | p4 | lfno | stable | lfno_paper | 3.50 | 3.60 | 3.57 | 85% | -0.07 | -3.50 |
| layer | p4 | lfno | stable | lfno_paper+plr10 | 20.31 | 29.40 | 35.71 | 0% | -0.16 | -20.30 |
| layer | p4 | lfno | warm | common | 0.18 | 0.74 | 4.00 | 0% | -0.02 | -0.00 |
| layer | p4 | lfno | warm | lfno_paper | 0.43 | 2.46 | 3.57 | 0% | +0.00 | -0.02 |
| layer | p4 | tied | band | common | 0.73 | 3.20 | 4.00 | 0% | -0.27 | -0.27 |
| layer | p4 | tied | band | common+plr10 | 1.35 | 3.90 | 39.96 | 0% | -0.18 | -1.23 |
| layer | p4 | tied | stable | common | 3.36 | 3.96 | 4.00 | 38% | -0.66 | -3.25 |
| layer | p4 | tied | stable | common+plr10 | 18.62 | 33.05 | 39.96 | 0% | -0.74 | -18.29 |
| layer | p4 | tied | warm | common | 0.14 | 1.07 | 4.00 | 0% | -0.02 | +0.00 |
| layer | p6 | alno | band | common | 0.33 | 1.53 | 4.00 | 0% | -0.20 | -0.09 |
| layer | p6 | alno | band | common+plr10 | 0.63 | 2.82 | 39.96 | 0% | -0.25 | -0.22 |
| layer | p6 | alno | stable | common | 3.83 | 4.84 | 4.00 | 70% | -0.54 | -3.71 |
| layer | p6 | alno | stable | common+plr10 | 18.00 | 32.40 | 39.96 | 0% | -0.23 | -18.00 |
| layer | p6 | alno | warm | common | 0.27 | 1.91 | 4.00 | 0% | -0.00 | +0.00 |
| layer | p6 | lfno | band | common | 0.41 | 1.26 | 4.00 | 0% | -0.26 | -0.11 |
| layer | p6 | lfno | band | common+plr10 | 0.66 | 3.00 | 39.96 | 0% | -0.33 | -0.09 |
| layer | p6 | lfno | band | lfno_paper | 1.38 | 3.18 | 3.57 | 0% | -0.02 | -1.37 |
| layer | p6 | lfno | band | lfno_paper+plr10 | 1.75 | 4.44 | 35.71 | 0% | +0.03 | -1.73 |
| layer | p6 | lfno | stable | common | 3.63 | 4.13 | 4.00 | 55% | -0.82 | -3.49 |
| layer | p6 | lfno | stable | common+plr10 | 18.37 | 32.24 | 39.96 | 0% | -0.61 | -18.36 |
| layer | p6 | lfno | stable | lfno_paper | 3.50 | 3.63 | 3.57 | 92% | -0.19 | -3.49 |
| layer | p6 | lfno | stable | lfno_paper+plr10 | 20.25 | 29.38 | 35.71 | 0% | -0.12 | -20.25 |
| layer | p6 | lfno | warm | common | 0.20 | 1.50 | 4.00 | 0% | +0.02 | +0.00 |
| layer | p6 | lfno | warm | lfno_paper | 1.22 | 2.97 | 3.57 | 0% | +0.27 | -0.00 |
| layer | p6 | tied | band | common | 0.46 | 1.47 | 4.00 | 0% | -0.28 | -0.10 |
| layer | p6 | tied | band | common+plr10 | 0.89 | 4.28 | 39.96 | 0% | -0.33 | -0.44 |
| layer | p6 | tied | stable | common | 3.78 | 5.02 | 4.00 | 63% | -1.21 | -3.49 |
| layer | p6 | tied | stable | common+plr10 | 19.94 | 32.60 | 39.96 | 0% | -0.66 | -19.86 |
| layer | p6 | tied | warm | common | 0.25 | 1.22 | 4.00 | 0% | -0.04 | +0.00 |
