# E4: power of the calibrated order test against a missed true pole

Per group: the `foh_eivo` report (exactly the true poles) minus one pole, refitted (EIV-weighted FOH), against the same reference as the order test (the run with the lowest full-model S). Ratio = ΔS / (κ χ²_Δp(0.99)); > 1 rejects the wrong report. κ = 1 (χ²), the pooled κ_g of `foh_eivc`, and the calibrated κ_o (16 EIV draws from the dropped-pole candidate). Reported model: its own calibrated ratio.

| condition | groups | reported model: ratio (max) | dropped pole | rejected χ² / pooled / calibrated | ratio calibrated (min) | κ_o (range) |
|---|---|---|---|---|---|---|
| p4_ms16_T20_n1e-2 | 4 | 0.51 | -0.4+3i | 4 / 4 / 4 of 4 | 7.65e+06 | 1.19–1.42 |
| | | | -0.2 | 4 / 4 / 4 of 4 | 2.33e+06 | 1.21–1.77 |
| | | | -1 | 4 / 4 / 4 of 4 | 2.18e+06 | 1.32–2.04 |
| p4_ms16_T20_n1e-2_u1e-1 | 8 | 0.91 | -0.4+3i | 8 / 8 / 8 of 8 | 374 | 174–340 |
| | | | -0.201 | 8 / 8 / 8 of 8 | 585 | 27.4–65.9 |
| | | | -0.993 | 8 / 8 / 8 of 8 | 563 | 25.7–67.4 |
| p4_ms16_T20_n1e-2_u1e-2 | 4 | 0.34 | -0.4+3i | 4 / 4 / 4 of 4 | 2.82e+06 | 1–1.38 |
| | | | -0.2 | 4 / 4 / 4 of 4 | 1.12e+06 | 1.01–2.39 |
| | | | -1 | 4 / 4 / 4 of 4 | 8.35e+05 | 1.28–2.96 |
| p4_ms16_T20_n1e-2_u3e-1 | 4 | 0.63 | -0.415+3.01i | 4 / 4 / 4 of 4 | 5.59 | 2.23e+03–3.2e+03 |
| | | | -0.207 | 4 / 4 / 4 of 4 | 7.85 | 467–705 |
| | | | -0.892 | 4 / 4 / 4 of 4 | 8.92 | 463–674 |
| p4_ms16_T20_n1e-2_u3e-2 | 4 | 0.31 | -0.4+3i | 4 / 4 / 4 of 4 | 3.92e+05 | 1.39–3.07 |
| | | | -0.2 | 4 / 4 / 4 of 4 | 1.28e+05 | 1.26–2.99 |
| | | | -1 | 4 / 4 / 4 of 4 | 1.45e+05 | 1.94–3.16 |
| p6_e1_T20_n1e-2 | 4 | 0.38 | -0.3+4i | 4 / 4 / 4 of 4 | 7.85e+05 | 1.63–1.9 |
| | | | -0.5+1.5i | 4 / 4 / 4 of 4 | 8.02e+05 | 1.74–1.87 |
| | | | -1 | 4 / 4 / 4 of 4 | 1.7e+03 | 1–1.13 |
| | | | -1.97 | 4 / 4 / 4 of 4 | 1.73e+03 | 1–1.12 |
| p6_e1_T20_n1e-2_u1e-2 | 4 | 0.44 | -0.5+1.5i | 4 / 4 / 4 of 4 | 4.48e+05 | 1.38–2 |
| | | | -0.3+4i | 4 / 4 / 4 of 4 | 4e+05 | 1.43–2.41 |
| | | | -1 | 4 / 4 / 4 of 4 | 692 | 1–1.81 |
| | | | -1.99 | 4 / 4 / 4 of 4 | 754 | 1–1.61 |
| p6_e1_T20_n1e-2_u3e-2 | 4 | 0.38 | -0.5+1.5i | 4 / 4 / 4 of 4 | 4.71e+04 | 2.61–3.6 |
| | | | -0.301+4i | 4 / 4 / 4 of 4 | 4.32e+04 | 2.6–4.84 |
| | | | -0.997 | 4 / 4 / 4 of 4 | 114 | 1.66–2.28 |
| | | | -2.03 | 4 / 4 / 4 of 4 | 112 | 2.07–3.16 |
