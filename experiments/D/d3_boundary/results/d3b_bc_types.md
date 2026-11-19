# (D) Boundary types: learned boundary-channel laws in k

D = 0.02. True laws: Dirichlet channel 2Dk/L (∝ k), Neumann flux channel 2/L (constant). Range over data seeds 0–4.

| case | model | test / oracle | D̂ | channel 0 law error | channel 1 law error |
|---|---|---|---|---|---|
| DD | bc | 1.00–1.02 | 0.01976–0.02006 | 1.5e-02–2.7e-01 | 1.5e-02–2.7e-01 |
| DD | nobc | 79.18–101.69 | – | – | – |
| NN | bc | 1.00–1.00 | 0.02000–0.02005 | 8.0e-03–7.4e-02 | 8.0e-03–7.4e-02 |
| NN | nobc | 239.62–inf | – | – | – |
| DN | bc | 1.00–1.02 | 0.01981–0.02007 | 7.2e-02–6.0e-01 | 4.9e-02–4.0e-01 |
| DN | nobc | 187.04–282.58 | – | – | – |
