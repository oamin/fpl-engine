# Correlated draws (scoreline → player xP)

Monte Carlo through Dixon–Coles λ shocks + optional Dirichlet shares.
Same match factor moves a whole team (and the opposition) together.

- Scenarios: **500**
- Max |mean(draw) − point μ|: **0.0900**
- Corr home M1–M2 total xP: **+0.111**
- Corr home M1–M2 goal xP: **-0.102**
- Corr home M1 vs away GK: **-0.672**

Independent `xp_se` cannot encode these; store the draw matrix for planning.

JSON: `data/processed/correlated_draws_demo.json`

