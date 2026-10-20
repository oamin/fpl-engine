# Team-form FDR — diagnostic only

Expanding mean of club Σ FPL points (assistant managers stripped).  
FDR = `team_mu_own − team_mu_opp` on schedule fixtures.

**Not** a decision board. L4 stays priced **t:t+2** hard-stop. Fills and terminals that pushed this into the MILP are parked (`reports/layer4_mupost_beyond_d.md`, `reports/layer4_mupost_beyond_d_adapt.md`, `reports/layer4_mupost_terminal.md`).

```bash
python -m src.strategy.team_form_diagnostic --season 2025-26 --gw 20 --write
```

Calibration screen (Spearman vs future margins): `reports/team_form_fdr.md`.
