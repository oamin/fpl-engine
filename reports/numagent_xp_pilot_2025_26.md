# `score_xp_v2` pilot — 2025/26

Pilot only. Not a gated four-season result. No holdout season. No climb. Candidate score column: **`score_xp_v2`** (method working title was Numagent 2.0).

## Protocol

- Pool: players with **minutes > 60** and an FD opening AH+OU fit.
- Minutes: **actual** minutes, **σ_m = 0** (stack test, not minutes tagging).
- Odds: football-data opening AH + OU 2.5 (BTTS weight 0).
- Priors / rates / α: walk-forward from GW `< t` only.
- Primary metric: Gaussian log score, σ = 3.0, on the **point** mean.
- Secondary: Numagent `xp_se` coverage only (not in the log-score).
- Baseline: published `score_xp` from `xp_engine.csv` on the same player-fixtures.

## Headline

- Player-fixtures: **7502** across GWs **2–38** (20 pool rows lacked an FD fit; join to `score_xp` kept 7502 of 7710 FD-matched rows).
- Mean GW Δlog-score (numagent − score_xp), all GWs: **+0.0047** [−0.0152, +0.0216] (single-season GW bootstrap, B=1000, seed 0) — covers zero.
- Same, GWs 5–38: **+0.0173** [+0.0020, +0.0318] — stays above zero.
- GWs 2–4 cold start: mean Δ **−0.138** (thin rates/priors). From GW 5, 23/34 weeks are positive.
- Row MAE: numagent **2.378** vs score_xp **2.347** (score_xp slightly sharper).
- Row bias (μ − points): numagent **−0.138** vs score_xp **−0.320** (numagent less low-biased).

## Secondary — Numagent SE

- Mean `xp_se`: **0.522**.
- `xp_se` is a **stability / reliability** band on the assembled mean (share, α, DefCon, bonus, card uncertainty in quadrature) — **not** a predictive σ for noisy FPL points. Larger `xp_se` ⇒ less reliable xP. Points coverage is not the calibration target; do not put `xp_se` in the log-score denominator.
- (Diagnostic only) fraction of rows with |points − xp| ≤ 1·SE / 2·SE: **0.109** / **0.249** — expected to be low; not a fail.

## Artefacts

- `data/processed/numagent_xp_pilot_2025_26.csv`
- `data/processed/numagent_xp_pilot_2025_26_gw.csv`
- `data/processed/numagent_xp_pilot_2025_26_alpha.csv`
- `data/processed/numagent_xp_pilot_2025_26_summary.json`
- `data/plots/numagent_xp_pilot_2025_26_dlog.png`

Positive Δ means Numagent has the higher log score. This pilot does **not** replace published `score_xp` and is **not** a gated four-season screen.
