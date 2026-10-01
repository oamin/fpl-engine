# Stage 4 — Player baseline + residual

- Rows (≥60′, ≥3 prior apps): **6973**
- Corr(baseline, points): **0.107**
- Corr(baseline, residual): **-0.282** (≈0 = calibrated)
- Top−bottom baseline decile points: **0.92**
- Mean |residual|: **2.41**

## By position

- DEF: n=2698 corr(base,pts)=0.100 mean_resid=0.413
- FWD: n=691 corr(base,pts)=0.166 mean_resid=0.708
- GKP: n=648 corr(base,pts)=0.042 mean_resid=-0.089
- MID: n=2936 corr(base,pts)=0.105 mean_resid=0.711

## Outputs

- `data/processed/player_baseline.csv`

Baseline = expanding mean of prior total_points; z = residual / prior resid std.
