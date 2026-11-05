# Next-GW event sequence dig

Discrete component-channel history (bar-plot ticks) vs fixture-only baseline.
**No** expanding inputs, **no** padding, **no** `total_points` in the history vector.

- K = **5** (first label GW = K+1)
- Split: `train ['2022-23', '2023-24', '2024-25'] → test 2025-26`
- Train rows: **57701**; test rows: **23478**
- History channels: `minutes, pts_appear, pts_goals, pts_assists, pts_cs, pts_gc, pts_saves, pts_defcon, pts_bonus, pts_cards, pts_own_goal, pts_pens`

## Arms

- **B**: position one-hot + was_home + opponent one-hot
- **C**: B + flat last-K discrete channel vector
- Model: sklearn MLPClassifier (64,32) + StandardScaler; one head per target

## Mean metrics (test)

| arm | mean log loss | mean Brier |
| --- | ---: | ---: |
| B | 0.3343 | 0.0698 |
| C | 0.2594 | 0.0483 |
| Δ(C−B) | -0.0748 | -0.0215 |

Negative Δ ⇒ sequence arm better.

## Per target (test)

| target | base rate | B ll | C ll | Δll | B brier | C brier | Δbrier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| y_goal | 0.032 | 0.1376 | 0.1126 | -0.0250 | 0.0316 | 0.0286 | -0.0030 |
| y_assist | 0.031 | 0.1418 | 0.1190 | -0.0228 | 0.0316 | 0.0291 | -0.0025 |
| y_appear60 | 0.265 | 0.5791 | 0.2682 | -0.3108 | 0.1952 | 0.0826 | -0.1126 |
| y_defcon | 0.051 | 0.7061 | 0.7061 | +0.0000 | 0.0511 | 0.0511 | +0.0000 |
| y_bonus | 0.041 | 0.1755 | 0.1419 | -0.0337 | 0.0406 | 0.0374 | -0.0032 |
| y_cs | 0.073 | 0.2655 | 0.2087 | -0.0568 | 0.0688 | 0.0611 | -0.0076 |

## Reading

Sequence arm **improves** mean log loss by 0.0748 vs fixture-only.

Most of the mean lift is **Appear60** (Δll −0.311). G / A / bonus / CS improve
modestly; DefCon is flat here (both arms ≈0.706 ll — weak head under this MLP).
Interpretation: discrete minutes/appearance history carries signal beyond
position×opponent; attacking-event sequence value is smaller in this setup.
Not a scorer promote.

Reproduce:

```bash
PYTHONPATH=. python3 -m src.eval.next_event_sequence_dig
```

