# Baudot gate autopsy — DEF blank and assist miss

Question: why does the locked full-remainder mixer paint **0 DEF goals** and underperform on assists?
Method: retrain the gate-family `BlockTokenMixer` + learnable Tversky (checkpoint mean score 0.218, α=0.447, β=0.553), then read soft ink on the 20% remainder holdout (15689 player-GW rows). Not a new learner trial.

## Verdict

1. **DEF goals = 0 is a role collapse, not a hard-threshold accident.** Soft goal ink on DEF is ~0.006 on both scoring and blank weeks; max bit ≈0.01. No DEF week crosses even 0.05, so every threshold recovers **0** of 103 DEF goals.
2. **The card still carries usable cues on DEF goal weeks** (more minutes on, slightly higher past goal rate), but the shared paint head does not turn them into separable ink.
3. **Assist “underpaint” on the gate table is mostly a placement / mix problem, not total volume.** Overall assist volume ≈ actual; DEF assists are fully blanked, FWD assists are heavily overpainted.

## Soft ink by position (20% remainder)

| pos | event rate | mean soft on event | mean soft on blank | mean max on event | % event max≥0.5 | painted weeks | hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| *goals* | | | | | | | |
| GKP | 0.000 | — | 0.0064 | — | — | 0 | 0 |
| DEF | 0.019 | 0.0063 | 0.0062 | 0.0098 | 0.0000 | 0 | 0 |
| MID | 0.054 | 0.0705 | 0.0211 | 0.2057 | 0.2107 | 370 | 83 |
| FWD | 0.125 | 0.1420 | 0.0453 | 0.4191 | 0.4764 | 331 | 111 |
| *assists* | | | | | | | |
| GKP | 0.003 | 0.0066 | 0.0067 | 0.0100 | 0.0000 | 0 | 0 |
| DEF | 0.033 | 0.0067 | 0.0065 | 0.0104 | 0.0000 | 0 | 0 |
| MID | 0.060 | 0.0574 | 0.0215 | 0.1660 | 0.1498 | 363 | 65 |
| FWD | 0.047 | 0.1590 | 0.0515 | 0.4699 | 0.5568 | 330 | 49 |

### Reading

- GKP/DEF goal soft mass sits at the paint-bias floor (~sigmoid(−3) after training stays tiny).
- MID/FWD show real separation (event max 0.21 / 0.42 vs blank 0.05 / 0.13).
- DEF event rate is 1.9% vs FWD 12.5%. Under global Tversky with β≥α, painting any DEF week is almost all FP cost, so the shared head learns **role → never paint** for DEF.

## Threshold search cannot rescue DEF

| pos | best thr (goals) | tp / fp / fn | recall |
|---|---:|---|---:|
| GKP | — | no events | — |
| DEF | 0.05 | 0/0/103 | 0.000 |
| MID | 0.05 | 166/763/228 | 0.421 |
| FWD | 0.15 | 135/287/98 | 0.579 |

DEF: even thr=0.05 paints **0** weeks — every DEF `g_max` is below 0.05. This is dead ranking, not a 0.5 cutoff mis-set.

True DEF goal weeks sit at mean percentile 0.25 of DEF `g_max` (p50 0.22): mild relative order among tiny scores, but every value stays below 0.05, so nothing is paintable.

## Cues on the card (DEF goal weeks vs DEF blanks)

| cue | DEF goal week | DEF blank week |
|---|---:|---:|
| mins_on | 5.6505 | 2.8168 |
| odds_left | 4.8447 | 4.5829 |
| odds_right | 4.4078 | 4.6179 |
| past_g_rate | 0.0347 | 0.0169 |
| past_g_sum | 0.2816 | 0.1384 |
| g_max | 0.0098 | 0.0087 |
| g_prob | 0.0063 | 0.0062 |

n_event=103, n_blank=5310. Minutes and past rate move; model `g_max` barely does.

## Assists

- DEF assists: same collapse as goals (soft ~0.0065, 0 painted, 181 actual).
- FWD assists: event max 0.470 but 330 painted weeks vs 88 actual — large overpaint; gate assist volume is propped up here.
- MID assists: weaker separation than MID goals (event max 0.166 vs 0.206); fewer hits per painted week.

So the familiar assist score gap (~0.16 vs goals ~0.26) is **missed DEF + weaker MID placement + FWD assist overpaint**, not a global refusal to paint assists.

## What this rules out / in

| hypothesis | call |
|---|---|
| 0.5 threshold too harsh for DEF | **out** — no DEF score reaches 0.05 |
| No minutes/odds signal on DEF goal weeks | **out** — minutes differ clearly |
| Shared paint head + rare DEF rate + β≥α Tversky → DEF never-paint | **in** |
| Role paint heads alone fix it | **out** (prior trial; soft collapse likely persists without new info) |
| FN-heavy DEF aux | paints volume, not ranking (prior 103/78/5) |
| Need fixture/share signal beyond 1X2 fill for DEF events | **open / best next** |

Artifacts: `/opt/cursor/artifacts/baudot_def_assist_autopsy.json`.
