# Baudot stepped-GW RL vs locked full-remainder gate

Same Baudot card + `BlockTokenMixer` encoder. Training is self-critical REINFORCE on next-GW Bernoulli paint; reward = channel `balanced_score` (hits × volume). Holdout = teacher-forced stepped table (thr 0.5), same cells as the locked GATE.

- Gate: `baudot_row_mixer_learnable_tversky` · mean **0.218**
- Parked stepped supervised (Tversky): mean **0.184** (Δ -0.034)
- Warm-start: 15 epochs next-GW Tversky, then SCST
- Entropy coef: 0.01

## Scoreboard

| arm | params | epochs | α | β | mean score | Δ gate | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate (full remainder one-shot) | — | — | 0.447 | 0.553 | 0.218 | +0.000 | — |
| stepped_next_gw (supervised, parked) | — | — | — | — | 0.184 | -0.034 | — |
| rl_step_bal | 38263 | 23 | 0.448 | 0.552 | 0.199 | -0.019 | 511.9 |

## Goals at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| rl_step_bal | 0/0/0 | 103/0/0 | 394/593/118 | 233/203/73 |

## Assists at 20% by position (actual / painted / hits)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| rl_step_bal | 4/0/0 | 181/0/0 | 434/591/114 | 88/203/31 |

## Holdout table (same cells as gate)

| arm | revealed | channel | actual | painted | hits | actual ±1 | painted ±1 | score | Δ gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gate | 20% | goals | 730 | 701 | 194 | 268 | 377 | 0.260 | +0.000 |
| gate | 20% | assists | 707 | 693 | 114 | 153 | 232 | 0.160 | +0.000 |
| gate | 40% | goals | 544 | 557 | 151 | 214 | 299 | 0.268 | +0.000 |
| gate | 40% | assists | 531 | 551 | 89 | 123 | 174 | 0.159 | +0.000 |
| gate | 60% | goals | 360 | 383 | 105 | 145 | 205 | 0.266 | +0.000 |
| gate | 60% | assists | 351 | 377 | 66 | 90 | 126 | 0.169 | +0.000 |
| gate | 80% | goals | 199 | 195 | 60 | 82 | 101 | 0.298 | +0.000 |
| gate | 80% | assists | 192 | 191 | 32 | 45 | 59 | 0.166 | +0.000 |
| rl_step_bal | 20% | goals | 730 | 796 | 191 | 253 | 420 | 0.230 | -0.031 |
| rl_step_bal | 20% | assists | 707 | 794 | 145 | 178 | 309 | 0.172 | +0.012 |
| rl_step_bal | 40% | goals | 544 | 618 | 142 | 191 | 317 | 0.215 | -0.053 |
| rl_step_bal | 40% | assists | 531 | 617 | 109 | 137 | 233 | 0.163 | +0.005 |
| rl_step_bal | 60% | goals | 360 | 397 | 91 | 119 | 201 | 0.218 | -0.048 |
| rl_step_bal | 60% | assists | 351 | 396 | 73 | 91 | 152 | 0.173 | +0.004 |
| rl_step_bal | 80% | goals | 199 | 198 | 46 | 59 | 95 | 0.231 | -0.068 |
| rl_step_bal | 80% | assists | 192 | 198 | 38 | 46 | 75 | 0.189 | +0.023 |

## Phase note

- Best checkpoint phase: `warm_tversky` (epoch 13)
- Warm-start best: 0.199
- SCST-phase best: 0.167

## Morning call

**Park.** Best holdout 0.199 is from the **warm_tversky** phase (epoch 13), not a SCST win. Warm best 0.199; SCST best 0.167. SCST did not improve on next-GW Tversky and does not beat the full-remainder gate (0.218, Δ -0.019). DEF goals/assists stay blank.

Disposition: park — SCST does not beat warm Tversky or the gate.

Read order:
1. Mean vs gate (primary).
2. Was the best checkpoint SCST or warm Tversky?
3. DEF goals/assists a/p/h at 20%.
4. Overpaint / GKP noise on the holdout table.
