# Baudot odds / minutes weight (precision & recall)

Same mixer, learnable Tversky, FOV, seed. Odds bits or minutes bits zeroed on every row. Judged on holdout **precision and recall** at 20%, not volume-matched balanced_score alone.

Gate locked mean is 0.218. This replay's intact arm is the within-run control.

## League presence at 20%

| arm | goals a/p/h | goals P | goals R | assists a/p/h | assists P | assists R | mean score |
|---|---|---:|---:|---|---:|---:|---:|
| intact | 730/701/194 | 0.277 | 0.266 | 707/693/114 | 0.165 | 0.161 | 0.218 |
| no_odds | 730/559/154 | 0.275 | 0.211 | 707/581/78 | 0.134 | 0.110 | 0.182 |
| no_mins | 730/613/128 | 0.209 | 0.175 | 707/615/76 | 0.124 | 0.107 | 0.115 |

## Goals at 20% by position (a/p/h · P · R)

| arm | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| intact | 0/0/0 · P — · R — | 103/0/0 · P — · R 0.000 | 394/370/83 · P 0.224 · R 0.211 | 233/331/111 · P 0.335 · R 0.476 |
| no_odds | 0/0/0 · P — · R — | 103/0/0 · P — · R 0.000 | 394/136/21 · P 0.154 · R 0.053 | 233/423/133 · P 0.314 · R 0.571 |
| no_mins | 0/0/0 · P — · R — | 103/1/0 · P 0.000 · R 0.000 | 394/403/60 · P 0.149 · R 0.152 | 233/209/68 · P 0.325 · R 0.292 |

## Average positional appearance (20% remainder)

Rate = share of present player-GW rows with ≥1 event. Painted rate is the model's thr-0.5 positive rate on that pool.

| arm | pos | n | mean mins | goal rate act | goal rate paint | assist rate act | assist rate paint |
|---|---|---:|---:|---:|---:|---:|---:|
| intact | GKP | 1168 | 46.2 | 0.000 | 0.000 | 0.003 | 0.000 |
| intact | DEF | 5413 | 41.7 | 0.019 | 0.000 | 0.033 | 0.000 |
| intact | MID | 7240 | 34.5 | 0.054 | 0.051 | 0.060 | 0.050 |
| intact | FWD | 1868 | 33.0 | 0.125 | 0.177 | 0.047 | 0.177 |
| no_odds | GKP | 1168 | 46.2 | 0.000 | 0.000 | 0.003 | 0.000 |
| no_odds | DEF | 5413 | 41.7 | 0.019 | 0.000 | 0.033 | 0.000 |
| no_odds | MID | 7240 | 34.5 | 0.054 | 0.019 | 0.060 | 0.020 |
| no_odds | FWD | 1868 | 33.0 | 0.125 | 0.226 | 0.047 | 0.232 |
| no_mins | GKP | 1168 | 46.2 | 0.000 | 0.000 | 0.003 | 0.000 |
| no_mins | DEF | 5413 | 41.7 | 0.019 | 0.000 | 0.033 | 0.000 |
| no_mins | MID | 7240 | 34.5 | 0.054 | 0.056 | 0.060 | 0.056 |
| no_mins | FWD | 1868 | 33.0 | 0.125 | 0.112 | 0.047 | 0.110 |

## Call

**Both strips carry placement weight. Minutes move precision; odds move who gets the recall.**

- Blanking minutes: goals P 0.277→0.209, R 0.266→0.175, mean score 0.218→0.115. FWD goal recall falls 0.476→0.292.
- Blanking 1X2 odds: goals P almost flat (0.277→0.275) but R 0.266→0.211 and mean 0.218→0.182. The loss is concentrated on MID (recall 0.211→0.053, painted 370→136) while FWD is over-painted (331→423).
- DEF goal recall stays 0 on every arm. Neither odds nor minutes, as currently drawn, unfreeze DEF.

So the 1X2 strip is doing real work for MID placement, and minutes are doing real work for precision. They are not a substitute for an opponent card: DEF stays blank, and odds-off paint just slides from MID onto FWD. Coupling and a role-aware loss are still open; this says they should be added on top of odds and minutes, not instead of them.
