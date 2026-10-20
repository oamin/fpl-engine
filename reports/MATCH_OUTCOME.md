# Match outcome — `M` and rolls

Part of the `score_xp_v2` decomposition. Module: `src/markets/match_outcome.py`.

## `match_outcome_probs`

Builds the Dixon–Coles scoreline grid

\[
M_{ij} = P(\text{home}=i,\ \text{away}=j)
\]

from bookie **AH + OU 2.5**, and **BTTS when available** (weight 0 if missing — same rule as historical FD screens).

| Input | Role |
| --- | --- |
| Fair `p_ah`, `ah_line` | Asian handicap cover |
| Fair `p_over_25` | Goals over 2.5 |
| Fair `p_btts` (optional) | Both teams to score |

Also: `match_outcome_probs_from_decimals` (Shin de-vig) and `match_outcome_probs_from_lambdas` (Opta / scenario λ → `M` without refitting markets).

Side moments (`xg` / `xcs` / `xgc`) sit on the returned object for player xP.

## `match_outcome_roll`

One categorical draw from `M`: every positive-mass cell is in the window; the
draw is not an independent Poisson outside `M`. Tail index `k_max+` is reported
as `goals = k_max` with `*_tail=True`.

```python
from src.markets.match_outcome import match_outcome_probs, match_outcome_roll
import random

probs = match_outcome_probs(p_ah=0.55, ah_line=-0.5, p_over_25=0.52)
roll = match_outcome_roll(probs, random.Random(0))
# roll.goals_home, roll.goals_away, roll.p
```

## Not yet rewired

Live pick / short-horizon still call `fit_scoreline` / λ±band paths directly.
Next decomposition step: route those call sites through this module, then
revisit cascade short/long rolls.
