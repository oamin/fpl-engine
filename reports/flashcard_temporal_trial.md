# Temporal error

The reader is the locked vision model. A goal week counts as exact only
when that appearance is inside the highlighted set. The next columns
are the other fixture in the same gameweek, one gameweek away, and two
gameweeks away. The shares are the mean across players, which is the
weighting of the capture score. Distance 0 in that weighting is the
top-10% score.

Wasserstein distance is computed per player on the gameweek axis, after
normalizing the actual goals and the predicted mass. It is the mean
number of gameweeks the mass has to move. Uniform spreads the same
total evenly. Permuted keeps the predicted spikes and shuffles which
week they land on.

Train: 2022-23, 2023-24, 2024-25. Test: 2025-26.
Gameweek-19 top 10/20/50: 24.7%, 41.5%, 80.1%
Players: 197. Mean hidden appearances: 18.8.
Locked default: 24.7%, 41.5%, 80.1%.

## Top slice of each player's weeks

Gameweeks 20–38. A player needs eight hidden appearances and one goal.

| highlights | exact | same GW | 1 GW | 2 GW | 3+ GW | within 1 | misses within 1 | misses within 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vision top 10% | 24.7% | 0.9% | 17.3% | 13.0% | 44.0% | 43.0% | 24.2% | 41.5% |
| minutes top 10% | 21.6% | 0.8% | 20.9% | 14.4% | 42.2% | 43.3% | 27.7% | 46.1% |
| random top 10% | 10.4% | 0.6% | 18.6% | 14.6% | 55.8% | 29.6% | 21.5% | 37.8% |
| vision top 20% | 41.5% | 1.4% | 23.7% | 13.6% | 19.9% | 66.6% | 42.9% | 66.0% |
| vision top 50% | 80.1% | 0.7% | 13.7% | 4.9% | 0.6% | 94.5% | 72.2% | 97.1% |

## Probability mass

Goal counts against expected goals, then goal bars against the probability of a bar.

| mass | mean GW | median GW | players ≤ 1 | players ≤ 2 | mass on a goal GW | mass within 1 | mass within 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| expected goals | 4.29 | 3.98 | 1.5% | 13.2% | 19.0% | 38.7% | 51.4% |
| uniform | 4.86 | 4.95 | 0.0% | 8.1% | 13.1% | 30.7% | 43.4% |
| permuted expected goals | 4.90 | 4.88 | 0.0% | 6.1% | 12.9% | 30.7% | 43.5% |
| bar probability | 4.29 | 3.94 | 1.5% | 12.2% | 19.0% | 38.7% | 51.4% |
| uniform bars | 4.86 | 4.95 | 0.0% | 8.1% | 13.1% | 30.7% | 43.4% |
| permuted bar probability | 4.90 | 4.88 | 0.0% | 6.1% | 12.9% | 30.7% | 43.5% |

## Ranked bars

The fixture budget from the calibrated decoder. Random draws the same number
of bars on that player's weeks with minutes.

| bars | exact | same GW | 1 GW | 2 GW | 3+ GW | within 1 | misses within 1 | misses within 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ranked | 17.9% | 0.3% | 7.7% | 3.0% | 71.0% | 25.9% | 9.7% | 13.4% |
| same count, random weeks | 13.0% | 0.2% | 8.7% | 4.4% | 73.7% | 21.9% | 10.2% | 15.2% |

## Assists

Players with eight hidden appearances and one assist: 238.
Vision top 10% exact 20.2%, within 1 39.5%, and 24.3% of the misses are within 1.
Expected-assist transport: model 4.67 gameweeks, uniform 5.16, permuted 5.19.

## Remainders from gameweek 1

| revealed | players | exact | within 1 | misses within 1 | random within 1 | model GW | uniform GW |
|---|---:|---:|---:|---:|---:|---:|---:|
| 80% | 124 | 23.2% | 50.3% | 35.4% | 34.2% | 1.84 | 2.17 |
| 20% | 265 | 24.2% | 44.2% | 26.4% | 28.0% | 6.41 | 7.42 |

The 80% row keeps a player with four hidden appearances, because that hidden
run is short. The 20% row keeps the eight-appearance rule.

## What the distances say

On gameweeks 20–38 the top 10% is exact for 24.7% of a player's goal weeks. Another 0.9% are the other fixture in that gameweek and 17.3% are one gameweek away. Of the weeks outside the top 10%, 24.2% sit within one gameweek and 41.5% sit within two. The largest share is farther out: 44.0% are three or more gameweeks from the nearest highlighted week.

A random top 10% already leaves 21.5% of its misses within one gameweek. Minutes leaves 27.7%. Vision leaves 24.2%. The exact week is where vision is ahead of minutes (24.7% against 21.6%). The misses that remain are spread about as widely as the misses of a minutes ranking.

The probability mass agrees. Matching expected goals to the actual goals costs 4.29 gameweeks on average. A flat spread costs 4.86 and a permutation of the same spikes costs 4.90. For 1.5% of players the move is one gameweek or less. Expected goals and the probability of a bar have the same mean transport, so a brace is not what moves the mass.

The ranked fixture bars are thinner. They land on 17.9% of a player's goal weeks. Of the other weeks, 9.7% are within one gameweek of that player's bar, against 10.2% when the same number of bars is placed at random. 71.0% are three or more gameweeks away. This average counts a player who was given no bar as a miss on every goal week, which is why it sits below the earlier event-weighted recall.

The capture gate stays at 24.7%.
Wall clock: 80 seconds on CPU.

