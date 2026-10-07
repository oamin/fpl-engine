Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Disagreement mechanisms

No change is made to score_xp.

The disagreement classification is a post-hoc description formulated after observing summary transfer characteristics; it does not replace the unconditional contrast or conditional intervals.

Each flag compares the two incoming players. A tie or a missing value does not fire. Realised points are not an input. Bonus points are not a class.

Disagreement weeks: 51. Weeks in 2023-24, 2024-25, and 2025-26: 28. That count is not a separate contrast.

The four slices of the minutes-versus-attack partition are mutually exclusive and sum to the 51 disagreement weeks; overlapping category gaps must not be summed.

## Overlapping flags

| flag | weeks | one-week gap | reading | three-week gap | reading |
|---|---:|---|---|---|---|
| minutes_to_exp | 24 | -1.1364 [-3.7284, +1.5466] | the interval covers zero | -4.1818 [-8.2739, -0.2727] | the interval stays below zero |
| attack_to_xp | 31 | -0.1481 [-2.1491, +1.7787] | the interval covers zero | -4.4815 [-9.3713, +0.1481] | the interval covers zero |
| cheaper_xp | 31 | -1.1935 [-2.9032, +0.4516] | the interval covers zero | -3.5806 [-6.9371, -0.5153] | the interval stays below zero |
| cs_defcon_to_xp | 31 | +0.3333 [-1.7778, +2.5204] | the interval covers zero | -0.8519 [-4.4824, +2.6676] | the interval covers zero |
| goals_assists_to_xp | 34 | +0.0625 [-1.8758, +2.1875] | the interval covers zero | -4.6562 [-8.7188, -0.4992] | the interval stays below zero |
| form_to_exp | 49 | -0.6327 [-2.0821, +1.0000] | the interval covers zero | -4.1020 [-7.0219, -1.0204] | the interval stays below zero |
| fixture_to_xp | 31 | -0.1481 [-2.1491, +1.7787] | the interval covers zero | -4.4815 [-9.3713, +0.1481] | the interval covers zero |
| other | 2 | undefined | undefined | undefined | undefined |

## Minutes versus attack

The table asks whether the realised gap sits on weeks where the expected-points buy has more expected minutes, or on weeks where the score_xp buy has more attack strength.

| slice | weeks | one-week gap | reading | three-week gap | reading |
|---|---:|---|---|---|---|
| both | 13 | undefined | undefined | undefined | undefined |
| minutes_only | 11 | undefined | undefined | undefined | undefined |
| attack_only | 18 | +1.0833 [-1.9167, +4.1687] | the interval covers zero | -2.6667 [-11.3354, +5.6687] | the interval covers zero |
| neither | 9 | undefined | undefined | undefined | undefined |

A slice interval that excludes zero is not a win and does not change score_xp.

On these 51 weeks the fixture flag fired on the same weeks as the attack flag, so it is not a second channel in this table.

The both slice and the minutes-only slice are undefined, because each has only one season with at least 5 weeks. The attack-only slice covers zero on both horizons.

No winner is declared between `score_xp` and `score_exp_points`, and `score_xp` is unchanged.

Gemini kept the flag definitions and reviewed the table ([mechanism](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).

Bootstrap 1000, seed 0. A season with fewer than 5 weeks in a group is omitted. Fewer than two such seasons leaves the interval undefined.
