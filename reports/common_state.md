Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. 2026-27 is not in this comparison.

# Common-state transfers

The common-state test evaluates transfer selection from an identical frozen fifteen built without reference to any player score.

Each week, every scoring rule evaluates candidate transfers from the exact same unmutated squad, ensuring zero state divergence.

Primary transfer contrasts pair realised in-minus-out gains unconditionally across all gameweeks, including weeks where both rules choose the same transfer or roll.

The question is whether `score_xp` chooses a better one-free-transfer move than `score_exp_points` when the squad, the bank, and the legal set are the same. A roll scores 0 predicted and 0 realised. The construction week is the first week that fills the fifteen (2022-23 GW5 (32 transfer weeks), 2023-24 GW5 (33 transfer weeks), 2024-25 GW5 (33 transfer weeks), 2025-26 GW5 (33 transfer weeks)). It is not a transfer week. No later transfer is written onto that squad.

Centre contrast, one-week realised, `score_xp` minus expected points: -0.2901 [-0.8702, +0.3132]. the interval covers zero.

| contrast | estimate | reading |
|---|---|---|
| R1 xp − R1 exp | -0.2901 [-0.8702, +0.3132] | the interval covers zero |
| R3 xp − R3 exp | -1.6947 [-2.9162, -0.5189] | the interval stays below zero |
| R1 xp − R1 roll3 | +0.2672 [-0.9849, +1.4055] | the interval covers zero |
| R1 xp − R1 shuffled xp | +3.9771 [+2.8319, +5.1305] | the interval stays above zero |
| concordance xp − exp | +0.0137 [+0.0074, +0.0192] | the interval stays above zero |

In accordance with protocol, if a paired transfer contrast interval covers zero, the result is inconclusive and no winner is declared.

No winner is declared from this batch. The published score, the hold, and the chip map are unchanged.

### One-week xp minus expected points

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 32 | -1.34 |
| 2023-24 | 33 | +0.58 |
| 2024-25 | 33 | +0.09 |
| 2025-26 | 33 | -0.52 |

On the one-week contrast the interval covers zero, so that comparison is inconclusive.

### Three-week xp minus expected points

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 32 | -3.34 |
| 2023-24 | 33 | -1.45 |
| 2024-25 | 33 | -1.15 |
| 2025-26 | 33 | -0.88 |

On the three-week contrast the interval stays below zero, and every season mean is negative. That reading is not a decision to replace score_xp.

Of the three-week gap, the decision week averages -0.29 and the next two weeks together average -1.40. That split has no interval.

### One-week xp minus the rolling three-week mean

The rolling three-week points baseline uses shift-1 historical data only, with zero current-week information.

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 32 | -1.28 |
| 2023-24 | 33 | +0.12 |
| 2024-25 | 33 | +1.03 |
| 2025-26 | 33 | +1.15 |

### One-week xp minus shuffled xp

Shuffled `score_xp` permutes eligible finite values inside that gameweek. Points stay on the player. The squad is not rebuilt.

| season | weeks | mean |
|---|---:|---:|
| 2022-23 | 32 | +2.66 |
| 2023-24 | 33 | +3.73 |
| 2024-25 | 33 | +6.82 |
| 2025-26 | 33 | +2.67 |

The one-week contrast against shuffled score_xp stays above zero. The squad was not rebuilt.

### Pairwise concordance

Pairwise concordance measures discrimination strictly across legal moves that satisfy position, price, club quota, and eligibility constraints; weeks without decisive pairs are undefined.

Defined-week means, not themselves contrasts: `score_xp` +0.643 on 131 defined weeks; expected points +0.630 on 131 defined weeks; roll3 +0.613 on 131 defined weeks; shuffled xp +0.582 on 131 defined weeks.
Weeks where xp concordance or exp concordance is undefined: 0. Those weeks are absent from the concordance contrast only.
Share of transfer weeks where `score_xp` and expected points pick the same players, or both roll: 0.611. That share is descriptive. Those weeks stay in the primary transfer contrasts, where the gap is zero.

Pairwise concordance of score_xp minus expected points stays above zero. The gap is +0.014. The defined-week means are +0.643 and +0.630.

Neither rule rolls. The two rules pick the same players on 61% of weeks. On the weeks they differ, the one-week gap averages -0.75 and the three-week gap averages -4.35. Means on the weeks they differ are descriptive only and were not bootstrapped.

| season | both defined | mean concordance gap |
|---|---:|---:|
| 2022-23 | 32 | +0.007 |
| 2023-24 | 33 | +0.013 |
| 2024-25 | 33 | +0.014 |
| 2025-26 | 33 | +0.021 |

The difference of +5.51 points per gameweek from the prior report is an arithmetic decomposition between separate baselines, not a statistical confidence interval.

## 2022-23 diverging path

The 2022–23 deficit of −14.45 points per gameweek is a property of multi-week squad divergence, not the sum of isolated transfer gains.

This table is the earlier replay, in which each score builds its own opening squad and then transfers. It is not a common-state choice. Attack strength is the mean of that player's rows in the week.

Weeks in the diverging replay: 33. `score_xp` transfers: 32. Expected-points transfers: 32.
Mean squad-point gap (xp greedy minus exp greedy): -14.45. Sum of those gaps: -477.0.
Mean of the same weeks' isolated one-week transfer gaps: +1.24. Sum of those isolated gaps: +41.0.
On the transfers `score_xp` made, mean predicted gain +6.57, mean one-week realised +3.78, mean attack_strength in minus out +0.08. Buy positions: MID 12, FWD 9, DEF 9, GKP 2.
On the transfers expected points made, mean predicted gain +5.09, mean one-week realised +2.50, mean attack_strength in minus out +0.00. Buy positions: MID 14, DEF 7, FWD 6, GKP 5.

| GW | XP out | XP in | XP pos | XP Δ | XP realised | Exp out | Exp in | Exp pos | Exp Δ | Exp realised | squad gap |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|
| 5 | roll | roll | — | +0.00 | +0.0 | roll | roll | — | +0.00 | +0.0 | +1.0 |
| 6 | Thomas Partey | Rodrigo Bentancur | MID | +2.34 | +1.0 | Ivan Perišić | João Cancelo | DEF | +1.40 | +1.0 | -19.0 |
| 8 | Alisson Ramses Becker | Nick Pope | GKP | +3.77 | +2.0 | Robert Sánchez | José Malheiro de Sá | GKP | +5.83 | +2.0 | -16.0 |
| 9 | Danny Ings | Roberto Firmino | FWD | +1.89 | +11.0 | Rodrigo Moreno | Alexis Mac Allister | MID | +6.50 | +0.0 | -45.0 |
| 10 | Andrew Robertson | Aaron Cresswell | DEF | +3.16 | +2.0 | Allan Saint-Maximin | Marcus Rashford | MID | +5.86 | +1.0 | -4.0 |
| 11 | John Stones | Ryan Sessegnon | DEF | +3.39 | +0.0 | Emerson Leite de Souza Junior | Kieran Trippier | DEF | +5.44 | +6.0 | +2.0 |
| 12 | Aaron Ramsdale | Robert Sánchez | GKP | +3.79 | +6.0 | Erling Haaland | Roberto Firmino | FWD | +6.11 | +2.0 | +6.0 |
| 13 | Ryan Sessegnon | Nathan Aké | DEF | +3.59 | +0.0 | Luis Díaz | Phil Foden | MID | +6.40 | +1.0 | -1.0 |
| 14 | Jamie Vardy | Brennan Johnson | FWD | +1.89 | -1.0 | Harry Kane | Erling Haaland | FWD | +3.80 | -1.0 | -3.0 |
| 15 | Nathan Aké | Aymeric Laporte | DEF | +3.67 | -2.0 | Phil Foden | Miguel Almirón Rejala | MID | +6.08 | +7.0 | -37.0 |
| 16 | Rúben Gato Alves Dias | John Stones | DEF | +3.69 | +1.0 | Erling Haaland | Harry Kane | FWD | +6.50 | +10.0 | -44.0 |
| 17 | Raheem Sterling | Wilfried Zaha | MID | +7.83 | -4.0 | Joël Veltman | Fabian Schär | DEF | +4.47 | +1.0 | -25.0 |
| 18 | Roberto Firmino | Darwin Núñez Ribeiro | FWD | +6.96 | +5.0 | Roberto Firmino | Erling Haaland | FWD | +4.47 | +6.0 | -19.0 |
| 19 | Rodrigo Bentancur | Raheem Sterling | MID | +6.79 | +1.0 | Alexis Mac Allister | Andreas Hoelgebaum Pereira | MID | +7.95 | +4.0 | -38.0 |
| 20 | Jorge Luiz Frello Filho | Marcus Rashford | MID | +8.56 | +10.0 | José Malheiro de Sá | David De Gea Quintana | GKP | +4.34 | -5.0 | -10.0 |
| 21 | Raheem Sterling | Lucas Tolentino Coelho de Lima | MID | +5.58 | +2.0 | Andreas Hoelgebaum Pereira | Rodrigo Moreno | MID | +1.20 | +0.0 | -33.0 |
| 22 | Joseph Gomez | Luke Shaw | DEF | +8.42 | +6.0 | João Cancelo | Maximilian Wöber | DEF | +7.00 | +2.0 | +16.0 |
| 23 | Darwin Núñez Ribeiro | Eddie Nketiah | FWD | +11.26 | +1.0 | Dean Henderson | Ederson Santana de Moraes | GKP | +7.35 | +3.0 | -21.0 |
| 24 | Wilfried Zaha | Pablo Sarabia | MID | +7.21 | +1.0 | Rodrigo Moreno | Solly March | MID | +4.90 | +2.0 | -8.0 |
| 25 | Marcus Rashford | Cody Gakpo | MID | +16.58 | +4.0 | David De Gea Quintana | Alisson Ramses Becker | GKP | +8.98 | +12.0 | -48.0 |
| 26 | Eddie Nketiah | João Félix Sequeira | FWD | +7.83 | +2.0 | Maximilian Wöber | Ben Mee | DEF | +1.42 | -1.0 | +3.0 |
| 27 | Mason Mount | Alexis Mac Allister | MID | +9.48 | +10.0 | Ederson Santana de Moraes | David Raya Martin | GKP | +5.48 | +3.0 | -44.0 |
| 28 | Alexis Mac Allister | Gabriel Martinelli Silva | MID | +5.21 | +7.0 | Erling Haaland | Ollie Watkins | FWD | +4.23 | +5.0 | +7.0 |
| 29 | Pablo Sarabia | Dango Ouattara | MID | +10.91 | +3.0 | Miguel Almirón Rejala | Mohamed Salah | MID | +11.85 | +7.0 | -69.0 |
| 30 | João Cancelo | Harry Souttar | DEF | +5.80 | +0.0 | William Saliba | Benjamin White | DEF | +4.45 | +0.0 | -21.0 |
| 31 | Che Adams | Harry Kane | FWD | +8.04 | +2.0 | Martin Ødegaard | Gabriel Martinelli Silva | MID | +0.47 | -5.0 | -3.0 |
| 32 | João Félix Sequeira | Ivan Toney | FWD | +5.99 | +9.0 | Pascal Groß | Eberechi Eze | MID | +3.87 | +3.0 | -3.0 |
| 33 | Dango Ouattara | Pablo Sarabia | MID | +5.02 | -1.0 | Ben Mee | Alexandre Moreno Lopera | DEF | +0.58 | +1.0 | -7.0 |
| 34 | Harry Kane | Erling Haaland | FWD | +12.06 | +8.0 | Eberechi Eze | Pascal Groß | MID | +4.60 | +5.0 | +6.0 |
| 35 | Aymeric Laporte | Pedro Porro | DEF | +5.39 | +10.0 | Harry Kane | Erling Haaland | FWD | +1.17 | -6.0 | +14.0 |
| 36 | Harry Souttar | Kieran Trippier | DEF | +8.57 | +10.0 | Solly March | Joe Willock | MID | +6.87 | +7.0 | -13.0 |
| 37 | Gabriel Martinelli Silva | Alexis Mac Allister | MID | +9.23 | +6.0 | Gabriel Martinelli Silva | Alexis Mac Allister | MID | +7.58 | +6.0 | -2.0 |
| 38 | Ivan Toney | Ollie Watkins | FWD | +6.24 | +9.0 | Marcus Rashford | Martin Ødegaard | MID | +5.65 | +1.0 | +1.0 |

Widest squad gaps, xp minus exp: GW29 -69, GW25 -48, GW9 -45, GW16 -44, GW27 -44.

In 2022-23 the diverging squad gap averages -14.45. The isolated one-week transfer gap on those weeks averages +1.24. The week's in-minus-out does not account for the squad gap.

Among 2022-23 transfers with a finite attack delta, the share with a higher attack strength on the buy is 0.52 for `score_xp` (14 of 27) and 0.52 for expected points (14 of 27).

Bootstrap 1000, seed 0. A season under 20 weeks is not pooled. Closed seasons: 2022-23, 2023-24, 2024-25, 2025-26.

Gemini reviewed these diagnostics ([common-state diagnostics](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).
