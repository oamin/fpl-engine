# What the square card uses

The full painter is the gate reader, trained with seed 0. The test
cut is 20% revealed on 2025-26. Each row removes one signal and paints
again. White is the empty mark. The placement still requires minutes,
so a player on zero minutes cannot receive a bar in any of these rows.

| signal removed | goal bars painted | goal same appearance | goal within 1 GW | assist same appearance | assist within 1 GW |
|---|---:|---:|---:|---:|---:|
| intact | 760 | 196 | 264 | 105 | 180 |
| white odds | 378 | 117 | 175 | 55 | 99 |
| white venue | 724 | 190 | 257 | 102 | 176 |
| white minutes | 95 | 28 | 61 | 9 | 26 |
| white goals | 697 | 183 | 249 | 96 | 169 |
| white assists | 692 | 183 | 247 | 95 | 167 |
| white position color | 735 | 192 | 261 | 104 | 175 |
| white picture | 33 | 10 | 29 | 4 | 12 |
| zero numeric minutes | 428 | 126 | 195 | 61 | 97 |
| zero position id | 665 | 12 | 25 | 10 | 18 |
| zero matchup | 705 | 185 | 252 | 96 | 171 |
| zero opponent id | 739 | 186 | 247 | 102 | 166 |
| no other squad | 717 | 188 | 254 | 99 | 175 |

The next rows keep the intact model's fixture budgets and change
only who receives them. Minutes still have to be above zero.

| ranking | goal bars painted | goal same appearance | goal within 1 GW |
|---|---:|---:|---:|
| minutes | 760 | 6 | 14 |
| prior goals | 760 | 159 | 183 |
| position then minutes | 760 | 203 | 245 |

These rows choose their own fixture budgets. A forward is 0.18,
a midfielder 0.10, a defender 0.04, and a goalkeeper 0.

| signal | goal bars painted | goal same appearance | goal within 1 GW |
|---|---:|---:|---:|
| minutes / 90 | 6524 | 622 | 679 |
| prior goals | 1391 | 223 | 240 |
| position rate | 750 | 165 | 224 |

On a hidden appearance with minutes, the mean chance of a goal bar is 8.4%.

| position | mean goal-bar chance | appearances |
|---|---:|---:|
| GK | 0.3% | 606 |
| DEF | 4.0% | 3113 |
| MID | 10.1% | 4162 |
| FWD | 18.5% | 1161 |

Mean absolute contribution to the zero-class logit on those appearances.
The picture term is whatever remains after the additive terms.

| term | mean absolute logit |
|---|---:|
| position | 0.131 |
| minutes | 0.087 |
| club | 0.132 |
| opponent | 0.175 |
| form | 0.066 |
| picture | 2.264 |

The picture carries the score. Whitening it leaves 33 goal bars and 10 on the same appearance. The additive terms are small beside that residual.

Two signals inside the picture set the fixture. Whitening the minute squares leaves 95 goal bars and 28 on the same appearance. Whitening the odds row leaves 378 goal bars and 117. Venue, past goals, past assists, the recent scoring rate, the opponent id, and the other squad each move the same-appearance count by about ten.

The position id decides the row. Zeroing it still paints 665 goal bars, and 12 land on a goal. The position color can be whitened and the count stays at 192, because the id is passed beside the picture. On a hidden appearance with minutes the reader puts a goal bar at 0.3% for a goalkeeper, 4.0% for a defender, 10.1% for a midfielder, and 18.5% for a forward.

The placement itself refuses a bar to a player on zero minutes. Among players who did play, handing the model's fixture budgets to the longest appearance hits 6 goals. Handing them to the highest position, with minutes as the tiebreak, hits 203, against 196 for the model's own ranking. The model's bars sit within one gameweek of 264 actual goals, against 245 for that position rule.

A flat position rate with its own budgets paints 750 goal bars and hits 165. The minute fraction as its own budget paints 6524 bars, so that hit count is the result of covering almost every appearance.

This run reproduces the goal gate: 760 painted, 196 on the same appearance, 264 within one gameweek. Assists come out at 105 and 180. The stored gate has 106 and 180.
Wall clock: 77 seconds on CPU.

