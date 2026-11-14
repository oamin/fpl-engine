# Stepped overlap

The score is the overlap of the gray card with the solid card: `2 * opacity on solid dots / (all opacity + solid dots)`. The loss on a step is one minus that overlap. Row treats the next gameweek as one picture. Column gives the players who scored half the score and the blank players the other half. Inside each half every player weighs the same, so a defender who scored weighs the same as a forward who scored. A blank player scores one minus their mean opacity. Combined gives the row overlap and the column overlap equal weight.

Training steps the next week, with the true past behind it. The table reads the same overlap 1, 2, 4 and 8 weeks ahead. The weeks in between stay blank.

## Next week and further

| arm | weeks ahead | picture overlap | player overlap |
|---|---:|---:|---:|
| row | 1 | 0.220 | 0.747 |
| row | 2 | 0.198 | 0.773 |
| row | 4 | 0.169 | 0.751 |
| row | 8 | 0.217 | 0.742 |
| column | 1 | 0.166 | 0.767 |
| column | 2 | 0.140 | 0.775 |
| column | 4 | 0.136 | 0.776 |
| column | 8 | 0.163 | 0.766 |
| combined | 1 | 0.214 | 0.759 |
| combined | 2 | 0.187 | 0.778 |
| combined | 4 | 0.164 | 0.765 |
| combined | 8 | 0.208 | 0.752 |

## Event marker, one week ahead

The first goal dot and the first assist dot, one week ahead. Solid counts are that week across the 20 holdout clubs.

| arm | revealed | channel | solid | on true | on false | precision | recall | overlap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| row | 20% | goals | 29 | 19.0 | 89.1 | 0.176 | 0.656 | 0.278 |
| row | 40% | goals | 26 | 20.5 | 92.9 | 0.181 | 0.788 | 0.294 |
| row | 60% | goals | 30 | 22.3 | 85.8 | 0.206 | 0.744 | 0.323 |
| row | 80% | goals | 27 | 16.6 | 95.0 | 0.149 | 0.615 | 0.240 |
| row | 20% | assists | 29 | 14.3 | 91.4 | 0.136 | 0.494 | 0.213 |
| row | 40% | assists | 28 | 13.4 | 97.9 | 0.121 | 0.480 | 0.193 |
| row | 60% | assists | 25 | 13.3 | 92.6 | 0.125 | 0.531 | 0.203 |
| row | 80% | assists | 25 | 19.1 | 90.5 | 0.174 | 0.763 | 0.283 |
| column | 20% | goals | 29 | 25.2 | 214.2 | 0.105 | 0.868 | 0.188 |
| column | 40% | goals | 26 | 23.0 | 212.9 | 0.097 | 0.883 | 0.175 |
| column | 60% | goals | 30 | 26.6 | 201.5 | 0.117 | 0.887 | 0.206 |
| column | 80% | goals | 27 | 23.4 | 216.4 | 0.097 | 0.865 | 0.175 |
| column | 20% | assists | 29 | 17.5 | 151.9 | 0.103 | 0.604 | 0.177 |
| column | 40% | assists | 28 | 16.4 | 154.3 | 0.096 | 0.584 | 0.165 |
| column | 60% | assists | 25 | 15.7 | 147.0 | 0.096 | 0.626 | 0.167 |
| column | 80% | assists | 25 | 18.0 | 154.8 | 0.104 | 0.718 | 0.182 |
| combined | 20% | goals | 29 | 20.4 | 115.6 | 0.150 | 0.704 | 0.247 |
| combined | 40% | goals | 26 | 20.9 | 110.1 | 0.159 | 0.803 | 0.266 |
| combined | 60% | goals | 30 | 24.4 | 103.9 | 0.190 | 0.813 | 0.308 |
| combined | 80% | goals | 27 | 20.2 | 112.9 | 0.152 | 0.747 | 0.252 |
| combined | 20% | assists | 29 | 14.8 | 109.6 | 0.119 | 0.510 | 0.193 |
| combined | 40% | assists | 28 | 13.9 | 107.7 | 0.114 | 0.495 | 0.185 |
| combined | 60% | assists | 25 | 14.6 | 103.4 | 0.124 | 0.584 | 0.204 |
| combined | 80% | assists | 25 | 18.8 | 104.1 | 0.153 | 0.753 | 0.255 |

## Call

Next-week picture overlap: row **0.220**, column **0.166**, combined **0.214**. Eight weeks ahead, with the weeks between left blank: row **0.217**, column **0.163**, combined **0.208**. The overlap does not fade with distance. Each future row still carries its own opponent, odds and minutes, so the step does not have to pass the week through the blank rows. The column arm covers more of the true goal dots and puts more gray on the blank ones, and the picture overlap falls. Combined stays between the two and does not pass the row.
