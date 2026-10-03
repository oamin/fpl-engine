# Opening-price horizon, Gameweeks 1–5

Diagnostic only. The published `score_xp` climb is unchanged. This arm starts from ojaminFC's Gameweek 1 fifteen. The current week keeps that week's closing-price xp. Gameweeks inside the three-week hold use the opening 1X2 (`AvgH`, `Avg>2.5`). Share and minutes stay on the deadline row. Gemini 3.8 Flash locked that split.

Published path: **327** points, hits 4.
Opening horizon: **331** points, hits 0.
ojaminFC: **350**.

João Pedro at the Gameweek 3 deadline, share frozen, opening λ on the later weeks:

- Gameweek 3: 2.94
- Gameweek 4: 5.36
- Gameweek 5: 3.82

Published squad still has him in Gameweek 3: False.
Opening-horizon squad still has him in Gameweek 3: True.
Still in the squad in Gameweek 4: True. Gameweek 5: True.

| GW | Published | Open horizon | ojaminFC | Published transfers | Open transfers |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 66 | 66 | 62 | none | none |
| 2 | 109 | 102 | 108 | in Mbeumo; out Gibbs-White | in Szoboszlai; out Cherki |
| 3 | 47 | 47 | 53 | in Thiago, Enzo; out João Pedro, Cherki | in Enzo; out Ødegaard |
| 4 | 62 | 80 | 79 | in Khalaili; out Shaw | none |
| 5 | 43 | 36 | 48 | none | none |

Opening prices matched 100 club-gameweeks. A week with a fixture and no opening price uses that club's earlier scoring rate. A blank week is zero. No Odds API call was made.

Gemini reviewed this run. Gameweek 3 is the case the rule was built for: 2.94 against Arsenal, 5.36 against Hull, and João Pedro stays. The five-week total is not a pass or a fail. Gameweek 2 and Gameweek 5 moved the other way. A full historical free-transfer season is what can accept or reject the arm. The formula was not changed after these weeks.

