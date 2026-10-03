# Opening-price horizon, Gameweeks 1–5

Diagnostic only. The published `score_xp` formula is unchanged. A missing week no longer borrows a later week's score. This arm starts from ojaminFC's Gameweek 1 fifteen. The current week keeps that week's closing-price xp. Gameweeks inside the three-week hold use the opening 1X2 (`AvgH`, `Avg>2.5`). Share and minutes stay on the deadline row.

Published path: **313** points, hits 0.
Opening horizon: **303** points, hits 0.
ojaminFC: **350**.

João Pedro at the Gameweek 3 deadline, share frozen, opening λ on the later weeks:

- Gameweek 3: 2.94
- Gameweek 4: 5.36
- Gameweek 5: 3.82

Published squad still has him in Gameweek 3: True.
Opening-horizon squad still has him in Gameweek 3: True.
Still in the squad in Gameweek 4: True. Gameweek 5: True.

| GW | Published | Open horizon | ojaminFC | Published transfers | Open transfers |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 68 | 68 | 62 | none | none |
| 2 | 89 | 82 | 108 | in Mbeumo; out Gibbs-White | in Szoboszlai; out Cherki |
| 3 | 43 | 37 | 53 | in Enzo; out Cherki | in Enzo; out Ødegaard |
| 4 | 73 | 80 | 79 | in Khalaili; out Shaw | none |
| 5 | 40 | 36 | 48 | none | none |

Opening prices matched 100 club-gameweeks. A week with a fixture and no opening price uses that club's earlier scoring rate. A blank week is zero. No Odds API call was made. The record of 327 in `reports/live_benchmark_2026.md` is the earlier run, which could fill a missing week from a later score.

Gemini accepted the stub fix. With past weeks only, the published rule also keeps João Pedro and takes no hit. The Gameweek 3 sale in that 327 record was the future score, not the Arsenal projection by itself. The opening-odds horizon still keeps him and scores 303 on this window, behind the corrected published path at 313.

