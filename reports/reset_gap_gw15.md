# Gameweeks 1–5, reset to the squad he held

Each week starts from the fifteen ojaminFC owned before that deadline, with that week's bank and free transfers. The search is the published rule: hold margin 1.25, switch penalty 1.0, a three-week opening horizon, and the early score capped at 6. The model plays no wildcard, free hit, or bench boost. Gameweek 1 triples the model's captain, because that is the chip he played. The squad is then discarded. The next week starts from the fifteen he actually fielded.

The search maximises a discounted three-week value. The score is one week of realised points, so a hit taken for a later week is charged here and those later points are not in the total. Gameweeks 4 and 5 look ahead to Gameweeks 6 and 7 on those fixtures' opening prices. Gameweek 6 is not in the sum.

The bar was locked before this total was read. A sum above 0 and at least 3 of 5 weeks non-negative is a gain on these five decisions. Anything else is the model not beating these five decisions.

The sum is -18 and 2 of 5 weeks are non-negative. The model did not beat these five decisions. Five weeks remain too few to call the rule reliable. Gemini kept the split on 2026-10-05 ([reset gap](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)).

| GW | Model | ojaminFC | Gap | Captain | Transfers | Lineup | Hits | Residual | Horizon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 61 | 62 | -1 | +0 | +9 | -2 | -8 | +0 | 1,2,3 |
| 2 | 98 | 108 | -10 | +0 | -10 | +0 | +0 | +0 | 2,3,4 |
| 3 | 53 | 53 | +0 | +0 | +0 | +0 | +0 | +0 | 3,4,5 |
| 4 | 79 | 79 | +0 | -5 | +8 | -3 | +0 | +0 | 4,5,6 |
| 5 | 41 | 48 | -7 | +0 | -7 | +0 | +0 | +0 | 5,6,7 |
| Total | 332 | 350 | -18 | -5 | +0 | -5 | -8 | +0 | |

Captain is the extra copy only. Gameweek 1 adds two extra copies, because both sides play Triple Captain. A player who finished in one scoring eleven is a transfer when he was bought or sold, and a lineup choice when both fifteens contained him. Hits are his charge minus the model's. The residual is the rebuilt week minus the official total. The five pieces sum to the gap.

The rebuilt week matches the official total in every week.

## Gameweek 1

Captaincy on B.Fernandes vs Haaland added +0.
Transferring Schade, Verbruggen in for Martinez added +9 net points.
Starting the rest of the eleven over Davis cost -2.
Hits cost -8.

## Gameweek 2

Captaincy on B.Fernandes vs B.Fernandes added +0.
Holding Martinez and transferring Szoboszlai in for Cherki, Lammens cost -10 net points.
Starting the same eleven added +0.
Hits added +0.

## Gameweek 3

Captaincy on Haaland vs Haaland added +0.
Transferring Enzo in for Cherki added +0 net points.
Starting the same eleven added +0.
Hits added +0.

## Gameweek 4

Captaincy on Khalaili vs João Pedro cost -5.
Holding Gibbs-White and transferring Enzo, Khalaili in for Cherki, Rogers added +8 net points.
Starting the rest of the eleven over Ødegaard cost -3.
Hits added +0.

## Gameweek 5

Captaincy on Haaland vs Haaland added +0.
Holding M.Sangaré and transferring Gakpo, Silva in for Barnes, Calvert-Lewin, Rogers cost -7 net points.
Starting the same eleven added +0.
Hits added +0.
