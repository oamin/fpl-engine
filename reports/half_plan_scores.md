# Half-season chip scores

Each crowd Gameweek 1 fifteen is climbed again with `plan_half` solved at every deadline before the transfer. The decision score is `score_xp`, one fixture. The empty-chip climb is the stored total in `data/processed/crowd_opening_scores.csv`. The lift is the chip climb minus that stored total, so a chip that changes the squad is part of the gap.

The seasons are 2022/23, 2023/24, and 2025/26. 2024/25 is absent until Assistant Manager is in the rules module. 2022/23 and 2023/24 are scored with the 2026 half-season wallet: one Wildcard, Free Hit, Bench Boost, and Triple Captain in each half. 2025/26 is the season that wallet matches. Free Hit needs a lead of 12 on the decision week. Wildcard needs a lead of 16 across the half, on the eleven. Bench Boost and Triple Captain play when this week is the best week left and the added points are above zero. Gameweek 1 cannot play Wildcard or Free Hit.

The decision week prices Wildcard and Free Hit from one legal fifteen, bought with the bank and the sell prices. A later Free Hit is the best starting eleven on that step's scores. The first gameweeks have no buy pool, so that eleven is taken from the rows in hand. Weeks after the third priced step reuse that step. A week with no clubs scores zero and stays out of the copied step. Realized points are Vaastav `total_points`, with the captain, automatic substitutes, the bench on Bench Boost, the extra captain copy on Triple Captain, and the hit deductions.

The published climb file is unchanged. These totals stay in this side report.

Gemini kept these totals on 2026-10-04 ([chip scores](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). All 12 climbs finished the same weeks as the stored empty-chip climb. The Gameweek 4 wildcard is the first week the buy pool exists, because a player needs three prior appearances. Bench Boost and Triple Captain cluster in the next few weeks because weeks past the third priced step reuse that step. Free Hit, which needs a lead of 12 on the legal fifteen, is played once.

## 2022/23

2026 wallet on this season

| Squad | Empty climb | Chip climb | Chip − empty | Chips |
| --- | --- | --- | --- | --- |
| Template | 1735 | 1639 | -96 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW21 wildcard; GW23 triple_captain; GW25 bench_boost |
| Premium | 1680 | 1610 | -70 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW21 wildcard; GW23 bench_boost; GW24 triple_captain |
| Next | 1714 | 1657 | -57 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW21 wildcard; GW23 bench_boost; GW24 triple_captain |
| Third | 1689 | 1649 | -40 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW21 wildcard; GW23 bench_boost; GW24 triple_captain |

Best chip lift in 2022/23: Third at -40.

## 2023/24

2026 wallet on this season

| Squad | Empty climb | Chip climb | Chip − empty | Chips |
| --- | --- | --- | --- | --- |
| Template | 2105 | 2090 | -15 | GW4 wildcard; GW6 triple_captain; GW8 bench_boost; GW20 wildcard; GW22 triple_captain; GW23 bench_boost |
| Premium | 1953 | 2053 | +100 | GW4 wildcard; GW6 triple_captain; GW8 bench_boost; GW20 wildcard; GW22 triple_captain; GW23 bench_boost |
| Next | 2092 | 1876 | -216 | GW4 wildcard; GW6 triple_captain; GW8 bench_boost; GW20 wildcard; GW22 triple_captain; GW23 bench_boost |
| Third | 1945 | 2073 | +128 | GW4 wildcard; GW6 triple_captain; GW8 bench_boost; GW20 wildcard; GW22 triple_captain; GW24 bench_boost; GW30 free_hit |

Best chip lift in 2023/24: Third at +128.

## 2025/26

matched wallet

| Squad | Empty climb | Chip climb | Chip − empty | Chips |
| --- | --- | --- | --- | --- |
| Template | 1988 | 2187 | +199 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW21 bench_boost; GW23 triple_captain; GW29 wildcard |
| Premium | 2084 | 2214 | +130 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW22 bench_boost; GW23 triple_captain; GW24 wildcard |
| Next | 2171 | 2205 | +34 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW22 bench_boost; GW24 wildcard; GW26 triple_captain |
| Third | 1999 | 2176 | +177 | GW4 wildcard; GW5 bench_boost; GW6 triple_captain; GW22 bench_boost; GW23 triple_captain; GW24 wildcard |

Best chip lift in 2025/26: Template at +199.

In 2022/23 every squad finishes below its empty climb. The best lift is Third at −40. The highest chip total is Next at 1657, and the Template empty climb is 1735. This season uses the 2026 wallet.

In 2023/24 the best lift is Third at +128. The highest chip total is Template at 2090, 15 below its empty climb of 2105. Template, Premium, and Next play the same six chip weeks, and the lifts on those three are −15, +100, and −216.

In 2025/26 every squad gains. This is the matched wallet. The best lift is Template at +199. The highest chip total is Premium at 2214. Next, the best empty path at 2171, gains 34 and finishes at 2205. The four chip climbs take 0 hits.

Transfers rise on every row, by 13 to 26. A wildcard in Gameweek 4 is in all 12 squads. Free Hit is Gameweek 30 on the 2023/24 Third squad. These totals stay in this side report.

A double stays one fixture on the decision score and inside the half plan. A Free Hit after the decision week has no budget cap, so a later Free Hit can look stronger than the fifteen the climb is able to buy. Weeks past the three priced steps reuse the last step, so a double or a blank that has not been announced is absent.

Season totals are in `data/processed/half_plan_scores.csv`.
