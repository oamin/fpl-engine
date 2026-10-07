# Crowd opening scores

Built from the Gameweek 1 fifteens in `reports/crowd_openings.md`. Each fifteen is scored twice on the published rule. The hold sets the margin high enough that a legal squad never transfers. The climb uses the published hold margin (1.25). The chip map is empty. The decision score is `score_xp`. Later weeks in the three-week window use that fixture's opening price. A double is one fixture on the decision score. Actual points are Vaastav `total_points` for the chosen eleven, with the captain extra, automatic substitutes, and hit deductions.

2022/23, 2023/24, 2024/25, and 2025/26 are all in this run. The chip map is empty, so 2024/25 does not need an Assistant Manager. Purchase price is the Gameweek 1 value. The bank is £100.0m minus that cost. A hold that transfers is a failure, and its sum is not a baseline. The best climb is named inside that season.

These are crowd templates from Gameweek 1 ownership. The ownership file is scraped after the gameweek. Nothing here replaces a published climb total.

Gemini kept the diagnostics ([crowd opening scores](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). A climb below its hold is an allowed result of the same scorer. The 2025/26 template hold stays a failure, and its climb stays. The 2024/25 third gap stays. The best climb is named inside the season.

## 2022/23

| Squad | Cost | Hold | Climb | Climb − hold | Climb transfers | Climb hits |
| --- | --- | --- | --- | --- | --- | --- |
| Template | £99.5m | 1786 | 1735 | -51 | 44 | 9 |
| Premium | £100.0m | 1731 | 1680 | -51 | 45 | 9 |
| Next | £100.0m | 1531 | 1714 | +183 | 43 | 7 |
| Third | £96.5m | 1536 | 1689 | +153 | 42 | 7 |

Best climb in 2022/23: Template at 1735.

## 2023/24

| Squad | Cost | Hold | Climb | Climb − hold | Climb transfers | Climb hits |
| --- | --- | --- | --- | --- | --- | --- |
| Template | £100.0m | 1859 | 2105 | +246 | 37 | 2 |
| Premium | £100.0m | 1533 | 1953 | +420 | 39 | 5 |
| Next | £99.5m | 1666 | 2092 | +426 | 39 | 4 |
| Third | £93.5m | 1563 | 1945 | +382 | 39 | 3 |

Best climb in 2023/24: Template at 2105.

## 2024/25

| Squad | Cost | Hold | Climb | Climb − hold | Climb transfers | Climb hits |
| --- | --- | --- | --- | --- | --- | --- |
| Template | £100.0m | 1961 | 2086 | +125 | 36 | 2 |
| Premium | £100.0m | 1853 | 2043 | +190 | 37 | 1 |
| Next | £100.0m | 2000 | 2041 | +41 | 35 | 1 |
| Third | £97.5m | 1325 | 1953 | +628 | 38 | 3 |

Best climb in 2024/25: Template at 2086.

## 2025/26

| Squad | Cost | Hold | Climb | Climb − hold | Climb transfers | Climb hits |
| --- | --- | --- | --- | --- | --- | --- |
| Template | £99.5m | failure | 1988 | failure | 37 | 3 |
| Premium | £100.0m | 1670 | 2084 | +414 | 36 | 1 |
| Next | £100.0m | 2183 | 2171 | -12 | 35 | 2 |
| Third | £98.5m | 1680 | 1999 | +319 | 36 | 2 |

Template — hold: hold transferred 3 times, 0 hits.

Best climb in 2025/26: Next at 2171.

## Reading

The best climb is the template in 2022/23, 2023/24, and 2024/25, and Next in 2025/26.

In 2022/23 the template and the premium both finish 51 below their hold. Those two fifteens share eight players. Next and third share no players with the template, and the same transfer rule adds 183 and 153.

In 2023/24 every climb gains. The template climb is 2105. Next is 13 behind, at 2092. The gains on Next and the premium are 426 and 420.

In 2024/25 the template climb is 2086. Next's hold is 2000 and the climb adds 41. Third's hold is 1325 and the climb brings it to 1953, still the lowest climb that season.

In 2025/26 Next is the best climb at 2171. The hold of that fifteen is 2183. The template hold failed. Marc Guiu moves from Sunderland to Chelsea in Gameweek 4, and the fifteen then has four Chelsea players. The published rule spends the three banked free transfers that week and takes no hit. The earlier weeks of that path are a hold. The raw sum is 1724, and that sum is not the comparison. The template climb scores 1988.

2022/23 has 37 played weeks. Gameweek 7 has no clubs. The other three seasons have 38. No chip was played. Every climb covered its played weeks.

Week rows are in `data/processed/crowd_opening_score_weeks.csv`. Season totals are in `data/processed/crowd_opening_scores.csv`.
