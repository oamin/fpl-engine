# Stage 33 — fast XI screen

Twelve scores, one setting each, on 2025/26 gameweeks 5–38.
Clearly behind means 100 or more points under expected points.
No transfer climb in this batch. `within_pos` rescales each position inside the gameweek, so a formation can win on tail shape rather than points.

| score | XI points | vs xp | queued |
|---|---:|---:|---|
| agree_min | 2104 | +21 | yes |
| starter | 2098 | +15 | yes |
| minutes | 2095 | +12 | yes |
| upside | 2092 | +9 | yes |
| premium | 2083 | +0 | yes |
| xp | 2083 | +0 | — |
| within_pos | 2065 | -18 | yes |
| goals_tilt | 1995 | -88 | yes |
| agree_max | 1989 | -94 | yes |
| no_deduction | 1980 | -103 | no |
| attack | 1963 | -120 | no |
| split | 1902 | -181 | no |
| per_million | 1787 | -296 | no |

Queued by the kill gap: agree_min, starter, minutes, upside, premium, within_pos, goals_tilt, agree_max.

The Co-PI agreed the screen. Premium matched expected points exactly, and it preserves order when expected points are non-negative, so it is not a new score. The next transfer climb is only agree_min, starter, minutes, and upside. within_pos, goals_tilt, and agree_max stay off that climb.
