# Opta−FD gap across closed seasons

Blind Opta ranking λ vs FD AH+OU centreline · GW8–38 · N=20 · wall 15.4s

## Verdict

- Pooled Opta−FD = **-0.024** (n=2498, sd=0.385) — **do not lean on the pooled mean**; season signs differ.
- Seasons with Opta > FD: 2022-23, 2025-26.
- Seasons with Opta < FD: 2023-24, 2024-25.
- Home: Opta usually **below** FD (book warmer at home).
- Away: Opta **above** FD in most seasons (place-blind away heat vs book).

## By season

| season | n | Opta−FD | sd | MAE | Opta−act | FD−act | corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 640 | +0.085 | 0.384 | 0.317 | +0.083 | -0.002 | 0.708 |
| 2023-24 | 620 | -0.167 | 0.356 | 0.298 | -0.247 | -0.080 | 0.794 |
| 2024-25 | 620 | -0.082 | 0.341 | 0.269 | -0.002 | +0.080 | 0.784 |
| 2025-26 | 618 | +0.066 | 0.400 | 0.308 | +0.121 | +0.054 | 0.705 |
| pooled | 2498 | -0.024 | 0.385 | 0.298 | -0.011 | +0.013 | 0.725 |

## By season × venue

| season | ha | Opta−FD | Opta | FD | act | Opta−act | FD−act |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | H | -0.068 | 1.500 | 1.567 | 1.581 | -0.082 | -0.014 |
| 2022-23 | A | +0.237 | 1.500 | 1.263 | 1.253 | +0.247 | +0.010 |
| 2023-24 | H | -0.320 | 1.428 | 1.749 | 1.845 | -0.417 | -0.097 |
| 2023-24 | A | -0.014 | 1.416 | 1.430 | 1.494 | -0.077 | -0.063 |
| 2024-25 | H | -0.251 | 1.475 | 1.726 | 1.526 | -0.051 | +0.200 |
| 2024-25 | A | +0.086 | 1.485 | 1.399 | 1.439 | +0.047 | -0.039 |
| 2025-26 | H | -0.099 | 1.511 | 1.610 | 1.524 | -0.013 | +0.086 |
| 2025-26 | A | +0.232 | 1.520 | 1.288 | 1.265 | +0.255 | +0.023 |

## Reproduce

Rows: `data/processed/cascade_long_fd_opta_gap_all_seasons_rows.csv`

