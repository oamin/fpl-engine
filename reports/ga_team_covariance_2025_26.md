# Team G/A covariance — 2025-26 (fixture-level)

Per-club player×player correlation of **goals** and **assists**.
Each **fixture** is one observation; DGWs with two match rows count as two.
Diagnostic only — not a scorer law.

- Team×fixture observations: **760** (team×GW 750; DGW extras **10**)
- Min appear fixtures: **5**; min pair n: **8**
- Attack positions: FWD, MID

## Team totals (ΣG vs ΣA per fixture)

Across clubs, corr(team goals, team assists) median **+0.922** (mean +0.916; n=20).

| team | corr(ΣG,ΣA) | mean ΣG | mean ΣA | fixtures |
| --- | ---: | ---: | ---: | ---: |
| West Ham | +0.968 | 1.16 | 1.13 | 38 |
| Brentford | +0.963 | 1.39 | 1.26 | 38 |
| Chelsea | +0.963 | 1.50 | 1.45 | 38 |
| Nott'm Forest | +0.960 | 1.21 | 1.18 | 38 |
| Wolves | +0.959 | 0.66 | 0.66 | 38 |
| Brighton | +0.945 | 1.34 | 1.29 | 38 |
| Everton | +0.928 | 1.18 | 1.16 | 38 |
| Burnley | +0.924 | 0.92 | 0.95 | 38 |
| Leeds | +0.924 | 1.26 | 1.16 | 38 |
| Aston Villa | +0.924 | 1.39 | 1.37 | 38 |
| Arsenal | +0.921 | 1.76 | 1.74 | 38 |
| Sunderland | +0.918 | 1.00 | 1.05 | 38 |
| Fulham | +0.914 | 1.13 | 1.16 | 38 |
| Crystal Palace | +0.908 | 1.05 | 0.95 | 38 |
| Man City | +0.885 | 1.95 | 1.79 | 38 |
| Bournemouth | +0.883 | 1.50 | 1.29 | 38 |
| Liverpool | +0.869 | 1.61 | 1.45 | 38 |
| Newcastle | +0.867 | 1.39 | 1.11 | 38 |
| Man Utd | +0.855 | 1.74 | 1.58 | 38 |
| Spurs | +0.839 | 1.21 | 1.05 | 38 |

## Pooled player pairs

Positive frac = share of pairs with corr > 0.

| measure | n pairs | mean | median | frac pos | p10 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attack G×G | 979 | -0.013 | -0.048 | 19.9% | -0.116 | +0.181 |
| Attack A×A | 1090 | -0.013 | -0.056 | 16.3% | -0.102 | +0.169 |
| Attack cross G_i×A_j | 2083 | +0.048 | -0.039 | 31.0% | -0.096 | +0.368 |
| Attack active cross (≥3G/≥3A) | 551 | +0.088 | +0.063 | 57.5% | -0.139 | +0.385 |
| Attack active G×G | 274 | -0.007 | -0.050 | 40.1% | -0.163 | +0.197 |
| Attack G×G (bin) | 979 | -0.012 | -0.048 | 20.9% | -0.122 | +0.192 |
| Attack A×A (bin) | 1090 | -0.012 | -0.056 | 16.4% | -0.102 | +0.175 |
| Attack cross bin | 2083 | +0.048 | -0.039 | 31.3% | -0.100 | +0.367 |
| Attack G×G both-played | 723 | -0.011 | -0.083 | 26.4% | -0.183 | +0.297 |
| Attack A×A both-played | 759 | -0.014 | -0.091 | 22.0% | -0.167 | +0.328 |
| Attack cross both-played | 1497 | +0.091 | -0.060 | 41.8% | -0.155 | +0.532 |
| Attack active cross both-played | 533 | +0.118 | +0.108 | 59.8% | -0.167 | +0.490 |
| All G×G | 1868 | -0.012 | -0.048 | 16.1% | -0.105 | +0.167 |
| All cross G_i×A_j | 4272 | +0.044 | -0.039 | 26.6% | -0.091 | +0.368 |
| Nailed G×G | 180 | +0.015 | -0.039 | 28.3% | -0.115 | +0.275 |
| Nailed cross G_i×A_j | 392 | +0.070 | -0.039 | 40.1% | -0.110 | +0.384 |
| Same-player G×A | 231 | +0.035 | -0.045 | 27.7% | -0.112 | +0.346 |
| Same-player G×A bin | 231 | +0.032 | -0.048 | 27.7% | -0.114 | +0.318 |

## Strongest *active* attack cross G→A (scorer≥3G, assister≥3A)

| team | scorer | pos | assister | pos | corr | n |
| --- | --- | --- | --- | --- | ---: | ---: |
| Aston Villa | Morgan Rogers | MID | Ollie Watkins | FWD | +0.715 | 38 |
| Brentford | Igor Thiago Nascimento Rodrigues | FWD | Kevin Schade | MID | +0.678 | 38 |
| Newcastle | Harvey Barnes | MID | Bruno Guimarães Rodriguez Moura | MID | +0.660 | 38 |
| Nott'm Forest | Chris Wood | FWD | Morgan Gibbs-White | MID | +0.650 | 38 |
| Brentford | Kevin Schade | MID | Mikkel Damsgaard | MID | +0.626 | 38 |
| Aston Villa | John McGinn | MID | Morgan Rogers | MID | +0.624 | 38 |
| Crystal Palace | Jørgen Strand Larsen | FWD | Adam Wharton | MID | +0.608 | 38 |
| Fulham | Emile Smith Rowe | MID | Harry Wilson | MID | +0.600 | 38 |
| Nott'm Forest | Taiwo Awoniyi | FWD | Morgan Gibbs-White | MID | +0.577 | 38 |
| Burnley | Zian Flemming | FWD | Hannibal Mejbri | MID | +0.577 | 38 |
| Arsenal | Martín Zubimendi Ibáñez | MID | Leandro Trossard | MID | +0.567 | 38 |
| Nott'm Forest | Igor Jesus Maciel da Cruz | FWD | Morgan Gibbs-White | MID | +0.557 | 38 |
| Bournemouth | Francisco Evanilson de Lima Barbosa | FWD | Marcus Tavernier | MID | +0.557 | 38 |
| West Ham | Valentín Castellanos | FWD | Jarrod Bowen | FWD | +0.550 | 38 |
| Chelsea | Cole Palmer | MID | João Pedro Junqueira de Jesus | FWD | +0.550 | 38 |

## Strongest *active* attack cross G→A (negative)

| team | scorer | pos | assister | pos | corr | n |
| --- | --- | --- | --- | --- | ---: | ---: |
| Man Utd | Matheus Santos Carneiro da Cunha | MID | Amad Diallo | MID | -0.233 | 38 |
| Arsenal | Bukayo Saka | MID | Martin Ødegaard | MID | -0.226 | 38 |
| Liverpool | Cody Gakpo | MID | Mohamed Salah | MID | -0.226 | 38 |
| Man Utd | Benjamin Sesko | FWD | Amad Diallo | MID | -0.222 | 38 |
| Man Utd | Carlos Henrique Casimiro | MID | Amad Diallo | MID | -0.217 | 38 |
| Fulham | Harry Wilson | MID | Saša Lukić | MID | -0.205 | 38 |
| Man Utd | Carlos Henrique Casimiro | MID | Matheus Santos Carneiro da Cunha | MID | -0.191 | 38 |
| Chelsea | João Pedro Junqueira de Jesus | FWD | Pedro Lomba Neto | MID | -0.190 | 38 |
| Brighton | Danny Welbeck | FWD | Yasin Ayari | MID | -0.190 | 38 |
| Arsenal | Bukayo Saka | MID | Gabriel Martinelli Silva | MID | -0.185 | 38 |
| Liverpool | Mohamed Salah | MID | Ryan Gravenberch | MID | -0.185 | 38 |
| Arsenal | Martín Zubimendi Ibáñez | MID | Declan Rice | MID | -0.179 | 38 |
| West Ham | Lucas Tolentino Coelho de Lima | MID | Jarrod Bowen | FWD | -0.178 | 38 |
| Newcastle | Anthony Gordon | MID | Bruno Guimarães Rodriguez Moura | MID | -0.177 | 38 |
| Man City | Antoine Semenyo | MID | Jérémy Doku | MID | -0.176 | 38 |

## Strongest *active* attack G×G (both ≥3G)

| team | A | pos | B | pos | corr | n |
| --- | --- | --- | --- | --- | ---: | ---: |
| Arsenal | Gabriel Fernando de Jesus | FWD | Noni Madueke | MID | +0.638 | 38 |
| Nott'm Forest | Taiwo Awoniyi | FWD | Igor Jesus Maciel da Cruz | FWD | +0.634 | 38 |
| Fulham | Emile Smith Rowe | MID | Samuel Chukwueze | MID | +0.486 | 38 |
| Leeds | Anton Stach | MID | Dominic Calvert-Lewin | FWD | +0.460 | 38 |
| Sunderland | Enzo Le Fée | MID | Brian Brobbey | FWD | +0.417 | 38 |
| Fulham | Alex Iwobi | MID | Samuel Chukwueze | MID | +0.408 | 38 |
| Arsenal | Gabriel Fernando de Jesus | FWD | Martín Zubimendi Ibáñez | MID | +0.383 | 38 |
| Man City | Tijjani Reijnders | MID | Rayan Cherki | MID | +0.374 | 38 |
| Man City | Omar Marmoush | MID | Antoine Semenyo | MID | +0.364 | 38 |
| Aston Villa | Ollie Watkins | FWD | John McGinn | MID | +0.352 | 38 |
| Arsenal | Viktor Gyökeres | FWD | Martín Zubimendi Ibáñez | MID | +0.306 | 38 |
| Nott'm Forest | Morgan Gibbs-White | MID | Igor Jesus Maciel da Cruz | FWD | +0.301 | 38 |

## Strongest *active* attack G×G negative

| team | A | pos | B | pos | corr | n |
| --- | --- | --- | --- | --- | ---: | ---: |
| Man City | Erling Haaland | FWD | Antoine Semenyo | MID | -0.356 | 38 |
| Man Utd | Bryan Mbeumo | MID | Benjamin Sesko | FWD | -0.324 | 38 |
| Arsenal | Viktor Gyökeres | FWD | Leandro Trossard | MID | -0.240 | 38 |
| Spurs | João Maria Lobo Alves Palhares Costa Palhinha Gonçalves | MID | Richarlison de Andrade | FWD | -0.222 | 38 |
| West Ham | Tomáš Souček | MID | Jarrod Bowen | FWD | -0.217 | 38 |
| Liverpool | Dominik Szoboszlai | MID | Mohamed Salah | MID | -0.206 | 38 |
| Newcastle | Anthony Gordon | MID | Nick Woltemade | FWD | -0.195 | 38 |
| Everton | Thierno Barry | FWD | Iliman Ndiaye | MID | -0.195 | 38 |
| Bournemouth | Junior Kroupi | FWD | Amine Adli | MID | -0.190 | 38 |
| Arsenal | Mikel Merino Zazón | MID | Viktor Gyökeres | FWD | -0.190 | 38 |
| Bournemouth | Antoine Semenyo | MID | Rayan Vitor Simplício Rocha | MID | -0.187 | 38 |
| Man City | Tijjani Reijnders | MID | Antoine Semenyo | MID | -0.185 | 38 |

## Reading

- Club **ΣG vs ΣA** is tightly coupled (median **+0.922**).
- Unfiltered attack **cross G_i×A_j** median **-0.039** (frac pos 31.0%) — most MID/FWD pairs are not linked; sparse zeros dominate.
- **Active** cross (volume filter) median **+0.063** (n=551, frac pos 57.5%).
- Attack **G×G** median **-0.048**; active **-0.050** (co-scoring is weak / slightly negative — pot sharing, not joint spikes).
- Both-played cross median **-0.060**.
- Unfiltered top ±1.0 pairs are one-hit artifacts; prefer the active tables.
- Implication for the scorer debate: team attack volume is shared, but player-level G/A co-movement is weak once sparse pairs are dropped — not strong evidence for a dense teammate covariance matrix on μ boards.

Reproduce:

```bash
PYTHONPATH=. python3 -m src.eval.ga_team_covariance --season 2025-26
```

Pairs CSV: `data/processed/ga_team_covariance_2025_26_pairs.csv`
JSON: `data/processed/ga_team_covariance_2025_26.json`

