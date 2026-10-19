# Minutes covariance — 2025-26 (fixture-level)

Per-club player×player correlation of minutes. **No position filter.**
Each **fixture** is one observation; DGWs with two match rows count as two.

- Team×fixture observations: **760** (team×GW would be 750; DGW extras **10**)
- Min fixtures with minutes>0: **5**
- Start indicator: minutes≥**60**

## Pooled off-diagonal pairs

| measure | n pairs | mean | median | frac neg | p10 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Minutes | 5193 | -0.040 | -0.041 | 55.3% | -0.371 | +0.286 |
| Start I(m≥60) | 4646 | -0.033 | -0.038 | 55.1% | -0.340 | +0.273 |

### Minutes corr by mean-minute stratum

| stratum | n pairs | median | frac neg | p10 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| both_nailed | 296 | -0.033 | 55.4% | -0.212 | +0.360 |
| both_rotation_band | 2513 | -0.054 | 57.1% | -0.437 | +0.300 |
| other | 2384 | -0.027 | 53.4% | -0.331 | +0.265 |

## By club (minutes corr)

| team | fixtures | players | DGW rows | median corr | frac neg | p10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arsenal | 38 | 24 | 2 | -0.034 | 55.1% | -0.343 |
| Aston Villa | 38 | 25 | 0 | -0.042 | 54.7% | -0.376 |
| Bournemouth | 38 | 23 | 2 | -0.052 | 54.5% | -0.363 |
| Brentford | 38 | 19 | 0 | -0.047 | 54.4% | -0.312 |
| Brighton | 38 | 24 | 2 | -0.034 | 55.7% | -0.320 |
| Burnley | 38 | 25 | 2 | -0.036 | 54.0% | -0.372 |
| Chelsea | 38 | 23 | 2 | -0.048 | 58.5% | -0.276 |
| Crystal Palace | 38 | 23 | 2 | -0.023 | 53.4% | -0.428 |
| Everton | 38 | 22 | 0 | -0.051 | 55.7% | -0.322 |
| Fulham | 38 | 24 | 0 | -0.035 | 55.3% | -0.373 |
| Leeds | 38 | 22 | 2 | -0.033 | 54.5% | -0.380 |
| Liverpool | 38 | 22 | 0 | -0.058 | 59.0% | -0.272 |
| Man City | 38 | 24 | 4 | -0.042 | 56.9% | -0.482 |
| Man Utd | 38 | 21 | 0 | -0.043 | 54.3% | -0.435 |
| Newcastle | 38 | 22 | 0 | -0.041 | 56.7% | -0.327 |
| Nott'm Forest | 38 | 26 | 0 | -0.035 | 53.5% | -0.362 |
| Spurs | 38 | 24 | 0 | -0.044 | 55.8% | -0.413 |
| Sunderland | 38 | 22 | 0 | -0.052 | 56.3% | -0.364 |
| West Ham | 38 | 28 | 0 | -0.043 | 53.4% | -0.391 |
| Wolves | 38 | 26 | 2 | -0.040 | 55.7% | -0.383 |

## Strongest negative minutes pairs (all clubs)

| team | player A | pos | player B | pos | corr | mean m A | mean m B |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Leeds | Karl Darlow | GK | Lucas Estella Perri | GK | -1.000 | 52.1 | 37.9 |
| Man Utd | Altay Bayındır | GK | Senne Lammens | GK | -1.000 | 14.2 | 75.8 |
| Spurs | Guglielmo Vicario | GK | Antonín Kinský | GK | -1.000 | 73.4 | 16.6 |
| Aston Villa | Emiliano Martínez Romero | GK | Marco Bizot | GK | -1.000 | 74.6 | 15.4 |
| West Ham | Alphonse Areola | GK | Mads Hermansen | GK | -1.000 | 47.4 | 42.6 |
| Newcastle | Nick Pope | GK | Aaron Ramsdale | GK | -1.000 | 63.6 | 26.4 |
| Chelsea | Robert Lynch Sánchez | GK | Filip Jörgensen | GK | -1.000 | 80.0 | 9.9 |
| Liverpool | Milos Kerkez | DEF | Andrew Robertson | DEF | -0.998 | 59.2 | 30.7 |
| Aston Villa | Lucas Digne | DEF | Ian Maatsen | DEF | -0.997 | 48.5 | 41.8 |
| West Ham | El Hadji Malick Diouf | DEF | Ollie Scarles | DEF | -0.974 | 71.2 | 17.3 |
| West Ham | Aaron Wan-Bissaka | DEF | Kyle Walker-Peters | DEF | -0.973 | 54.7 | 36.3 |
| Everton | Thierno Barry | FWD | Norberto Bercique Gomes Betuncal | FWD | -0.969 | 49.9 | 40.8 |
| Wolves | Hugo Bueno López | DEF | David Møller Wolfe | DEF | -0.965 | 62.1 | 27.4 |
| Fulham | Calvin Bassey | DEF | Jorge Cuenca Barreno | DEF | -0.922 | 66.6 | 24.6 |
| Man City | John Stones | DEF | Gianluigi Donnarumma | GK | -0.922 | 11.5 | 80.5 |

## Strongest positive minutes pairs

| team | player A | pos | player B | pos | corr | mean m A | mean m B |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Man City | Antoine Semenyo | MID | Marc Guéhi | DEF | +0.903 | 36.8 | 35.5 |
| Nott'm Forest | Nikola Milenković | DEF | Elliot Anderson | MID | +0.832 | 88.8 | 87.7 |
| Aston Villa | Leon Bailey | MID | Douglas Luiz Soares de Paulo | MID | +0.829 | 11.8 | 15.8 |
| Everton | Kiernan Dewsbury-Hall | MID | Iliman Ndiaye | MID | +0.813 | 69.2 | 73.2 |
| West Ham | Mads Hermansen | GK | Axel Disasi | DEF | +0.805 | 42.6 | 33.0 |
| Man Utd | Bryan Mbeumo | MID | Bruno Borges Fernandes | MID | +0.788 | 68.7 | 80.7 |
| Leeds | Wilfried Gnonto | MID | Daniel James | MID | +0.774 | 13.7 | 14.4 |
| West Ham | Tomáš Souček | MID | Valentín Castellanos | FWD | +0.772 | 57.6 | 36.4 |
| West Ham | Valentín Castellanos | FWD | Axel Disasi | DEF | +0.758 | 36.4 | 33.0 |
| Leeds | Jack Harrison | MID | Sean Longstaff | MID | +0.755 | 6.9 | 26.4 |

## Example: Arsenal

- Fixtures **38**, players **24**, DGW fixture rows **2**

Most negative minutes:

| A | B | corr |
| --- | --- | ---: |
| David Raya Martín (GK) | Christian Nørgaard (MID) | -0.856 |
| Bukayo Saka (MID) | Noni Madueke (MID) | -0.830 |
| David Raya Martín (GK) | Max Dowman (MID) | -0.780 |
| Gabriel Martinelli Silva (MID) | Leandro Trossard (MID) | -0.750 |
| Declan Rice (MID) | Christian Nørgaard (MID) | -0.706 |
| William Saliba (DEF) | Cristhian Mosquera (DEF) | -0.683 |

Most positive minutes:

| A | B | corr |
| --- | --- | ---: |
| Christian Nørgaard (MID) | Max Dowman (MID) | +0.631 |
| David Raya Martín (GK) | Declan Rice (MID) | +0.593 |
| Myles Lewis-Skelly (DEF) | Max Dowman (MID) | +0.569 |
| Noni Madueke (MID) | Gabriel Martinelli Silva (MID) | +0.490 |
| Christian Nørgaard (MID) | Gabriel Fernando de Jesus (FWD) | +0.439 |
| Gabriel Fernando de Jesus (FWD) | Max Dowman (MID) | +0.435 |

## Example: Liverpool

- Fixtures **38**, players **22**, DGW fixture rows **0**

Most negative minutes:

| A | B | corr |
| --- | --- | ---: |
| Milos Kerkez (DEF) | Andrew Robertson (DEF) | -0.998 |
| Alisson Becker (GK) | Giorgi Mamardashvili (GK) | -0.872 |
| Jeremie Frimpong (DEF) | Mohamed Salah (MID) | -0.522 |
| Alexander Isak (FWD) | Hugo Ekitiké (FWD) | -0.483 |
| Dominik Szoboszlai (MID) | Endo Wataru (MID) | -0.477 |
| Federico Chiesa (MID) | Dominik Szoboszlai (MID) | -0.465 |

Most positive minutes:

| A | B | corr |
| --- | --- | ---: |
| Trey Nyoni (MID) | Rio Ngumoha (MID) | +0.508 |
| Alisson Becker (GK) | Hugo Ekitiké (FWD) | +0.452 |
| Joe Gomez (DEF) | Trey Nyoni (MID) | +0.389 |
| Giorgi Mamardashvili (GK) | Rio Ngumoha (MID) | +0.370 |
| Joe Gomez (DEF) | Rio Ngumoha (MID) | +0.325 |
| Cody Gakpo (MID) | Dominik Szoboszlai (MID) | +0.321 |

## Example: Man City

- Fixtures **38**, players **24**, DGW fixture rows **4**

Most negative minutes:

| A | B | corr |
| --- | --- | ---: |
| John Stones (DEF) | Gianluigi Donnarumma (GK) | -0.922 |
| Abdukodir Khusanov (DEF) | Rúben dos Santos Gato Alves Dias (DEF) | -0.844 |
| Rodrigo 'Rodri' Hernandez Cascante (MID) | Nico González Iglesias (MID) | -0.815 |
| Matheus Nunes (DEF) | Rico Lewis (DEF) | -0.798 |
| Rico Lewis (DEF) | Gianluigi Donnarumma (GK) | -0.691 |
| Antoine Semenyo (MID) | Joško Gvardiol (DEF) | -0.684 |

Most positive minutes:

| A | B | corr |
| --- | --- | ---: |
| Antoine Semenyo (MID) | Marc Guéhi (DEF) | +0.903 |
| Joško Gvardiol (DEF) | Phil Foden (MID) | +0.754 |
| John Stones (DEF) | Oscar Bobb (MID) | +0.737 |
| Rúben dos Santos Gato Alves Dias (DEF) | Phil Foden (MID) | +0.641 |
| Rayan Aït-Nouri (DEF) | Omar Marmoush (MID) | +0.635 |
| Nico O'Reilly (DEF) | Gianluigi Donnarumma (GK) | +0.628 |

## Verdict

Fixture-level panel: 10 extra obs vs GW-collapse. Pooled minutes corr median -0.04 (55% negative) — full squad matrix mixes XI co-starts with bench competition. Both-nailed median -0.03; rotation-band median -0.05 (p10 -0.44). Start I(m≥60) median -0.04. Clear GK/FB competition pairs near −1 — minutes covariance is real and club-matrix shaped.

Pairs CSV: `data/processed/minutes_covariance_2025_26_pairs.csv`
JSON: `data/processed/minutes_covariance_2025_26.json`

