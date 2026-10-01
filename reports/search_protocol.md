# Search protocol

Screen 2025-26, gameweeks 5–38. Ranked by fast XI points versus xp. A candidate is clearly behind at 100 points. Top 2 architectures go to the free-transfer grid. Spearman, MAE, and bias do not decide advancement.

Holdout season: 2023-24. A winner must beat that season's xp climb by 34.

## Tier 1

| candidate | params | XI points | vs xp | spearman | MAE | bias |
|---|---|---:|---:|---:|---:|---:|
| blend | alpha=0.75 | 2101 | +18 | 0.229 | 2.28 | 0.10 |
| xp | column=score_xp | 2083 | +0 | 0.229 | 2.29 | 0.13 |
| tail_risk | lambda=0.5 | 2025 | -58 | 0.219 | 2.25 | -0.05 |
| ownership | weight=0.25 | 2012 | -71 | 0.227 | 2.27 | 0.07 |
| blend | alpha=0.5 | 2011 | -72 | 0.213 | 2.29 | 0.07 |
| linear_risk | lambda=0.1 | 1998 | -85 | 0.228 | 2.20 | -0.16 |
| tail_risk | lambda=1.0 | 1998 | -85 | 0.186 | 2.25 | -0.22 |
| linear_risk | lambda=0.25 | 1983 | -100 | 0.219 | 2.12 | -0.58 |
| linear_risk | lambda=0.5 | 1983 | -100 | 0.185 | 2.15 | -1.30 |
| team_prior | column=score_team_prior | 1977 | -106 | 0.178 | 2.32 | 0.12 |
| price | column=score_price | 1973 | -110 | 0.122 | 51.16 | 51.16 |
| ownership | weight=0.5 | 1959 | -124 | 0.221 | 2.26 | 0.01 |
| blend | alpha=0.25 | 1947 | -136 | 0.191 | 2.30 | 0.04 |
| ownership | weight=1.0 | 1871 | -212 | 0.186 | 2.25 | -0.10 |
| exp_points | column=score_exp_points | 1845 | -238 | 0.169 | 2.33 | 0.01 |
| roll3 | column=score_roll3_points | 1709 | -374 | 0.113 | 2.57 | 0.17 |
| sharpe | eps=2.0 | 1696 | -387 | 0.146 | 2.67 | -2.58 |
| xmi | column=score_xmi | 1577 | -506 | 0.148 | 74.88 | 74.88 |
| sharpe | eps=1.0 | 1515 | -568 | 0.112 | 2.52 | -2.36 |
| sharpe | eps=0.5 | 1375 | -708 | 0.091 | 2.43 | -2.18 |

Advanced: **blend, tail_risk**.

## Tier 2

| candidate | params | FT points | vs xp_ft | transfers/GW | hits |
|---|---|---:|---:|---:|---:|
| blend | alpha=0.5 | 1890 | +22 | 0.88 | 0 |
| blend | alpha=0.25 | 1842 | -26 | 0.82 | 0 |
| tail_risk | lambda=1.0 | 1787 | -81 | 1.03 | 3 |
| blend | alpha=0.75 | 1770 | -98 | 0.94 | 1 |
| tail_risk | lambda=0.5 | 1755 | -113 | 1.00 | 3 |

## Holdout

| method | total |
|---|---:|
| xp_ft | 1635 |
| P* | 1822 |

On the protocol's single holdout, blend alpha=0.5 was +187 on 2023/24. That was the best of the search. It does not replace the expected-points climb.

P* is blend alpha=0.5. On the 2025/26 transfer climb that chose it, the same setting was only +22, which is under the +34 bar.

The Co-PI reviewed this after the run. Agreed: do not replace the expected-points comparator. The fast screen preferred alpha=0.75, which then lost 98 points on the transfer climb, while alpha=0.5 was the transfer-climb pick. The 2023/24 gain came with fewer transfers and no hits, against a baseline that took 6 hits. A second holdout on 2024/25, same alpha and the same +34 bar, was required before any further claim.

That second holdout failed. 2024/25 paired expected-points climb was 1889 (1.24 transfers per gameweek, 9 hits). The same blend scored 1866 (0.88 transfers per gameweek, 0 hits), a gap of −23. The blend is not carried forward.

Transfer-value candidates are listed in the matrix and are not ranked here.

- `data/processed/search_tier1.csv`
- `data/processed/search_tier2.csv`
- `data/processed/search_holdout.csv`
