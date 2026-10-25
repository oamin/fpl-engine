# Cascade stepped xi_ft vs L3 / L4 (2025-26 GW8–12)

As of `2026-10-25T08:13:02Z` (UTC).

## Arms

- **cascade_step** — `cascade_mean_xi_ft_step_pool_resid` commits (full-club shares; ≥60′ selection refreshed GW8…12; projected to GW19). Captain = Haaland if in XI.
- **l3 / l4_mupost** — locked climb weeks from `data/processed/layer4_mupost_gamma1.json` (μ_post priced γ=1).

**Not gated.** Different boards (Opta generative long vs priced t:t+1).

![pts_net](/opt/cursor/artifacts/cascade_step_vs_l3_l4.png)

## Totals

| arm | Σ pts_net | Σ XI+C | hits | transfers |
| --- | ---: | ---: | ---: | ---: |
| l4_mupost | 399 | 403 | 1 | 6 |
| l3 | 398 | 398 | 0 | 5 |
| cascade_step | 276 | 276 | 0 | 4 |

- cascade − L3: **-122** (-24.4/GW)
- cascade − L4: **-123** (-24.6/GW)

## Per GW (pts_net)

| GW | cascade | L3 | L4 | casc−L3 | casc−L4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 84 | 162 | 162 | -78 | -78 |
| 9 | 42 | 81 | 79 | -39 | -37 |
| 10 | 72 | 75 | 89 | -3 | -17 |
| 11 | 42 | 38 | 35 | +4 | +7 |
| 12 | 36 | 42 | 34 | -6 | +2 |

## Cascade squads (reminder)

- **GW8** C=Erling Haaland · net=84
  - XI: Gabriel dos Santos Magalhães, Jurriën Timber, Marcos Senesi Barón, Antoine Semenyo, Josh Cullen, João Pedro Junqueira de Jesus, Erling Haaland, Bruno Borges Fernandes, Robin Roefs, Omar Alderete, Nick Woltemade
  - Bench: Kevin Schade, Joachim Andersen, Ryan Sessegnon, Nick Pope
- **GW9** C=Erling Haaland · net=42
  - XI: Gabriel dos Santos Magalhães, Jurriën Timber, Marcos Senesi Barón, Antoine Semenyo, Josh Cullen, João Pedro Junqueira de Jesus, Erling Haaland, Bruno Borges Fernandes, Nick Pope, Nordi Mukiele, Nick Woltemade
  - Bench: Kevin Schade, Joachim Andersen, Ryan Sessegnon, Robin Roefs
- **GW10** C=Erling Haaland · net=72
  - XI: Gabriel dos Santos Magalhães, Jurriën Timber, Antoine Semenyo, Josh Cullen, Ryan Sessegnon, Dominik Szoboszlai, Erling Haaland, Bruno Borges Fernandes, Robin Roefs, Nordi Mukiele, Nick Woltemade
  - Bench: Marcos Senesi Barón, João Pedro Junqueira de Jesus, Joachim Andersen, Nick Pope
- **GW11** C=Erling Haaland · net=42
  - XI: Gabriel dos Santos Magalhães, Jurriën Timber, Marcos Senesi Barón, Antoine Semenyo, Josh Cullen, João Pedro Junqueira de Jesus, Ryan Sessegnon, Dominik Szoboszlai, Erling Haaland, Bruno Borges Fernandes, Nick Pope
  - Bench: Igor Thiago Nascimento Rodrigues, Joachim Andersen, Robin Roefs, Nordi Mukiele
- **GW12** C=Erling Haaland · net=36
  - XI: Gabriel dos Santos Magalhães, Jurriën Timber, Marcos Senesi Barón, Antoine Semenyo, Igor Thiago Nascimento Rodrigues, Joachim Andersen, Ryan Sessegnon, Dominik Szoboszlai, Erling Haaland, Bruno Borges Fernandes, Robin Roefs
  - Bench: Josh Cullen, João Pedro Junqueira de Jesus, Nick Pope, Nordi Mukiele

## Verdict

On this window cascade stepped trails locked L3/L4 by **-122** / **-123** net pts (mostly GW8: 84 vs 162). CS/GA calibration on the cascade path does not yet translate to beating the priced μ_post climb.

## Reproduce

```bash
PYTHONPATH=. python3 -m src.eval.cascade_step_vs_l3_l4
```

