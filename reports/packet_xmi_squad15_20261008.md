# Packet → xmi on ojaminFC (GW6)

Entry 2632584. Fifteen from the Gameweek 5 picks. Sidecar only: `data/live/xmi_gw6.csv` is not overwritten.

An `ask` (no packet) now writes the mean of the last three gameweeks, and a 0-minute week stays in that window. `50/50` is still half the appearance prior.

| | Player | Last 3 games | Tag | Packet xmi | LLM xmi |
| --- | --- | --- | --- | ---: | ---: |
| XI | Lammens | 90, 90, 90 | ask | 90 | 90 |
| XI | Davis | 90, 90, 90 | ask | 90 | 90 |
| XI | Calafiori | 66, 90, 90 | ask | 82 | 90 |
| XI | Guéhi | 90, 90, 90 | ask | 90 | 90 |
| XI | Barnes | 90, 90, 90 | ask | 90 | 85 |
| XI | B.Fernandes (V) | 90, 90, 90 | ask | 90 | 90 |
| XI | Rogers | 86, 90, 90 | ask | 88.7 | 85 |
| XI | Cherki | 65, 45, 84 | ask | 64.7 | 75 |
| XI | Ødegaard | 76, 84, 72 | ask | 77.3 | 80 |
| XI | Calvert-Lewin | 72, 77, 90 | ask | 79.7 | 80 |
| XI | Haaland (C) | 90, 90, 90 | 50/50 | 45 | 90 |
| BN | Forster | 0, 0, 0 | ask | 0 | 0 |
| BN | van Ewijk | 90, 90, 0 | 50/50 | 45 | 60 |
| BN | Shaw | 90, 0, 83 | ask | 57.7 | 80 |
| BN | Scarlett | 0, 0, 0 | transferred | 0 | 0 |

Shaw is the row the window changes. Dropping the 0-minute week would leave the mean of 90 and 83. Keeping it, as a game in the window, gives 57.7.
