# Packet → xmi compile demo (high-profile)

Deterministic SUPPORT phrases → `news_tags.minutes_for_tag`. Does **not** overwrite `data/live/xmi_gw6.csv`.

## Provenance withdrawal (2026-10-08)

Three hand-written packets used `*-example` URLs and invented copy. They are deleted and the loader now rejects that URL shape.

- `mancity_fc:gw06:guardiola-on-haaland` quoted Pep Guardiola confirming Haaland would start, with `published_at_utc` 2026-10-08. Guardiola left Manchester City in May 2026. Enzo Maresca was appointed on 29 June 2026. The note was not a scraped article; it was a demo sentence stamped as this week. It does not enter `xmi`.
- `arsenal_fc:gw06:saliba-injury-update` and `bbc_sport:gw06:van-ewijk-remains-a-doubt` were the same class (example URLs, no article). Withdrawn with it.

No replacement quote is invented. Haaland has no admissible packet, so the compile returns `ask` and writes no minutes.

## Context the compile sees

### Player: Saliba (id 6)
- Club: Arsenal | Position: DEF
- Prior minutes (compile base): none
- FPL flag: status `i`, chance 0
- FPL news: Back injury - Unknown return date

#### Audited packets (pre-deadline)
- (none)

### Player: van Ewijk (id 175)
- Club: Coventry City | Position: DEF
- Prior minutes (compile base): 90.0
- FPL flag: status `d`, chance 75
- FPL news: Hamstring injury - 75% chance of playing

#### Audited packets (pre-deadline)
- **[fpl_bootstrap:gw06:175-hamstring-injury-75-chance-of]** *fpl_bootstrap* (2026-09-19T16:00:09.486751Z)
  > van Ewijk: Hamstring injury - 75% chance of playing
  > status d. chance 75. club Coventry City.

### Player: Haaland (id 411)
- Club: Man City | Position: FWD
- Prior minutes (compile base): 90.0
- FPL flag: status `a`, chance blank
- FPL news: (none)

#### Audited packets (pre-deadline)
- (none)

## Compiled rows

| Player | Tag | Prior | Chance | xmi |
| --- | --- | ---: | ---: | ---: |
| Saliba | ask | none | 0.0 | no write |
| van Ewijk | injured | 90.0 | 75.0 | 67.5 |
| Haaland | ask | 90.0 | blank | no write |
