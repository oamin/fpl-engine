# Packet → xmi compile demo (high-profile)

Deterministic SUPPORT phrases → `news_tags.minutes_for_tag`. Does **not** overwrite `data/live/xmi_gw6.csv`.

## Context the compile sees

### Player: Saliba (id 6)
- Club: Arsenal | Position: DEF
- Prior minutes (compile base): none
- FPL flag: status `i`, chance 0
- FPL news: Back injury - Unknown return date

#### Audited packets (pre-deadline)
- **[arsenal_fc:gw06:saliba-injury-update]** *arsenal_fc* (2026-10-08T14:00:00Z)
  > Saliba injury update
  > William Saliba remains out with a back injury and is suspended from contention until a further assessment. The club says the injury is ongoing.

### Player: van Ewijk (id 175)
- Club: Coventry City | Position: DEF
- Prior minutes (compile base): 90.0
- FPL flag: status `d`, chance 75
- FPL news: Hamstring injury - 75% chance of playing

#### Audited packets (pre-deadline)
- **[fpl_bootstrap:gw06:175-hamstring-injury-75-chance-of]** *fpl_bootstrap* (2026-09-19T16:00:09.486751Z)
  > van Ewijk: Hamstring injury - 75% chance of playing
  > status d. chance 75. club Coventry City.
- **[bbc_sport:gw06:van-ewijk-remains-a-doubt]** *bbc_sport* (2026-10-08T16:00:00Z)
  > van Ewijk remains a doubt
  > Coventry defender Milan van Ewijk remains a doubt for Saturday with a hamstring knock after feeling tightness in training.

### Player: Haaland (id 411)
- Club: Man City | Position: FWD
- Prior minutes (compile base): 90.0
- FPL flag: status `a`, chance blank
- FPL news: (none)

#### Audited packets (pre-deadline)
- **[mancity_fc:gw06:guardiola-on-haaland]** *mancity_fc* (2026-10-08T15:30:00Z)
  > Guardiola on Haaland
  > Pep Guardiola confirmed Erling Haaland will start and is first choice up front. Haaland trained fully and is named in the team plans for the weekend.

## Compiled rows

| Player | Tag | Prior | Chance | xmi |
| --- | --- | ---: | ---: | ---: |
| Saliba | injured | None | 0.0 | 0.0 |
| van Ewijk | injured | 90.0 | 75.0 | 67.5 |
| Haaland | firm_starter | 90.0 | None | 90.0 |
