# Packet → xmi compile demo (high-profile)

Deterministic SUPPORT phrases → `news_tags.minutes_for_tag`. Does **not** overwrite `data/live/xmi_gw6.csv`.

## Sources admitted 2026-10-08

PI called the GW6 table reliable. Whitelisted and filed, each with its own URL and time:

`bbc_sport`, `mancity_fc`, `guardian`, `skysports` (already on the list), plus `manchestereveningnews`, `sportsmole`, `football_london`, `the_standard`, `coventry_telegraph`, `chronicle_live`, `haaglanden_voetbal`.

The latest `published_at_utc` decides the tag. A sentence that says the injury eased, or that he is not nursing an injury, does not count as `injured`. Bodies are short cites, not full articles.

## Context the compile sees

### Player: Saliba (id 6)
- Club: Arsenal | Position: DEF
- Prior minutes (compile base): none
- FPL flag: status `i`, chance 0
- FPL news: Back injury - Unknown return date

#### Audited packets (pre-deadline)
- **[guardian:gw06:arsenal-reveal-saliba-out-for-an-extended-period]** *guardian* (2026-07-22T21:05:00Z)
  > Arsenal reveal Saliba out for an extended period with back injury
  > The Guardian, 22 Jul 2026: Arsenal said reviews confirmed a back injury needing rehabilitation, surgery is not recommended, and he is expected to be out for an extended period.
- **[skysports:gw06:saliba-will-miss-an-extended-period-with-a-back-]** *skysports* (2026-07-23T08:00:00Z)
  > Saliba will miss an extended period with a back problem
  > Sky Sports, 23 Jul 2026: Arsenal have confirmed William Saliba will be out for an extended period with a back injury. Surgery is not expected.
- **[football_london:gw06:saliba-still-out-for-leeds-with-a-back-injury]** *football_london* (2026-10-06T05:00:00Z)
  > Saliba still out for Leeds with a back injury
  > Football London, 6 Oct 2026: William Saliba is yet to feature this season because of a back injury from the World Cup. Surgery was not needed. A return from November is suggested.
- **[sportsmole:gw06:saliba-out-for-leeds-with-a-back-injury]** *sportsmole* (2026-10-08T11:18:00Z)
  > Saliba out for Leeds with a back injury
  > Sports Mole, updated 8 Oct 2026: William Saliba status out, back injury, possible return 12 December versus Bournemouth. Arteta has set no number of weeks.
- **[the_standard:gw06:saliba-still-out-with-a-long-term-back-injury]** *the_standard* (2026-10-08T13:53:40Z)
  > Saliba still out with a long-term back injury
  > The Standard, 8 Oct 2026: Saliba has yet to play this season due to a long-term back injury worsened at the World Cup. He avoided surgery. No new official update. French reports say he is running and targeting November or December.

### Player: van Ewijk (id 175)
- Club: Coventry City | Position: DEF
- Prior minutes (compile base): 90.0
- FPL flag: status `d`, chance 75
- FPL news: Hamstring injury - 75% chance of playing

#### Audited packets (pre-deadline)
- **[fpl_bootstrap:gw06:175-hamstring-injury-75-chance-of]** *fpl_bootstrap* (2026-09-19T16:00:09.486751Z)
  > van Ewijk: Hamstring injury - 75% chance of playing
  > status d. chance 75. club Coventry City.
- **[haaglanden_voetbal:gw06:van-ewijk-hamstring-is-not-serious]** *haaglanden_voetbal* (2026-09-24T00:00:00Z)
  > van Ewijk hamstring is not serious
  > Haaglanden Voetbal, 24 Sep 2026: Frank Lampard confirmed the hamstring injury is not a bad tear. Coventry expect van Ewijk to be available after the international break.
- **[chronicle_live:gw06:van-ewijk-expected-to-be-fit-for-newcastle]** *chronicle_live* (2026-10-08T05:00:00Z)
  > van Ewijk expected to be fit for Newcastle
  > Chronicle Live, 8 Oct 2026: Milan van Ewijk is expected to be fit again this week after missing the Forest game with a hamstring issue.
- **[coventry_telegraph:gw06:lampard-says-van-ewijk-is-back-in-training]** *coventry_telegraph* (2026-10-08T15:14:00Z)
  > Lampard says van Ewijk is back in training
  > Coventry Telegraph, 8 Oct 2026: Frank Lampard said Milan van Ewijk is back in training ahead of Newcastle after a hamstring issue. He hopes van Ewijk will be alright if he comes through the next few days.

### Player: Haaland (id 411)
- Club: Man City | Position: FWD
- Prior minutes (compile base): 90.0
- FPL flag: status `a`, chance blank
- FPL news: (none)

#### Audited packets (pre-deadline)
- **[mancity_fc:gw06:dias-beats-haaland-in-nations-league]** *mancity_fc* (2026-10-04T21:00:00Z)
  > Dias beats Haaland in Nations League
  > Portugal beat Norway 2-1 in Porto. Haaland was replaced in the 67th minute.
- **[bbc_sport:gw06:haaland-an-injury-doubt-after-limping-off-agains]** *bbc_sport* (2026-10-05T14:54:39Z)
  > Haaland an injury doubt after limping off against Portugal
  > BBC Sport, 5 Oct 2026: Erling Haaland is an injury doubt after he limped off in Norway's match against Portugal. He asked to be substituted on 67 minutes and left camp on Monday.
- **[manchestereveningnews:gw06:haaland-granted-extra-rest-before-liverpool]** *manchestereveningnews* (2026-10-06T09:49:00Z)
  > Haaland granted extra rest before Liverpool
  > Manchester Evening News, 6 Oct 2026: Stale Solbakken said Haaland was thoroughly spent. He was withdrawn with suspected muscle fatigue, and concerns over a serious injury eased. City hope to have him available.
- **[sportsmole:gw06:haaland-a-minor-doubt-for-liverpool-with-fatigue]** *sportsmole* (2026-10-08T07:10:00Z)
  > Haaland a minor doubt for Liverpool with fatigue
  > Sports Mole, 8 Oct 2026: status minor doubt, fatigue, possible return 11 October versus Liverpool. Haaland is not nursing an injury and should be available for selection on Sunday.

## Compiled rows

| Player | Tag | Prior | Chance | xmi | Deciding packet |
| --- | --- | ---: | ---: | ---: | --- |
| Saliba | injured | none | 0.0 | 0.0 | the_standard:gw06:saliba-still-out-with-a-long-term-back-injury |
| van Ewijk | ask | 90.0 | 75.0 | no write | coventry_telegraph:gw06:lampard-says-van-ewijk-is-back-in-training |
| Haaland | ask | 90.0 | blank | no write | sportsmole:gw06:haaland-a-minor-doubt-for-liverpool-with-fatigue |
