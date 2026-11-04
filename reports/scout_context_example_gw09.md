# Scout context example — GW9

Source: `https://www.fantasyfootballscout.co.uk/team-news` · fetched `2026-10-31T07:45:32.860375+00:00` · ρ=0.85 · overlays **124**

Combined signals: FFScout Out/Banned/Doubts + predicted-XI omissions + FPL `news`/status + club Latest News blurb parse.

## Counts

- `doubt`: 30
- `injured_expected_back`: 4
- `injured_unknown_return`: 50
- `suspected_single_week_rotation`: 36
- `suspended`: 4

- with `return_gws`: 44
- with blurb evidence: 7
- with FPL news: 53

## Highlights (inspect these first)

- **Lacroix** (Chelsea): `injured, expected back in 2 weeks` — m0=0.0, return_gws=2, force={'0': 0.0, '1': 0.0}
- **Ruggeri** (Aston Villa): `injured, expected back in 3 weeks` — m0=0.0, return_gws=3, force={'0': 0.0, '1': 0.0, '2': 0.0}
- **Scott** (Bournemouth): `injured, expected back in 1 week` — m0=0.0, return_gws=1, force={'0': 0.0}
- **Wan-Bissaka** (Aston Villa): `injured, expected back in 3 weeks` — m0=0.0, return_gws=3, force={'0': 0.0, '1': 0.0, '2': 0.0}
- **A.Becker** (Liverpool): `suspected single-week rotation` — m0=0.0, return_gws=1, force={'0': 0.0}
- **Aina** (Nott'm Forest): `suspected single-week rotation` — m0=15.0, return_gws=1, force={'0': 15.0}
- **Anthony** (Brentford): `suspected single-week rotation` — m0=15.0, return_gws=1, force={'0': 15.0}
- **Barkley** (Aston Villa): `suspected single-week rotation` — m0=15.0, return_gws=1, force={'0': 15.0}
- **Yarmoliuk** (Brentford): `suspended, expected back in 3 weeks` — m0=0.0, return_gws=3, force={'0': 0.0, '1': 0.0, '2': 0.0}
- **Bentancur** (Spurs): `suspended` — m0=0.0, return_gws=None, force={}
- **Dowman** (Arsenal): `suspended` — m0=0.0, return_gws=None, force={}
- **Sessegnon** (Fulham): `suspended` — m0=0.0, return_gws=None, force={}

## Overlay table (all)

| player | club | bucket/tag | context | return_gws | m0 | force | evidence |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| Tzolis | Arsenal | injured/ffscout | doubt (75% chance) | 0 | 60 | — | ffscout:FFScout Doubt: Tzolis (75%); fpl:Hamstring injury - Expected back 1 Nov |
| Yohanna | Brighton | injured/ffscout | doubt (25% chance) | 0 | 3 | — | ffscout:FFScout Doubt: Yohanna (25%); fpl:Unspecified injury - Expected back 18 Oct |
| Lukić | Ipswich | injured/ffscout | doubt (75% chance) | 0 | 60 | — | ffscout:FFScout Doubt: Lukic (75%); blurb:No surprise that O'Neil didn't commit one way or another regarding a start … |
| Ruggeri | Aston Villa | injured/ffscout | injured, expected back in 3 weeks | 3 | 0 | 0:0,1:0,2:0 | ffscout:FFScout Out: Ruggeri; fpl:Hamstring injury - Unknown return date |
| Wan-Bissaka | Aston Villa | injured/ffscout | injured, expected back in 3 weeks | 3 | 0 | 0:0,1:0,2:0 | ffscout:FFScout Out: Wan-Bissaka; blurb:The broadcast section of Unai Emery’s presser was short on team news, so we’r… |
| Scott | Bournemouth | injured/ffscout | injured, expected back in 1 week | 1 | 0 | 0:0 | ffscout:FFScout Out: Scott; fpl:Thigh injury - Unknown return date |
| Lacroix | Chelsea | injured/ffscout | injured, expected back in 2 weeks | 2 | 0 | 0:0,1:0 | ffscout:FFScout Out: Lacroix; blurb:Xabi Alonso confirmed that Maxence Lacroix is out of the next two Gameweeks with … |
| Gabriel | Arsenal | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Konsa | Arsenal | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=56.2; recent=[90.0, 90.0, 90.0] |
| Lewis-Skelly | Arsenal | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=54.2; recent=[66.0, 45.0, 0.0] |
| Raya | Arsenal | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 0 | 0:0 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Barkley | Aston Villa | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=61.6; recent=[90.0, 62.0, 6.0] |
| Buendía | Aston Villa | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=80.2; recent=[74.0, 62.0, 90.0] |
| George Hemmings | Aston Villa | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=59.4; recent=[74.0, 45.0, 18.0] |
| Mings | Aston Villa | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=52.2; recent=[90.0, 65.0, 90.0] |
| Anthony | Brentford | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=53.4; recent=[45.0, 59.0, 80.0] |
| Damsgaard | Brentford | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=56.6; recent=[85.0, 30.0, 86.0] |
| Schuster | Brentford | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=50.6; recent=[73.0, 90.0, 90.0] |
| Yalcouyé | Brighton | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=47.2; recent=[90.0, 64.0, 5.0] |
| Gusto | Chelsea | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=54.4; recent=[30.0, 77.0, 0.0] |
| Lavia | Chelsea | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=51.6; recent=[59.0, 68.0, 0.0] |
| Thomas-Asante | Coventry | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=45.4; recent=[19.0, 17.0, 90.0] |
| Yirenkyi | Coventry | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=53.0; recent=[27.0, 17.0, 90.0] |
| Yeremy | Crystal Palace | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=61.4; recent=[88.0, 36.0, 81.0] |
| Maitland-Niles | Everton | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=59.0; recent=[13.0, 74.0, 90.0] |
| Gonzalo | Fulham | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=85.0; recent=[90.0, 76.0, 79.0] |
| Palacios | Fulham | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=45.2; recent=[60.0, 0.0, 19.0] |
| Greaves | Ipswich | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=71.4; recent=[90.0, 87.0, 0.0] |
| Scherpen | Ipswich | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 0 | 0:0 | expand_m=72.0; recent=[90.0, 90.0, 0.0] |
| A.Becker | Liverpool | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 0 | 0:0 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Virgil | Liverpool | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Cherki | Man City | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=60.4; recent=[65.0, 45.0, 84.0] |
| Enzo | Man City | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=54.2; recent=[75.0, 81.0, 90.0] |
| Guéhi | Man City | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Rúben | Man City | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Tielemans | Man Utd | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=80.4; recent=[90.0, 66.0, 90.0] |
| Botman | Newcastle | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Thiaw | Newcastle | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
| Aina | Nott'm Forest | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=61.2; recent=[58.0, 90.0, 0.0] |
| Gray | Spurs | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=61.4; recent=[0.0, 84.0, 71.0] |
| Sávio | Spurs | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=48.0; recent=[60.0, 90.0, 90.0] |
| Danso | Sunderland | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=54.0; recent=[90.0, 90.0, 90.0] |
| Meunier | Sunderland | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=79.2; recent=[63.0, 75.0, 78.0] |
| Yarmoliuk | Brentford | injured/ffscout | suspended, expected back in 3 weeks | 3 | 0 | 0:0,1:0,2:0 | ffscout:FFScout Banned: Yarmolyuk; blurb:Yehor Yarmolyuk serves game three of a three-match ban, too, but Vitaly Jane… |
| Calafiori | Arsenal | injured/ffscout | doubt (75% chance) |  | 62 | — | ffscout:FFScout Doubt: Calafiori (75%) |
| Madueke | Arsenal | injured/ffscout | doubt (75% chance) |  | 9 | — | ffscout:FFScout Doubt: Madueke (75%) |
| Rice | Arsenal | injured/ffscout | doubt (75% chance) |  | 65 | — | ffscout:FFScout Doubt: Rice (75%) |
| Milosavljević | Bournemouth | injured/ffscout | doubt (25% chance) |  | 22 | — | ffscout:FFScout Doubt: Milosavljevic (25%); fpl:Thigh injury - Unknown return date |
| Dasilva | Brentford | injured/ffscout | doubt (25% chance) |  | 22 | — | ffscout:FFScout Doubt: Dasilva (25%); fpl:Hamstring injury - Unknown return date |
| Milambo | Brentford | injured/ffscout | doubt (25% chance) |  | 22 | — | ffscout:FFScout Doubt: Milambo (25%); fpl:Knee injury - Unknown return date |
| Andrés | Brighton | injured/ffscout | doubt (75% chance) |  | 41 | — | ffscout:FFScout Doubt: Chema Andrés (75%) |
| F.Kadıoğlu | Brighton | injured/ffscout | doubt (75% chance) |  | 60 | — | ffscout:FFScout Doubt: Kadioglu (75%) |
| Tzimas | Brighton | injured/ffscout | doubt (25% chance) |  | 22 | — | ffscout:FFScout Doubt: Tzimas (25%); fpl:Knee injury - Unknown return date |
| Thomas | Coventry | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Thomas (75%) |
| Strand Larsen | Crystal Palace | injured/ffscout | doubt (75% chance) |  | 52 | — | ffscout:FFScout Doubt: Strand Larsen (75%) |
| Timber | Crystal Palace | injured/ffscout | doubt (75% chance) |  | 63 | — | ffscout:FFScout Doubt: Timber (75%) |
| Branthwaite | Everton | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Branthwaite (75%) |
| Grealish | Everton | injured/ffscout | doubt (50% chance) |  | 21 | — | ffscout:FFScout Doubt: Grealish (50%) |
| Butland | Hull | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Butland (75%); fpl:Arm injury - Unknown return date |
| Stach | Leeds | injured/ffscout | doubt (75% chance) |  | 67 | — | ffscout:FFScout Doubt: Stach (75%) |
| Munoz | Liverpool | injured/ffscout | doubt (50% chance) |  | 23 | — | ffscout:FFScout Doubt: Víctor Muñoz (50%) |
| Szoboszlai | Liverpool | injured/ffscout | doubt (50% chance) |  | 43 | — | ffscout:FFScout Doubt: Szoboszlai (50%); fpl:Hamstring injury - 25% chance of playing |
| Mainoo | Man Utd | injured/ffscout | doubt (75% chance) |  | 63 | — | ffscout:FFScout Doubt: Mainoo (75%); fpl:Unspecified injury - 75% chance of playing |
| Rashford | Man Utd | injured/ffscout | doubt (25% chance) |  | 19 | — | ffscout:FFScout Doubt: Rashford (25%) |
| Fernandez-Pardo | Newcastle | injured/ffscout | doubt (75% chance) |  | 39 | — | ffscout:FFScout Doubt: Fernandez-Pardo (75%) |
| L.Miley | Newcastle | injured/ffscout | doubt (75% chance) |  | 58 | — | ffscout:FFScout Doubt: Miley (75%) |
| Osula | Newcastle | injured/ffscout | doubt (75% chance) |  | 41 | — | ffscout:FFScout Doubt: Osula (75%) |
| Dominguez | Nott'm Forest | injured/ffscout | doubt (75% chance) |  | 1 | — | ffscout:FFScout Doubt: Nicolás Domínguez (75%) |
| Ndoye | Nott'm Forest | injured/ffscout | doubt (75% chance) |  | 48 | — | ffscout:FFScout Doubt: Ndoye (75%) |
| Isidor | Sunderland | injured/ffscout | doubt (75% chance) |  | 17 | — | ffscout:FFScout Doubt: Isidor (75%); fpl:Knock - 50% chance of playing |
| Mukiele | Sunderland | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Mukiele (75%) |
| Saka | Arsenal | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Saka |
| Saliba | Arsenal | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Saliba; fpl:Back injury - Unknown return date |
| Onana | Aston Villa | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mvom Onana; fpl:Knee injury - Unknown return date |
| J.Araujo | Bournemouth | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Araujo; fpl:Thigh injury - Unknown return date |
| Tóth.A | Bournemouth | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Tóth |
| Carvalho | Brentford | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Carvalho |
| Furo | Brentford | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Furo; fpl:Unspecified injury - Unknown return date |
| Henry | Brentford | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Henry |
| Azeez | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Azeez; fpl:Hamstring injury - Expected back 18 Oct |
| Ferguson | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ferguson; fpl:Ankle injury - Expected back 18 Oct |
| Mitoma | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mitoma; fpl:Hamstring injury - Expected back 18 Oct |
| Struijk | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Struijk |
| Anselmino | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Anselmino |
| Gittens | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Gittens; fpl:Ankle injury - Unknown return date |
| Hato | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Hato |
| James | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: James; fpl:Knee injury - Unknown return date |
| João Pedro | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: João Pedro |
| Martinez | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Martinez |
| Amenda | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Amenda; fpl:Calf injury - Unknown return date |
| Eccles | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Eccles; fpl:Ankle injury - Unknown return date |
| Woolfenden | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Woolfenden; fpl:Knee injury - Unknown return date |
| Wright | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Wright; fpl:Thigh injury - Unknown return date |
| Chadi Riad | Crystal Palace | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Riad; fpl:Knee injury - Unknown return date |
| Sarr | Crystal Palace | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ismaila Sarr |
| Tomiyasu | Crystal Palace | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Tomiyasu; fpl:Calf injury - Expected back 18 Oct |
| Ansah | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ansah; fpl:Back injury - Unknown return date |
| Hughes | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Hughes; fpl:Groin injury - Unknown return date |
| Matazo | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Matazo; fpl:Knee injury - Unknown return date |
| Targett | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Targett; fpl:Ankle injury - Unknown return date |
| Mateo Joseph | Leeds | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mateo Joseph; fpl:Knee injury - Unknown return date |
| Rodon | Leeds | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Rodon; fpl:Hamstring injury - Expected back 21 Nov |
| Bradley | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Bradley; fpl:Knee injury - Unknown return date |
| Ekitiké | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ekitiké; fpl:Achilles injury - Unknown return date |
| Frimpong | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Frimpong |
| Leoni | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Leoni; fpl:Knee injury - Unknown return date |
| Semenyo | Man City | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Semenyo |
| Dorgu | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Dorgu; fpl:Hamstring injury - Expected back 25 Oct |
| Shaw | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Shaw; fpl:Unspecified injury - 75% chance of playing |
| Ugarte | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ugarte; fpl:Knee injury - Unknown return date |
| Šeško | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Sesko; fpl:Shin injury - Expected back 18 Oct |
| Jaouen | Newcastle | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Jaouen; fpl:Hamstring injury - Unknown return date |
| Livramento | Newcastle | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Livramento; fpl:Unspecified injury - 50% chance of playing |
| Igor Jesus | Nott'm Forest | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Igor Jesus; fpl:Calf injury - Unknown return date |
| Savona | Nott'm Forest | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Savona; fpl:Knee injury - Unknown return date |
| Bergvall | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Bergvall |
| Mudryk | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mudryk; fpl:Ankle injury - Unknown return date |
| Odobert | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Odobert; fpl:Knee injury - Unknown return date |
| Tosin | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Tosin |
| Ballard | Sunderland | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ballard; fpl:Hamstring injury - Unknown return date |
| Mundle | Sunderland | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mundle; fpl:Knee injury - Unknown return date |
| Dowman | Arsenal | injured/ffscout | suspended |  | 0 | — | ffscout:FFScout Banned: Dowman |
| Sessegnon | Fulham | injured/ffscout | suspended |  | 0 | — | ffscout:FFScout Banned: Sessegnon (Ryan) |
| Bentancur | Spurs | injured/ffscout | suspended |  | 0 | — | ffscout:FFScout Banned: Bentancur; fpl:Suspended until 7 Nov |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.live.scout_minutes --gw 9 --horizon 8 --example-md reports/scout_context_example_gw09.md
```
