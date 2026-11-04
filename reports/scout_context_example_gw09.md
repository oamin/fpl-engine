# Scout context example — GW9

Source: `https://www.fantasyfootballscout.co.uk/team-news` · fetched `2026-10-31T07:45:32.860375+00:00` · ρ=0.85 · overlays **114**

Combined signals: FFScout Out/Banned/Doubts + predicted-XI omissions + FPL `news`/status + club Latest News blurb parse.

## Counts

- `doubt`: 30
- `injured_expected_back`: 2
- `injured_unknown_return`: 53
- `suspected_single_week_rotation`: 25
- `suspended`: 4

- with `return_gws`: 31
- with blurb evidence: 7
- with FPL news: 71

## Overlay table (all)

| player | club | bucket/tag | context | return_gws | m0 | force | evidence |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| Milosavljević | Bournemouth | injured/ffscout | doubt (25% chance) | 0 | 22 | — | ffscout:FFScout Doubt: Milosavljevic (25%); fpl:Thigh injury - Expected back 31 Oct |
| Yohanna | Brighton | injured/ffscout | doubt (25% chance) | 0 | 3 | — | ffscout:FFScout Doubt: Yohanna (25%); fpl:Unspecified injury - Expected back 22 Nov |
| Lukić | Ipswich | injured/ffscout | doubt (75% chance) | 0 | 60 | — | ffscout:FFScout Doubt: Lukic (75%); fpl:Groin injury - 75% chance of playing |
| Ruggeri | Aston Villa | injured/ffscout | injured, expected back in 11 weeks | 11 | 0 | 0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0 | ffscout:FFScout Out: Ruggeri; fpl:Hamstring injury - Expected back 2 Jan |
| Lacroix | Chelsea | injured/ffscout | injured, expected back in 2 weeks | 2 | 0 | 0:0,1:0 | ffscout:FFScout Out: Lacroix; fpl:Hamstring injury - Unknown return date |
| Lewis-Skelly | Arsenal | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=54.2; recent=[66.0, 45.0, 0.0] |
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
| Yirenkyi | Coventry | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=53.0; recent=[27.0, 17.0, 90.0] |
| Maitland-Niles | Everton | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=59.0; recent=[13.0, 74.0, 90.0] |
| Palacios | Fulham | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=45.2; recent=[60.0, 0.0, 19.0] |
| Greaves | Ipswich | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=71.4; recent=[90.0, 87.0, 0.0] |
| Cherki | Man City | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=60.4; recent=[65.0, 45.0, 84.0] |
| Guéhi | Man City | benched/ffscout_xi_omit | suspected single-week rotation | 1 | 15 | 0:15 | expand_m=90.0; recent=[90.0, 90.0, 90.0] |
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
| Tzolis | Arsenal | injured/ffscout | doubt (75% chance) |  | 60 | — | ffscout:FFScout Doubt: Tzolis (75%) |
| Dasilva | Brentford | injured/ffscout | doubt (25% chance) |  | 22 | — | ffscout:FFScout Doubt: Dasilva (25%); fpl:Hamstring injury - Unknown return date |
| Milambo | Brentford | injured/ffscout | doubt (25% chance) |  | 22 | — | ffscout:FFScout Doubt: Milambo (25%); fpl:Knee injury - Unknown return date |
| Andrés | Brighton | injured/ffscout | doubt (75% chance) |  | 41 | — | ffscout:FFScout Doubt: Chema Andrés (75%) |
| F.Kadıoğlu | Brighton | injured/ffscout | doubt (75% chance) |  | 60 | — | ffscout:FFScout Doubt: Kadioglu (75%); fpl:Hip injury - 75% chance of playing |
| Tzimas | Brighton | injured/ffscout | doubt (25% chance) |  | 22 | — | ffscout:FFScout Doubt: Tzimas (25%); fpl:Knee injury - Unknown return date |
| Thomas | Coventry | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Thomas (75%); fpl:Calf injury - 50% chance of playing |
| Strand Larsen | Crystal Palace | injured/ffscout | doubt (75% chance) |  | 52 | — | ffscout:FFScout Doubt: Strand Larsen (75%); fpl:Ankle injury - 75% chance of playing |
| Timber | Crystal Palace | injured/ffscout | doubt (75% chance) |  | 63 | — | ffscout:FFScout Doubt: Timber (75%); fpl:Hamstring injury - 75% chance of playing |
| Branthwaite | Everton | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Branthwaite (75%) |
| Grealish | Everton | injured/ffscout | doubt (50% chance) |  | 21 | — | ffscout:FFScout Doubt: Grealish (50%) |
| Butland | Hull | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Butland (75%); fpl:Arm injury - 50% chance of playing |
| Stach | Leeds | injured/ffscout | doubt (75% chance) |  | 67 | — | ffscout:FFScout Doubt: Stach (75%) |
| Munoz | Liverpool | injured/ffscout | doubt (50% chance) |  | 23 | — | ffscout:FFScout Doubt: Víctor Muñoz (50%) |
| Szoboszlai | Liverpool | injured/ffscout | doubt (50% chance) |  | 43 | — | ffscout:FFScout Doubt: Szoboszlai (50%) |
| Mainoo | Man Utd | injured/ffscout | doubt (75% chance) |  | 63 | — | ffscout:FFScout Doubt: Mainoo (75%) |
| Rashford | Man Utd | injured/ffscout | doubt (25% chance) |  | 19 | — | ffscout:FFScout Doubt: Rashford (25%); fpl:Hip injury - 75% chance of playing |
| Fernandez-Pardo | Newcastle | injured/ffscout | doubt (75% chance) |  | 39 | — | ffscout:FFScout Doubt: Fernandez-Pardo (75%); fpl:Illness - 75% chance of playing |
| L.Miley | Newcastle | injured/ffscout | doubt (75% chance) |  | 58 | — | ffscout:FFScout Doubt: Miley (75%); fpl:Muscular injury - 75% chance of playing |
| Osula | Newcastle | injured/ffscout | doubt (75% chance) |  | 41 | — | ffscout:FFScout Doubt: Osula (75%) |
| Dominguez | Nott'm Forest | injured/ffscout | doubt (75% chance) |  | 1 | — | ffscout:FFScout Doubt: Nicolás Domínguez (75%); fpl:Muscular injury - 75% chance of playing |
| Ndoye | Nott'm Forest | injured/ffscout | doubt (75% chance) |  | 48 | — | ffscout:FFScout Doubt: Ndoye (75%) |
| Isidor | Sunderland | injured/ffscout | doubt (75% chance) |  | 17 | — | ffscout:FFScout Doubt: Isidor (75%) |
| Mukiele | Sunderland | injured/ffscout | doubt (75% chance) |  | 68 | — | ffscout:FFScout Doubt: Mukiele (75%); fpl:Concussion - 75% chance of playing |
| Saka | Arsenal | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Saka; fpl:Thigh injury - Unknown return date |
| Saliba | Arsenal | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Saliba |
| Onana | Aston Villa | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mvom Onana; fpl:Knee injury - Unknown return date |
| Wan-Bissaka | Aston Villa | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Wan-Bissaka; fpl:Hamstring injury - Expected back 21 Nov |
| J.Araujo | Bournemouth | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Araujo; fpl:Thigh injury - Unknown return date |
| Scott | Bournemouth | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Scott; fpl:Thigh injury - Expected back 21 Nov |
| Tóth.A | Bournemouth | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Tóth; fpl:Hamstring injury - Unknown return date |
| Carvalho | Brentford | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Carvalho; fpl:Knock - Expected back 23 Nov |
| Furo | Brentford | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Furo; fpl:Unspecified injury - Unknown return date |
| Henry | Brentford | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Henry; fpl:Hip injury - Unknown return date |
| Azeez | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Azeez; fpl:Hamstring injury - Expected back 8 Nov |
| Ferguson | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ferguson; fpl:Ankle injury - Unknown return date |
| Mitoma | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mitoma; fpl:Hamstring injury - Expected back 22 Nov |
| Struijk | Brighton | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Struijk; fpl:Hamstring injury - Expected back 8 Nov |
| Anselmino | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Anselmino; fpl:Unspecified injury - Unknown return date |
| Gittens | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Gittens; fpl:Ankle injury - Unknown return date |
| Hato | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Hato; fpl:Unspecified injury - Unknown return date |
| James | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: James; fpl:Knee injury - Unknown return date |
| João Pedro | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: João Pedro; fpl:Thigh injury - Unknown return date |
| Martinez | Chelsea | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Martinez; fpl:Head injury - Unknown return date |
| Amenda | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Amenda; fpl:Calf injury - Unknown return date |
| Eccles | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Eccles; fpl:Ankle injury - Unknown return date |
| Woolfenden | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Woolfenden; fpl:Knee injury - Unknown return date |
| Wright | Coventry | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Wright; fpl:Thigh injury - Unknown return date |
| Chadi Riad | Crystal Palace | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Riad; fpl:Knee injury - Unknown return date |
| Sarr | Crystal Palace | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ismaila Sarr; fpl:Groin injury - Expected back 8 Nov |
| Tomiyasu | Crystal Palace | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Tomiyasu; fpl:Calf injury - Expected back 8 Nov |
| Ansah | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ansah; fpl:Back injury - Unknown return date |
| Hughes | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Hughes; fpl:Groin injury - Unknown return date |
| Matazo | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Matazo; fpl:Knee injury - Unknown return date |
| Targett | Hull | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Targett; fpl:Ankle injury - Unknown return date |
| Mateo Joseph | Leeds | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mateo Joseph; fpl:Knee injury - Unknown return date |
| Rodon | Leeds | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Rodon; fpl:Hamstring injury - Expected back 21 Nov |
| Barcola | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Bradley |
| Ekitiké | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ekitiké; fpl:Achilles injury - Unknown return date |
| Frimpong | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Frimpong; fpl:Hamstring injury - Unknown return date |
| Leoni | Liverpool | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Leoni; fpl:Knee injury - Unknown return date |
| Semenyo | Man City | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Semenyo; fpl:Knee injury - Unknown return date |
| Dorgu | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Dorgu; fpl:Hamstring injury - Unknown return date |
| Shaw | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Shaw; fpl:Unspecified injury - Unknown return date |
| Ugarte | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ugarte; fpl:Knee injury - Unknown return date |
| Šeško | Man Utd | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Sesko; fpl:Shin injury - Unknown return date |
| Jaouen | Newcastle | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Jaouen; fpl:Hamstring injury - Unknown return date |
| Livramento | Newcastle | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Livramento; fpl:Knee injury - Unknown return date |
| Igor Jesus | Nott'm Forest | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Igor Jesus; fpl:Calf injury - Unknown return date |
| Savona | Nott'm Forest | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Savona; fpl:Knee injury - Unknown return date |
| Bergvall | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Bergvall; fpl:Knee injury - Expected back 31 Oct |
| Mudryk | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mudryk; fpl:Ankle injury - Unknown return date |
| Odobert | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Odobert; fpl:Knee injury - Unknown return date |
| Tosin | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Tosin; fpl:Calf injury - Unknown return date |
| Xavi | Spurs | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Simons; fpl:Knee injury - Unknown return date |
| Ballard | Sunderland | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Ballard; fpl:Hamstring injury - Unknown return date |
| Mundle | Sunderland | injured/ffscout | injured, unknown return |  | 0 | — | ffscout:FFScout Out: Mundle; fpl:Knee injury - Unknown return date |
| Dowman | Arsenal | injured/ffscout | suspended |  | 0 | — | ffscout:FFScout Banned: Dowman; fpl:Suspended until 29 Nov |
| Sessegnon | Fulham | injured/ffscout | suspended |  | 0 | — | ffscout:FFScout Banned: Sessegnon (Ryan) |
| Bentancur | Spurs | injured/ffscout | suspended |  | 0 | — | ffscout:FFScout Banned: Bentancur |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.live.scout_minutes --gw 9 --horizon 8 --example-md reports/scout_context_example_gw09.md
```
