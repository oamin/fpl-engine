# Scout context example — GW9

Source: `https://www.fantasyfootballscout.co.uk/team-news` · fetched `2026-10-31T07:45:32.860375+00:00` · ρ=0.85 · overlays **105**

Combined signals: FFScout Out/Banned/Doubts + predicted-XI omissions + FPL `news`/status + club Latest News blurb parse.

## Counts

- `doubt`: 30
- `injured_expected_back`: 2
- `injured_unknown_return`: 53
- `suspected_single_week_rotation`: 16
- `suspended`: 4

- with `return_gws`: 22
- with blurb evidence: 7
- with FPL news: 70

## Overlay table (all)

| player | club | bucket/tag | context | partner | return_gws | m0 | force |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Milosavljević | Bournemouth | injured/ffscout | doubt (25% chance) | — | 0 | 22 | — |
| Yohanna | Brighton | injured/ffscout | doubt (25% chance); Minutes anti-correlated with Andrés (ρ=-0.42) — when one … | Andrés (ρ=-0.42) | 0 | 3 | — |
| Lukić | Ipswich | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Matusiwa (ρ=-0.81) — when on… | Matusiwa (ρ=-0.81) | 0 | 60 | — |
| Ruggeri | Aston Villa | injured/ffscout | injured, expected back in 11 weeks; Minutes anti-correlated with Maatsen (ρ=-… | Maatsen (ρ=-0.66) | 11 | 0 | 0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0 |
| Lacroix | Chelsea | injured/ffscout | injured, expected back in 2 weeks; Minutes anti-correlated with James (ρ=-0.4… | James (ρ=-0.48) | 2 | 0 | 0:0,1:0 |
| Lewis-Skelly | Arsenal | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Bruno G. (ρ=-0.9… | Bruno G. (ρ=-0.93) | 1 | 15 | 0:15 |
| Buendía | Aston Villa | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Mbaye (ρ=-0.82) … | Mbaye (ρ=-0.82) | 1 | 15 | 0:15 |
| Mings | Aston Villa | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Pau (ρ=-0.99) — … | Pau (ρ=-0.99) | 1 | 15 | 0:15 |
| Anthony | Brentford | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with O.Dango (ρ=-0.86… | O.Dango (ρ=-0.86) | 1 | 15 | 0:15 |
| Yirenkyi | Coventry | benched/ffscout_xi_omit | suspected single-week rotation | — | 1 | 15 | 0:15 |
| Maitland-Niles | Everton | benched/ffscout_xi_omit | suspected single-week rotation | — | 1 | 15 | 0:15 |
| Thomas | Hull | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Slater (ρ=-0.93)… | Slater (ρ=-0.93) | 1 | 15 | 0:15 |
| Greaves | Ipswich | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Ouattara (ρ=-0.7… | Ouattara (ρ=-0.77) | 1 | 15 | 0:15 |
| Cherki | Man City | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Kovačić (ρ=-0.76… | Kovačić (ρ=-0.76) | 1 | 15 | 0:15 |
| Guéhi | Man City | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Lewis (ρ=-0.56) … | Lewis (ρ=-0.56) | 1 | 15 | 0:15 |
| Tielemans | Man Utd | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Lacey (ρ=-0.68) … | Lacey (ρ=-0.68) | 1 | 15 | 0:15 |
| Botman | Newcastle | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Burn (ρ=-1.00) —… | Burn (ρ=-1.00) | 1 | 15 | 0:15 |
| Thiaw | Newcastle | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Burn (ρ=-1.00) —… | Burn (ρ=-1.00) | 1 | 15 | 0:15 |
| Aina | Nott'm Forest | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Jair Cunha (ρ=-0… | Jair Cunha (ρ=-0.54) | 1 | 15 | 0:15 |
| Gray | Spurs | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Mudryk (ρ=-0.96)… | Mudryk (ρ=-0.96) | 1 | 15 | 0:15 |
| Danso | Sunderland | benched/ffscout_xi_omit | suspected single-week rotation; Minutes anti-correlated with Hume (ρ=-0.74) —… | Hume (ρ=-0.74) | 1 | 15 | 0:15 |
| Yarmoliuk | Brentford | injured/ffscout | suspended, expected back in 3 weeks; Minutes anti-correlated with O.Dango (ρ=… | O.Dango (ρ=-0.81) | 3 | 0 | 0:0,1:0,2:0 |
| Calafiori | Arsenal | injured/ffscout | doubt (75% chance); Minutes anti-correlated with White (ρ=-0.45) — when one s… | White (ρ=-0.45) |  | 48 | — |
| Madueke | Arsenal | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Tzolis (ρ=-0.86) — when one … | Tzolis (ρ=-0.86) |  | 56 | — |
| Rice | Arsenal | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Lewis-Skelly (ρ=-0.62) — whe… | Lewis-Skelly (ρ=-0.62) |  | 65 | — |
| Tzolis | Arsenal | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Madueke (ρ=-0.86) — when one… | Madueke (ρ=-0.86) |  | 60 | — |
| Dasilva | Brentford | injured/ffscout | doubt (25% chance) | — |  | 22 | — |
| Milambo | Brentford | injured/ffscout | doubt (25% chance) | — |  | 22 | — |
| Andrés | Brighton | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Osman (ρ=-0.69) — when one s… | Osman (ρ=-0.69) |  | 58 | — |
| F.Kadıoğlu | Brighton | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Costinha (ρ=-0.75) — when on… | Costinha (ρ=-0.75) |  | 46 | — |
| Tzimas | Brighton | injured/ffscout | doubt (25% chance) | — |  | 22 | — |
| Thomas | Coventry | injured/ffscout | doubt (75% chance) | — |  | 60 | — |
| Strand Larsen | Crystal Palace | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Mateta (ρ=-0.85) — when one … | Mateta (ρ=-0.85) |  | 56 | — |
| Timber | Crystal Palace | injured/ffscout | doubt (75% chance); Minutes anti-correlated with McNeil (ρ=-0.85) — when one … | McNeil (ρ=-0.85) |  | 56 | — |
| Branthwaite | Everton | injured/ffscout | doubt (75% chance); Minutes anti-correlated with O'Brien (ρ=-0.98) — when one… | O'Brien (ρ=-0.98) |  | 68 | — |
| Grealish | Everton | injured/ffscout | doubt (50% chance); Minutes anti-correlated with George (ρ=-0.88) — when one … | George (ρ=-0.88) |  | 41 | — |
| Butland | Hull | injured/ffscout | doubt (75% chance) | — |  | 68 | — |
| Stach | Leeds | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Wilson (ρ=-0.55) — when one … | Wilson (ρ=-0.55) |  | 51 | — |
| Munoz | Liverpool | injured/ffscout | doubt (50% chance); Minutes anti-correlated with Gakpo (ρ=-0.68) — when one s… | Gakpo (ρ=-0.68) |  | 31 | — |
| Szoboszlai | Liverpool | injured/ffscout | doubt (50% chance); Minutes anti-correlated with Barcola (ρ=-0.69) — when one… | Barcola (ρ=-0.69) |  | 34 | — |
| Mainoo | Man Utd | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Andrey Santos (ρ=-1.00) — wh… | Andrey Santos (ρ=-1.00) |  | 65 | — |
| Rashford | Man Utd | injured/ffscout | doubt (25% chance); Minutes anti-correlated with Amad (ρ=-0.93) — when one si… | Amad (ρ=-0.93) |  | 13 | — |
| Fernandez-Pardo | Newcastle | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Osula (ρ=-0.50) — when one s… | Osula (ρ=-0.50) |  | 54 | — |
| L.Miley | Newcastle | injured/ffscout | doubt (75% chance); Minutes anti-correlated with J.Murphy (ρ=-0.50) — when on… | J.Murphy (ρ=-0.50) |  | 60 | — |
| Osula | Newcastle | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Wissa (ρ=-0.69) — when one s… | Wissa (ρ=-0.69) |  | 30 | — |
| Dominguez | Nott'm Forest | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Gibbs-White (ρ=-1.00) — when… | Gibbs-White (ρ=-1.00) |  | 28 | — |
| Ndoye | Nott'm Forest | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Gibbs-White (ρ=-0.39) — when… | Gibbs-White (ρ=-0.39) |  | 53 | — |
| Isidor | Sunderland | injured/ffscout | doubt (75% chance); Minutes anti-correlated with Brobbey (ρ=-0.37) — when one… | Brobbey (ρ=-0.37) |  | 33 | — |
| Mukiele | Sunderland | injured/ffscout | doubt (75% chance); Minutes anti-correlated with O'Nien (ρ=-0.78) — when one … | O'Nien (ρ=-0.78) |  | 58 | — |
| Saka | Arsenal | injured/ffscout | injured, unknown return; Minutes anti-correlated with Merino (ρ=-0.89) — when… | Merino (ρ=-0.89) |  | 0 | — |
| Saliba | Arsenal | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Onana | Aston Villa | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Wan-Bissaka | Aston Villa | injured/ffscout | injured, unknown return; Minutes anti-correlated with Cash (ρ=-0.63) — when o… | Cash (ρ=-0.63) |  | 0 | — |
| J.Araujo | Bournemouth | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Scott | Bournemouth | injured/ffscout | injured, unknown return; Minutes anti-correlated with Gannon-Doak (ρ=-0.68) —… | Gannon-Doak (ρ=-0.68) |  | 0 | — |
| Tóth.A | Bournemouth | injured/ffscout | injured, unknown return; Minutes anti-correlated with Adams (ρ=-0.66) — when … | Adams (ρ=-0.66) |  | 0 | — |
| Carvalho | Brentford | injured/ffscout | injured, unknown return; Minutes anti-correlated with M.Sangaré (ρ=-0.60) — w… | M.Sangaré (ρ=-0.60) |  | 0 | — |
| Furo | Brentford | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Henry | Brentford | injured/ffscout | injured, unknown return; Minutes anti-correlated with Hickey (ρ=-0.55) — when… | Hickey (ρ=-0.55) |  | 0 | — |
| Azeez | Brighton | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Ferguson | Brighton | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Mitoma | Brighton | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Struijk | Brighton | injured/ffscout | injured, unknown return; Minutes anti-correlated with Dunk (ρ=-0.99) — when o… | Dunk (ρ=-0.99) |  | 0 | — |
| Anselmino | Chelsea | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Gittens | Chelsea | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Hato | Chelsea | injured/ffscout | injured, unknown return; Minutes anti-correlated with Chavarria (ρ=-0.94) — w… | Chavarria (ρ=-0.94) |  | 0 | — |
| James | Chelsea | injured/ffscout | injured, unknown return; Minutes anti-correlated with Palestra (ρ=-0.51) — wh… | Palestra (ρ=-0.51) |  | 0 | — |
| João Pedro | Chelsea | injured/ffscout | injured, unknown return; Minutes anti-correlated with Welbeck (ρ=-0.63) — whe… | Welbeck (ρ=-0.63) |  | 0 | — |
| Martinez | Chelsea | injured/ffscout | injured, unknown return; Minutes anti-correlated with Sánchez (ρ=-0.65) — whe… | Sánchez (ρ=-0.65) |  | 0 | — |
| Amenda | Coventry | injured/ffscout | injured, unknown return; Minutes anti-correlated with Pinnock (ρ=-0.58) — whe… | Pinnock (ρ=-0.58) |  | 0 | — |
| Eccles | Coventry | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Woolfenden | Coventry | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Wright | Coventry | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Chadi Riad | Crystal Palace | injured/ffscout | injured, unknown return; Minutes anti-correlated with Khalaili (ρ=-0.96) — wh… | Khalaili (ρ=-0.96) |  | 0 | — |
| Sarr | Crystal Palace | injured/ffscout | injured, unknown return; Minutes anti-correlated with McNeil (ρ=-0.51) — when… | McNeil (ρ=-0.51) |  | 0 | — |
| Tomiyasu | Crystal Palace | injured/ffscout | injured, unknown return; Minutes anti-correlated with Canvot (ρ=-0.40) — when… | Canvot (ρ=-0.40) |  | 0 | — |
| Ansah | Hull | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Hughes | Hull | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Matazo | Hull | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Targett | Hull | injured/ffscout | injured, unknown return; Minutes anti-correlated with Ajayi (ρ=-1.00) — when … | Ajayi (ρ=-1.00) |  | 0 | — |
| Mateo Joseph | Leeds | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Rodon | Leeds | injured/ffscout | injured, unknown return; Minutes anti-correlated with Gudmundsson (ρ=-0.87) —… | Gudmundsson (ρ=-0.87) |  | 0 | — |
| Barcola | Liverpool | injured/ffscout | injured, unknown return; Minutes anti-correlated with Szoboszlai (ρ=-0.69) — … | Szoboszlai (ρ=-0.69) |  | 0 | — |
| Ekitiké | Liverpool | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Frimpong | Liverpool | injured/ffscout | injured, unknown return; Minutes anti-correlated with Jacquet (ρ=-0.98) — whe… | Jacquet (ρ=-0.98) |  | 0 | — |
| Leoni | Liverpool | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Semenyo | Man City | injured/ffscout | injured, unknown return; Minutes anti-correlated with Doku (ρ=-0.95) — when o… | Doku (ρ=-0.95) |  | 0 | — |
| Dorgu | Man Utd | injured/ffscout | injured, unknown return; Minutes anti-correlated with Amad (ρ=-0.55) — when o… | Amad (ρ=-0.55) |  | 0 | — |
| Shaw | Man Utd | injured/ffscout | injured, unknown return; Minutes anti-correlated with Dalot (ρ=-0.58) — when … | Dalot (ρ=-0.58) |  | 0 | — |
| Ugarte | Man Utd | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Šeško | Man Utd | injured/ffscout | injured, unknown return; Minutes anti-correlated with Zirkzee (ρ=-0.57) — whe… | Zirkzee (ρ=-0.57) |  | 0 | — |
| Jaouen | Newcastle | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Livramento | Newcastle | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Igor Jesus | Nott'm Forest | injured/ffscout | injured, unknown return; Minutes anti-correlated with Delap (ρ=-0.91) — when … | Delap (ρ=-0.91) |  | 0 | — |
| Savona | Nott'm Forest | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Bergvall | Spurs | injured/ffscout | injured, unknown return; Minutes anti-correlated with Fernandes (ρ=-0.39) — w… | Fernandes (ρ=-0.39) |  | 0 | — |
| Mudryk | Spurs | injured/ffscout | injured, unknown return; Minutes anti-correlated with Gray (ρ=-0.96) — when o… | Gray (ρ=-0.96) |  | 0 | — |
| Odobert | Spurs | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Tosin | Spurs | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Xavi | Spurs | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Ballard | Sunderland | injured/ffscout | injured, unknown return; Minutes anti-correlated with Hume (ρ=-0.63) — when o… | Hume (ρ=-0.63) |  | 0 | — |
| Mundle | Sunderland | injured/ffscout | injured, unknown return | — |  | 0 | — |
| Dowman | Arsenal | injured/ffscout | suspended; Minutes anti-correlated with Saka (ρ=-0.72) — when one sits, the o… | Saka (ρ=-0.72) |  | 0 | — |
| Sessegnon | Fulham | injured/ffscout | suspended; Minutes anti-correlated with Robinson (ρ=-0.96) — when one sits, t… | Robinson (ρ=-0.96) |  | 0 | — |
| Bentancur | Spurs | injured/ffscout | suspended; Minutes anti-correlated with Maddison (ρ=-0.80) — when one sits, t… | Maddison (ρ=-0.80) |  | 0 | — |

## Reproduce

```bash
PYTHONPATH=. python3 -m src.live.scout_minutes --gw 9 --horizon 8 --example-md reports/scout_partner_blurb_example_gw09.md
```
