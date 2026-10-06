# Named elevens who played zero minutes, Gameweeks 1–5

A starter for the model is one of the eleven named before the deadline, before automatic substitutes. The slot stored for a manager is not that. The public picks rewrite the slot after the gameweek: a player who was substituted out sits on the bench, and the player who came in sits in the eleven. The list that shows this is `automatic_subs`. The files in `data/entry` do not keep that list. Reading slot 11 or earlier as the submitted eleven counts the scored eleven.

On that scored eleven the managers have 0 starters who played 0 minutes. Putting the substituted players back, the submitted elevens have 10. All 10 were replaced. The model named 23, and all 23 were replaced. The substitutes on the model side scored 97 points over the 22 manager-weeks that had one of these starts. Adam Whitting's Gameweek 5 had two.

The 10 human starters were João Pedro in Gameweek 5 (Will Morrison, Paul Mitchell, Jess Bernstein, Adam Whitting), O'Reilly in Gameweek 3 (Cameron Scott), Palestra in Gameweek 1 (Will Morrison), Watkins in Gameweeks 1 and 2 (Jess Bernstein), Pedro Porro in Gameweek 4 (Adam Whitting), and Caicedo in Gameweek 1 (Sion Jones).

The 23 are six players. They are not one fault.

## João Pedro, Gameweek 5: 14 of the 23

Every model squad named him. The carried score was 5.30. He had played 90 minutes in each of Gameweeks 1–4, for 11, 9, 1, and 12. Gameweek 5 was his first zero. A score built from earlier minutes still starts him.

Seven of the 14 still owned him. Four started him, and the picks then moved him to the bench:

| manager | named shape | substitute |
|---|---|---|
| Will Morrison | 3-4-3 | Groß, midfielder |
| Paul Mitchell | 3-4-3 | Groß, midfielder |
| Adam Whitting | 3-4-3 | E.Le Fée, midfielder |
| Jess Bernstein | 4-4-2, and she captained him | Slater, midfielder |

Sion Jones, Viniii Denie, and Tom Anderson had no automatic substitute that week, and João Pedro was not their captain, so those three did leave him on the bench. The other seven did not own him. The sheets still have no status taken before the deadline. Four managers started him anyway.

## Rico Lewis, the repeat: 3 of the 23

He played 56 minutes in Gameweek 1, then 0 in Gameweeks 2–5. Adam Whitting's model named him in Gameweeks 2, 3, 4, and 5, still on the Gameweek 1 score of 3.99.

Gameweek 2 is the first zero. It is not knowable at that deadline. Gameweeks 3–5 are. A row with 0 minutes is dropped before the priors are built, so an owned player who is missing from that week's score keeps the last score he had. The zeros never enter expected minutes. None of the 14 owned him.

## The other six are first zeros

| player | week | squads | carried score | what the owners did |
|---|---:|---|---:|---|
| Rico Lewis | 2 | Adam Whitting | 3.99 | not owned |
| Mosquera | 3 | Viniii Denie, elevenify.com | 3.71 | elevenify had benched him in Gameweek 2; both had sold him by Gameweek 3 |
| Rodon | 3 | daniel bowes | 3.44 | bowes benched him |
| Colwill | 3 | Jess Bernstein | 4.13 | Jess benched him in all five weeks |
| Shaw | 4 | daniel bowes | 3.50 | Adam Whitting benched him; Paul Mitchell no longer owned him |

Twenty of the 770 model starts came after that player had already recorded a zero earlier in 2026/27. Three of those twenty were also zeros this week, and all three are Lewis in Gameweeks 3–5. The other seventeen played. Enzo missed Gameweek 2 and then played 75, 81, and 90. Elvedi missed Gameweek 1 and then played every week. A rule that treats one missed week as the next week's minutes would have cut both.

## The substitute from three forwards

A squad has three forwards. If all three are named and one plays 0 minutes, two forwards remain, so the next substitute can be a midfielder or a defender. `bank_squad_gw` orders the bench by `score_xp` before the deadline, goalkeeper first and then the outfield. `apply_autosubs` walks that outfield order. It skips a substitute who played 0 minutes. It skips a substitute who would leave the eleven illegal, which includes leaving it with no forward. The first player who played and keeps a legal eleven comes in.

On a 3-4-3 with João Pedro on 0 minutes, that player is the highest-scoring outfield substitute. In the checked case he is a midfielder, the eleven becomes 3-5-2, and his points are the substitute points. If that midfielder also played 0 minutes, the next defender comes in and the eleven becomes 4-4-2. If João Pedro is the only forward named, a midfielder or a defender would leave no forward, so he stays and the scored eleven keeps the zero.

That is the same replacement the four managers received. Will, Paul, and Adam named 3-4-3 and a midfielder came in. Jess named two forwards, so a midfielder was still legal, and one came in.

## What this does not change

The scored points already use the eleven after substitutes, and the official total. The gap of −27.43 is that comparison. What was wrong was reading the stored slot as the eleven named before the deadline. Writing a missed week into the next minutes prior would move Lewis from Gameweek 3. It would not have benched João Pedro. It is not a change to `score_xp`, and it is not fit on these five weeks.
