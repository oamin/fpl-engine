# Opponent-card lock (2026-11-14)

Superseded the same day. The default card is the opacity reading of the Baudot row card (`reports/LOCKED.md`). This note records the lock that was replaced.

The default flashcard card and method, for this lock only, was the opponent cross-read.

- Module: `src/models/flashcard_baudot_opp_train.py`
- Table: `flashcard_card_score.GATE_OPP_2026_11_14`
- Run: `reports/baudot_opp_train.md` (mean **0.192**, seed 0, best epoch 12)

The card is `club8 · opp8 · odds12 · gc6` plus the player blocks. Each remainder row attends the opponent card. Earlier opponent rows stay true. This fixture's minutes are revealed on the opponent card and are not copied across. Goals, assists, and goals conceded for the match stay blank on both cards. Later opponent kickoffs stay hidden. The loss on this lock is learnable Tversky on the assessed club's remainder events.

`flashcard_card_score.GATE` remains the 2026-11-13 mixer table (mean **0.218**). This lock does not rewrite that score. Screens that change the opponent-card method compare to `GATE_OPP_2026_11_14` and still quote `GATE`.
