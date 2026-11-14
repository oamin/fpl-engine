# Opponent-card cross-read (training step)

One masked GW on the club being painted attends over the opponent card's earlier rows. Illustrated on Arsenal 2025-26 GW13 away at Chelsea.

![Opponent cross-read](baudot_opponent_crossread.png)

## Rule

1. Pair club-season cards by fixture id.
2. Keys: opponent rows with an earlier kickoff. Goals, assists, and gc on those rows are true.
3. Both cards hide the current match. The painted club's gc that week is the opponent's goals, so it is not an input.
4. The painted row keeps who, minutes, and odds. Its G/A and gc are blank. Attention reads the opponent's past rows, not the opponent's XI in this match.
5. The match-up signal is the opponent's past row-sums (goals scored, goals conceded).
6. Loss is the painted club's masked G/A only. A remainder cut repeats the step once per masked GW, each with its own opponent.

Chelsea had 12 earlier rows before this kickoff (mean gc 0.92). The last five conceded counts are 0, 2, 0, 0, 0.
