# Current comparison

The reportable results are `reports/procedure_audit.md`, `reports/decision_layer.md`, `reports/decision_placebo.md`, `reports/common_state.md`, `reports/disagreement.md`, and `reports/loso.md`.

`src/eval` is the default path. A result is written only by `write_gated_report`, after the as-of audit and the paired intervals. The certified historical comparison is score_xp against expected points. Scraped Vaastav `xP` is not a benchmark. A test that reads official xP fails unless every row has a capture time strictly before the canonical deadline. The live encompassing decision is held. At 20 live weeks an interval that covers zero is undetermined, and the test continues through gameweek 38. The decision-layer replay uses `score_xp` and does not name a winner against `ep_next` or against expected points. Greedy minus a squad that never transfers is not the skill comparison. The paired contrast against the shuffled-score greedy squad is in `reports/decision_placebo.md`. The common-state test, one frozen squad and the same legal moves, is in `reports/common_state.md`. The disagreement weeks, the shuffle leg for expected points, and the frozen opening fifteens are in `reports/disagreement.md`. Nothing in those files was used to change the rule. `reports/loso.md` re-aggregates stored weeks and does not replace the four-season intervals. From gameweek 6 the live choice primary is `ep_next`, with `score_xp` logged beside it. The scorer is not rewired. An interval that covers zero at 20 live weeks stays undetermined through gameweek 38.

Stages 14–48 on the played-only pool are listed in `reports/SUPERSEDED.md`.

Pull requests 33–37 and 39–56 are closed. Pull requests 1 and 6–32 are closed as superseded by that later path. Their branches are kept. #57 is merged. #58 stays a draft. A chip simulator stays deferred. Merging pull request 58 onto `main` is a PI action on GitHub. This checkout does not merge it.

The clean 2026/27 holdout starts at gameweek 6. Gameweeks 1–5 were already read. Pre-deadline forecasts are timestamped files under `data/predictions/`.
