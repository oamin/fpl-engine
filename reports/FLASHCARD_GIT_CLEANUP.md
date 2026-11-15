# Flashcard git cleanup

**Date:** 2026-11-15 (Europe/London). **Updated** with cement consolidation.

## Decision

Product on main stays **Numagent 2.0**. Flashcard research is a separate track whose
source of truth is the **typed tensor card** ([`FLASHCARD_CEMENT.md`](FLASHCARD_CEMENT.md)).

## Consolidation (this turn)

1. Land typed-tensor modules + cement docs on `main` via `cursor/flashcard-cement-c985`.
2. **Close** intermediate typed-tensor stack PRs (#185–#190) as superseded by that consolidation.
3. **Close** historical Baudot / pre-tensor flashcard PRs (#139–#184 and kin) as archaeology.
   Do not merge them. Branches may remain on the remote as read-only history; do not extend them.
4. New work: branch from `main` after cement merges — `cursor/flashcard-<topic>-c985`.

## Retired

| Item | Status |
| --- | --- |
| Baudot binary planes as training targets | retired |
| Hand-built LineupNet / FormColumn | rejected |
| Slice × horizon mixing as a product path | rejected |
| Lag-1 as default slice minutes | rejected (honesty reference only) |
| Stepped FOV painter as default | parked (screen loss) |
| Reservoir / ZIP / SMOTE / SAE for this painter | parked |
| 2026/27 holdout fit | frozen |

## Do not

- Reopen Baudot bit training.
- Mass-reopen closed archaeological PRs onto main.
- Ship flashcard → Numagent comparison until a later pre-registered gate says so.
