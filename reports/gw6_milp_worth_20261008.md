# Worth running MILP on GW6?

Gemini ([forecast horizon](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): **DROP for now.** Defer a fresh `wildcard_plan` MILP to the T−1h window on the Mac.

## Why not now

- The logged dry run already has wildcard ahead of 1FT by **114.12** discounted XI+C (GW6 piece **39.00**). Filling GW8–9 with κ=0.5 outrights inflates both sides; it will not flip the call under the 16 hurdle.
- Wednesday capture and minutes are stale before Friday pressers. A solve today is not submitable.
- Betfair artifacts are Mac-local; US cloud cannot pull or re-price the slate.

## What not to run

- No ad-hoc multi-period transfer MILP for `plan_half`.
- Live XV stays on T−1h `ep_next`, not engine `score_xp`.

## When it is worth it

Saturday T−1h on the Mac: fresh `slot_t1` capture, post-presser minutes, local Betfair folder → `python -m src.live.wildcard_plan --capture <official_t1>` and log the squad. That updates personnel, not the wild vs 1FT call.
