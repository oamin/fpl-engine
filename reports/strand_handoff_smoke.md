# Strand hand-off smoke (priced → Opta)

Plumbing only. Short-horizon centreline `M` → `k ∼ M` (no λ jitter) updates Opta-initialised `ClubPath` rates before the long-horizon roll continues.

- Season / GW: **2024-25** / **10**
- Priced fixtures: **10**
- Strands: **4** · mean |Δatt| vs pure Opta: **0.5316**

## Decision

**plumbing_only_no_likelihood_arm**

Register a gated path-V or player-GW log comparison before promoting hand-off into the live / cascade default.

## Outputs

- `data/processed/strand_handoff_smoke.json`

Reproduce: `PYTHONPATH=. python3 -m src.models.strand_handoff`.
