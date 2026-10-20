"""Priced t:t+1 → Opta t+2:t+n strand hand-off (plumbing).

Short-horizon scenarios sample FD Dixon–Coles λ ±15%. The long-horizon
cascade initialises ``ClubPath`` from Opta ranks. This module joins them:

1. Init paths from Opta (as today).
2. For each MC strand, sample priced-week λ from the FD fit (same band as
   ``short_horizon_cascade``) and ``observe`` those scoreline means on the
   path so t+2… carry short-horizon information.
3. Remaining fixtures stay on the Opta path roll.

This is **plumbing + smoke**, not a likelihood promote. A gated path-V /
player-log arm stays deferred until pre-registered.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.models.multistage_mc_milp import (
    ClubPath,
    init_paths_from_opta,
    pot_from_sampled_lam,
)
from src.models.short_horizon_cascade import MARKET_BAND, sample_fixture_moments
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


def apply_priced_observations(
    paths: dict[str, ClubPath],
    *,
    fixtures: Sequence[Mapping[str, Any]],
    rng: random.Random,
    band: float = MARKET_BAND,
) -> list[dict[str, Any]]:
    """Sample λ for each priced fixture and update expanding ClubPath rates.

    ``fixtures`` entries need ``home``, ``away``, ``lam_h``, ``lam_a``, and
    optionally ``rho``. Returns the sampled scoreline records.
    """
    records: list[dict[str, Any]] = []
    for fx in fixtures:
        home = norm_team(str(fx["home"]))
        away = norm_team(str(fx["away"]))
        if home not in paths or away not in paths:
            continue
        fit = {
            "lam_h": float(fx["lam_h"]),
            "lam_a": float(fx["lam_a"]),
            "rho": float(fx.get("rho", 0.0) or 0.0),
            "fixture": fx.get("fixture", 0),
        }
        sampled = sample_fixture_moments(fit, rng, band=band)
        # Soft Poisson means as observed rates (fractional observe via rounding).
        gh = int(round(float(sampled["lam_h"])))
        ga = int(round(float(sampled["lam_a"])))
        paths[home].observe(gh, ga)
        paths[away].observe(ga, gh)
        records.append(
            {
                "home": home,
                "away": away,
                "lam_h": float(sampled["lam_h"]),
                "lam_a": float(sampled["lam_a"]),
                "goals_h": gh,
                "goals_a": ga,
                "pot_home": pot_from_sampled_lam(
                    float(sampled["lam_h"]), float(sampled["lam_a"])
                ),
                "pot_away": pot_from_sampled_lam(
                    float(sampled["lam_a"]), float(sampled["lam_h"])
                ),
            }
        )
    return records


def handoff_paths(
    *,
    priced_fixtures: Sequence[Mapping[str, Any]],
    ranks: Mapping[str, float] | None = None,
    seed: int = 0,
    band: float = MARKET_BAND,
) -> tuple[dict[str, ClubPath], list[dict[str, Any]]]:
    """Opta-init paths after absorbing one priced-week scenario draw."""
    paths = init_paths_from_opta(ranks=ranks)
    rng = random.Random(int(seed))
    recs = apply_priced_observations(paths, fixtures=priced_fixtures, rng=rng, band=band)
    return paths, recs


def smoke_handoff(
    *,
    season: str = "2024-25",
    decision_gw: int = 10,
    n_strands: int = 4,
    seed: int = 7,
) -> dict[str, Any]:
    """Smoke: priced GW ``decision_gw`` FD fits → path shift vs pure Opta."""
    from src.models.short_horizon_cascade import centreline_fixture_fits, fixtures_for_gw
    from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

    merged = _load_merged_enriched(season)
    fd = _fd_index(season)
    fx = fixtures_for_gw(merged, decision_gw)
    priced = centreline_fixture_fits(fx, fd)
    if not priced:
        raise RuntimeError(f"no FD-priced fixtures for {season} GW{decision_gw}")
    # Normalise keys for apply_priced_observations.
    priced = [
        {
            "home": r["home"],
            "away": r["away"],
            "lam_h": r["lam_h"],
            "lam_a": r["lam_a"],
            "rho": r.get("rho", 0.0),
            "fixture": r.get("fixture"),
        }
        for r in priced
    ]

    base = init_paths_from_opta()
    shifts: list[dict[str, Any]] = []
    for s in range(n_strands):
        paths, recs = handoff_paths(priced_fixtures=priced, seed=seed + s)
        # Mean |Δatt| vs Opta init over clubs that played.
        played = {r["home"] for r in recs} | {r["away"] for r in recs}
        d_att = [
            abs(paths[c].att - base[c].att) for c in played if c in paths and c in base
        ]
        shifts.append(
            {
                "strand": s,
                "n_priced": len(recs),
                "n_clubs_touched": len(played),
                "mean_abs_att_shift": float(sum(d_att) / len(d_att)) if d_att else 0.0,
                "mean_path_n": float(
                    sum(paths[c].n for c in played) / max(len(played), 1)
                ),
            }
        )

    payload = {
        "task": "strand_handoff_smoke",
        "season": season,
        "decision_gw": decision_gw,
        "n_strands": n_strands,
        "band": MARKET_BAND,
        "n_priced_fixtures": len(priced),
        "strands": shifts,
        "mean_abs_att_shift": float(
            sum(r["mean_abs_att_shift"] for r in shifts) / max(len(shifts), 1)
        ),
        "decision": "plumbing_only_no_likelihood_arm",
        "note": (
            "Hand-off updates ClubPath from priced λ samples before Opta "
            "t+2… rolls. Not a promote screen — register a path-V / player-log "
            "comparison before claiming value."
        ),
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "strand_handoff_smoke.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Strand hand-off smoke (priced → Opta)",
        "",
        "Plumbing only. Short-horizon FD λ samples (±15%) update Opta-initialised "
        "`ClubPath` rates before the long-horizon roll continues.",
        "",
        f"- Season / GW: **{season}** / **{decision_gw}**",
        f"- Priced fixtures: **{len(priced)}**",
        f"- Strands: **{n_strands}** · mean |Δatt| vs pure Opta: "
        f"**{payload['mean_abs_att_shift']:.4f}**",
        "",
        "## Decision",
        "",
        "**plumbing_only_no_likelihood_arm**",
        "",
        "Register a gated path-V or player-GW log comparison before promoting "
        "hand-off into the live / cascade default.",
        "",
        "## Outputs",
        "",
        "- `data/processed/strand_handoff_smoke.json`",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.strand_handoff`.",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "strand_handoff_smoke.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    out = smoke_handoff()
    print(json.dumps({k: out[k] for k in (
        "decision", "season", "decision_gw", "n_priced_fixtures", "mean_abs_att_shift"
    )}, indent=2))


if __name__ == "__main__":
    main()
