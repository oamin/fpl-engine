#!/usr/bin/env python3
"""Demo: Opta GF/GA → match λ → player forecast_xp for Arsenal vs Fulham.

Uses the two 2026/27 fixtures (GW18 Fulham home, GW26 Arsenal home).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.opta_gf_ga_horizon import (
    bootstrap_shares,
    forecast_pots_from_opta_gf_ga,
    load_opta_ranks,
    match_prediction,
    player_forecast_xp,
)
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "data/predictions/2026-27/gw06/betfair_20261008/bootstrap.json"
FIX = ROOT / "data/predictions/2026-27/gw06/betfair_20261008/fixtures.json"
OUT_JSON = ROOT / "reports/figures/opta_gf_ga_horizon_arsenal_fulham.json"
OUT_MD = ROOT / "reports/opta_gf_ga_horizon_arsenal_fulham.md"


def main() -> None:
    boot = json.loads(BOOT.read_text())
    fixtures = json.loads(FIX.read_text())
    teams = {int(t["id"]): str(t["name"]) for t in boot["teams"]}
    team_id = {norm_team(n): i for i, n in teams.items()}

    pairs = []
    for f in fixtures:
        h, a = teams[int(f["team_h"])], teams[int(f["team_a"])]
        if {norm_team(h), norm_team(a)} == {"arsenal", "fulham"}:
            pairs.append(f)

    ranks = load_opta_ranks()
    rank_map = {r["club_norm"]: r for r in ranks}

    fixtures_out = []
    for f in pairs:
        home, away = teams[int(f["team_h"])], teams[int(f["team_a"])]
        pred = match_prediction(home, away)
        home_players = bootstrap_shares(boot, team_id=team_id[norm_team(home)])
        away_players = bootstrap_shares(boot, team_id=team_id[norm_team(away)])
        # Only players with minutes this season for a readable XI-ish list.
        home_xi = [p for p in home_players if p["minutes_season"] > 0] or home_players
        away_xi = [p for p in away_players if p["minutes_season"] > 0] or away_players
        home_xp = player_forecast_xp(home_xi, pred["pot_home"])[:8]
        away_xp = player_forecast_xp(away_xi, pred["pot_away"])[:8]
        fixtures_out.append(
            {
                "gw": int(f["event"]),
                "kickoff": f.get("kickoff_time"),
                "home": home,
                "away": away,
                "prediction": {
                    k: pred[k]
                    for k in (
                        "home_rank",
                        "away_rank",
                        "home_season",
                        "away_season",
                        "lam_home",
                        "lam_away",
                        "lam_home_lo",
                        "lam_home_hi",
                        "lam_away_lo",
                        "lam_away_hi",
                        "e_total",
                        "p_cs_home",
                        "p_cs_away",
                    )
                },
                "top_forecast_xp_home": home_xp,
                "top_forecast_xp_away": away_xp,
            }
        )

    # Also emit full unpriced pot map for GW15–26 as a smoke check.
    pots = forecast_pots_from_opta_gf_ga(
        fixtures,
        teams,
        start=15,
        end=26,
        priced_weeks=set(),
    )

    payload = {
        "method": "Opta rank → OLS GF/GA (±MAE bands) → Dixon–Coles λ → xp_on_pot",
        "opta_as_of": ranks[0].get("as_of"),
        "arsenal_rank": rank_map["arsenal"]["rank"],
        "fulham_rank": rank_map["fulham"]["rank"],
        "fixtures": fixtures_out,
        "n_pots_gw15_26": len(pots),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Long-horizon match pots from Opta GF/GA — Arsenal vs Fulham",
        "",
        "Pipeline: **Opta predicted rank** → season GF/GA with ±MAE bands "
        "(closed-season OLS) → Dixon–Coles match λ → standard "
        "``xp_on_pot`` with bootstrap player shares.",
        "",
        f"- Opta as-of: **{ranks[0].get('as_of')}**",
        f"- Arsenal rank **{rank_map['arsenal']['rank']}**, "
        f"Fulham rank **{rank_map['fulham']['rank']}**",
        f"- Unpriced pots emitted for GW15–26 keys: **{len(pots)}**",
        "",
    ]
    for fx in fixtures_out:
        p = fx["prediction"]
        hs, aws = p["home_season"], p["away_season"]
        lines += [
            f"## GW{fx['gw']}: {fx['home']} vs {fx['away']}",
            "",
            f"Kickoff: `{fx['kickoff']}`",
            "",
            "| side | Opta rk | Season GF | Season GA | λ (match) | λ band | P(CS) |",
            "|---|---:|---:|---:|---:|---|---:|",
            f"| {fx['home']} | {p['home_rank']:.0f} | "
            f"{hs['gf']:.0f} [{hs['gf_lo']:.0f},{hs['gf_hi']:.0f}] | "
            f"{hs['ga']:.0f} [{hs['ga_lo']:.0f},{hs['ga_hi']:.0f}] | "
            f"{p['lam_home']:.2f} | [{p['lam_home_lo']:.2f},{p['lam_home_hi']:.2f}] | "
            f"{p['p_cs_home']:.2f} |",
            f"| {fx['away']} | {p['away_rank']:.0f} | "
            f"{aws['gf']:.0f} [{aws['gf_lo']:.0f},{aws['gf_hi']:.0f}] | "
            f"{aws['ga']:.0f} [{aws['ga_lo']:.0f},{aws['ga_hi']:.0f}] | "
            f"{p['lam_away']:.2f} | [{p['lam_away_lo']:.2f},{p['lam_away_hi']:.2f}] | "
            f"{p['p_cs_away']:.2f} |",
            "",
            f"Implied scoreline centre ≈ **{p['lam_home']:.2f}–{p['lam_away']:.2f}** "
            f"(e_total={p['e_total']:.2f}).",
            "",
            f"### Top {fx['home']} forecast_xp (appeared)",
            "",
            "| player | pos | share_xG | xmi | forecast_xp |",
            "|---|---|---:|---:|---:|",
        ]
        for r in fx["top_forecast_xp_home"]:
            lines.append(
                f"| {r['name']} | {r['position']} | {r['share_xG']:.3f} | "
                f"{r['xmi']:.0f} | {r['forecast_xp']:.2f} |"
            )
        lines += [
            "",
            f"### Top {fx['away']} forecast_xp (appeared)",
            "",
            "| player | pos | share_xG | xmi | forecast_xp |",
            "|---|---|---:|---:|---:|",
        ]
        for r in fx["top_forecast_xp_away"]:
            lines.append(
                f"| {r['name']} | {r['position']} | {r['share_xG']:.3f} | "
                f"{r['xmi']:.0f} | {r['forecast_xp']:.2f} |"
            )
        lines.append("")

    lines += [
        "## Integration",
        "",
        "```python",
        "from src.models.opta_gf_ga_horizon import forecast_pots_from_opta_gf_ga",
        "pots = forecast_pots_from_opta_gf_ga(",
        "    fixtures, team_names, start=gw, end=gw+H, priced_weeks=priced,",
        ")",
        "# then compute_player_forecast / make_forecast_steps as today",
        "```",
        "",
        "Priced weeks keep opening/Betfair pots; Opta GF/GA fills the unpriced "
        "horizon (same slot as `forecast_pots_from_outrights`).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({k: payload[k] for k in ("arsenal_rank", "fulham_rank", "n_pots_gw15_26")}, indent=2))
    for fx in fixtures_out:
        p = fx["prediction"]
        print(
            f"GW{fx['gw']} {fx['home']} vs {fx['away']}: "
            f"λ {p['lam_home']:.2f}-{p['lam_away']:.2f} "
            f"band H[{p['lam_home_lo']:.2f},{p['lam_home_hi']:.2f}] "
            f"A[{p['lam_away_lo']:.2f},{p['lam_away_hi']:.2f}]"
        )
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
