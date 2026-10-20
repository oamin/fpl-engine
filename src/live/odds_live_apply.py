"""Apply live Odds API BTTS moments + anytime props onto ``score_team_side``.

Default path is offline from a probe snapshot (no new credits)::

  python -m src.live.odds_live_apply --from-snapshot data/scratch/gw07_odds_probe.json

Builds matched FPL ids, scores each priced side with and without prop rescale,
and writes a short report. Does not touch the holdout freeze or published climbs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.live.odds_props import match_anytime_props, side_moments_from_fit
from src.markets import numagent_xp

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "data" / "scratch"
DEFAULT_BOOT = (
    ROOT / "data" / "predictions" / "2026-27" / "gw06" / "bootstrap_20261007T220914Z.json"
)
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _club_frame(
    bootstrap: dict[str, Any],
    club: str,
    *,
    prop_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Minimal Numagent side table: matched prop players + rest of club squad.

    Rates are flat placeholders for the wiring smoke (live path will pass
    rolling history). One low-rate 'returner' among prop-matched FWDs/MIDs
    demonstrates the Havertz-style lift when that player also has a prop.
    """
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    rows: list[dict[str, Any]] = []
    for el in bootstrap["elements"]:
        if teams.get(int(el["team"])) != club:
            continue
        pid = int(el["id"])
        pos = POS_MAP.get(int(el.get("element_type") or 3), "MID")
        rows.append(
            {
                "player_id": str(pid),
                "element": pid,
                "name": str(el.get("web_name") or ""),
                "position": pos,
                "rG": 0.12 if pos in {"MID", "FWD"} else 0.04,
                "drG": 0.05,
                "rA": 0.08 if pos in {"MID", "FWD"} else 0.03,
                "drA": 0.04,
                "m": 0.0,
                "sigma_m": 10.0,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Availability from FPL status / chance
    for i, el_id in enumerate(frame["element"].tolist()):
        el = next(e for e in bootstrap["elements"] if int(e["id"]) == int(el_id))
        status = str(el.get("status") or "a")
        chance = el.get("chance_of_playing_this_round")
        if status in {"i", "s", "u"} or chance == 0:
            m = 0.0
        elif status == "d" or (chance is not None and float(chance) < 75):
            m = 45.0
        else:
            m = 75.0
        frame.loc[frame.index[i], "m"] = m
    # Depress one prop-covered attacker if present (returner case).
    if prop_ids:
        cand = frame.index[
            frame["element"].isin(list(prop_ids))
            & frame["position"].isin(["MID", "FWD"])
            & (frame["m"] > 0)
        ].tolist()
        if cand:
            i = cand[-1]
            frame.loc[i, "rG"] = 0.02
            frame.loc[i, "m"] = 70.0
    return frame.reset_index(drop=True)


def apply_snapshot(
    doc: dict[str, Any],
    bootstrap: dict[str, Any],
    *,
    alpha_g: float | None = None,
    alpha_a: float | None = None,
) -> dict[str, Any]:
    if alpha_g is None or alpha_a is None:
        ag, aa, _, _ = numagent_xp.load_alphas()
        alpha_g = ag if alpha_g is None else alpha_g
        alpha_a = aa if alpha_a is None else alpha_a

    out: dict[str, Any] = {
        "ok": bool(doc.get("ok")),
        "source": "snapshot",
        "fetched_at": doc.get("fetched_at"),
        "fixtures": [],
    }
    props_block = doc.get("props") or {}
    for match in doc.get("matches") or []:
        if not match.get("ok"):
            continue
        home = str(match["home"])
        away = str(match["away"])
        key = f"{home} v {away}"
        moments = side_moments_from_fit(match)
        prop_raw = (props_block.get(key) or {}).get("top") or []
        # Prefer full list if stored; probe stores top only — use top.
        all_props = (props_block.get(key) or {}).get("all") or prop_raw
        matched = match_anytime_props(
            all_props,
            home=home,
            away=away,
            bootstrap=bootstrap,
        )
        fx: dict[str, Any] = {
            "match": key,
            "has_btts": match.get("has_btts"),
            "w_btts": match.get("w_btts"),
            "moments": moments,
            "props_match": {
                "n_matched": matched.get("n_matched"),
                "n_props": matched.get("n_props"),
                "n_unmatched": matched.get("n_unmatched"),
                "unmatched_sample": (matched.get("unmatched") or [])[:8],
            },
            "sides": {},
        }
        by_id = matched.get("by_id") or {}
        for side_name, club_key in (("home", "home"), ("away", "away")):
            club = matched.get(club_key) or (home if side_name == "home" else away)
            side_ids = {
                int(r["element"])
                for r in matched.get("rows") or []
                if r.get("side") == side_name
            }
            club_df = _club_frame(bootstrap, str(club), prop_ids=side_ids)
            if club_df.empty:
                fx["sides"][side_name] = {"ok": False, "reason": "empty_club"}
                continue
            mom = moments[side_name]
            base = numagent_xp.score_team_side(
                club_df,
                xg=mom["xg"],
                xcs=mom["xcs"],
                xgc=mom["xgc"],
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
            )
            with_props = numagent_xp.score_team_side(
                club_df,
                xg=mom["xg"],
                xcs=mom["xcs"],
                xgc=mom["xgc"],
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
                prop_p_by_player=by_id,
            )
            # Highlight largest |Δ E_G| among prop-covered rows
            deltas: list[dict[str, Any]] = []
            for i in range(len(club_df)):
                pid = str(club_df.iloc[i]["player_id"])
                if pid not in by_id:
                    continue
                deltas.append(
                    {
                        "element": int(pid),
                        "name": str(club_df.iloc[i]["name"]),
                        "p_anytime": float(by_id[pid]),
                        "E_G0": float(base.iloc[i]["E_G"]),
                        "E_G1": float(with_props.iloc[i]["E_G"]),
                        "sG0": float(base.iloc[i]["sG"]),
                        "sG1": float(with_props.iloc[i]["sG"]),
                        "xp0": float(base.iloc[i]["xp"]),
                        "xp1": float(with_props.iloc[i]["xp"]),
                    }
                )
            deltas.sort(key=lambda r: abs(r["E_G1"] - r["E_G0"]), reverse=True)
            fx["sides"][side_name] = {
                "ok": True,
                "club": club,
                "n_players": len(club_df),
                "xg": mom["xg"],
                "G_team": float(with_props.iloc[0]["G_team"]) if len(with_props) else None,
                "prop_covered": float(with_props.iloc[0]["prop_covered"])
                if "prop_covered" in with_props.columns and len(with_props)
                else 0.0,
                "top_deltas": deltas[:8],
            }
        out["fixtures"].append(fx)

    out["ok"] = bool(out["fixtures"])
    return out


def write_report(doc: dict[str, Any], path: Path) -> None:
    lines = [
        "# Live odds apply — BTTS moments + prop share rescale",
        "",
        f"Source snapshot fetched_at=`{doc.get('fetched_at')}`",
        "",
    ]
    for fx in doc.get("fixtures") or []:
        lines += [
            f"## {fx['match']}",
            "",
            f"BTTS weighted: `{fx.get('has_btts')}` (w=`{fx.get('w_btts')}`)",
            f"Props matched: `{fx.get('props_match', {}).get('n_matched')}` / "
            f"`{fx.get('props_match', {}).get('n_props')}` "
            f"(unmatched `{fx.get('props_match', {}).get('n_unmatched')}`)",
            "",
        ]
        um = fx.get("props_match", {}).get("unmatched_sample") or []
        if um:
            lines.append("Unmatched sample: " + ", ".join(um))
            lines.append("")
        for side, block in (fx.get("sides") or {}).items():
            if not block.get("ok"):
                lines.append(f"### {side}: {block.get('reason')}")
                continue
            lines += [
                f"### {side} — {block.get('club')} (xG={block.get('xg'):.3f}, "
                f"G≈{block.get('G_team'):.3f}, prop_covered≈{block.get('prop_covered'):.3f})",
                "",
                "| Player | p | E[G]0 | E[G]1 | s0 | s1 | xp0 | xp1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for r in block.get("top_deltas") or []:
                lines.append(
                    f"| {r['name']} | {r['p_anytime']:.3f} | {r['E_G0']:.3f} | "
                    f"{r['E_G1']:.3f} | {r['sG0']:.3f} | {r['sG1']:.3f} | "
                    f"{r['xp0']:.2f} | {r['xp1']:.2f} |"
                )
            lines.append("")
    lines += [
        "## Notes",
        "",
        "- Moments come from the probe fit (BTTS weight 1 when priced).",
        "- Player rates here are flat placeholders; live wiring will pass rolling history.",
        "- Prop rescale pins covered λ; residual pot stays with baseline shares.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-snapshot",
        type=Path,
        default=SCRATCH / "gw07_odds_probe.json",
    )
    ap.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOT)
    ap.add_argument("--out", type=Path, default=SCRATCH / "gw07_odds_apply.json")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "live_odds_apply.md")
    args = ap.parse_args(argv)

    doc = json.loads(args.from_snapshot.read_text(encoding="utf-8"))
    bootstrap = json.loads(args.bootstrap.read_text(encoding="utf-8"))
    applied = apply_snapshot(doc, bootstrap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(applied, indent=2, default=str), encoding="utf-8")
    write_report(applied, args.report)
    print(
        json.dumps(
            {
                "ok": applied.get("ok"),
                "n_fixtures": len(applied.get("fixtures") or []),
                "props": [
                    {
                        "match": f["match"],
                        "matched": f.get("props_match", {}).get("n_matched"),
                        "unmatched": f.get("props_match", {}).get("n_unmatched"),
                    }
                    for f in applied.get("fixtures") or []
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")
    return 0 if applied.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
