"""Selectable player pool for short- and long-horizon squad picks.

Filters (all required unless noted):

1. Expanding mean minutes **> 60** (as-of; leakage-free prior when a GW history
   sheet is supplied; otherwise bootstrap minutes/starts proxy).
2. FFScout forecast **firm / nailed_on** — in the public predicted XI and not
   listed Out, Banned, or Doubts (see ``src.live.ffscout_team_news``).
3. Every club keeps a **firm_selection** (predicted XI minus Out/Banned) even
   when some XI names fail the minutes cut; only players passing (1)+(2) enter
   the selectable pool used by the solver / horizon.

The pool is the only set that short- and long-horizon selection should
consider.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from src.live.ffscout_team_news import (
    match_ffs_name_to_fpl,
    parse_team_news,
    pull_team_news,
)
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
MIN_EXPANDING_MINUTES = 60.0
MIN_APPS = 2


def expanding_mean_minutes_from_history(
    history_rows: list[Mapping[str, Any]],
    *,
    before_gw: int | None = None,
) -> dict[str, dict[str, float]]:
    """player_id → {mean_minutes, n_apps, mean_when_played}.

    Uses only rows with ``gw < before_gw`` when ``before_gw`` is set.
    """
    by: dict[str, list[float]] = defaultdict(list)
    for row in history_rows:
        gw = row.get("gw", row.get("GW", row.get("round")))
        if before_gw is not None and gw is not None and int(gw) >= int(before_gw):
            continue
        pid = str(row.get("player_id") or row.get("element") or "")
        if not pid:
            continue
        minutes = float(row.get("minutes") or 0.0)
        by[pid].append(minutes)
    out: dict[str, dict[str, float]] = {}
    for pid, vals in by.items():
        played = [m for m in vals if m > 0]
        out[pid] = {
            "mean_minutes": float(sum(vals) / len(vals)) if vals else 0.0,
            "mean_minutes_played": float(sum(played) / len(played)) if played else 0.0,
            "n_apps": float(len(vals)),
            "n_played": float(len(played)),
        }
    return out


def expanding_mean_minutes_from_bootstrap(
    bootstrap: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Proxy when no GW sheet: minutes / max(starts,1), n_apps = starts."""
    out: dict[str, dict[str, float]] = {}
    for el in bootstrap.get("elements") or []:
        pid = str(el["id"])
        minutes = float(el.get("minutes") or 0.0)
        starts = int(el.get("starts") or 0)
        apps = max(starts, 1) if minutes > 0 else 0
        mean = minutes / starts if starts > 0 else 0.0
        out[pid] = {
            "mean_minutes": mean,
            "mean_minutes_played": mean,
            "n_apps": float(starts),
            "n_played": float(starts),
        }
    return out


def build_selectable_pool(
    *,
    bootstrap: Mapping[str, Any],
    ffscout: Mapping[str, Any] | None = None,
    html: str | None = None,
    history_rows: list[Mapping[str, Any]] | None = None,
    before_gw: int | None = None,
    min_minutes: float = MIN_EXPANDING_MINUTES,
    min_apps: int = MIN_APPS,
) -> dict[str, Any]:
    """Return selectable pool + per-club firm selections."""
    if ffscout is None:
        if html is not None:
            clubs = [asdict(c) for c in parse_team_news(html)]
            ffscout = {"clubs": clubs, "availability": {"mapped": True}}
        else:
            ffscout = pull_team_news()

    teams = {int(t["id"]): t for t in bootstrap.get("teams") or []}
    elements = list(bootstrap.get("elements") or [])
    by_team: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for el in elements:
        by_team[int(el["team"])].append(el)

    if history_rows is not None:
        minutes_map = expanding_mean_minutes_from_history(
            history_rows, before_gw=before_gw
        )
        minutes_source = "history"
    else:
        minutes_map = expanding_mean_minutes_from_bootstrap(bootstrap)
        minutes_source = "bootstrap_proxy"

    club_rows = []
    pool_rows = []
    unmatched_firm = []

    for club in ffscout.get("clubs") or []:
        code = club["code"]
        club_norm = club.get("club_norm") or norm_team(club["club"])
        # Map FFScout club to FPL team id via norm_team on bootstrap names.
        team_id = None
        for tid, t in teams.items():
            if norm_team(t.get("name") or "") == club_norm:
                team_id = tid
                break
        if team_id is None:
            # try short_name
            for tid, t in teams.items():
                if norm_team(t.get("short_name") or "") == club_norm:
                    team_id = tid
                    break
        candidates = by_team.get(team_id or -1, [])

        firm_fpl = []
        firm_selection_fpl = []
        seen_unmatched: set[tuple[str, str]] = set()
        for label, dest, names in (
            ("firm", firm_fpl, club.get("firm") or []),
            ("firm_selection", firm_selection_fpl, club.get("firm_selection") or []),
        ):
            for name in names:
                hit = match_ffs_name_to_fpl(name, candidates)
                if hit is None:
                    key = (club["club"], name)
                    if key not in seen_unmatched:
                        seen_unmatched.add(key)
                        unmatched_firm.append(
                            {"club": club["club"], "ffs_name": name, "bucket": label}
                        )
                    continue
                dest.append(hit)

        # Deduplicate by id preserving order
        def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen: set[int] = set()
            out = []
            for r in rows:
                i = int(r["id"])
                if i in seen:
                    continue
                seen.add(i)
                out.append(r)
            return out

        firm_fpl = _dedupe(firm_fpl)
        firm_selection_fpl = _dedupe(firm_selection_fpl)

        club_firm_ids = []
        for el in firm_fpl:
            pid = str(el["id"])
            mm = minutes_map.get(pid, {})
            mean_m = float(mm.get("mean_minutes_played") or mm.get("mean_minutes") or 0.0)
            n_apps = float(mm.get("n_apps") or 0.0)
            minutes_ok = mean_m > min_minutes and n_apps >= min_apps
            row = {
                "player_id": pid,
                "name": el.get("web_name"),
                "full_name": f"{el.get('first_name','')} {el.get('second_name','')}".strip(),
                "team": club["club"],
                "team_norm": club_norm,
                "position": POS_MAP.get(int(el.get("element_type") or 0), "?"),
                "ffs_status": "firm",
                "mean_minutes": round(mean_m, 2),
                "n_apps": int(n_apps),
                "minutes_ok": minutes_ok,
                "selectable": minutes_ok,
                "now_cost": int(el.get("now_cost") or 0),
                "status": el.get("status"),
            }
            club_firm_ids.append(row)
            if minutes_ok:
                pool_rows.append(row)

        club_rows.append(
            {
                "code": code,
                "club": club["club"],
                "club_norm": club_norm,
                "opponent": club.get("opponent"),
                "ffs_xi": club.get("predicted_xi"),
                "ffs_firm": club.get("firm"),
                "ffs_firm_selection": club.get("firm_selection"),
                "ffs_out": club.get("out"),
                "ffs_doubts": club.get("doubts"),
                "n_firm_matched": len(firm_fpl),
                "n_firm_selection_matched": len(firm_selection_fpl),
                "n_selectable": sum(1 for r in club_firm_ids if r["selectable"]),
                "firm_players": club_firm_ids,
                "firm_selection_ids": [str(el["id"]) for el in firm_selection_fpl],
            }
        )

    # Sort pool by team then position
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    pool_rows.sort(
        key=lambda r: (r["team_norm"], pos_order.get(r["position"], 9), r["name"] or "")
    )

    return {
        "rules": {
            "min_expanding_minutes": min_minutes,
            "min_apps": min_apps,
            "ffs_statuses": ["firm", "nailed_on"],
            "ffs_mapping": (
                "Public FFScout team-news has no nailed/firm labels; "
                "firm/nailed_on := predicted XI \\ {Out, Banned, Doubts}."
            ),
            "minutes_source": minutes_source,
            "before_gw": before_gw,
        },
        "ffscout_meta": {
            "source": ffscout.get("source"),
            "fetched_at": ffscout.get("fetched_at"),
            "availability": ffscout.get("availability"),
            "n_clubs": ffscout.get("n_clubs") or len(club_rows),
        },
        "n_selectable": len(pool_rows),
        "n_clubs_with_firm": sum(1 for c in club_rows if c["n_firm_matched"] > 0),
        "n_clubs_with_selectable": sum(1 for c in club_rows if c["n_selectable"] > 0),
        "unmatched_ffs_names": unmatched_firm,
        "clubs": club_rows,
        "pool": pool_rows,
        "selectable_ids": [r["player_id"] for r in pool_rows],
    }


def load_selectable_ids(path: Path | None = None) -> set[str]:
    """Load ``selectable_ids`` from a pool JSON (default live snapshot)."""
    p = path or (ROOT / "data/predictions/2026-27/selectable_pool/selectable_pool.json")
    raw = json.loads(p.read_text())
    return {str(x) for x in raw.get("selectable_ids") or []}


def apply_selectable_pool(
    feat: Any,
    selectable_ids: set[str] | None,
    *,
    id_col: str = "player_id",
    eligible_col: str = "eligible",
) -> Any:
    """Restrict buy eligibility to the FFScout+minutes pool.

    Owned players are kept by the climber's ``_gw_pool`` (eligible | owned).
    Players outside the pool are marked ``eligible=False`` so short- and
    long-horizon transfer search will not buy them.
    """
    if selectable_ids is None:
        return feat
    out = feat.copy()
    pids = out[id_col].astype(str)
    # Strip season prefixes like "2025-26:123" for id comparison.
    bare = pids.str.split(":").str[-1]
    in_pool = bare.isin(selectable_ids) | pids.isin(selectable_ids)
    if eligible_col not in out.columns:
        out[eligible_col] = in_pool
    else:
        out[eligible_col] = out[eligible_col].astype(bool) & in_pool
    return out


def write_pool_outputs(
    result: dict[str, Any],
    *,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "selectable_pool.json"
    csv_path = out_dir / "selectable_pool.csv"
    report_path = ROOT / "reports" / "selectable_pool_ffscout.md"

    json_path.write_text(json.dumps(result, indent=2) + "\n")

    import csv

    fields = [
        "player_id",
        "name",
        "team",
        "position",
        "ffs_status",
        "mean_minutes",
        "n_apps",
        "now_cost",
        "status",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in result["pool"]:
            w.writerow(row)

    lines = [
        "# Selectable pool — expanding minutes + FFScout firm",
        "",
        "## FFScout availability",
        "",
        "Public page: [team-news](https://www.fantasyfootballscout.co.uk/team-news).",
        "",
        "| Available | Notes |",
        "|---|---|",
        "| Predicted XI (11/club) | Public HTML |",
        "| Out / Doubts (%) / Banned | Public HTML |",
        "| Club news blurb | Public HTML |",
        "| Explicit `nailed_on` / `firm` labels | **Not published** — we map firm/nailed := XI − Out − Banned − Doubts |",
        "| Members projections / Rate My Team | Login wall — not used |",
        "",
        f"Fetched: `{result['ffscout_meta'].get('fetched_at')}` · "
        f"clubs with firm XI match: **{result['n_clubs_with_firm']}** · "
        f"selectable players: **{result['n_selectable']}** "
        f"(mean minutes > {result['rules']['min_expanding_minutes']}, "
        f"apps ≥ {result['rules']['min_apps']}, minutes via "
        f"`{result['rules']['minutes_source']}`).",
        "",
        "## Per-club firm selection",
        "",
        "| Club | FFS firm | Matched | Selectable | Opponent |",
        "|---|---:|---:|---:|---|",
    ]
    for c in result["clubs"]:
        lines.append(
            f"| {c['club']} | {len(c.get('ffs_firm') or [])} | "
            f"{c['n_firm_matched']} | {c['n_selectable']} | {c.get('opponent') or ''} |"
        )
    if result["unmatched_ffs_names"]:
        lines += [
            "",
            f"## Unmatched FFScout names ({len(result['unmatched_ffs_names'])})",
            "",
        ]
        for u in result["unmatched_ffs_names"][:30]:
            lines.append(f"- {u['club']}: `{u['ffs_name']}` ({u['bucket']})")

    lines += [
        "",
        "## Selectable pool (head)",
        "",
        "| id | name | team | pos | mean mins | apps |",
        "|---:|---|---|---|---:|---:|",
    ]
    for r in result["pool"][:40]:
        lines.append(
            f"| {r['player_id']} | {r['name']} | {r['team']} | {r['position']} | "
            f"{r['mean_minutes']:.0f} | {r['n_apps']} |"
        )
    if len(result["pool"]) > 40:
        lines.append(f"| … | ({len(result['pool']) - 40} more) | | | | |")

    lines += [
        "",
        "## Use",
        "",
        "Short- and long-horizon squad selection should restrict candidates to "
        "`selectable_ids` in `selectable_pool.json`.",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.models.selectable_pool",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines))
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "report": str(report_path),
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bootstrap",
        type=Path,
        default=ROOT
        / "data/predictions/2026-27/gw06/betfair_20261008/bootstrap.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/predictions/2026-27/selectable_pool",
    )
    p.add_argument("--html", type=Path, default=None, help="Offline FFScout HTML")
    args = p.parse_args()
    bootstrap = json.loads(args.bootstrap.read_text())
    html = args.html.read_text(encoding="utf-8") if args.html else None
    snap = args.out
    if html is None:
        ffscout = pull_team_news(out_dir=snap)
        result = build_selectable_pool(bootstrap=bootstrap, ffscout=ffscout)
    else:
        result = build_selectable_pool(bootstrap=bootstrap, html=html)
        snap.mkdir(parents=True, exist_ok=True)
    paths = write_pool_outputs(result, out_dir=snap)
    print(json.dumps({
        "n_selectable": result["n_selectable"],
        "n_clubs_with_selectable": result["n_clubs_with_selectable"],
        "n_unmatched": len(result["unmatched_ffs_names"]),
        **paths,
    }, indent=2))


if __name__ == "__main__":
    main()
