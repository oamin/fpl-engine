"""Match Odds API anytime-scorer props to FPL element ids and side moments.

Used by the live t:t+1 path: BTTS-on Dixon–Coles moments supply xG/xCS/xGC;
matched P(anytime) lifts goal shares via ``prop_shares.rescale_goal_shares``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.live.news_packets import match_element_id
from src.teams import norm_team


def bootstrap_team_names(bootstrap: Mapping[str, Any]) -> dict[int, str]:
    return {int(t["id"]): str(t["name"]) for t in bootstrap.get("teams") or []}


def fpl_club_for_book_name(
    book_team: str,
    team_names: Mapping[int, str],
) -> str | None:
    """Map an Odds API club string to the bootstrap display name via ``norm_team``."""
    want = norm_team(book_team)
    for name in team_names.values():
        if norm_team(name) == want:
            return str(name)
    return None


def match_anytime_props(
    props: Sequence[Mapping[str, Any]],
    *,
    home: str,
    away: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Map fair P(anytime) → FPL element id for home/away clubs.

    Tries home club first, then away. Ambiguous or unmatched names are listed
    under ``unmatched`` (no credit spent; offline from a probe snapshot).
    """
    elements = list(bootstrap.get("elements") or [])
    team_names = bootstrap_team_names(bootstrap)
    home_club = fpl_club_for_book_name(home, team_names)
    away_club = fpl_club_for_book_name(away, team_names)
    if home_club is None or away_club is None:
        return {
            "ok": False,
            "reason": f"club_map home={home!r}→{home_club} away={away!r}→{away_club}",
            "by_id": {},
            "rows": [],
            "unmatched": [str(p.get("player") or "") for p in props],
        }

    by_id: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for prop in props:
        name = str(prop.get("player") or "").strip()
        p = float(prop.get("p_anytime") or 0.0)
        if not name or p <= 0.0:
            continue
        pid = match_element_id(name, home_club, elements, team_names)
        club = home_club
        side = "home"
        if pid is None:
            pid = match_element_id(name, away_club, elements, team_names)
            club = away_club
            side = "away"
        if pid is None:
            unmatched.append(name)
            continue
        # Prefer higher p if duplicate quotes collide on the same id.
        if pid in by_id and by_id[pid] >= p:
            continue
        by_id[pid] = p
        rows.append(
            {
                "player": name,
                "element": int(pid),
                "club": club,
                "side": side,
                "p_anytime": p,
                "n_books": prop.get("n_books"),
            }
        )

    return {
        "ok": True,
        "home": home_club,
        "away": away_club,
        "by_id": {str(k): float(v) for k, v in by_id.items()},
        "n_matched": len(by_id),
        "n_props": len(props),
        "n_unmatched": len(unmatched),
        "rows": rows,
        "unmatched": unmatched[:40],
    }


def side_moments_from_fit(fit: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Home/away xG, xCS, xGC from a ``fit_match`` / probe match row."""
    xgc_h = fit.get("xgc_home")
    xgc_a = fit.get("xgc_away")
    # Older probe snapshots omit xGC — fall back to opponent xG as events proxy.
    if xgc_h is None:
        xgc_h = max(0.0, float(fit.get("xg_away") or 0.0) / 2.0)
    if xgc_a is None:
        xgc_a = max(0.0, float(fit.get("xg_home") or 0.0) / 2.0)
    return {
        "home": {
            "xg": float(fit["xg_home"]),
            "xcs": float(fit["xcs_home"]),
            "xgc": float(xgc_h),
        },
        "away": {
            "xg": float(fit["xg_away"]),
            "xcs": float(fit["xcs_away"]),
            "xgc": float(xgc_a),
        },
    }


def _id_key(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).split(":")[-1]


def prop_p_for_frame(
    players: Any,
    by_id: Mapping[str | int, float],
    *,
    id_key: str = "player_id",
) -> dict[int, float]:
    """Build row-index → P(anytime) for ``rescale_goal_shares`` / ``score_team_side``."""
    lookup = {str(k): float(v) for k, v in by_id.items()}
    out: dict[int, float] = {}
    if hasattr(players, "reset_index"):
        frame = players.reset_index(drop=True)
        for i in range(len(frame)):
            raw = frame.iloc[i].get(id_key)
            if raw is None and "element" in frame.columns:
                raw = frame.iloc[i].get("element")
            key = _id_key(raw)
            if key in lookup:
                out[i] = lookup[key]
        return out
    for i, row in enumerate(players):
        key = _id_key(row.get(id_key, row.get("element")))
        if key in lookup:
            out[i] = lookup[key]
    return out
