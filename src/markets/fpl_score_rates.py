"""FPL award scalings α_G and α_A: player goals / assists per match goal.

Canonical store: ``data/processed/fpl_goal_assist_rates.json``.

Numagent 2.0 pots:
    G_team = α_G * xG_M
    A_team = α_A * xG_M

The 2025/26 closed season is the prior. **During the live season these
scalings must be updated continually** — after each completed GW, add that
GW's clean counts to the prior and refresh ``pooled``:

    α_G = (player_goals_prior + player_goals_live) / (match_goals_prior + match_goals_live)
    α_A = (assists_prior + assists_live) / (match_goals_prior + match_goals_live)

Do not freeze the initial 0.9615 / 0.9027 point estimates for the whole year.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
STORE = PROCESSED / "fpl_goal_assist_rates.json"

PRIOR_SEASON = "2025-26"
PRIOR_MERGED = CACHE / "merged_gw_2025_26.csv"
LIVE_SEASON = "2026-27"
LIVE_PLAYER_GW = CACHE / "player_gw_2026_27.csv"
LIVE_FD = CACHE / "E0_2627.csv"


def _fixture_sides_from_merged(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, low_memory=False)
    gcol = "goals_scored" if "goals_scored" in m.columns else "goals"
    fid = "fixture" if "fixture" in m.columns else "fixture_id"
    home_flag = "was_home" if "was_home" in m.columns else "is_home"
    rows: list[dict[str, Any]] = []
    for (f, wh), g in m.groupby([fid, home_flag]):
        wh_b = bool(g[home_flag].iloc[0])
        pg = float(pd.to_numeric(g[gcol], errors="coerce").fillna(0).sum())
        pa = float(pd.to_numeric(g["assists"], errors="coerce").fillna(0).sum())
        hs = g["team_h_score"].iloc[0]
        aws = g["team_a_score"].iloc[0]
        if pd.isna(hs) or pd.isna(aws):
            continue
        mg = float(hs if wh_b else aws)
        rows.append(
            {
                "fixture_id": f,
                "team": str(g["team"].iloc[0]),
                "player_goals": pg,
                "assists": pa,
                "match_goals": mg,
            }
        )
    return pd.DataFrame(rows)


def clean_sides(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bugged sides where FPL player goals exceed the scoreline."""
    out = df[
        (df["player_goals"] <= df["match_goals"])
        & (df["match_goals"] >= 0)
        & (df["player_goals"] >= 0)
        & (df["assists"] >= 0)
    ].copy()
    return out


def _bootstrap_se(clean: pd.DataFrame, *, n_boot: int = 4000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    fix_ids = clean["fixture_id"].unique()
    parts = {fid: clean[clean.fixture_id == fid] for fid in fix_ids}
    bp: list[float] = []
    ba: list[float] = []
    for _ in range(n_boot):
        samp = rng.choice(fix_ids, size=len(fix_ids), replace=True)
        sub = pd.concat([parts[i] for i in samp], ignore_index=True)
        mg = float(sub["match_goals"].sum())
        if mg <= 0:
            continue
        bp.append(float(sub["player_goals"].sum() / mg))
        ba.append(float(sub["assists"].sum() / mg))
    bp_a = np.asarray(bp)
    ba_a = np.asarray(ba)
    return {
        "player_g_per_match_g_se": float(bp_a.std(ddof=1)),
        "player_g_per_match_g_ci95_lo": float(np.quantile(bp_a, 0.025)),
        "player_g_per_match_g_ci95_hi": float(np.quantile(bp_a, 0.975)),
        "a_per_match_g_se": float(ba_a.std(ddof=1)),
        "a_per_match_g_ci95_lo": float(np.quantile(ba_a, 0.025)),
        "a_per_match_g_ci95_hi": float(np.quantile(ba_a, 0.975)),
        "n_bootstrap": int(n_boot),
    }


def counts_from_sides(clean: pd.DataFrame) -> dict[str, Any]:
    mg = float(clean["match_goals"].sum())
    pg = float(clean["player_goals"].sum())
    pa = float(clean["assists"].sum())
    n_fix = int(clean["fixture_id"].nunique())
    return {
        "n_matches": n_fix,
        "match_goals": mg,
        "player_goals": pg,
        "assists": pa,
        "player_g_per_match_g": pg / mg if mg else float("nan"),
        "a_per_match_g": pa / mg if mg else float("nan"),
    }


def build_prior_2025_26(*, n_boot: int = 4000) -> dict[str, Any]:
    raw = _fixture_sides_from_merged(PRIOR_MERGED)
    n_bugged = int((raw["player_goals"] > raw["match_goals"]).sum())
    clean = clean_sides(raw)
    counts = counts_from_sides(clean)
    unc = _bootstrap_se(clean, n_boot=n_boot)
    return {
        "schema": "fpl_goal_assist_rates.v1",
        "description": (
            "FPL player-goals and assists per scoreline match goal. "
            "Prior = cleaned 2025/26. Pool with live season by summing counts."
        ),
        "method": {
            "drop_if": "player_goals > match_goals on a fixture-side",
            "uncertainty": "bootstrap over matches (both sides move together)",
            "pool": "add prior counts to live-to-date counts",
        },
        "prior": {
            "season": PRIOR_SEASON,
            "source": str(PRIOR_MERGED.relative_to(ROOT)),
            "n_bugged_sides_dropped": n_bugged,
            **counts,
            **unc,
        },
        "live": None,
        "pooled": None,
    }


def pool_counts(prior: dict[str, Any], live: dict[str, Any] | None) -> dict[str, Any]:
    mg = float(prior["match_goals"]) + float((live or {}).get("match_goals") or 0.0)
    pg = float(prior["player_goals"]) + float((live or {}).get("player_goals") or 0.0)
    pa = float(prior["assists"]) + float((live or {}).get("assists") or 0.0)
    # Pooled SE: binomial approx on combined match goals (conservative, ignores clustering).
    p = pg / mg if mg else float("nan")
    a = pa / mg if mg else float("nan")
    se_p = math.sqrt(p * (1.0 - p) / mg) if mg and 0 < p < 1 else float("nan")
    a_clip = min(max(a, 1e-9), 1.0 - 1e-9) if mg else float("nan")
    se_a = math.sqrt(a_clip * (1.0 - a_clip) / mg) if mg else float("nan")
    return {
        "match_goals": mg,
        "player_goals": pg,
        "assists": pa,
        "player_g_per_match_g": p,
        "a_per_match_g": a,
        "player_g_per_match_g_se_binomial": se_p,
        "a_per_match_g_se_binomial": se_a,
        "includes_live": bool(live and float((live or {}).get("match_goals") or 0) > 0),
    }


def _live_fixture_sides(
    *,
    before_gw: int | None = None,
    player_gw: Path = LIVE_PLAYER_GW,
    fd_path: Path = LIVE_FD,
) -> pd.DataFrame:
    """Fixture-sides for the live season: player goals/assists + FD match goals.

    ``before_gw`` keeps GW ``< before_gw`` only (as-of). Match goals come from
    football-data FTHG/FTAG — never from summing player goals (that would
    force α_G ≈ 1).
    """
    from src.teams import norm_team

    if not player_gw.is_file() or not fd_path.is_file():
        return pd.DataFrame(
            columns=["fixture_id", "team", "player_goals", "assists", "match_goals", "gw"]
        )

    pg = pd.read_csv(player_gw, low_memory=False)
    gcol = "goals_scored" if "goals_scored" in pg.columns else "goals"
    pg["gw"] = pd.to_numeric(pg["gw"], errors="coerce")
    if before_gw is not None:
        pg = pg.loc[pg["gw"].notna() & (pg["gw"] < int(before_gw))].copy()
    if pg.empty:
        return pd.DataFrame(
            columns=["fixture_id", "team", "player_goals", "assists", "match_goals", "gw"]
        )

    pg = pg.copy()
    pg["date_iso"] = pd.to_datetime(pg["date"], errors="coerce").dt.normalize()
    pg["team_n"] = pg["team"].map(norm_team)
    club = (
        pg.groupby(["gw", "date_iso", "team_n", "team", "is_home"], as_index=False)
        .agg(
            player_goals=(gcol, "sum"),
            assists=("assists", "sum"),
        )
    )

    fd = pd.read_csv(fd_path, low_memory=False)
    fd = fd.copy()
    fd["date_iso"] = pd.to_datetime(fd["Date"], dayfirst=True, errors="coerce").dt.normalize()
    fd["home_n"] = fd["HomeTeam"].map(norm_team)
    fd["away_n"] = fd["AwayTeam"].map(norm_team)
    fd["fixture_id"] = (
        fd["date_iso"].dt.strftime("%Y-%m-%d")
        + ":"
        + fd["home_n"].astype(str)
        + ":"
        + fd["away_n"].astype(str)
    )

    rows: list[dict[str, Any]] = []
    for row in club.to_dict(orient="records"):
        is_home = bool(row["is_home"]) if pd.notna(row["is_home"]) else True
        d = row["date_iso"]
        team_n = row["team_n"]
        if is_home:
            hit = fd.loc[(fd["date_iso"] == d) & (fd["home_n"] == team_n)]
            mg = float(hit["FTHG"].iloc[0]) if not hit.empty else float("nan")
            fid = str(hit["fixture_id"].iloc[0]) if not hit.empty else f"{d}:{team_n}:?"
        else:
            hit = fd.loc[(fd["date_iso"] == d) & (fd["away_n"] == team_n)]
            mg = float(hit["FTAG"].iloc[0]) if not hit.empty else float("nan")
            fid = str(hit["fixture_id"].iloc[0]) if not hit.empty else f"{d}:?:{team_n}"
        if not math.isfinite(mg):
            continue
        rows.append(
            {
                "fixture_id": fid,
                "team": str(row["team"]),
                "player_goals": float(row["player_goals"]),
                "assists": float(row["assists"]),
                "match_goals": mg,
                "gw": int(row["gw"]) if pd.notna(row["gw"]) else None,
            }
        )
    return pd.DataFrame(rows)


def live_counts(*, before_gw: int | None = None) -> dict[str, Any] | None:
    """Clean live-to-date counts (GW ``< before_gw`` if set). ``None`` if empty."""
    raw = _live_fixture_sides(before_gw=before_gw)
    if raw.empty:
        return None
    n_bugged = int((raw["player_goals"] > raw["match_goals"]).sum())
    clean = clean_sides(raw)
    if clean.empty:
        return None
    counts = counts_from_sides(clean)
    max_gw = int(pd.to_numeric(raw["gw"], errors="coerce").max())
    return {
        "season": LIVE_SEASON,
        "source": str(LIVE_PLAYER_GW.relative_to(ROOT)),
        "fd_source": str(LIVE_FD.relative_to(ROOT)),
        "before_gw": before_gw,
        "through_gw": max_gw if before_gw is None else int(before_gw) - 1,
        "n_bugged_sides_dropped": n_bugged,
        **counts,
    }


def refresh_pooled(
    *,
    before_gw: int | None = None,
    path: Path = STORE,
    write: bool = True,
) -> dict[str, Any]:
    """Prior 2025/26 + live 2026/27 counts → pooled α_G / α_A.

    Call after each completed live GW (or with ``before_gw`` for as-of).
    """
    doc = load_store(path)
    live = live_counts(before_gw=before_gw)
    doc["live"] = live
    doc["pooled"] = pool_counts(doc["prior"], live)
    doc["description"] = (
        "FPL award scalings α_G = player_goals/match_goals and α_A = assists/match_goals. "
        "Prior = cleaned 2025/26. Live season updates continually: after each completed GW, "
        "add clean live counts to the prior and refresh pooled α."
    )
    doc["method"] = {
        "drop_if": "player_goals > match_goals on a fixture-side",
        "uncertainty": "bootstrap over matches on prior; binomial SE on pooled",
        "pool": "add prior counts to live-to-date counts after every completed GW",
        "live_update": "required — recompute pooled α_G and α_A continuously during 2026/27",
        "share_history": "player shares stay same-season only; α is the only prior-season blend",
    }
    if write:
        write_store(doc, path)
    return doc


def write_store(doc: dict[str, Any], path: Path = STORE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if doc.get("prior") and not doc.get("pooled"):
        doc["pooled"] = pool_counts(doc["prior"], doc.get("live"))
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def load_store(path: Path = STORE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rates(path: Path = STORE) -> tuple[float, float]:
    """Return ``(player_g_per_match_g, a_per_match_g)`` from pooled block."""
    doc = load_store(path)
    pooled = doc.get("pooled") or pool_counts(doc["prior"], doc.get("live"))
    return float(pooled["player_g_per_match_g"]), float(pooled["a_per_match_g"])


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--refresh-live",
        action="store_true",
        help="update live counts from player_gw_2026_27 + E0_2627 and rewrite pooled α",
    )
    ap.add_argument(
        "--before-gw",
        type=int,
        default=None,
        help="as-of: only live GWs strictly before this (default: all completed in log)",
    )
    ap.add_argument("--rebuild-prior", action="store_true", help="rebuild 2025/26 prior block")
    args = ap.parse_args(argv)

    if args.rebuild_prior:
        doc = build_prior_2025_26()
        doc["pooled"] = pool_counts(doc["prior"], None)
        path = write_store(doc)
        p = doc["prior"]
        print(f"wrote {path}")
        print(
            f"prior {p['season']}: player_G/match_G={p['player_g_per_match_g']:.4f} "
            f"± {p['player_g_per_match_g_se']:.4f}  "
            f"A/match_G={p['a_per_match_g']:.4f} ± {p['a_per_match_g_se']:.4f}  "
            f"(dropped {p['n_bugged_sides_dropped']} bugged sides)"
        )
        return

    if args.refresh_live:
        doc = refresh_pooled(before_gw=args.before_gw, write=True)
        live = doc.get("live") or {}
        pooled = doc["pooled"]
        print(f"wrote {STORE}")
        print(
            f"live through GW{live.get('through_gw')}: "
            f"match_G={live.get('match_goals')} player_G={live.get('player_goals')} "
            f"A={live.get('assists')} (dropped {live.get('n_bugged_sides_dropped')})"
        )
        print(
            f"pooled α_G={pooled['player_g_per_match_g']:.4f}  "
            f"α_A={pooled['a_per_match_g']:.4f}  includes_live={pooled['includes_live']}"
        )
        return

    # Default: rebuild prior only (legacy entrypoint)
    main(["--rebuild-prior"])


if __name__ == "__main__":
    main()
