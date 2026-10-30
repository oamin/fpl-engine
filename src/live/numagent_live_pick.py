"""Numagent 2.0 live pick: xmi + book pots → L4 centreline horizon on held 15.

Product ``protocol.numagent_2_0`` (supersedes Numagent 1.0 / ep_next freeze).
Primary scorer is ``score_xp_v2``. Forecast μ:
``protocol.l3_centreline_forecast`` — priced near (t, t+1) + Opta blind far
(default ``--centreline-horizon 5``).

Live book = Betfair Exchange (``python3 -m src.live.betfair_lines``). The Odds
API is banned. Offline rehearsal may pass a legacy odds snapshot via
``--from-snapshot``; do not refresh Odds API keys.

Does not touch HOLDOUT_FREEZE.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from src.live.odds_live_probe import (
    MATCH_MARKETS,
    PROP_MARKET,
    REGION,
    SPORT,
    _load_odds_key,
    extract_anytime_props,
    extract_match_targets,
    fit_match,
)
from src.live.odds_props import match_anytime_props, side_moments_from_fit
from src.live.xmi import load_xmi
from src.markets import numagent_xp
from src.rules.fpl_2026 import sell_price
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer3_week import held_from_frame, solve_week_transfers
from src.strategy.layer4_week import solve_horizon_transfers
from src.teams import norm_team

# Locked L4: γ=1, no terminal lump. Near = priced t:t+1; far = Opta blind
# under protocol.l3_centreline_forecast (default horizon 5).
L4_GAMMA = 1.0
BOOKIE_DEPTH = 2
CENTRELINE_HORIZON = 5  # t … t+4

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "data" / "scratch"
CACHE = ROOT / "data" / "cache"
PRED = ROOT / "data" / "predictions" / "2026-27"
ENTRY_ID = 2632584
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _get_json(url: str) -> Any:
    with httpx.Client(timeout=60.0) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def fetch_bootstrap() -> dict[str, Any]:
    return _get_json("https://fantasy.premierleague.com/api/bootstrap-static/")


def fetch_entry_bundle(entry_id: int, played_gw: int) -> dict[str, Any]:
    base = f"https://fantasy.premierleague.com/api/entry/{int(entry_id)}"
    return {
        "entry": _get_json(f"{base}/"),
        "picks": _get_json(f"{base}/event/{int(played_gw)}/picks/"),
        "transfers": _get_json(f"{base}/transfers/"),
        "fixtures": _get_json(
            f"https://fantasy.premierleague.com/api/fixtures/?event={int(played_gw) + 1}"
        ),
        "live_played": _get_json(
            f"https://fantasy.premierleague.com/api/event/{int(played_gw)}/live/"
        ),
    }


def append_live_gw_history(
    base: pd.DataFrame,
    *,
    bootstrap: dict[str, Any],
    live: dict[str, Any],
    gw: int,
) -> pd.DataFrame:
    """Append one finished GW from FPL ``event/N/live`` onto player_gw-shaped rows."""
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    els = {int(e["id"]): e for e in bootstrap["elements"]}
    rows: list[dict[str, Any]] = []
    for block in live.get("elements") or []:
        eid = int(block["id"])
        el = els.get(eid)
        if el is None:
            continue
        st = block.get("stats") or {}
        mins = float(st.get("minutes") or 0)
        team = teams.get(int(el["team"]), "")
        rows.append(
            {
                "date": "",
                "gw": int(gw),
                "player_id": eid,
                "player_name": str(el.get("web_name") or ""),
                "team": team,
                "team_short": "",
                "team_norm": norm_team(team),
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "is_home": False,
                "minutes": mins,
                "total_points": float(st.get("total_points") or 0),
                "goals": float(st.get("goals_scored") or 0),
                "assists": float(st.get("assists") or 0),
                "clean_sheets": float(st.get("clean_sheets") or 0),
                "goals_conceded": float(st.get("goals_conceded") or 0),
                "saves": float(st.get("saves") or 0),
                "bonus": float(st.get("bonus") or 0),
                "bps": float(st.get("bps") or 0),
                "xG": float(st.get("expected_goals") or 0),
                "xA": float(st.get("expected_assists") or 0),
                "defcon": float(st.get("defensive_contribution") or 0),
                "yellow_cards": float(st.get("yellow_cards") or 0),
                "red_cards": float(st.get("red_cards") or 0),
                "starts": float(st.get("starts") or 0),
                "value": int(el.get("now_cost") or 0),
                "defcon_raw": float(st.get("defensive_contribution") or 0),
            }
        )
    extra = pd.DataFrame(rows)
    if base.empty:
        return extra
    keep = base.loc[pd.to_numeric(base["gw"], errors="coerce") < int(gw)].copy()
    return pd.concat([keep, extra], ignore_index=True)


def history_for_numagent(history: pd.DataFrame) -> pd.DataFrame:
    """Map live player_gw columns → numagent rate/prior helpers."""
    h = history.copy()
    h["element"] = pd.to_numeric(h["player_id"], errors="coerce").astype("Int64")
    h["GW"] = pd.to_numeric(h["gw"], errors="coerce")
    h["goals_scored"] = pd.to_numeric(h.get("goals", 0), errors="coerce").fillna(0.0)
    h["assists"] = pd.to_numeric(h["assists"], errors="coerce").fillna(0.0)
    h["minutes"] = pd.to_numeric(h["minutes"], errors="coerce").fillna(0.0)
    h["bonus"] = pd.to_numeric(h["bonus"], errors="coerce").fillna(0.0)
    h["yellow_cards"] = pd.to_numeric(h["yellow_cards"], errors="coerce").fillna(0.0)
    h["red_cards"] = pd.to_numeric(h["red_cards"], errors="coerce").fillna(0.0)
    h["saves"] = pd.to_numeric(h.get("saves", 0), errors="coerce").fillna(0.0)
    dc = h["defcon_raw"] if "defcon_raw" in h.columns else h.get("defcon", 0)
    h["defensive_contribution"] = pd.to_numeric(dc, errors="coerce").fillna(0.0)
    h["name"] = h.get("player_name", h["element"].astype(str))
    h["team"] = h["team"].astype(str)
    h["season"] = "2026-27"
    return h


def reconstruct_purchases(
    *,
    held_ids: list[int],
    transfers: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    opening: list[dict[str, Any]] | None = None,
) -> dict[int, int]:
    """Best-effort purchase tenths: last in_cost, else opening, else now_cost."""
    now = {int(e["id"]): int(e["now_cost"]) for e in bootstrap["elements"]}
    purchase: dict[int, int] = {}
    if opening:
        for p in opening:
            purchase[int(p["id"])] = int(now.get(int(p["id"]), 0))
    # Chronological transfers overwrite
    for t in sorted(transfers, key=lambda r: str(r.get("time") or "")):
        purchase[int(t["element_in"])] = int(t["element_in_cost"])
        purchase.pop(int(t["element_out"]), None)
    for eid in held_ids:
        if eid not in purchase:
            purchase[eid] = int(now.get(eid, 0))
    return {eid: purchase[eid] for eid in held_ids}


def load_xmi_map(path: Path, gw: int) -> dict[int, float]:
    frame = load_xmi(path, gw)
    out: dict[int, float] = {}
    for row in frame.itertuples(index=False):
        pid = int(str(row.player_id).split(":")[-1])
        out[pid] = float(row.xmi)
    return out


def sigma_m_from_history(hist: pd.DataFrame, element: int) -> float:
    rows = hist.loc[hist["element"] == int(element)].sort_values("GW")
    if rows.empty:
        return 15.0
    mins = pd.to_numeric(rows["minutes"], errors="coerce").fillna(0.0).tail(3)
    if len(mins) < 2:
        return 15.0
    return float(max(5.0, mins.std(ddof=1)))


def build_player_table(
    *,
    bootstrap: dict[str, Any],
    hist: pd.DataFrame,
    before_gw: int,
    xmi: dict[int, float],
    club: str | None = None,
    minutes_mode: str = "xmi_then_rolling",
) -> pd.DataFrame:
    asof = numagent_xp.asof_share_history(hist, before_gw=before_gw, season="2026-27")
    rates = numagent_xp.player_rates_from_history(asof, id_col="element")
    priors = numagent_xp.history_priors_for_xp(asof, id_col="element")
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    rows: list[dict[str, Any]] = []
    for el in bootstrap["elements"]:
        team = teams[int(el["team"])]
        if club is not None and norm_team(team) != norm_team(club):
            continue
        eid = int(el["id"])
        pos = POS_MAP.get(int(el["element_type"]), "MID")
        past = asof.loc[asof["element"] == eid]
        roll = 0.0
        if not past.empty:
            roll = float(
                pd.to_numeric(past["minutes"], errors="coerce")
                .fillna(0.0)
                .tail(3)
                .mean()
            )
        if minutes_mode == "rolling3":
            # L4 t+1: as-of mean minutes (no future xmi tags).
            m, m_src = roll, "rolling3"
        elif eid in xmi:
            m, m_src = float(xmi[eid]), "xmi"
        else:
            # Probe sheets are often club-subset; fill rest with rolling 3-GW mean.
            m, m_src = roll, "rolling3"
        rows.append(
            {
                "element": eid,
                "player_id": str(eid),
                "name": str(el.get("web_name") or ""),
                "position": pos,
                "club": team,
                "team_norm": norm_team(team),
                "value": int(el["now_cost"]),
                "m": m,
                "m_source": m_src,
                "sigma_m": sigma_m_from_history(asof, eid),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if not rates.empty:
        frame = frame.merge(rates, on="element", how="left")
    if not priors.empty:
        frame = frame.merge(
            priors.drop(columns=["position"], errors="ignore"),
            on="element",
            how="left",
        )
    for c, default in (
        ("rG", 0.0),
        ("drG", 0.0),
        ("rA", 0.0),
        ("drA", 0.0),
        ("bonus_mu", 0.0),
        ("bonus_se", 0.0),
        ("card_mu", 0.0),
        ("card_se", 0.0),
        ("saves_mu", 0.0),
        ("saves_se", 0.0),
        ("defcon_mu", 0.0),
        ("defcon_sig", numagent_xp.SIG_FLOOR),
        ("defcon_n", 0),
    ):
        if c not in frame.columns:
            frame[c] = default
        else:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(default)
    return frame.reset_index(drop=True)


def fetch_odds_for_fixtures(
    fixtures: list[tuple[str, str]],
    *,
    with_props: bool = True,
) -> dict[str, Any]:
    """Pull AH+OU+BTTS (+ anytime) for named home/away pairs. region=us."""
    key = _load_odds_key()
    out: dict[str, Any] = {
        "ok": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "region": REGION,
        "match_markets": MATCH_MARKETS,
        "prop_market": PROP_MARKET if with_props else None,
        "matches": [],
        "props": {},
        "remaining_before": None,
        "remaining_after": None,
    }
    if not key:
        out["reason"] = "no_odds_api_key"
        return out
    want = {(norm_team(h), norm_team(a)) for h, a in fixtures}
    n = len(want)
    estimate = (4 if with_props else 3) * n
    out["estimated_credits"] = estimate
    with httpx.Client(timeout=90.0) as client:
        ev = client.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/events",
            params={"apiKey": key},
        )
        out["remaining_before"] = ev.headers.get("x-requests-remaining")
        if ev.status_code != 200:
            out["reason"] = f"events_http_{ev.status_code}"
            return out
        rem = int(out["remaining_before"] or "0")
        if rem < estimate + 30:
            out["reason"] = f"low_credits remaining={rem} need≈{estimate}+30"
            return out
        events = ev.json()
        picked: list[dict[str, Any]] = []
        for e in events:
            key_pair = (norm_team(str(e.get("home_team"))), norm_team(str(e.get("away_team"))))
            if key_pair in want:
                picked.append(e)
        out["n_events_listed"] = len(events)
        out["n_matched_fixtures"] = len(picked)
        markets = f"{MATCH_MARKETS},{PROP_MARKET}" if with_props else MATCH_MARKETS
        for event in picked:
            rid = event.get("id")
            odds = client.get(
                f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{rid}/odds",
                params={
                    "apiKey": key,
                    "regions": REGION,
                    "markets": markets,
                    "oddsFormat": "decimal",
                },
            )
            out["remaining_after"] = odds.headers.get("x-requests-remaining")
            name = f"{event.get('home_team')} v {event.get('away_team')}"
            if odds.status_code != 200:
                out["matches"].append(
                    {
                        "home": event.get("home_team"),
                        "away": event.get("away_team"),
                        "id": rid,
                        "ok": False,
                        "reason": f"odds_http_{odds.status_code}",
                    }
                )
                continue
            payload = odds.json()
            targets = extract_match_targets(payload)
            if targets is None:
                out["matches"].append(
                    {
                        "home": event.get("home_team"),
                        "away": event.get("away_team"),
                        "id": rid,
                        "ok": False,
                        "reason": "missing_ah_ou",
                    }
                )
                continue
            fitted = fit_match(targets)
            fitted["ok"] = True
            fitted["id"] = rid
            fitted["home"] = event.get("home_team")
            fitted["away"] = event.get("away_team")
            out["matches"].append(fitted)
            if with_props:
                props = extract_anytime_props(payload)
                out["props"][name] = {
                    "ok": True,
                    "n_players": len(props),
                    "top": props[:12],
                    "all": props,
                }
    out["ok"] = any(m.get("ok") for m in out["matches"])
    out["reason"] = "live"
    return out


def _merge_odds_docs(docs: list[dict[str, Any]]) -> dict[str, Any]:
    if not docs:
        return {"ok": False, "matches": [], "props": {}}
    out = dict(docs[0])
    matches: list[dict[str, Any]] = []
    props: dict[str, Any] = {}
    seen: set[str] = set()
    for doc in docs:
        for m in doc.get("matches") or []:
            key = f"{m.get('home')} v {m.get('away')}"
            if key in seen:
                continue
            seen.add(key)
            matches.append(m)
        props.update(doc.get("props") or {})
    out["matches"] = matches
    out["props"] = props
    out["ok"] = any(m.get("ok") for m in matches)
    out["fetched_at"] = docs[-1].get("fetched_at") or out.get("fetched_at")
    out["remaining_after"] = docs[-1].get("remaining_after") or out.get("remaining_after")
    return out


def score_pool_from_odds(
    *,
    bootstrap: dict[str, Any],
    hist: pd.DataFrame,
    before_gw: int,
    xmi: dict[int, float],
    odds_doc: dict[str, Any],
    fixture_pairs: list[tuple[str, str]] | None = None,
    minutes_mode: str = "xmi_then_rolling",
    use_props: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    alpha_g, alpha_a, _, _ = numagent_xp.load_alphas()
    meta: dict[str, Any] = {
        "alpha_g": alpha_g,
        "alpha_a": alpha_a,
        "minutes_mode": minutes_mode,
        "use_props": bool(use_props),
        "fixtures_scored": [],
        "fixtures_failed": [],
    }
    allow = None
    if fixture_pairs is not None:
        allow = {(norm_team(h), norm_team(a)) for h, a in fixture_pairs}

    base = build_player_table(
        bootstrap=bootstrap,
        hist=hist,
        before_gw=before_gw,
        xmi=xmi,
        minutes_mode=minutes_mode,
    )
    xp = pd.Series(0.0, index=base["element"].astype(int))
    se = pd.Series(0.0, index=base["element"].astype(int))
    props_block = odds_doc.get("props") or {}

    for match in odds_doc.get("matches") or []:
        home = str(match.get("home") or "")
        away = str(match.get("away") or "")
        pair = (norm_team(home), norm_team(away))
        if allow is not None and pair not in allow:
            continue
        if not match.get("ok"):
            meta["fixtures_failed"].append(
                {"match": f"{home} v {away}", "reason": match.get("reason")}
            )
            continue
        key = f"{home} v {away}"
        moments = side_moments_from_fit(match)
        by_id: dict[Any, float] = {}
        matched: dict[str, Any] = {"n_matched": 0, "n_props": 0}
        if use_props:
            prop_raw = (props_block.get(key) or {}).get("all") or (
                (props_block.get(key) or {}).get("top") or []
            )
            matched = match_anytime_props(
                prop_raw, home=home, away=away, bootstrap=bootstrap
            )
            by_id = matched.get("by_id") or {}
        scored_sides = 0
        for side_name, club in (("home", home), ("away", away)):
            club_df = build_player_table(
                bootstrap=bootstrap,
                hist=hist,
                before_gw=before_gw,
                xmi=xmi,
                club=club,
                minutes_mode=minutes_mode,
            )
            if club_df.empty:
                continue
            mom = moments[side_name]
            scored = numagent_xp.score_team_side(
                club_df,
                xg=mom["xg"],
                xcs=mom["xcs"],
                xgc=mom["xgc"],
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
                prop_p_by_player=by_id or None,
            )
            for i in range(len(scored)):
                eid = int(club_df.iloc[i]["element"])
                xp.loc[eid] = float(xp.loc[eid]) + float(scored.iloc[i]["xp"])
                se.loc[eid] = float(
                    math.sqrt(
                        float(se.loc[eid]) ** 2
                        + float(scored.iloc[i].get("xp_se", 0.0) or 0.0) ** 2
                    )
                )
            scored_sides += 1
        meta["fixtures_scored"].append(
            {
                "match": key,
                "sides": scored_sides,
                "props_matched": matched.get("n_matched") if use_props else 0,
                "props_n": matched.get("n_props") if use_props else 0,
            }
        )

    out = base.copy()
    out["score_xp_v2"] = out["element"].map(lambda e: float(xp.get(int(e), 0.0)))
    out["xp_se"] = out["element"].map(lambda e: float(se.get(int(e), 0.0)))
    meta["n_m_xmi"] = int((out["m_source"] == "xmi").sum()) if "m_source" in out.columns else 0
    meta["n_m_rolling3"] = (
        int((out["m_source"] == "rolling3").sum()) if "m_source" in out.columns else 0
    )
    meta["n_fixtures_ok"] = len(meta["fixtures_scored"])
    return out, meta


def _plan_block(plan, *, els: dict[int, dict], label: str) -> dict[str, Any]:
    def _name(pid: str) -> str:
        eid = int(str(pid).split(":")[-1])
        return str(els.get(eid, {}).get("web_name") or pid)

    xi_rows = [
        {
            "player_id": str(r["player_id"]),
            "name": _name(str(r["player_id"])),
            "position": r["position"],
            "club": r["club"],
            "mu_post": float(r["mu"]),
            "captain": str(r["player_id"]) == str(plan.captain_id),
        }
        for _, r in plan.xi.iterrows()
    ]
    return {
        **plan.summary(),
        "label": label,
        "captain_name": _name(plan.captain_id),
        "sold_names": [_name(p) for p in plan.sold],
        "bought_names": [_name(p) for p in plan.bought],
        "xi": xi_rows,
        "bench": [
            {
                "player_id": str(r["player_id"]),
                "name": _name(str(r["player_id"])),
                "position": r["position"],
                "mu_post": float(r["mu"]),
            }
            for _, r in plan.squad.loc[~plan.squad["in_xi"]].iterrows()
        ],
    }


def run_pick(
    *,
    gw: int,
    xmi_path: Path,
    odds_path: Path | None,
    odds_t1_path: Path | None,
    fetch_odds: bool,
    entry_id: int = ENTRY_ID,
    time_limit: float = 60.0,
    use_l4: bool = True,
    use_props: bool = False,
    centreline_horizon: int = CENTRELINE_HORIZON,
) -> dict[str, Any]:
    played_gw = int(gw) - 1
    bootstrap = fetch_bootstrap()
    bundle = fetch_entry_bundle(entry_id, played_gw)
    entry = bundle["entry"]
    picks = bundle["picks"]
    transfers = bundle["transfers"]
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    els = {int(e["id"]): e for e in bootstrap["elements"]}

    fx_t = _get_json(f"https://fantasy.premierleague.com/api/fixtures/?event={int(gw)}")
    fx_t1 = _get_json(
        f"https://fantasy.premierleague.com/api/fixtures/?event={int(gw) + 1}"
    )
    pairs_t = [(teams[int(f["team_h"])], teams[int(f["team_a"])]) for f in fx_t]
    pairs_t1 = [(teams[int(f["team_h"])], teams[int(f["team_a"])]) for f in fx_t1]

    held_ids = [int(p["element"]) for p in picks.get("picks") or []]
    if len(held_ids) != 15:
        raise RuntimeError(f"expected 15 picks, got {len(held_ids)}")

    hist_path = CACHE / "player_gw_2026_27.csv"
    hist_raw = pd.read_csv(hist_path)
    hist_raw = append_live_gw_history(
        hist_raw,
        bootstrap=bootstrap,
        live=bundle["live_played"],
        gw=played_gw,
    )
    hist = history_for_numagent(hist_raw)
    scratch_hist = SCRATCH / f"player_gw_2026_27_through_gw{played_gw:02d}.csv"
    hist_raw.to_csv(scratch_hist, index=False)

    xmi = load_xmi_map(xmi_path, gw)

    if fetch_odds:
        raise RuntimeError(
            "Odds API fetch is banned. Pull Betfair with "
            "`python3 -m src.live.betfair_lines --out …` or pass "
            "`--from-snapshot` for offline rehearsal."
        )
    if odds_path is None or not odds_path.is_file():
        raise FileNotFoundError(
            "need --from-snapshot odds JSON (legacy rehearsal) or a Betfair "
            "slate wired into the scorer"
        )
    odds_docs: list[dict[str, Any]] = [
        json.loads(odds_path.read_text(encoding="utf-8"))
    ]
    if use_l4 and odds_t1_path is not None and odds_t1_path.is_file():
        odds_docs.append(json.loads(odds_t1_path.read_text(encoding="utf-8")))
    odds_doc = _merge_odds_docs(odds_docs)

    pool_t, meta_t = score_pool_from_odds(
        bootstrap=bootstrap,
        hist=hist,
        before_gw=gw,
        xmi=xmi,
        odds_doc=odds_doc,
        fixture_pairs=pairs_t,
        minutes_mode="xmi_then_rolling",
        use_props=use_props,
    )
    work_t, _ = attach_mu_post(pool_t, mu_col="score_xp_v2", se_col="xp_se", mode="none")

    work_t1 = None
    meta_t1: dict[str, Any] | None = None
    if use_l4:
        pool_t1, meta_t1 = score_pool_from_odds(
            bootstrap=bootstrap,
            hist=hist,
            before_gw=gw,
            xmi=xmi,
            odds_doc=odds_doc,
            fixture_pairs=pairs_t1,
            minutes_mode="rolling3",
            use_props=use_props,
        )
        work_t1, _ = attach_mu_post(
            pool_t1, mu_col="score_xp_v2", se_col="xp_se", mode="none"
        )
        if int(meta_t1.get("n_fixtures_ok") or 0) < 1:
            print("L4 skipped: no priced t+1 fixtures", flush=True)
            use_l4 = False
            work_t1 = None

    eh = picks.get("entry_history") or {}
    bank = int(eh.get("bank", entry.get("last_deadline_bank") or 0))
    ft = 1
    purchase = reconstruct_purchases(
        held_ids=held_ids,
        transfers=transfers,
        bootstrap=bootstrap,
    )
    held_rows = []
    for eid in held_ids:
        el = els[eid]
        held_rows.append(
            {
                "player_id": str(eid),
                "element": eid,
                "name": el["web_name"],
                "position": POS_MAP[int(el["element_type"])],
                "club": teams[int(el["team"])],
                "value": int(el["now_cost"]),
                "purchase": int(purchase[eid]),
                "mu_post": float(
                    work_t.loc[work_t["element"] == eid, "mu_post"].iloc[0]
                )
                if (work_t["element"] == eid).any()
                else 0.0,
            }
        )
    held = held_from_frame(pd.DataFrame(held_rows), bank=bank, ft=ft)

    missing = [p for p in held.player_ids if p not in set(work_t["player_id"].astype(str))]
    if missing:
        raise RuntimeError(f"held missing from scored pool: {missing}")

    l3 = solve_week_transfers(
        work_t, held, mu_col="mu_post", max_hits=15, time_limit=time_limit
    )
    l3_block = _plan_block(l3, els=els, label="L3")

    l4_block = None
    centreline_meta: dict[str, Any] = {}
    if use_l4 and work_t1 is not None:
        pools: list[pd.DataFrame] = [work_t, work_t1]
        H = max(BOOKIE_DEPTH, int(centreline_horizon))
        if H > BOOKIE_DEPTH:
            try:
                from src.markets.numagent_xp_pilot import _asof_alphas
                from src.models.opta_gf_ga_horizon import (
                    league_avg_gf,
                    rank_gf_ga_profile_excluding,
                    season_final_ranks,
                )
                from src.models.short_horizon_cascade import build_frozen_side_table
                from src.strategy.l3_centreline_horizon import (
                    _opta_e0,
                    opta_fits_from_fixture_pairs,
                    score_opta_fits_on_side,
                )
                from src.strategy.layer4_mupost_priced import _align_to_week0

                e0 = _opta_e0("2026-27")
                ranks = season_final_ranks(e0)
                profile = rank_gf_ga_profile_excluding(e0)
                league_gf = float(league_avg_gf(profile))
                alpha_g, alpha_a = _asof_alphas(hist)
                side, alpha_g, alpha_a = build_frozen_side_table(
                    hist,
                    alpha_g=float(alpha_g),
                    alpha_a=float(alpha_a),
                    season="2026-27",
                    historical_rate_baseline=True,
                )
                opta_gws: list[int] = []
                for step in range(BOOKIE_DEPTH, H):
                    gw_f = int(gw) + step
                    fx_f = _get_json(
                        f"https://fantasy.premierleague.com/api/fixtures/?event={gw_f}"
                    )
                    if not fx_f:
                        break
                    pairs_f = [
                        (teams[int(f["team_h"])], teams[int(f["team_a"])])
                        for f in fx_f
                        if f.get("team_h") is not None and f.get("team_a") is not None
                    ]
                    fits = opta_fits_from_fixture_pairs(
                        pairs_f,
                        ranks=ranks,
                        profile=profile,
                        league_gf=league_gf,
                    )
                    raw = score_opta_fits_on_side(
                        side, fits, alpha_g=float(alpha_g), alpha_a=float(alpha_a)
                    )
                    if raw.empty:
                        break
                    # Align element ids onto week-0 universe.
                    raw = raw.copy()
                    raw["player_id"] = raw["element"].astype(str)
                    raw["mu_post"] = raw["score_xp_v2"]
                    if "xp_se" not in raw.columns:
                        raw["xp_se"] = 0.0
                    pools.append(_align_to_week0(work_t, raw))
                    opta_gws.append(gw_f)
                centreline_meta = {
                    "protocol": "l3_centreline_forecast",
                    "horizon": len(pools),
                    "opta_gws": opta_gws,
                    "opta_e0": e0,
                    "near": "priced_odds_t_t1",
                    "far": "opta_blind",
                }
                print(
                    f"L4 centreline horizon={len(pools)} "
                    f"(Opta far GWs {opta_gws or '—'})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — fall back to priced-only
                print(f"L4 centreline Opta extend skipped: {exc}", flush=True)
                centreline_meta = {
                    "protocol": "l3_centreline_forecast",
                    "horizon": BOOKIE_DEPTH,
                    "opta_gws": [],
                    "error": str(exc),
                    "fallback": "priced_only",
                }
        hz = solve_horizon_transfers(
            pools,
            held,
            mu_col="mu_post",
            horizon=len(pools),
            gamma=L4_GAMMA,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=15,
            time_limit=time_limit,
        )
        l4_block = _plan_block(hz.week0, els=els, label="L4")
        l4_block["horizon"] = int(hz.horizon)
        l4_block["gamma"] = float(hz.gamma)
        l4_block["horizon_objective"] = float(hz.objective)
        l4_block["week_xi_mu"] = list(hz.week_xi_mu)
        l4_block["week_captain_mu"] = list(hz.week_captain_mu)
        l4_block["status_horizon"] = hz.status
        l4_block["centreline"] = centreline_meta

    out_dir = PRED / f"gw{gw:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    board_path = out_dir / f"numagent_board_{stamp}.csv"
    work_t.to_csv(board_path, index=False)
    board_t1_path = None
    if work_t1 is not None:
        board_t1_path = out_dir / f"numagent_board_t1_{stamp}.csv"
        work_t1.to_csv(board_t1_path, index=False)

    primary = l4_block if l4_block is not None else l3_block
    result = {
        "ok": True,
        "gw": gw,
        "primary": "L4" if l4_block is not None else "L3",
        "entry_id": entry_id,
        "team_name": entry.get("name"),
        "played_through": played_gw,
        "bank": bank,
        "ft": ft,
        "active_chip_last_gw": picks.get("active_chip"),
        "odds_source": "live" if fetch_odds else "snapshot",
        "odds_fetched_at": odds_doc.get("fetched_at"),
        "odds_remaining_after": odds_doc.get("remaining_after"),
        "n_odds_ok": sum(1 for m in odds_doc.get("matches") or [] if m.get("ok")),
        "n_fixtures_t": len(pairs_t),
        "n_fixtures_t1": len(pairs_t1),
        "score_meta_t": meta_t,
        "score_meta_t1": meta_t1,
        "xmi_path": str(xmi_path),
        "history_scratch": str(scratch_hist),
        "board_path": str(board_path),
        "board_t1_path": str(board_t1_path) if board_t1_path else None,
        "held": [
            {
                "id": eid,
                "name": els[eid]["web_name"],
                "purchase": purchase[eid],
                "now_cost": int(els[eid]["now_cost"]),
                "mu_post_t": float(
                    work_t.loc[work_t["element"] == eid, "mu_post"].iloc[0]
                ),
                "xmi": float(xmi.get(eid, float("nan"))),
            }
            for eid in held_ids
        ],
        "recommendation": primary,
        "l3": l3_block,
        "l4": l4_block,
    }
    return result


def _write_plan_section(lines: list[str], title: str, plan: dict[str, Any]) -> None:
    lines += [
        f"## {title}",
        "",
        f"- Transfers: **{plan.get('n_transfers')}** "
        f"(hits {plan.get('hits')}, −{plan.get('hit_points')} pts)",
        f"- Sold: {', '.join(plan.get('sold_names') or []) or '—'}",
        f"- Bought: {', '.join(plan.get('bought_names') or []) or '—'}",
        f"- Captain: **{plan.get('captain_name')}**",
        f"- Formation: {plan.get('formation')}",
        f"- Week-0 objective (μ − hits): {plan.get('objective'):.2f} "
        f"(XI {plan.get('xi_mu'):.2f} + C {plan.get('captain_mu'):.2f})",
    ]
    if plan.get("horizon_objective") is not None:
        lines.append(
            f"- Horizon objective (γ={plan.get('gamma')}, H={plan.get('horizon')}): "
            f"**{float(plan['horizon_objective']):.2f}** "
            f"(week μ {plan.get('week_xi_mu')} + C {plan.get('week_captain_mu')})"
        )
    lines += [
        f"- Bank after: {plan.get('bank_after')} · FT after: {plan.get('ft_after')}",
        "",
        "### XI",
        "",
        "| Player | Pos | Club | μ | C |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for r in plan.get("xi") or []:
        lines.append(
            f"| {r['name']} | {r['position']} | {r['club']} | {r['mu_post']:.2f} | "
            f"{'C' if r.get('captain') else ''} |"
        )
    lines += ["", "### Bench", "", "| Player | Pos | μ |", "| --- | --- | ---: |"]
    for r in plan.get("bench") or []:
        lines.append(f"| {r['name']} | {r['position']} | {r['mu_post']:.2f} |")
    lines.append("")


def write_report(doc: dict[str, Any], path: Path) -> None:
    primary = doc.get("recommendation") or doc.get("l4") or doc.get("l3") or {}
    meta_t = doc.get("score_meta_t") or {}
    meta_t1 = doc.get("score_meta_t1") or {}
    lines = [
        f"# Numagent 2.0 live pick — GW{doc.get('gw')}",
        "",
        f"Primary solver: **{doc.get('primary')}**. "
        f"Team `{doc.get('team_name')}` (entry {doc.get('entry_id')}). "
        f"Bank {doc.get('bank')} tenths, FT {doc.get('ft')}. "
        f"Odds: `{doc.get('odds_source')}` "
        f"({doc.get('n_odds_ok')} priced events; "
        f"t={meta_t.get('n_fixtures_ok')}/{doc.get('n_fixtures_t')}, "
        f"t+1={meta_t1.get('n_fixtures_ok')}/{doc.get('n_fixtures_t1')}).",
        "",
        f"α_G={meta_t.get('alpha_g'):.4f}, α_A={meta_t.get('alpha_a'):.4f}. "
        f"Week-t minutes: xmi `{meta_t.get('n_m_xmi')}` / rolling3 `{meta_t.get('n_m_rolling3')}`. "
        f"Week-t+1 minutes: rolling3 only (L4 as-of law).",
        "",
    ]
    _write_plan_section(
        lines,
        f"Recommendation ({doc.get('primary')})",
        primary,
    )
    if doc.get("l4") and doc.get("l3") and doc.get("primary") == "L4":
        l3 = doc["l3"]
        lines += [
            "## L3 contrast (week t only)",
            "",
            f"- Transfers: {l3.get('n_transfers')} "
            f"({', '.join(l3.get('sold_names') or []) or '—'} → "
            f"{', '.join(l3.get('bought_names') or []) or '—'})",
            f"- Captain: {l3.get('captain_name')}",
            f"- Objective: {l3.get('objective'):.2f}",
            "",
        ]
    lines += ["## Fixtures scored (week t)", ""]
    for fx in meta_t.get("fixtures_scored") or []:
        lines.append(
            f"- {fx['match']}: props {fx.get('props_matched')}/{fx.get('props_n')}"
        )
    if meta_t1:
        lines += ["", "## Fixtures scored (week t+1)", ""]
        for fx in meta_t1.get("fixtures_scored") or []:
            lines.append(
                f"- {fx['match']}: props {fx.get('props_matched')}/{fx.get('props_n')}"
            )
    cl = (doc.get("l4") or {}).get("centreline") or {}
    lines += [
        "",
        "## Notes",
        "",
        "- `mu_post` ≡ `score_xp_v2` (no shrink).",
        (
            f"- L4 centreline horizon H={cl.get('horizon', '—')} "
            f"(`protocol.l3_centreline_forecast`): priced t:t+1"
            + (
                f" + Opta blind {cl.get('opta_gws')}"
                if cl.get("opta_gws")
                else ""
            )
            + "; γ=1; no terminal."
        ),
        "- Shares: 2026/27 only, GW < decision week; α pooled prior+live.",
        (
            "- Props: lift-only anytime overlay."
            if (doc.get("score_meta_t") or {}).get("use_props")
            else "- Props: off (AH+OU+BTTS pots only)."
        ),
        "- Week t minutes: xmi probe then rolling3; t+1 / Opta far: rolling3 / frozen side.",
        "- Live book: Betfair Exchange (Odds API banned).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gw", type=int, default=7)
    ap.add_argument("--entry", type=int, default=ENTRY_ID)
    ap.add_argument(
        "--xmi",
        type=Path,
        default=SCRATCH / "gw07_xmi_probe.csv",
    )
    ap.add_argument(
        "--from-snapshot",
        type=Path,
        default=None,
        help="odds JSON for week t (and t+1 if merged / plus --odds-t1)",
    )
    ap.add_argument(
        "--odds-t1",
        type=Path,
        default=None,
        help="optional separate odds JSON for week t+1",
    )
    ap.add_argument(
        "--fetch-odds",
        action="store_true",
        help="DEPRECATED / blocked — Odds API banned; use Betfair folders",
    )
    ap.add_argument(
        "--l3-only",
        action="store_true",
        help="skip L4 (week t only)",
    )
    ap.add_argument(
        "--centreline-horizon",
        type=int,
        default=CENTRELINE_HORIZON,
        help=(
            "L4 centreline horizon under l3_centreline_forecast "
            f"(default {CENTRELINE_HORIZON}: priced t:t+1 + Opta far)"
        ),
    )
    ap.add_argument(
        "--with-props",
        action="store_true",
        help="apply anytime-scorer prop overlay (default: off)",
    )
    ap.add_argument("--time-limit", type=float, default=90.0)
    args = ap.parse_args(argv)

    if args.fetch_odds:
        raise SystemExit(
            "--fetch-odds is blocked (Odds API ban). "
            "Pull Betfair with: python3 -m src.live.betfair_lines "
            "--out data/predictions/2026-27/gwNN/betfair_*"
        )

    snap = args.from_snapshot
    if snap is None:
        for cand in (
            SCRATCH / f"gw{int(args.gw):02d}_odds_bookie_depth2.json",
            SCRATCH / f"gw{int(args.gw):02d}_odds_full.json",
            SCRATCH / f"gw{int(args.gw):02d}_odds_probe.json",
        ):
            if cand.is_file():
                snap = cand
                break

    doc = run_pick(
        gw=int(args.gw),
        xmi_path=args.xmi,
        odds_path=snap,
        odds_t1_path=args.odds_t1,
        fetch_odds=False,
        entry_id=int(args.entry),
        time_limit=float(args.time_limit),
        use_l4=not bool(args.l3_only),
        use_props=bool(args.with_props),
        centreline_horizon=int(args.centreline_horizon),
    )
    out_json = PRED / f"gw{int(args.gw):02d}" / "numagent_pick.json"
    out_md = ROOT / "reports" / f"numagent_live_pick_gw{int(args.gw):02d}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    rec = doc.get("recommendation") or {}
    print(
        json.dumps(
            {
                "ok": doc.get("ok"),
                "primary": doc.get("primary"),
                "transfers": rec.get("n_transfers"),
                "sold": rec.get("sold_names"),
                "bought": rec.get("bought_names"),
                "captain": rec.get("captain_name"),
                "horizon_objective": rec.get("horizon_objective"),
                "report": str(out_md),
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0 if doc.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
