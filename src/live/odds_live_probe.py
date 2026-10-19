"""Live Odds API probe: AH+OU+BTTS → M (BTTS on), optional anytime props → share rescale.

Quota (estimate, region=us only). BTTS is **not** on bulk ``/odds`` — use
per-event ``/events/{id}/odds``:

  - match markets ``spreads,totals,btts`` → ~3 credits / event
  - props ``player_goal_scorer_anytime`` → ~1 credit / event (same endpoint)

  python -m src.live.odds_live_probe --credits-only
  python -m src.live.odds_live_probe --matches 2 --props-matches 1
  python -m src.live.odds_live_probe --from-snapshot data/scratch/gw07_odds_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.markets.prop_shares import rescale_goal_shares
from src.markets.scoreline import fair_pair_from_decimals, fit_scoreline, team_moments
from src.markets.shin import shin_pair

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "data" / "scratch"
SPORT = "soccer_epl"
REGION = "us"
MATCH_MARKETS = "spreads,totals,btts"
PROP_MARKET = "player_goal_scorer_anytime"


def _load_odds_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _mean_decimal(prices: list[float]) -> float | None:
    vals = [float(p) for p in prices if p and p > 1.0 and math.isfinite(p)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def extract_match_targets(event: dict[str, Any]) -> dict[str, Any] | None:
    """Fair AH / OU / BTTS from an Odds API event payload. Prefers OU 2.5."""
    ah_home: list[float] = []
    ah_away: list[float] = []
    ah_line: float | None = None
    over: list[float] = []
    under: list[float] = []
    ou_point: float | None = None
    yes: list[float] = []
    no: list[float] = []

    for bm in event.get("bookmakers") or []:
        for market in bm.get("markets") or []:
            key = market.get("key")
            outcomes = market.get("outcomes") or []
            if key == "spreads":
                home_o = next((o for o in outcomes if o.get("name") == event.get("home_team")), None)
                away_o = next((o for o in outcomes if o.get("name") == event.get("away_team")), None)
                if not home_o or not away_o:
                    continue
                line = float(home_o.get("point"))
                if ah_line is None:
                    ah_line = line
                if abs(line - ah_line) > 1e-9:
                    continue
                ah_home.append(float(home_o["price"]))
                ah_away.append(float(away_o["price"]))
            elif key == "totals":
                over_o = next((o for o in outcomes if str(o.get("name")).lower() == "over"), None)
                under_o = next((o for o in outcomes if str(o.get("name")).lower() == "under"), None)
                if not over_o or not under_o:
                    continue
                point = float(over_o.get("point"))
                # Prefer 2.5; otherwise take the first seen line.
                if ou_point is None:
                    ou_point = point
                if abs(point - 2.5) < 1e-9:
                    ou_point = 2.5
                    over.append(float(over_o["price"]))
                    under.append(float(under_o["price"]))
                elif abs(point - float(ou_point)) < 1e-9 and abs(float(ou_point) - 2.5) > 1e-9:
                    over.append(float(over_o["price"]))
                    under.append(float(under_o["price"]))
            elif key == "btts":
                y = next((o for o in outcomes if str(o.get("name")).lower() == "yes"), None)
                n = next((o for o in outcomes if str(o.get("name")).lower() == "no"), None)
                if y and n:
                    yes.append(float(y["price"]))
                    no.append(float(n["price"]))

    # If we preferred 2.5 but collected none, re-scan for any OU line.
    if not over:
        for bm in event.get("bookmakers") or []:
            for market in bm.get("markets") or []:
                if market.get("key") != "totals":
                    continue
                outcomes = market.get("outcomes") or []
                over_o = next((o for o in outcomes if str(o.get("name")).lower() == "over"), None)
                under_o = next((o for o in outcomes if str(o.get("name")).lower() == "under"), None)
                if not over_o or not under_o:
                    continue
                ou_point = float(over_o.get("point"))
                over.append(float(over_o["price"]))
                under.append(float(under_o["price"]))
                break
            if over:
                break

    if ah_line is None or not ah_home or not over:
        return None
    d_ah_h = _mean_decimal(ah_home)
    d_ah_a = _mean_decimal(ah_away)
    d_over = _mean_decimal(over)
    d_under = _mean_decimal(under)
    if None in (d_ah_h, d_ah_a, d_over, d_under):
        return None
    p_ah, _ = shin_pair(d_ah_h, d_ah_a)  # type: ignore[arg-type]
    p_over, _ = fair_pair_from_decimals(d_over, d_under)  # type: ignore[arg-type]
    has_btts = bool(yes and no)
    if has_btts:
        d_yes = _mean_decimal(yes)
        d_no = _mean_decimal(no)
        if d_yes is None or d_no is None:
            has_btts = False
            p_btts = 0.5
            w_btts = 0.0
        else:
            p_btts, _ = fair_pair_from_decimals(d_yes, d_no)
            w_btts = 1.0
    else:
        p_btts = 0.5
        w_btts = 0.0

    # Note: fitter still uses P(total≥3) functional; non-2.5 OU is approximate.
    return {
        "home": event.get("home_team"),
        "away": event.get("away_team"),
        "id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "ah_line": ah_line,
        "ou_point": ou_point,
        "p_ah": p_ah,
        "p_over_25": p_over,
        "p_btts": p_btts,
        "w_btts": w_btts,
        "n_ah_books": len(ah_home),
        "n_ou_books": len(over),
        "n_btts_books": len(yes),
        "has_btts": has_btts,
    }


def extract_anytime_props(event: dict[str, Any]) -> list[dict[str, Any]]:
    """De-vigged P(anytime scorer) per player from US books (mean across books)."""
    # player -> list of yes decimals
    buckets: dict[str, list[float]] = {}
    for bm in event.get("bookmakers") or []:
        for market in bm.get("markets") or []:
            if market.get("key") != PROP_MARKET:
                continue
            for o in market.get("outcomes") or []:
                # Odds API: description = player name, name = "Yes"/"No" or Just Yes
                name = str(o.get("description") or o.get("name") or "").strip()
                label = str(o.get("name") or "").lower()
                price = float(o.get("price") or 0.0)
                if not name or price <= 1.0:
                    continue
                if label in {"no", "over", "under"}:
                    continue
                # "Yes" outcomes carry description=player; some books use player as name
                if label == "yes" or o.get("description"):
                    player = str(o.get("description") or name)
                    buckets.setdefault(player, []).append(price)
                elif label not in {"yes", "no"}:
                    buckets.setdefault(name, []).append(price)

    rows: list[dict[str, Any]] = []
    for player, prices in sorted(buckets.items()):
        d_yes = _mean_decimal(prices)
        if d_yes is None:
            continue
        # One-sided market: fair p ≈ 1/d with soft vig trim via max with implied
        # (no reliable No side on all books) — clip.
        p = min(0.95, max(0.01, 1.0 / d_yes))
        rows.append({"player": player, "decimal_yes": d_yes, "p_anytime": p, "n_books": len(prices)})
    rows.sort(key=lambda r: -float(r["p_anytime"]))
    return rows


def fit_match(targets: dict[str, Any]) -> dict[str, Any]:
    w_btts = float(targets.get("w_btts") or 0.0)
    fit = fit_scoreline(
        p_ah=float(targets["p_ah"]),
        ah_line=float(targets["ah_line"]),
        p_over_25=float(targets["p_over_25"]),
        p_btts=float(targets["p_btts"]),
        weights=(1.0, 1.0, w_btts),
    )
    moments = team_moments(fit.matrix)
    summary = fit.summary()
    summary.update(
        {
            "home": targets["home"],
            "away": targets["away"],
            "id": targets.get("id"),
            "ah_line": targets["ah_line"],
            "ou_point": targets.get("ou_point"),
            "has_btts": targets.get("has_btts"),
            "w_btts": w_btts,
            "n_ah_books": targets.get("n_ah_books"),
            "n_ou_books": targets.get("n_ou_books"),
            "n_btts_books": targets.get("n_btts_books"),
            "xg_home": moments["xg_home"],
            "xg_away": moments["xg_away"],
            "xcs_home": moments["xcs_home"],
            "xcs_away": moments["xcs_away"],
            "xgc_home": moments["xgc_home"],
            "xgc_away": moments["xgc_away"],
        }
    )
    return summary


def demo_prop_rescale(props: list[dict[str, Any]], G_team: float) -> dict[str, Any]:
    """Toy side: equal baseline shares among top props + one low-rate 'Havertz'."""
    if not props:
        return {"ok": False, "reason": "no_props"}
    # Take top 5 props as the named side; inject a depressed rolling player.
    top = props[:5]
    names = [r["player"] for r in top] + ["RollingReturner"]
    n = len(names)
    # Baseline: returner has tiny share (injured rolling); others split rest
    s0 = np.full(n, 0.18)
    s0[-1] = 0.05
    s0 = s0 / s0.sum()
    prop_p = {i: float(top[i]["p_anytime"]) for i in range(len(top))}
    # Suppose returner also has a strong prop (Havertz case)
    if props:
        prop_p[n - 1] = max(float(props[0]["p_anytime"]) * 0.9, 0.25)
    s1, diag = rescale_goal_shares(s0, G_team=G_team, prop_p=prop_p)
    return {
        "ok": True,
        "G_team": G_team,
        "players": [
            {
                "name": names[i],
                "s_baseline": float(s0[i]),
                "eg_baseline": float(s0[i] * G_team),
                "p_anytime": prop_p.get(i),
                "s_rescaled": float(s1[i]),
                "eg_rescaled": float(s1[i] * G_team),
            }
            for i in range(n)
        ],
        "diag": diag,
    }


def credits_only() -> dict[str, Any]:
    import httpx

    key = _load_odds_key()
    if not key:
        return {"ok": False, "reason": "no_odds_api_key"}
    with httpx.Client(timeout=30.0) as client:
        r = client.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/events",
            params={"apiKey": key},
        )
    return {
        "ok": r.status_code == 200,
        "status": r.status_code,
        "remaining": r.headers.get("x-requests-remaining"),
        "used": r.headers.get("x-requests-used"),
        "n_events": len(r.json()) if r.status_code == 200 else 0,
    }


def fetch_probe(*, n_matches: int, n_props: int) -> dict[str, Any]:
    import httpx

    key = _load_odds_key()
    out: dict[str, Any] = {
        "ok": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "region": REGION,
        "match_markets": MATCH_MARKETS,
        "prop_market": PROP_MARKET,
        "remaining_before": None,
        "remaining_after": None,
        "matches": [],
        "props": {},
    }
    if not key:
        out["reason"] = "no_odds_api_key"
        return out

    n_m = max(1, int(n_matches))
    n_p = max(0, int(n_props))
    # Per-event: ~3 for AH+OU+BTTS; +1 when anytime props are requested on that event.
    estimate = 3 * n_m + max(0, min(n_p, n_m))
    out["estimated_credits"] = estimate

    with httpx.Client(timeout=60.0) as client:
        ev = client.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/events",
            params={"apiKey": key},
        )
        out["remaining_before"] = ev.headers.get("x-requests-remaining")
        if ev.status_code != 200:
            out["reason"] = f"events_http_{ev.status_code}"
            return out
        try:
            rem = int(out["remaining_before"] or "0")
        except ValueError:
            rem = 0
        if rem < estimate + 20:
            out["reason"] = f"low_credits remaining={rem} need≈{estimate}+20 reserve"
            return out

        events = ev.json()
        now = datetime.now(timezone.utc)

        def days(e: dict[str, Any]) -> float:
            kick = datetime.fromisoformat(str(e["commence_time"]).replace("Z", "+00:00"))
            return (kick - now).total_seconds() / 86400.0

        ranked = sorted(events, key=lambda e: abs(days(e) - 1.0))
        picked = ranked[:n_m]
        prop_budget = n_p
        last_costs: list[str] = []

        for event in picked:
            rid = event.get("id")
            markets = MATCH_MARKETS
            want_props = prop_budget > 0
            if want_props:
                markets = f"{MATCH_MARKETS},{PROP_MARKET}"
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
            last = odds.headers.get("x-requests-last")
            if last:
                last_costs.append(str(last))
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
            targets["id"] = rid
            fitted = fit_match(targets)
            fitted["ok"] = True
            fitted["id"] = rid
            fitted["days"] = round(days(event), 2)
            out["matches"].append(fitted)

            if want_props:
                prop_budget -= 1
                props = extract_anytime_props(payload)
                G_home = float(fitted.get("xg_home") or 1.2) * 0.96
                demo = demo_prop_rescale(props, G_team=G_home)
            out["props"][name] = {
                "ok": True,
                "n_players": len(props),
                "top": props[:12],
                "all": props,
                "rescale_demo": demo,
            }

        out["last_cost_odds"] = "+".join(last_costs) if last_costs else None

    out["ok"] = True
    out["reason"] = "live"
    return out


def write_report(doc: dict[str, Any], path: Path) -> None:
    lines = [
        "# Live odds probe — BTTS in M + anytime props",
        "",
        f"Fetched: `{doc.get('fetched_at')}` · region=`{doc.get('region')}`",
        f"Credits: before=`{doc.get('remaining_before')}` after=`{doc.get('remaining_after')}` "
        f"est=`{doc.get('estimated_credits')}` last_odds=`{doc.get('last_cost_odds')}` "
        f"last_props=`{doc.get('last_cost_props')}`",
        f"Reason: `{doc.get('reason')}`",
        "",
        "## Match markets → Dixon–Coles M (BTTS weight 1 when present)",
        "",
        "| Match | days | AH | OU pt | BTTS? | λH | λA | ρ | xG_h | xG_a | SSE |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in doc.get("matches") or []:
        if not m.get("ok"):
            lines.append(
                f"| {m.get('home')} v {m.get('away')} |  |  |  |  |  |  |  |  |  | {m.get('reason')} |"
            )
            continue
        lines.append(
            f"| {m['home']} v {m['away']} | {m.get('days')} | {m.get('ah_line')} | "
            f"{m.get('ou_point')} | {'yes' if m.get('has_btts') else 'no'} | "
            f"{m.get('lam_h', float('nan')):.3f} | {m.get('lam_a', float('nan')):.3f} | "
            f"{m.get('rho', float('nan')):+.3f} | {m.get('xg_home', float('nan')):.2f} | "
            f"{m.get('xg_away', float('nan')):.2f} | {m.get('sse', float('nan')):.4f} |"
        )
    lines += ["", "## Anytime scorer props → share rescale demo", ""]
    props = doc.get("props") or {}
    if not props:
        lines.append("No props pulled.")
    for match, block in props.items():
        lines.append(f"### {match}")
        lines.append("")
        if not block.get("ok"):
            lines.append(f"Failed: `{block.get('reason')}`")
            lines.append("")
            continue
        lines.append(f"Players with Yes quotes: **{block.get('n_players')}**")
        lines.append("")
        lines.append("| Player | p_anytime | books |")
        lines.append("| --- | ---: | ---: |")
        for row in block.get("top") or []:
            lines.append(
                f"| {row['player']} | {row['p_anytime']:.3f} | {row['n_books']} |"
            )
        demo = block.get("rescale_demo") or {}
        if demo.get("ok"):
            lines.append("")
            lines.append(
                f"Rescale demo on home pot G≈{demo['G_team']:.2f} "
                f"(covered={demo['diag'].get('covered'):.2f}, residual={demo['diag'].get('residual'):.2f}):"
            )
            lines.append("")
            lines.append("| Player | s0 | E[G]0 | p | s1 | E[G]1 |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for row in demo.get("players") or []:
                p = row.get("p_anytime")
                ptxt = "" if p is None else f"{p:.3f}"
                lines.append(
                    f"| {row['name']} | {row['s_baseline']:.3f} | {row['eg_baseline']:.3f} | "
                    f"{ptxt} | {row['s_rescaled']:.3f} | {row['eg_rescaled']:.3f} |"
                )
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- Historical FD screens keep BTTS weight 0; this live path turns it on when priced.",
        "- Prop rescale: P(anytime)→λ=−log(1−p); pin covered E[G]; residual pot by baseline shares.",
        "- OU functional in the fitter is still P(total≥3); non-2.5 lines are approximate until generalised.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--credits-only", action="store_true")
    ap.add_argument("--matches", type=int, default=4, help="how many near fixtures to fit")
    ap.add_argument("--props-matches", type=int, default=1, help="how many events for anytime props")
    ap.add_argument("--from-snapshot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=SCRATCH / "gw07_odds_probe.json")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "live_odds_btts_props_probe.md")
    args = ap.parse_args(argv)

    if args.credits_only:
        meta = credits_only()
        print(json.dumps(meta, indent=2))
        return 0 if meta.get("ok") else 1

    if args.from_snapshot:
        doc = json.loads(args.from_snapshot.read_text(encoding="utf-8"))
    else:
        est = 3 * max(1, args.matches) + max(0, min(args.props_matches, max(1, args.matches)))
        print(
            f"Odds API probe: ~{est} credits "
            f"(per-event {MATCH_MARKETS} ×{args.matches}"
            f"{f' + {PROP_MARKET} ×{args.props_matches}' if args.props_matches else ''}, "
            f"region={REGION})",
            flush=True,
        )
        doc = fetch_probe(n_matches=args.matches, n_props=args.props_matches)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, args.report)
    print(json.dumps({k: doc.get(k) for k in ("ok", "reason", "remaining_before", "remaining_after", "estimated_credits")}, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")
    return 0 if doc.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
