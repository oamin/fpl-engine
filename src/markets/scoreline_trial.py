"""Trial Dixon–Coles scoreline fits on a fixed AH/OU/BTTS test set.

Synthetic cases are always run. Optional ``--live N`` pulls up to N upcoming
EPL events from The Odds API (region=us, markets=spreads,totals,btts).
Cost ≈ 3 credits × events with data.

Usage::

    python3 -m src.markets.scoreline_trial
    python3 -m src.markets.scoreline_trial --live 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.markets.scoreline import fair_pair_from_decimals, fit_scoreline
from src.markets.shin import shin_pair

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "scoreline_ah_ou_btts_trial.md"
SCRATCH = ROOT / "data" / "scratch" / "scoreline_trial.json"

# Hand-built fair-prob scenarios (already de-vigged).
SYNTHETIC: list[dict[str, Any]] = [
    {"name": "even_low", "ah_line": 0.0, "p_ah": 0.50, "p_over_25": 0.42, "p_btts": 0.48},
    {"name": "even_high", "ah_line": 0.0, "p_ah": 0.50, "p_over_25": 0.58, "p_btts": 0.55},
    {"name": "home_fav_05", "ah_line": -0.5, "p_ah": 0.58, "p_over_25": 0.52, "p_btts": 0.50},
    {"name": "home_big_fav", "ah_line": -1.0, "p_ah": 0.55, "p_over_25": 0.60, "p_btts": 0.48},
    {"name": "home_quarter", "ah_line": -0.25, "p_ah": 0.54, "p_over_25": 0.50, "p_btts": 0.51},
    {"name": "away_fav_05", "ah_line": 0.5, "p_ah": 0.45, "p_over_25": 0.47, "p_btts": 0.49},
    {"name": "under_no_btts", "ah_line": -0.5, "p_ah": 0.60, "p_over_25": 0.35, "p_btts": 0.38},
    {"name": "over_btts", "ah_line": 0.0, "p_ah": 0.50, "p_over_25": 0.62, "p_btts": 0.60},
    {"name": "stress_high_btts_low_ou", "ah_line": -0.5, "p_ah": 0.55, "p_over_25": 0.40, "p_btts": 0.58},
    {"name": "stress_low_btts_high_ou", "ah_line": -0.5, "p_ah": 0.55, "p_over_25": 0.60, "p_btts": 0.40},
]


def _load_odds_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # zsh profile export (local Mac)
    try:
        import subprocess

        out = subprocess.check_output(
            ["zsh", "-lc", "source ~/.zshrc >/dev/null 2>&1; print -r -- $ODDS_API_KEY"],
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def run_synthetic() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in SYNTHETIC:
        fit = fit_scoreline(
            p_ah=float(case["p_ah"]),
            ah_line=float(case["ah_line"]),
            p_over_25=float(case["p_over_25"]),
            p_btts=float(case["p_btts"]),
        )
        summary = fit.summary()
        summary["name"] = case["name"]
        summary["ah_line"] = case["ah_line"]
        rows.append(summary)
    return rows


def _mean_decimal(prices: list[float]) -> float | None:
    vals = [float(p) for p in prices if p and p > 1.0 and math.isfinite(p)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _extract_live_targets(event: dict[str, Any]) -> dict[str, Any] | None:
    """Build fair AH/OU/BTTS targets from an Odds API event-odds payload."""
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
                # Prefer the main line closest to 0 among books; take first book's line then match.
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
                if abs(point - 2.5) > 1e-9:
                    continue  # only 2.5 for this trial
                ou_point = 2.5
                over.append(float(over_o["price"]))
                under.append(float(under_o["price"]))
            elif key == "btts":
                y = next((o for o in outcomes if str(o.get("name")).lower() == "yes"), None)
                n = next((o for o in outcomes if str(o.get("name")).lower() == "no"), None)
                if y and n:
                    yes.append(float(y["price"]))
                    no.append(float(n["price"]))

    if ah_line is None or not ah_home or not over or not yes:
        return None
    d_ah_h = _mean_decimal(ah_home)
    d_ah_a = _mean_decimal(ah_away)
    d_over = _mean_decimal(over)
    d_under = _mean_decimal(under)
    d_yes = _mean_decimal(yes)
    d_no = _mean_decimal(no)
    if None in (d_ah_h, d_ah_a, d_over, d_under, d_yes, d_no):
        return None
    p_ah, _ = shin_pair(d_ah_h, d_ah_a)  # type: ignore[arg-type]
    p_over, _ = fair_pair_from_decimals(d_over, d_under)  # type: ignore[arg-type]
    p_btts, _ = fair_pair_from_decimals(d_yes, d_no)  # type: ignore[arg-type]
    return {
        "home": event.get("home_team"),
        "away": event.get("away_team"),
        "commence_time": event.get("commence_time"),
        "ah_line": ah_line,
        "ou_point": ou_point,
        "n_ah_books": len(ah_home),
        "n_ou_books": len(over),
        "n_btts_books": len(yes),
        "p_ah": p_ah,
        "p_over_25": p_over,
        "p_btts": p_btts,
        "decimals": {
            "ah_home": d_ah_h,
            "ah_away": d_ah_a,
            "over": d_over,
            "under": d_under,
            "btts_yes": d_yes,
            "btts_no": d_no,
        },
    }


def run_live(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import httpx

    key = _load_odds_key()
    meta: dict[str, Any] = {"ok": False, "reason": "", "remaining": "", "used": ""}
    if not key:
        meta["reason"] = "no_odds_api_key"
        return [], meta
    if limit <= 0:
        meta["reason"] = "live_limit_zero"
        return [], meta

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=45.0) as client:
        ev = client.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/events",
            params={"apiKey": key},
        )
        meta["remaining"] = ev.headers.get("x-requests-remaining", "")
        meta["used"] = ev.headers.get("x-requests-used", "")
        if ev.status_code != 200:
            meta["reason"] = f"events_http_{ev.status_code}"
            return [], meta
        events = ev.json()
        # Prefer fixtures >1 day out (US depth), then fill with nearer if needed.
        now = datetime.now(timezone.utc)

        def days_out(e: dict[str, Any]) -> float:
            kick = datetime.fromisoformat(str(e["commence_time"]).replace("Z", "+00:00"))
            return (kick - now).total_seconds() / 86400.0

        ranked = sorted(events, key=lambda e: abs(days_out(e) - 8.0))
        picked = ranked[: int(limit)]
        for event in picked:
            rid = event["id"]
            odds = client.get(
                f"https://api.the-odds-api.com/v4/sports/soccer_epl/events/{rid}/odds",
                params={
                    "apiKey": key,
                    "regions": "us",
                    "markets": "spreads,totals,btts",
                    "oddsFormat": "decimal",
                },
            )
            meta["remaining"] = odds.headers.get("x-requests-remaining", meta["remaining"])
            meta["last_cost"] = odds.headers.get("x-requests-last", "")
            if odds.status_code != 200:
                rows.append(
                    {
                        "name": f"{event.get('home_team')} v {event.get('away_team')}",
                        "ok": False,
                        "reason": f"odds_http_{odds.status_code}",
                    }
                )
                continue
            payload = odds.json()
            targets = _extract_live_targets(payload)
            if targets is None:
                rows.append(
                    {
                        "name": f"{event.get('home_team')} v {event.get('away_team')}",
                        "ok": False,
                        "reason": "missing_ah_ou25_or_btts",
                        "days": round(days_out(event), 2),
                    }
                )
                continue
            fit = fit_scoreline(
                p_ah=float(targets["p_ah"]),
                ah_line=float(targets["ah_line"]),
                p_over_25=float(targets["p_over_25"]),
                p_btts=float(targets["p_btts"]),
            )
            summary = fit.summary()
            summary.update(
                {
                    "name": f"{targets['home']} v {targets['away']}",
                    "days": round(days_out(event), 2),
                    "ah_line": targets["ah_line"],
                    "n_ah_books": targets["n_ah_books"],
                    "n_ou_books": targets["n_ou_books"],
                    "n_btts_books": targets["n_btts_books"],
                    "ok": True,
                }
            )
            rows.append(summary)
    meta["ok"] = True
    meta["reason"] = "live"
    return rows, meta


def _fmt_res(row: dict[str, Any]) -> str:
    r = row.get("residual") or {}
    return (
        f"AH {r.get('p_ah', float('nan')):+.3f} / "
        f"OU {r.get('p_over_25', float('nan')):+.3f} / "
        f"BTTS {r.get('p_btts', float('nan')):+.3f}"
    )


def write_report(
    synthetic: list[dict[str, Any]],
    live: list[dict[str, Any]],
    live_meta: dict[str, Any],
) -> None:
    lines = [
        "# Scoreline matrix trial (AH / OU / BTTS)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "Fits a Dixon–Coles grid so market functionals match fair AH cover, "
        "P(over 2.5), and P(BTTS). No neutral shrinkage.",
        "",
        "## Synthetic battery",
        "",
        "| Case | AH line | λH | λA | ρ | SSE | Residuals (AH/OU/BTTS) | ok |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in synthetic:
        lines.append(
            f"| {row['name']} | {row['ah_line']} | {row['lam_h']:.3f} | {row['lam_a']:.3f} | "
            f"{row['rho']:+.3f} | {row['sse']:.5f} | {_fmt_res(row)} | "
            f"{'yes' if row.get('success') else 'no'} |"
        )
    syn_ok = sum(1 for r in synthetic if r.get("success"))
    lines += [
        "",
        f"Synthetic hits (SSE residual gate in fitter `success`): **{syn_ok}/{len(synthetic)}**.",
        "",
        "## Live US Odds API (optional)",
        "",
    ]
    if not live:
        lines.append(
            f"No live rows. reason=`{live_meta.get('reason')}`; "
            f"remaining=`{live_meta.get('remaining')}`."
        )
    else:
        lines += [
            f"Meta: reason=`{live_meta.get('reason')}`, remaining=`{live_meta.get('remaining')}`, "
            f"last_cost=`{live_meta.get('last_cost')}`.",
            "",
            "| Match | days | AH line | books AH/OU/BTTS | λH | λA | ρ | SSE | Residuals |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in live:
            if not row.get("ok") and "lam_h" not in row:
                lines.append(
                    f"| {row.get('name')} | {row.get('days', '')} |  |  |  |  |  |  | "
                    f"{row.get('reason')} |"
                )
                continue
            books = f"{row.get('n_ah_books')}/{row.get('n_ou_books')}/{row.get('n_btts_books')}"
            lines.append(
                f"| {row['name']} | {row.get('days')} | {row.get('ah_line')} | {books} | "
                f"{row['lam_h']:.3f} | {row['lam_a']:.3f} | {row['rho']:+.3f} | "
                f"{row['sse']:.5f} | {_fmt_res(row)} |"
            )
    lines += [
        "",
        "## Read",
        "",
        "A small SSE means the three fair probs are jointly feasible under Dixon–Coles. "
        "Stress cases that force high BTTS with low OU (or the reverse) are the ones that "
        "should strain ρ and show larger residuals.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        type=int,
        default=0,
        help="Pull this many upcoming EPL events (us spreads+totals+btts). 0 = synthetic only.",
    )
    args = parser.parse_args(argv)
    synthetic = run_synthetic()
    live: list[dict[str, Any]] = []
    live_meta: dict[str, Any] = {"ok": False, "reason": "not_requested"}
    if args.live > 0:
        live, live_meta = run_live(args.live)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": synthetic,
        "live": live,
        "live_meta": live_meta,
    }
    SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(synthetic, live, live_meta)
    syn_ok = sum(1 for r in synthetic if r.get("success"))
    print(f"synthetic_ok={syn_ok}/{len(synthetic)} report={REPORT}")
    if args.live > 0:
        live_ok = sum(1 for r in live if r.get("lam_h") is not None)
        print(
            f"live_fitted={live_ok}/{len(live)} reason={live_meta.get('reason')} "
            f"remaining={live_meta.get('remaining')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
