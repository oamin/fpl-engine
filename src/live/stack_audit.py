"""Pre-deadline sensitivity audit for the live three-part stack.

Gemini 2026-10-08 (bc-90156d5d): KEEP as input sensitivity, DROP as accuracy.
Arms compare Odds API lines vs Betfair, ± contextual minutes tags, ± outright
κ=0.5 forecast. Higher expected points is not \"better performance\".

Requires Mac-synced Betfair derived files under
``data/predictions/2026-27/gw06/betfair_*/`` for the Exchange arms.
Does not call the Odds API or Betfair. Does not touch ``data/live/`` freeze.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.live.deadline import (
    BOOTSTRAP_PATH,
    ENTRY_PATH,
    FIXTURES_PATH,
    LOG_PATH,
    ODDS_PATH,
    apply_live_minutes,
    current_costs,
    final_players,
    gameweek_values,
    holdings_state,
    last_observed_minutes,
    load_minutes,
    load_odds_frame,
    purchase_source,
    reconstruct_purchases,
    resolve_holdings,
)
from src.live.lines import LINES_PATH
from src.live.scorer import player_key, price_half

ROOT = Path(__file__).resolve().parents[2]
MINUTES_PATH = ROOT / "data" / "live" / "xmi_gw6.csv"
DEFAULT_CAPTURE = (
    ROOT / "data" / "predictions" / "2026-27" / "gw06" / "official_20261007T220914Z.csv"
)
DEFAULT_BETFAIR = ROOT / "data" / "predictions" / "2026-27" / "gw06" / "betfair_20261008"
REPORT_PATH = ROOT / "reports" / "stack_audit_gw6.md"
DECISION_GW = 6


@dataclass(frozen=True)
class ArmSpec:
    """One ablation configuration."""

    key: str
    label: str
    use_tags: bool
    use_betfair_lines: bool
    use_outrights: bool


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("arm0_baseline", "Odds API + rolling minutes + GW7 copy", False, False, False),
    ArmSpec("arm1_tags", "Odds API + contextual tags + GW7 copy", True, False, False),
    ArmSpec("arm2_betfair", "Betfair lines + tags + GW7 copy", True, True, False),
    ArmSpec("arm3_full", "Betfair lines + tags + outright forecast", True, True, True),
)


def _minute_map(
    bootstrap: Mapping[str, Any],
    logs: pd.DataFrame,
    *,
    use_tags: bool,
    minutes_path: Path,
    gw: int,
) -> dict[str, float]:
    supplied = None
    if use_tags:
        if not minutes_path.is_file():
            raise FileNotFoundError(f"minutes file missing: {minutes_path}")
        supplied = load_minutes(minutes_path, int(gw))
    resolved = apply_live_minutes(
        [int(element["id"]) for element in bootstrap["elements"]],
        supplied,
        last_observed_minutes(logs),
    )
    return {player_key(int(row["player_id"])): float(row["xmi"]) for row in resolved}


def _artifacts_for_arm(
    spec: ArmSpec, betfair_dir: Path | None, empty_dir: Path
) -> Path:
    """Always return a path so ``price_half`` does not auto-discover overlays.

    Outrights / TO_SCORE load only when the Betfair folder is passed. Lines
    come from the odds frame, not from this directory, except for arm3 which
    also needs ``outrights_ranks.json`` beside the pull.
    """
    if spec.use_outrights and betfair_dir is not None and betfair_dir.is_dir():
        return betfair_dir
    return empty_dir


def betfair_ready(betfair_dir: Path | None) -> bool:
    """Derived Exchange pull is present (not raw scratch)."""
    if betfair_dir is None or not betfair_dir.is_dir():
        return False
    return (betfair_dir / "gw_lines.csv").is_file()


def run_arm(
    spec: ArmSpec,
    *,
    gw: int = DECISION_GW,
    entry_path: Path = ENTRY_PATH,
    log_path: Path = LOG_PATH,
    odds_path: Path = ODDS_PATH,
    bootstrap_path: Path = BOOTSTRAP_PATH,
    fixtures_path: Path = FIXTURES_PATH,
    minutes_path: Path = MINUTES_PATH,
    odds_api_lines: Path = LINES_PATH,
    betfair_dir: Path | None = DEFAULT_BETFAIR,
    capture_path: Path | None = DEFAULT_CAPTURE,
    empty_artifacts: Path | None = None,
) -> dict[str, Any]:
    """Price one arm. Returns a JSON-serialisable summary."""
    if spec.use_betfair_lines and not betfair_ready(betfair_dir):
        return {
            "key": spec.key,
            "label": spec.label,
            "skipped": True,
            "reason": f"Betfair derived folder missing: {betfair_dir}",
        }
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    logs = pd.read_csv(log_path)
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    live_path = (
        betfair_dir / "gw_lines.csv"  # type: ignore[operator]
        if spec.use_betfair_lines
        else odds_api_lines
    )
    odds = load_odds_frame(odds_path, live_path if live_path.is_file() else None)
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    holdings = resolve_holdings(
        players, purchases, current_costs(bootstrap), None, sources
    )
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    minutes = _minute_map(
        bootstrap, logs, use_tags=spec.use_tags, minutes_path=minutes_path, gw=gw
    )
    played = {
        int(row["gw"]): str(row["chip"]) for row in entry.get("chips_played") or []
    }
    blank = empty_artifacts if empty_artifacts is not None else Path(tempfile.mkdtemp())
    blank.mkdir(parents=True, exist_ok=True)
    artifacts = _artifacts_for_arm(spec, betfair_dir, blank)
    if spec.use_outrights and not (artifacts / "outrights_ranks.json").is_file():
        return {
            "key": spec.key,
            "label": spec.label,
            "skipped": True,
            "reason": "outrights_ranks.json missing from Betfair folder",
        }
    # Engine sensitivity only: do not mix held outlooks with captured ep_next.
    _ = capture_path  # documented dry-run stamp in the report payload
    scored = price_half(
        gw=int(gw),
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=bootstrap,
        state=state,
        minutes=minutes,
        played=played,
        artifacts_dir=artifacts,
    )
    owned_keys = [row.key for row in holdings]
    names = {
        player_key(int(el["id"])): str(el.get("web_name") or el.get("second_name") or el["id"])
        for el in bootstrap["elements"]
    }
    owned_gw6 = {
        names.get(pid, pid): float(scored.step_scores[int(gw)].get(pid, 0.0))
        for pid in owned_keys
    }
    outlooks = [
        {
            "gw": int(row.gw),
            "held_xi": float(row.held.xi_xp),
            "held_bench": float(row.held.bench_xp),
            "rebuilt_xi": float(row.rebuilt.xi_xp if row.rebuilt is not None else 0.0),
            "fh_xi": float(row.fh_xi or 0.0),
        }
        for row in scored.weeks
        if int(row.gw) in scored.horizon_weeks
    ]
    rebuild_gap = None
    if outlooks:
        first = outlooks[0]
        rebuild_gap = float(first["rebuilt_xi"]) - float(first["held_xi"])
    return {
        "key": spec.key,
        "label": spec.label,
        "skipped": False,
        "use_tags": spec.use_tags,
        "use_betfair_lines": spec.use_betfair_lines,
        "use_outrights": spec.use_outrights,
        "line_weeks": list(scored.line_weeks),
        "horizon_weeks": list(scored.horizon_weeks),
        "copy_note": scored.copy_note,
        "chip": scored.plan.chip,
        "rebuild_gap_gw": rebuild_gap,
        "outlooks": outlooks,
        "owned_gw6": owned_gw6,
        "owned_gw6_sum": float(sum(owned_gw6.values())),
    }


def _delta_owned(
    baseline: Mapping[str, float], other: Mapping[str, float]
) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(set(baseline) | set(other)):
        a = float(baseline.get(name, 0.0))
        b = float(other.get(name, 0.0))
        if abs(b - a) < 1e-9:
            continue
        rows.append({"player": name, "baseline": a, "arm": b, "delta": b - a})
    rows.sort(key=lambda r: -abs(float(r["delta"])))
    return rows


def run_audit(
    *,
    betfair_dir: Path | None = DEFAULT_BETFAIR,
    capture_path: Path | None = DEFAULT_CAPTURE,
    minutes_path: Path = MINUTES_PATH,
    odds_api_lines: Path = LINES_PATH,
    gw: int = DECISION_GW,
) -> dict[str, Any]:
    """Run all four arms and attach deltas vs arm0."""
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as folder:
        empty = Path(folder)
        for spec in ARMS:
            results.append(
                run_arm(
                    spec,
                    gw=gw,
                    minutes_path=minutes_path,
                    odds_api_lines=odds_api_lines,
                    betfair_dir=betfair_dir,
                    capture_path=capture_path,
                    empty_artifacts=empty,
                )
            )
    by_key = {row["key"]: row for row in results}
    baseline = by_key.get("arm0_baseline") or {}
    base_owned = baseline.get("owned_gw6") or {}
    deltas = {}
    for row in results:
        if row.get("skipped") or row["key"] == "arm0_baseline":
            continue
        deltas[row["key"]] = {
            "owned_sum_delta": float(row["owned_gw6_sum"]) - float(baseline.get("owned_gw6_sum") or 0.0),
            "rebuild_gap_delta": (
                None
                if row.get("rebuild_gap_gw") is None or baseline.get("rebuild_gap_gw") is None
                else float(row["rebuild_gap_gw"]) - float(baseline["rebuild_gap_gw"])
            ),
            "owned_movers": _delta_owned(base_owned, row.get("owned_gw6") or {})[:12],
            "copy_note": row.get("copy_note") or "",
            "chip": row.get("chip"),
        }
    return {
        "gw": int(gw),
        "capture": str(capture_path) if capture_path else None,
        "betfair_dir": str(betfair_dir) if betfair_dir else None,
        "betfair_ready": betfair_ready(betfair_dir),
        "arms": results,
        "deltas_vs_arm0": deltas,
        "note": (
            "Sensitivity only. Do not read higher xP as better performance. "
            "Realised check waits until GW6 points land."
        ),
    }


def render_report(payload: Mapping[str, Any]) -> str:
    """Markdown for the PI. No accuracy claims."""
    lines = [
        "# Gameweek 6 stack audit (sensitivity)",
        "",
        "Gemini ([stack audit](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): "
        "pre-deadline ablation of Exchange odds, coarse outright forecast, and "
        "contextual tags. This is input sensitivity, not realised performance.",
        "",
        f"Capture: `{payload.get('capture')}`.",
        f"Betfair folder ready: **{payload.get('betfair_ready')}** "
        f"(`{payload.get('betfair_dir')}`).",
        "",
        "## Arms",
        "",
        "| Arm | Config | Owned GW6 Σ | Rebuild−held GW6 | Chip | Line weeks |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for arm in payload.get("arms") or []:
        if arm.get("skipped"):
            lines.append(
                f"| {arm['key']} | {arm['label']} | — | — | skipped | {arm.get('reason', '')} |"
            )
            continue
        lines.append(
            f"| {arm['key']} | {arm['label']} | {float(arm['owned_gw6_sum']):.2f} | "
            f"{float(arm['rebuild_gap_gw'] or 0.0):.2f} | {arm.get('chip')} | "
            f"{arm.get('line_weeks')} |"
        )
    lines.extend(
        [
            "",
            "## Horizon outlooks (held XI)",
            "",
            "| Arm | GW6 | GW7 | GW8 | Copy note |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for arm in payload.get("arms") or []:
        if arm.get("skipped"):
            lines.append(f"| {arm['key']} | — | — | — | skipped |")
            continue
        by_gw = {int(row["gw"]): float(row["held_xi"]) for row in arm.get("outlooks") or []}
        note = arm.get("copy_note") or "(none — outrights filled unpriced weeks)"
        lines.append(
            f"| {arm['key']} | {by_gw.get(6, 0.0):.2f} | {by_gw.get(7, 0.0):.2f} | "
            f"{by_gw.get(8, 0.0):.2f} | {note} |"
        )
    lines.extend(["", "## Deltas vs arm0 (not performance)", ""])
    deltas = payload.get("deltas_vs_arm0") or {}
    if not deltas:
        lines.append("No completed comparison arms yet.")
    for key, block in deltas.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(
            f"Owned GW6 Σ Δ = {float(block['owned_sum_delta']):+.3f}. "
            f"Rebuild gap Δ = {block['rebuild_gap_delta']}. "
            f"Chip = {block.get('chip')}."
        )
        if block.get("copy_note"):
            lines.append(f"Copy note: {block['copy_note']}")
        movers = block.get("owned_movers") or []
        if movers:
            lines.append("")
            lines.append("| Player | Baseline | Arm | Δ |")
            lines.append("| --- | ---: | ---: | ---: |")
            for row in movers:
                lines.append(
                    f"| {row['player']} | {row['baseline']:.3f} | "
                    f"{row['arm']:.3f} | {row['delta']:+.3f} |"
                )
        lines.append("")
    if not payload.get("betfair_ready"):
        lines.extend(
            [
                "## Sync needed for Exchange arms",
                "",
                "On the Mac, commit the derived files under "
                "`data/predictions/2026-27/gw06/betfair_20261008/` "
                "(`gw_lines.csv`, `outrights_ranks.json`, `betfair_to_score.json`, "
                "`betfair_meta.json`). Do not commit `.env` or `data/scratch/betfair/`. "
                "Then re-run:",
                "",
                "```bash",
                "python3 -m src.live.stack_audit",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## What this is not",
            "",
            "- Not a claim that higher xP is better.",
            "- Not a chip-rule test (the 114 wildcard lead is separate).",
            "- Not a historical Exchange backtest (no books on file).",
            "- Realised XI / Spearman wait until GW6 points are final.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betfair", type=Path, default=DEFAULT_BETFAIR)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--minutes", type=Path, default=MINUTES_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "data" / "predictions" / "2026-27" / "gw06" / "stack_audit.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_audit(
        betfair_dir=args.betfair,
        capture_path=args.capture,
        minutes_path=args.minutes,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    text = render_report(payload)
    args.report.write_text(text, encoding="utf-8")
    print(f"wrote {args.report}")
    print(f"wrote {args.json_out}")
    print(f"betfair_ready={payload['betfair_ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
