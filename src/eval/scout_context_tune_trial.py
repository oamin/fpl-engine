"""Trial scout-context rotation tune vs pre-tune packet + GW actual minutes.

  PYTHONPATH=. python3 -m src.eval.scout_context_tune_trial \\
    --gw 9 \\
    --before data/predictions/2026-27/gw09/scout_minutes_context_before_tune.json \\
    --ffs data/scratch/ffscout_gw09/ffscout_team_news.json \\
    --bootstrap data/scratch/bootstrap_full.json \\
    --live-event data/scratch/event9_live.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.scout_context import (
    CTX_ROTATION_SINGLE,
    CTX_ROTATION_SUSTAINED,
    score_rotation_against_minutes,
)
from src.live.scout_minutes import build_scout_packet_from_ffscout

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "scout_context_tune_trial_gw09.md"
REPORT_JSON = ROOT / "reports" / "scout_context_tune_trial_gw09.json"


def _counts(overlays: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ov in overlays.values():
        ctx = str(ov.get("context") or "")
        out[ctx] = out.get(ctx, 0) + 1
    return dict(sorted(out.items()))


def _rotation_names(overlays: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ov in overlays.values():
        if ov.get("context") not in {CTX_ROTATION_SINGLE, CTX_ROTATION_SUSTAINED}:
            continue
        exp = None
        recent = None
        for e in ov.get("evidence") or []:
            if str(e).startswith("expand_m="):
                exp = float(str(e).split("=", 1)[1])
            if str(e).startswith("recent="):
                recent = str(e).split("=", 1)[1]
        rows.append(
            {
                "element": ov.get("element"),
                "name": ov.get("name"),
                "club": ov.get("club"),
                "context_label": ov.get("context_label"),
                "m0": ov.get("m0"),
                "expand_m": exp,
                "recent": recent,
            }
        )
    return sorted(rows, key=lambda r: (-(r.get("expand_m") or 0), r["name"]))


def _load_live_minutes(path: Path) -> dict[int, float]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, float] = {}
    for el in doc.get("elements") or []:
        out[int(el["id"])] = float((el.get("stats") or {}).get("minutes") or 0.0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gw", type=int, default=9)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--ffs", type=Path, required=True)
    ap.add_argument("--bootstrap", type=Path, required=True)
    ap.add_argument("--hist", type=Path, default=ROOT / "data" / "cache" / "player_gw_2026_27.csv")
    ap.add_argument("--live-event", type=Path, required=True)
    ap.add_argument(
        "--after-out",
        type=Path,
        default=ROOT / "data" / "predictions" / "2026-27" / "gw09" / "scout_minutes_context_tuned.json",
    )
    args = ap.parse_args(argv)

    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    hist = pd.read_csv(args.hist)
    boot = json.loads(Path(args.bootstrap).read_text(encoding="utf-8"))
    packet = build_scout_packet_from_ffscout(
        decision_gw=int(args.gw),
        hist=hist,
        bootstrap=boot,
        ffs_json=Path(args.ffs),
        horizon=8,
        season="2026-27",
    )
    after_path = Path(args.after_out)
    packet.write_json(after_path)
    after = packet.to_dict()

    actual = _load_live_minutes(Path(args.live_event))
    # Score from context rows rebuilt from overlays
    from src.live.scout_context import ScoutContextRow

    after_rows = [
        ScoutContextRow(
            element=int(ov["element"]),
            name=str(ov["name"]),
            club=str(ov["club"]),
            bucket="xi_omit" if ov.get("source") == "ffscout_xi_omit" else "overlay",
            context=str(ov.get("context") or ""),
            context_detail=str(ov.get("context_detail") or ""),
            tag=str(ov.get("tag") or ""),
            m0=float(ov.get("m0") or 0.0),
            expand_m=None,
            recent_minutes=[],
        )
        for ov in after["overlays"].values()
    ]
    for r, ov in zip(after_rows, after["overlays"].values()):
        for e in ov.get("evidence") or []:
            if str(e).startswith("expand_m="):
                r.expand_m = float(str(e).split("=", 1)[1])
    before_rows = [
        ScoutContextRow(
            element=int(ov["element"]),
            name=str(ov["name"]),
            club=str(ov["club"]),
            bucket="xi_omit" if ov.get("source") == "ffscout_xi_omit" else "overlay",
            context=str(ov.get("context") or ""),
            context_detail="",
            tag=str(ov.get("tag") or ""),
            m0=float(ov.get("m0") or 0.0),
            expand_m=None,
        )
        for ov in before["overlays"].values()
    ]

    score_before = score_rotation_against_minutes(before_rows, actual_minutes=actual)
    score_after = score_rotation_against_minutes(after_rows, actual_minutes=actual)

    before_rot = _rotation_names(before["overlays"])
    after_rot = _rotation_names(after["overlays"])
    removed = {
        (r["name"], r["club"]) for r in before_rot
    } - {(r["name"], r["club"]) for r in after_rot}
    added = {
        (r["name"], r["club"]) for r in after_rot
    } - {(r["name"], r["club"]) for r in before_rot}

    payload = {
        "gw": int(args.gw),
        "before_counts": _counts(before["overlays"]),
        "after_counts": _counts(after["overlays"]),
        "before_n_overlays": len(before["overlays"]),
        "after_n_overlays": len(after["overlays"]),
        "before_rotation_n": len(before_rot),
        "after_rotation_n": len(after_rot),
        "removed_rotation": sorted(
            [{"name": n, "club": c} for n, c in removed],
            key=lambda x: x["name"],
        ),
        "added_rotation": sorted(
            [{"name": n, "club": c} for n, c in added],
            key=lambda x: x["name"],
        ),
        "after_rotation_list": after_rot,
        "score_before": score_before,
        "score_after": score_after,
        "xi_match_quality": (after.get("meta") or {}).get("context", {}).get(
            "xi_match_quality"
        ),
        "tune": (after.get("meta") or {}).get("context", {}).get("rotation_tune"),
        "after_packet": str(after_path),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _prec(block: dict[str, Any], key: str) -> str:
        b = block.get(key) or {}
        if not b.get("n"):
            return "n=0"
        return (
            f"n={b['n']}, low(≤30′)={b['n_low_minutes']}, "
            f"precision={b['precision']:.0%}"
        )

    lines = [
        f"# Scout context rotation tune trial — GW{args.gw}",
        "",
        "Trial of the five tune points: XI→id resolve, club match-quality gate, "
        "GK slot guard, rest/cup blurb veto, single vs sustained, plus precision "
        f"vs actual GW{args.gw} minutes.",
        "",
        "## Counts",
        "",
        "| | before | after |",
        "| --- | ---: | ---: |",
        f"| overlays | {payload['before_n_overlays']} | {payload['after_n_overlays']} |",
        f"| rotation labels | {payload['before_rotation_n']} | {payload['after_rotation_n']} |",
        "",
        "### Context breakdown",
        "",
        "| context | before | after |",
        "| --- | ---: | ---: |",
    ]
    keys = sorted(
        set(payload["before_counts"]) | set(payload["after_counts"])
    )
    for k in keys:
        lines.append(
            f"| `{k}` | {payload['before_counts'].get(k, 0)} | "
            f"{payload['after_counts'].get(k, 0)} |"
        )

    lines += [
        "",
        "## Precision vs actual minutes (≤30′ = hit)",
        "",
        f"- **Before** single-week: {_prec(score_before, CTX_ROTATION_SINGLE)}",
        f"- **After** single-week: {_prec(score_after, CTX_ROTATION_SINGLE)}",
        f"- **Before** sustained: {_prec(score_before, CTX_ROTATION_SUSTAINED)}",
        f"- **After** sustained: {_prec(score_after, CTX_ROTATION_SUSTAINED)}",
        "",
        "### After false positives (played >30′)",
        "",
    ]
    fps = (score_after.get(CTX_ROTATION_SINGLE) or {}).get("false_positives") or []
    if not fps:
        lines.append("_None._")
    else:
        lines += [
            "| player | club | actual′ | expand′ |",
            "| --- | --- | ---: | ---: |",
        ]
        for r in fps:
            exp = r.get("expand_m")
            exp_s = "" if exp is None else f"{float(exp):.0f}"
            lines.append(
                f"| {r['name']} | {r['club']} | {r['actual_m']:.0f} | {exp_s} |"
            )

    lines += [
        "",
        "## Removed by tune (were rotation, now not)",
        "",
    ]
    if not payload["removed_rotation"]:
        lines.append("_None._")
    else:
        for r in payload["removed_rotation"]:
            lines.append(f"- {r['name']} ({r['club']})")

    lines += [
        "",
        "## Remaining rotation labels (after)",
        "",
        "| player | club | label | m0 | expand | recent |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in after_rot:
        lines.append(
            f"| {r['name']} | {r['club']} | {r['context_label']} | "
            f"{r['m0']} | {r.get('expand_m') or ''} | {r.get('recent') or ''} |"
        )

    lines += [
        "",
        "## XI match quality (after)",
        "",
        "| club | matched | ok | unmatched sample |",
        "| --- | --- | --- | --- |",
    ]
    for q in payload.get("xi_match_quality") or []:
        lines.append(
            f"| {q['club']} | {q['n_matched']}/{q['n_xi']} | {q['ok']} | "
            f"{', '.join(q.get('unmatched') or []) or '—'} |"
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.scout_context_tune_trial \\",
        f"  --gw {args.gw} \\",
        f"  --before {args.before} \\",
        f"  --ffs {args.ffs} \\",
        f"  --bootstrap {args.bootstrap} \\",
        f"  --live-event {args.live_event}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT),
        "json": str(REPORT_JSON),
        "before_rotation_n": payload["before_rotation_n"],
        "after_rotation_n": payload["after_rotation_n"],
        "precision_before": (score_before.get(CTX_ROTATION_SINGLE) or {}).get("precision"),
        "precision_after": (score_after.get(CTX_ROTATION_SINGLE) or {}).get("precision"),
        "removed_n": len(payload["removed_rotation"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
