"""Gated climb sanity after rate-baseline + full-share μ wire.

Pre-registered: ``protocol.l4_opta_joint_path_mu_wire_climb``.

Arms (locked joint-path planner; Opta far ρ=0.85→0):

* ``joint_path_decay_legacy`` — prior ``joint_path_decay`` dumps (pre-wire μ)
* ``l4_opta_h12_wired`` — L4 oneshot on wired centreline boards
* ``joint_path_decay_wired`` — joint path + decay on wired boards

Comparisons:

* wired decay − legacy decay  (μ wire effect under locked strategiser)
* wired decay − wired L4      (strategiser still holds on new μ)

  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_wire_climb_gate
  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_wire_climb_gate --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.eval.l4_opta_joint_path_climb_gate import (
    BASELINE as L4_ARM,
    GAMMA,
    GW_END,
    GW_START,
    H,
    MIP_REL_GAP,
    N_BOOT,
    NEAR_DEPTH,
    RHO,
    SEED,
    TIME_LIMIT_L4,
    TIME_LIMIT_PATH,
    TREAT_B as DECAY_ARM,
    climb_season,
)
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_mupost_priced import PROCESSED

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "l4_opta_joint_path_mu_wire_climb_gate.md"
STATE = ROOT / "REPORT_STATE.json"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
ROWS_DIR = PROCESSED / "l4_opta_joint_path_mu_wire_rows"

LEGACY = "joint_path_decay_legacy"
WIRED_L4 = "l4_opta_h12_wired"
WIRED_DECAY = "joint_path_decay_wired"
ARMS = (LEGACY, WIRED_L4, WIRED_DECAY)

# Map climb_season arm names → product arm labels in rows.
_CLIMB_ARM = {
    WIRED_L4: L4_ARM,
    WIRED_DECAY: DECAY_ARM,
}


def _now() -> float:
    return time.time()


def _remap_legacy(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for r in blob.get("rows") or []:
        if str(r.get("arm")) != DECAY_ARM:
            continue
        row = dict(r)
        row["arm"] = LEGACY
        out.append(row)
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    xfer = df.loc[df["phase"] == "transfer"].copy()
    pairs = (
        (WIRED_DECAY, LEGACY),
        (WIRED_DECAY, WIRED_L4),
    )
    comparisons: dict[str, Any] = {}
    by_season_all: dict[str, list[dict[str, Any]]] = {}

    for treat, base in pairs:
        piv = xfer.pivot_table(
            index=["season", "gw"], columns="arm", values="pts_net"
        )
        deltas: dict[str, np.ndarray] = {}
        for season in CLOSED_SEASONS:
            if season not in piv.index.get_level_values(0):
                continue
            block = piv.loc[season]
            if treat not in block.columns or base not in block.columns:
                continue
            d = (block[treat] - block[base]).dropna().to_numpy(float)
            if d.size:
                deltas[season] = d
        if set(deltas) != set(CLOSED_SEASONS):
            raise RuntimeError(
                f"{treat}−{base}: missing season deltas; have {sorted(deltas)}"
            )
        for season, arr in deltas.items():
            if arr.size < 20:
                raise RuntimeError(
                    f"{treat}−{base} {season}: fewer than 20 GWs ({arr.size})"
                )
        interval = cluster_interval(
            deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED
        )
        interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
        interval["by_season_mean"] = {
            s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS
        }
        lo, hi = float(interval["lo"]), float(interval["hi"])
        if lo > 0:
            verdict = "promote"
        elif hi < 0:
            verdict = "park_clear_lose"
        else:
            verdict = "inconclusive"
        key = comparison_key(treat, base)
        comparisons[key] = {
            **{k: interval[k] for k in ("mean", "lo", "hi", "n_gws")},
            "incomplete_seasons": {},
            "covers_zero": interval["covers_zero"],
            "by_season_mean": interval["by_season_mean"],
            "verdict": verdict,
            "treatment": treat,
            "baseline": base,
        }
        season_rows = []
        for season in CLOSED_SEASONS:
            sub = xfer.loc[xfer["season"] == season]
            b_pts = float(sub.loc[sub["arm"] == base, "pts_net"].sum())
            t_pts = float(sub.loc[sub["arm"] == treat, "pts_net"].sum())
            season_rows.append(
                {
                    "season": season,
                    "n_gw": int(sub.loc[sub["arm"] == base, "gw"].nunique()),
                    "base_pts": b_pts,
                    "treat_pts": t_pts,
                    "d_pts": t_pts - b_pts,
                    "base_hits": int(
                        sub.loc[sub["arm"] == base, "hit_points"].sum()
                    ),
                    "treat_hits": int(
                        sub.loc[sub["arm"] == treat, "hit_points"].sum()
                    ),
                    "base_transfers": int(
                        sub.loc[sub["arm"] == base, "n_transfers"].sum()
                    ),
                    "treat_transfers": int(
                        sub.loc[sub["arm"] == treat, "n_transfers"].sum()
                    ),
                    "mean_d_gw": float(np.mean(deltas[season])),
                }
            )
        by_season_all[key] = season_rows

    return {
        "comparisons": comparisons,
        "by_season": by_season_all,
        "weeks": df.to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=GW_END)
    ap.add_argument("--time-limit-l4", type=float, default=TIME_LIMIT_L4)
    ap.add_argument("--time-limit-path", type=float, default=TIME_LIMIT_PATH)
    ap.add_argument("--mip-rel-gap", type=float, default=MIP_REL_GAP)
    ap.add_argument("--arms", default=",".join(a for a in ARMS if a != LEGACY))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rows-out", default="")
    ap.add_argument("--from-rows", default="")
    ap.add_argument(
        "--reuse-legacy",
        default="",
        help="comma paths of prior joint_path season dumps (remap joint_path_decay)",
    )
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("l4_opta_joint_path_mu_wire_climb") or {}
    if not screen:
        raise RuntimeError(
            "protocol.l4_opta_joint_path_mu_wire_climb missing — pre-register first"
        )

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"unknown arm {a!r}; choose from {ARMS}")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    gw_start = int(args.gw_start)
    gw_end = int(args.gw_end)
    if args.smoke:
        seasons = ["2025-26"]
        gw_start = 8
        gw_end = 10
        print("[mu-wire-climb] SMOKE 2025-26 GW8–10", flush=True)

    print("[mu-wire-climb] as-of audit…", flush=True)
    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}

    t0 = _now()
    all_rows: list[dict[str, Any]] = []

    if args.reuse_legacy:
        for p in str(args.reuse_legacy).split(","):
            path = Path(p.strip())
            if not path:
                continue
            remapped = _remap_legacy(path)
            all_rows.extend(remapped)
            print(
                f"[mu-wire-climb] reused {len(remapped)} legacy rows from {path}",
                flush=True,
            )

    if args.from_rows:
        paths = [Path(p.strip()) for p in str(args.from_rows).split(",") if p.strip()]
        for path in paths:
            blob = json.loads(path.read_text(encoding="utf-8"))
            all_rows.extend(blob.get("rows") or [])
        print(f"[mu-wire-climb] merged {len(all_rows)} rows", flush=True)
        seasons = sorted({str(r["season"]) for r in all_rows})
        arms = tuple(sorted({str(r["arm"]) for r in all_rows}))
        gw_start = int(min(int(r["gw"]) for r in all_rows))
        gw_end = int(max(int(r["gw"]) for r in all_rows))
    else:
        print("loading closed score_xp_v2 …", flush=True)
        from src.strategy.shrink_climb import SeasonStore

        v2_all = pd.read_csv(CLOSED_V2)
        already = {(str(r["season"]), str(r["arm"])) for r in all_rows}
        for season in seasons:
            merged = _load_merged_enriched(season)
            fd = _fd_index(season)
            prices = pd.read_csv(
                PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
                usecols=[
                    "element",
                    "gw",
                    "value",
                    "team_norm",
                    "position",
                    "player_id",
                ],
            )
            prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
            store = SeasonStore(season, v2_all)
            for arm in arms:
                if arm == LEGACY:
                    continue
                if (season, arm) in already:
                    print(f"skip {season} arm={arm}", flush=True)
                    continue
                climb_arm = _CLIMB_ARM[arm]
                print(f"climbing {season} arm={arm} (as {climb_arm}) …", flush=True)
                rows = climb_season(
                    store,
                    arm=climb_arm,
                    merged=merged,
                    fd=fd,
                    prices=prices,
                    gw_start=gw_start,
                    gw_end=gw_end,
                    time_limit_l4=float(args.time_limit_l4),
                    time_limit_path=float(args.time_limit_path),
                    rho=float(RHO),
                    mip_rel_gap=float(args.mip_rel_gap),
                )
                for r in rows:
                    r["arm"] = arm
                all_rows.extend(rows)

    elapsed = _now() - t0

    if (args.smoke or args.rows_out) and not args.from_rows:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out_path = (
            Path(args.rows_out)
            if args.rows_out
            else PROCESSED / "l4_opta_joint_path_mu_wire_climb_smoke.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "smoke": bool(args.smoke),
                    "seasons": seasons,
                    "arms": list(arms),
                    "gw_start": gw_start,
                    "gw_end": gw_end,
                    "elapsed_s": elapsed,
                    "rows": all_rows,
                    "rho": float(RHO),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        STATE.write_text(
            json.dumps(
                {
                    "task": "l4_opta_joint_path_mu_wire_climb",
                    "status": "rows_dump" if args.rows_out else "smoke_ok",
                    "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "elapsed_s": elapsed,
                    "rows_out": str(out_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "smoke": bool(args.smoke),
                    "rows_out": str(out_path),
                    "elapsed_s": elapsed,
                    "n_rows": len(all_rows),
                },
                indent=2,
            )
        )
        return 0

    need = set(ARMS)
    have = {str(r["arm"]) for r in all_rows}
    if have != need:
        raise RuntimeError(f"full gate needs arms {need}; have {have}")
    if set(seasons) != set(CLOSED_SEASONS):
        raise RuntimeError(f"full gate requires {CLOSED_SEASONS}; got {seasons}")

    doc = summarise(all_rows)
    comparisons = {
        **_certified_protocol_comparison(),
        **doc["comparisons"],
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "l4_opta_joint_path_mu_wire_climb",
    }
    pair_keys = [
        comparison_key(WIRED_DECAY, LEGACY),
        comparison_key(WIRED_DECAY, WIRED_L4),
    ]
    verdicts = {k: doc["comparisons"][k]["verdict"] for k in pair_keys}
    doc["defs"] = {
        "protocol": "l4_opta_joint_path_mu_wire_climb",
        "arms": list(ARMS),
        "H": H,
        "gamma": GAMMA,
        "rho": float(RHO),
        "near_depth": NEAR_DEPTH,
        "wire": "rate_baseline + full_club_shares on published/centreline",
        "elapsed_s": elapsed,
        "verdicts": verdicts,
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_json = PROCESSED / "l4_opta_joint_path_mu_wire_climb_gate.json"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (PROCESSED / "l4_opta_joint_path_mu_wire_climb_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Joint path μ-wire climb sanity — gated",
        "",
        "Pre-registered: `protocol.l4_opta_joint_path_mu_wire_climb`.",
        "",
        "After likelihood promotes for rate baseline + full-club shares, "
        "sanity-check locked `joint_path_opta_decay` on wired boards.",
        "",
        f"- `{LEGACY}`: prior `joint_path_decay` dumps (pre-wire μ).",
        f"- `{WIRED_L4}`: L4 oneshot on wired centreline μ.",
        f"- `{WIRED_DECAY}`: joint path + Opta far ρ={RHO}→0 on wired μ.",
        f"- Metric: paired GW XI−hits. Wall {elapsed:.0f}s.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Results",
        "",
    ]
    labels = {
        comparison_key(WIRED_DECAY, LEGACY): f"`{WIRED_DECAY}` − `{LEGACY}` (μ wire)",
        comparison_key(WIRED_DECAY, WIRED_L4): f"`{WIRED_DECAY}` − `{WIRED_L4}` (planner)",
    }
    for key in pair_keys:
        iv = doc["comparisons"][key]
        lines += [
            f"### {labels[key]}",
            "",
            f"- Mean Δ pts/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
            f"- Covers 0: **{iv['covers_zero']}**",
            f"- Verdict: **{iv['verdict']}**",
            "",
            "| season | n GW | base | treat | Δ | mean Δ/GW |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in doc["by_season"][key]:
            lines.append(
                f"| {r['season']} | {r['n_gw']} | {r['base_pts']:.0f} | "
                f"{r['treat_pts']:.0f} | {r['d_pts']:+.0f} | {r['mean_d_gw']:+.3f} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Positive μ-wire Δ = wired boards beat pre-wire dumps under the same "
        "joint-path decay strategiser.",
        "- Positive planner Δ = joint path still beats L4 oneshot on wired μ.",
        "- Park the wire if μ-wire interval hi < 0 despite likelihood promote.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_wire_climb_gate",
        "```",
        "",
        f"JSON: `{out_json.relative_to(ROOT)}`",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    STATE.write_text(
        json.dumps(
            {
                "task": "l4_opta_joint_path_mu_wire_climb",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "verdicts": {
                    k: {
                        "verdict": doc["comparisons"][k]["verdict"],
                        "mean": doc["comparisons"][k]["mean"],
                        "lo": doc["comparisons"][k]["lo"],
                        "hi": doc["comparisons"][k]["hi"],
                    }
                    for k in pair_keys
                },
                "report": str(REPORT.relative_to(ROOT)),
                "elapsed_s": elapsed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdicts": verdicts,
                "means": {k: doc["comparisons"][k]["mean"] for k in pair_keys},
                "report": str(REPORT),
                "elapsed_s": elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
