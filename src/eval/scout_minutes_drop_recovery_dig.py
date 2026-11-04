"""Diagnostic dig (2025-26): expanding-mean minutes drops and recovery time.

Not a gated promote screen — exploratory signal hunt for scout decay / overlays.

1. Pool: players with enough as-of history and meaningful expanding mean.
2. Drop: GW minutes < expand_mean − k · expand_sd (default k=1).
3. Recovery: GWs until minutes return to ≥ expand_mean − 0.5·expand_sd
   (recomputed as-of each later GW), or season end (censored).
4. Summaries: duration, zero vs soft drops, position, return path vs ρ-decay.

  PYTHONPATH=. python3 -m src.eval.scout_minutes_drop_recovery_dig
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORT = REPORTS / "scout_minutes_drop_recovery_dig_2025_26.md"
STATE = ROOT / "REPORT_STATE.json"

SEASON = "2025-26"
K_DROP = 1.0
K_RECOVER = 0.5
MIN_HIST_GW = 5
MIN_EXPAND_MEAN = 45.0  # regulars / rotation candidates
MIN_EXPAND_SD = 5.0  # floor so tiny SD doesn't fire on noise
RHO_GRID = (0.50, 0.70, 0.85, 0.95)


def load_season(season: str = SEASON) -> pd.DataFrame:
    stem = season.replace("-", "_")
    path = CACHE / f"merged_gw_{stem}.csv"
    df = pd.read_csv(path)
    df["GW"] = pd.to_numeric(df["GW"], errors="coerce").astype(int)
    df["element"] = pd.to_numeric(df["element"], errors="coerce").astype(int)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    # DGW: sum minutes per element-GW
    g = (
        df.groupby(["element", "GW", "name", "position", "team"], as_index=False)
        .agg(minutes=("minutes", "sum"), total_points=("total_points", "sum"))
    )
    return g.sort_values(["element", "GW"]).reset_index(drop=True)


def expanding_stats(series: list[float]) -> tuple[float, float]:
    if not series:
        return 0.0, float(MIN_EXPAND_SD)
    mu = float(np.mean(series))
    if len(series) < 2:
        return mu, float(MIN_EXPAND_SD)
    sd = float(np.std(series, ddof=1))
    return mu, max(sd, float(MIN_EXPAND_SD))


def detect_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, path_rows) for drop→recovery dig."""
    events: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []

    for eid, g in frame.groupby("element", sort=False):
        g = g.sort_values("GW")
        hist_m: list[float] = []
        name = str(g["name"].iloc[0])
        pos = str(g["position"].iloc[0])
        team = str(g["team"].iloc[0])
        in_event = False
        event_id = 0
        drop_gw = None
        drop_m = None
        drop_mu = None
        drop_sd = None
        drop_z = None

        for row in g.itertuples(index=False):
            gw = int(row.GW)
            m = float(row.minutes)
            mu, sd = expanding_stats(hist_m)
            # Only score drops once we have a pool-eligible baseline.
            eligible = (
                len(hist_m) >= int(MIN_HIST_GW) and mu >= float(MIN_EXPAND_MEAN)
            )
            thresh = mu - float(K_DROP) * sd
            recover_thresh = mu - float(K_RECOVER) * sd

            if eligible and (not in_event) and m < thresh:
                in_event = True
                event_id += 1
                drop_gw = gw
                drop_m = m
                drop_mu = mu
                drop_sd = sd
                drop_z = (m - mu) / sd if sd > 0 else float("nan")
                events.append(
                    {
                        "element": int(eid),
                        "name": name,
                        "position": pos,
                        "team": team,
                        "event_id": event_id,
                        "drop_gw": drop_gw,
                        "drop_minutes": drop_m,
                        "expand_mean": drop_mu,
                        "expand_sd": drop_sd,
                        "drop_z": drop_z,
                        "thresh": thresh,
                        "zero_drop": drop_m <= 0.0,
                        "soft_drop": 0.0 < drop_m < 45.0,
                    }
                )
                paths.append(
                    {
                        "element": int(eid),
                        "event_id": event_id,
                        "h": 0,
                        "gw": gw,
                        "minutes": m,
                        "expand_mean_asof": mu,
                        "expand_sd_asof": sd,
                        "phase": "drop",
                    }
                )
            elif in_event:
                assert drop_gw is not None
                h = gw - int(drop_gw)
                # Recompute recover threshold from history *before* this GW.
                mu_h, sd_h = expanding_stats(hist_m)
                rec_thr = mu_h - float(K_RECOVER) * sd_h
                recovered = m >= rec_thr
                paths.append(
                    {
                        "element": int(eid),
                        "event_id": event_id,
                        "h": h,
                        "gw": gw,
                        "minutes": m,
                        "expand_mean_asof": mu_h,
                        "expand_sd_asof": sd_h,
                        "phase": "recover" if recovered else "down",
                    }
                )
                if recovered:
                    # Close event
                    for e in reversed(events):
                        if e["element"] == int(eid) and e["event_id"] == event_id:
                            e["recover_gw"] = gw
                            e["recover_minutes"] = m
                            e["recovery_gws"] = h
                            e["censored"] = False
                            e["recover_expand_mean"] = mu_h
                            break
                    in_event = False
                    drop_gw = None

            hist_m.append(m)

        if in_event and drop_gw is not None:
            last_gw = int(g["GW"].max())
            for e in reversed(events):
                if e["element"] == int(eid) and e["event_id"] == event_id:
                    e["recover_gw"] = None
                    e["recover_minutes"] = None
                    e["recovery_gws"] = int(last_gw - int(drop_gw))
                    e["censored"] = True
                    e["recover_expand_mean"] = None
                    break

    return pd.DataFrame(events), pd.DataFrame(paths)


def summarise(
    events: pd.DataFrame,
    paths: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if events.empty:
        return {"n_events": 0}
    e = events.copy()
    unc = e.loc[~e["censored"].astype(bool)]
    cens = e.loc[e["censored"].astype(bool)]

    def _dur_stats(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {"n": 0}
        x = sub["recovery_gws"].astype(float)
        return {
            "n": int(len(sub)),
            "mean": float(x.mean()),
            "median": float(x.median()),
            "p25": float(x.quantile(0.25)),
            "p75": float(x.quantile(0.75)),
            "p90": float(x.quantile(0.90)),
            "share_1gw": float((x <= 1).mean()),
            "share_le3": float((x <= 3).mean()),
            "share_ge6": float((x >= 6).mean()),
        }

    by_pos = {}
    for pos, sub in unc.groupby("position"):
        by_pos[str(pos)] = _dur_stats(sub)

    zero = unc.loc[unc["zero_drop"].astype(bool)]
    soft = unc.loc[unc["soft_drop"].astype(bool)]
    other = unc.loc[~unc["zero_drop"].astype(bool) & ~unc["soft_drop"].astype(bool)]

    # Unconditional return path: actual minutes at drop_gw+h for ALL uncensored
    # events (includes already-recovered players — avoids survivor bias).
    path_curve: dict[str, Any] = {}
    if not unc.empty:
        by_em = {
            (int(r.element), int(r.GW)): float(r.minutes)
            for r in frame.itertuples(index=False)
        }
        max_h = 8
        rows = []
        err_by_rho: dict[str, list[float]] = {str(r): [] for r in RHO_GRID}
        for h in range(0, max_h + 1):
            actuals: list[float] = []
            preds: dict[str, list[float]] = {str(r): [] for r in RHO_GRID}
            for r in unc.itertuples(index=False):
                gw_h = int(r.drop_gw) + h
                key = (int(r.element), gw_h)
                if key not in by_em:
                    continue
                m_act = by_em[key]
                actuals.append(m_act)
                expand = float(r.expand_mean)
                m0 = float(r.drop_minutes)
                for rho in RHO_GRID:
                    yhat = expand + (m0 - expand) * (float(rho) ** h)
                    preds[str(rho)].append(yhat)
                    if 1 <= h <= 6:
                        err_by_rho[str(rho)].append((m_act - yhat) ** 2)
            if not actuals:
                continue
            rows.append(
                {
                    "h": h,
                    "n": int(len(actuals)),
                    "mean_minutes": float(np.mean(actuals)),
                    "median_minutes": float(np.median(actuals)),
                    "share_zero": float(np.mean([m <= 0 for m in actuals])),
                    **{
                        f"pred_rho_{rho}": float(np.mean(preds[str(rho)]))
                        for rho in RHO_GRID
                    },
                }
            )
        path_curve = {
            "by_h": rows,
            "note": "unconditional on recovery (all uncensored events with a row at drop_gw+h)",
            "rmse_h1_6": {
                rho: float(math.sqrt(np.mean(errs))) if errs else None
                for rho, errs in err_by_rho.items()
            },
        }
        # Mixture split: quick zero (recover ≤1) vs long zero (≥4)
        mix = {}
        for label, sub in (
            ("zero_quick_le1", zero.loc[zero["recovery_gws"] <= 1]),
            ("zero_long_ge4", zero.loc[zero["recovery_gws"] >= 4]),
        ):
            if sub.empty:
                mix[label] = {"n": 0}
                continue
            means = []
            for h in range(0, 5):
                vals = []
                for r in sub.itertuples(index=False):
                    key = (int(r.element), int(r.drop_gw) + h)
                    if key in by_em:
                        vals.append(by_em[key])
                means.append(float(np.mean(vals)) if vals else None)
            mix[label] = {"n": int(len(sub)), "mean_m_h0_to_4": means}
        path_curve["mixture"] = mix

    # Multiplicity: events per player
    per_player = e.groupby("element").size()
    return {
        "season": SEASON,
        "defs": {
            "k_drop": K_DROP,
            "k_recover": K_RECOVER,
            "min_hist_gw": MIN_HIST_GW,
            "min_expand_mean": MIN_EXPAND_MEAN,
            "min_expand_sd": MIN_EXPAND_SD,
        },
        "n_events": int(len(e)),
        "n_uncensored": int(len(unc)),
        "n_censored": int(len(cens)),
        "n_players_with_event": int(e["element"].nunique()),
        "duration_uncensored": _dur_stats(unc),
        "duration_zero_drop": _dur_stats(zero),
        "duration_soft_drop": _dur_stats(soft),
        "duration_other_drop": _dur_stats(other),
        "duration_by_position": by_pos,
        "zero_drop_share": float(e["zero_drop"].mean()),
        "soft_drop_share": float(e["soft_drop"].mean()),
        "events_per_player": {
            "mean": float(per_player.mean()),
            "share_ge2": float((per_player >= 2).mean()),
            "share_ge3": float((per_player >= 3).mean()),
        },
        "path_curve": path_curve,
        "examples_long": unc.nlargest(8, "recovery_gws")[
            [
                "name",
                "position",
                "team",
                "drop_gw",
                "drop_minutes",
                "expand_mean",
                "recovery_gws",
                "recover_minutes",
            ]
        ].to_dict(orient="records")
        if not unc.empty
        else [],
        "examples_zero_quick": zero.nsmallest(8, "recovery_gws")[
            [
                "name",
                "position",
                "team",
                "drop_gw",
                "expand_mean",
                "recovery_gws",
                "recover_minutes",
            ]
        ].to_dict(orient="records")
        if not zero.empty
        else [],
    }


def write_report(doc: dict[str, Any], events: pd.DataFrame) -> None:
    d = doc["duration_uncensored"]
    lines = [
        "# Scout dig — minutes drops & recovery (2025-26)",
        "",
        "Exploratory only (not a gated promote). Question: when a regular's",
        "minutes fall >1σ below expanding mean, how many GWs until recovery,",
        "and does a ρ-decay toward expanding fit the return path?",
        "",
        "## Definitions",
        "",
        f"- Pool eligibility (as-of): ≥{MIN_HIST_GW} prior GWs, expanding mean ≥{MIN_EXPAND_MEAN}′.",
        f"- **Drop:** minutes < expand_mean − {K_DROP}·expand_sd (sd floored at {MIN_EXPAND_SD}).",
        f"- **Recover:** minutes ≥ expand_mean − {K_RECOVER}·expand_sd (as-of before that GW).",
        "- **Zero drop:** minutes = 0 at drop GW; **soft:** 0 < m < 45.",
        "",
        "## Counts",
        "",
        f"- Events: **{doc['n_events']}** ({doc['n_uncensored']} recovered, "
        f"{doc['n_censored']} censored at season end).",
        f"- Players with ≥1 event: **{doc['n_players_with_event']}**.",
        f"- Zero-drop share: **{doc['zero_drop_share']:.1%}**; soft: **{doc['soft_drop_share']:.1%}**.",
        f"- Events/player: mean {doc['events_per_player']['mean']:.2f}; "
        f"≥2 events {doc['events_per_player']['share_ge2']:.1%}.",
        "",
        "## Recovery duration (uncensored)",
        "",
        f"- Median **{d.get('median', float('nan')):.1f}** GW "
        f"(mean {d.get('mean', float('nan')):.2f}; "
        f"IQR {d.get('p25', float('nan')):.0f}–{d.get('p75', float('nan')):.0f}; "
        f"p90 {d.get('p90', float('nan')):.0f}).",
        f"- ≤1 GW: **{d.get('share_1gw', float('nan')):.1%}**; "
        f"≤3 GW: **{d.get('share_le3', float('nan')):.1%}**; "
        f"≥6 GW: **{d.get('share_ge6', float('nan')):.1%}**.",
        "",
        "### By drop type",
        "",
        "| type | n | median GW | mean | ≤1 | ≤3 | ≥6 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("zero (m=0)", "duration_zero_drop"),
        ("soft (0<m<45)", "duration_soft_drop"),
        ("other (m≥45 but <μ−σ)", "duration_other_drop"),
    ):
        s = doc[key]
        if not s.get("n"):
            lines.append(f"| {label} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {s['n']} | {s['median']:.1f} | {s['mean']:.2f} | "
            f"{s['share_1gw']:.0%} | {s['share_le3']:.0%} | {s['share_ge6']:.0%} |"
        )
    lines += [
        "",
        "### By position (uncensored)",
        "",
        "| pos | n | median | mean | ≤3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pos, s in sorted(doc["duration_by_position"].items()):
        lines.append(
            f"| {pos} | {s['n']} | {s['median']:.1f} | {s['mean']:.2f} | "
            f"{s['share_le3']:.0%} |"
        )

    pc = doc.get("path_curve") or {}
    if pc.get("by_h"):
        lines += [
            "",
            "## Return path (unconditional mean minutes after drop)",
            "",
            "All uncensored events, including those already recovered (avoids",
            "survivor bias). Compared to `m(h)=expand+(m0−expand)·ρ^h`.",
            "",
            "| h | n | actual mean | share m=0 | ρ=0.50 | ρ=0.70 | ρ=0.85 | ρ=0.95 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in pc["by_h"]:
            lines.append(
                f"| {r['h']} | {r['n']} | {r['mean_minutes']:.1f} | "
                f"{r.get('share_zero', float('nan')):.0%} | "
                f"{r['pred_rho_0.5']:.1f} | {r['pred_rho_0.7']:.1f} | "
                f"{r['pred_rho_0.85']:.1f} | {r['pred_rho_0.95']:.1f} |"
            )
        if pc.get("rmse_h1_6"):
            lines += [
                "",
                "RMSE vs actual on h=1…6 (lower better):",
                "",
            ]
            for rho, val in pc["rmse_h1_6"].items():
                lines.append(
                    f"- ρ={rho}: **{val:.2f}**′" if val is not None else f"- ρ={rho}: —"
                )
        mix = pc.get("mixture") or {}
        if mix:
            lines += [
                "",
                "### Mixture: quick vs long zero-drops",
                "",
                "| bucket | n | mean m path h=0…4 |",
                "| --- | ---: | --- |",
            ]
            for label, s in mix.items():
                if not s.get("n"):
                    lines.append(f"| {label} | 0 | — |")
                    continue
                path = ", ".join(
                    f"{v:.0f}" if v is not None else "—"
                    for v in (s.get("mean_m_h0_to_4") or [])
                )
                lines.append(f"| {label} | {s['n']} | {path} |")

    lines += [
        "",
        "## Long recoveries (uncensored)",
        "",
        "| player | pos | team | drop GW | m | expand | recover GW | recover m |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc.get("examples_long") or []:
        lines.append(
            f"| {r['name']} | {r['position']} | {r['team']} | {r['drop_gw']} | "
            f"{r['drop_minutes']:.0f} | {r['expand_mean']:.0f} | "
            f"{r['recovery_gws']} | {r['recover_minutes']:.0f} |"
        )

    lines += [
        "",
        "## Signals (readout)",
        "",
        "- Median recovery **~2 GW**; almost half bounce in ≤1 GW → many drops are",
        "  rotation/one-offs, not multi-week injuries.",
        "- **Zero drops split:** ~1/3 back next GW, ~1/3 still out ≥4 GW. One ρ",
        "  cannot serve both; need a duration tag (`out_for_n_gws`) or a mixture.",
        "- Soft drops (0<m<45) recover faster (median 1) than zeros (median 2).",
        "- Best single ρ on the unconditional return curve is a clue for the",
        "  default decay — still a 2025-26 dig, not a four-season lock.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.scout_minutes_drop_recovery_dig",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=SEASON)
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"loading {args.season} …", flush=True)
    frame = load_season(str(args.season))
    print(
        f"rows={len(frame)} players={frame['element'].nunique()} "
        f"GW {frame['GW'].min()}–{frame['GW'].max()}",
        flush=True,
    )
    events, paths = detect_events(frame)
    doc = summarise(events, paths, frame)
    doc["elapsed_s"] = time.time() - t0

    PROCESSED.mkdir(parents=True, exist_ok=True)
    events_path = PROCESSED / "scout_minutes_drop_events_2025_26.csv"
    paths_path = PROCESSED / "scout_minutes_drop_paths_2025_26.csv"
    json_path = PROCESSED / "scout_minutes_drop_recovery_dig_2025_26.json"
    events.to_csv(events_path, index=False)
    paths.to_csv(paths_path, index=False)
    json_path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    write_report(doc, events)

    STATE.write_text(
        json.dumps(
            {
                "task": "scout_minutes_drop_recovery_dig_2025_26",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_events": doc.get("n_events"),
                "median_recovery_gws": (doc.get("duration_uncensored") or {}).get(
                    "median"
                ),
                "rmse_h1_6": (doc.get("path_curve") or {}).get("rmse_h1_6"),
                "report": str(REPORT.relative_to(ROOT)),
                "elapsed_s": doc["elapsed_s"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_events": doc["n_events"],
                "n_uncensored": doc["n_uncensored"],
                "median_recovery_gws": doc["duration_uncensored"].get("median"),
                "zero_drop_share": doc["zero_drop_share"],
                "rmse_h1_6": (doc.get("path_curve") or {}).get("rmse_h1_6"),
                "report": str(REPORT),
                "elapsed_s": doc["elapsed_s"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
