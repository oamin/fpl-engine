"""Gated likelihood: scout-m share recompute vs freeze.

Pre-registered: ``protocol.score_xp_v2_scout_m_share_policy``.
**Locked 2026-11-01 (PI):** freeze under scout m —
``reports/SCOUT_M_SHARE_FREEZE_LOCK.md``.

Same as-of scout ``m`` = rolling-3 for both arms. Treatment freezes ``sG``/``sA``
from expanding as-of ``m``; baseline recomputes shares with scout ``m``.
Appear/CS/GC always use scout ``m``.

  PYTHONPATH=. python3 -m src.markets.scout_m_share_policy_likelihood_gate
  PYTHONPATH=. python3 -m src.markets.scout_m_share_policy_likelihood_gate --smoke
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
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import (
    GW_EVAL_START,
    MIN_MINUTES,
    SIGMA,
    _asof_alphas,
    load_fd_moments,
    load_merged,
)
from src.markets.rate_baseline import apply_historical_rate_baseline
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison
from src.strategy.layer4_mupost_priced import (
    expanding_minutes_by_element,
    scout_rolling_minutes_by_element,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "score_xp_v2_scout_m_share_policy.md"
STATE = ROOT / "REPORT_STATE.json"

BASE = "share_recompute"
TREAT = "share_freeze"
N_BOOT = 1000
SEED = 0
SCOUT_WINDOW = 3


def _attach_rates(
    apps: pd.DataFrame, hist: pd.DataFrame, *, season: str
) -> pd.DataFrame:
    out = apps.copy()
    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    if rates.empty:
        for c in ("rG", "drG", "rA", "drA"):
            out[c] = 0.0
    else:
        out = out.merge(rates, on="element", how="left")
        for c in ("rG", "drG", "rA", "drA"):
            out[c] = out[c].fillna(0.0)
    if priors.empty:
        for c in (
            "bonus_mu",
            "bonus_se",
            "card_mu",
            "card_se",
            "saves_mu",
            "saves_se",
            "defcon_mu",
            "defcon_sig",
            "defcon_n",
        ):
            out[c] = 0.0
        out["defcon_sig"] = numagent_xp.SIG_FLOOR
    else:
        out = out.merge(
            priors.drop(columns=["position"], errors="ignore"),
            on="element",
            how="left",
        )
        for c, d in (
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
            if c not in out.columns:
                out[c] = d
            else:
                out[c] = out[c].fillna(d)
        out["defcon_n"] = out["defcon_n"].fillna(0).astype(int)
    out = apply_historical_rate_baseline(out, season=season, hist=hist)
    out["player_id"] = out["element"]
    out["sigma_m"] = 0.0
    return out


def _map_m(frame: pd.DataFrame, mmap: dict[int, float]) -> pd.DataFrame:
    out = frame.copy()
    out["m"] = (
        out["element"]
        .astype(int)
        .map(lambda e: float(mmap.get(int(e), 0.0)))
        .astype(float)
    )
    return out


def score_gameweek_arms(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    season: str,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    if week.empty:
        return pd.DataFrame()
    numagent_xp.assert_single_season_rate_history(hist)
    w = week.merge(fd, on=["team_n", "date_iso"], how="left")
    w = w[w["xg"].notna()].copy()
    if w.empty:
        return pd.DataFrame()
    apps = w[w["minutes"] > 0].copy()
    if apps.empty:
        return pd.DataFrame()

    scout_m = scout_rolling_minutes_by_element(hist, window=SCOUT_WINDOW)
    expand_m = expanding_minutes_by_element(hist)
    if not scout_m:
        return pd.DataFrame()

    base = _attach_rates(apps, hist, season=season)
    parts: list[pd.DataFrame] = []

    for (fx, team_n), g0 in base.groupby(["fixture", "team_n"], sort=False):
        g0 = g0.drop_duplicates(subset=["element"], keep="first").copy()
        xg = float(g0["xg"].iloc[0])
        xcs = float(g0["xcs"].iloc[0])
        xgc = float(g0["xgc"].iloc[0])

        g_scout = _map_m(g0, scout_m)
        g_expand = _map_m(g0, expand_m)

        # Full-club share donors: same fixture side rows with mapped m.
        # Recompute: shares from scout m among m>0 mapped players on the side.
        scored_recompute = numagent_xp.score_team_side(
            g_scout,
            xg=xg,
            xcs=xcs,
            xgc=xgc,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_players=None,
        )

        # Freeze: shares from expanding m, then appear/CS under scout m.
        scored_expand = numagent_xp.score_team_side(
            g_expand,
            xg=xg,
            xcs=xcs,
            xgc=xgc,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_players=None,
        )
        share_by = scored_expand.drop_duplicates("element").set_index("element")
        order = g_scout["element"].astype(int).to_numpy()
        sG = np.array(
            [
                float(share_by.at[e, "sG"]) if e in share_by.index else 0.0
                for e in order
            ],
            dtype=float,
        )
        sA = np.array(
            [
                float(share_by.at[e, "sA"]) if e in share_by.index else 0.0
                for e in order
            ],
            dtype=float,
        )
        # Renorm overrides to sum 1 among positive mass (score_team_side expects compositional).
        sG_sum = float(sG.sum())
        sA_sum = float(sA.sum())
        if sG_sum > 0:
            sG = sG / sG_sum
        if sA_sum > 0:
            sA = sA / sA_sum
        scored_freeze = numagent_xp.score_team_side(
            g_scout,
            xg=xg,
            xcs=xcs,
            xgc=xgc,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_players=None,
            sG_override=sG,
            sA_override=sA,
        )

        thin_ids = set(
            g0.loc[
                pd.to_numeric(g0["minutes"], errors="coerce").fillna(0.0) > MIN_MINUTES,
                "element",
            ]
            .astype(int)
            .tolist()
        )
        if not thin_ids:
            continue
        sr = scored_recompute.loc[
            scored_recompute["element"].astype(int).isin(thin_ids)
        ].copy()
        sf = scored_freeze.loc[
            scored_freeze["element"].astype(int).isin(thin_ids)
        ].copy()
        if sr.empty or sf.empty:
            continue
        m = sr[
            ["element", "total_points", "minutes", "position", "xp", "xp_se"]
        ].rename(columns={"xp": "xp_recompute", "xp_se": "xp_se_recompute"})
        m = m.merge(
            sf[["element", "xp", "xp_se"]].rename(
                columns={"xp": "xp_freeze", "xp_se": "xp_se_freeze"}
            ),
            on="element",
            how="inner",
        )
        m["fixture"] = int(fx)
        m["team_n"] = str(team_n)
        m["gw"] = int(week["GW"].iloc[0])
        parts.append(m)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.loc[
        pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0) > MIN_MINUTES
    ].copy()
    return out


def run_season(season: str, *, gw_end: int | None = None) -> pd.DataFrame:
    stem = season.replace("-", "_")
    merged = load_merged(stem)
    fd = load_fd_moments(stem)
    parts: list[pd.DataFrame] = []
    gws = sorted(int(g) for g in merged["GW"].unique() if g >= 2)
    for gw in gws:
        if gw < GW_EVAL_START:
            continue
        if gw_end is not None and gw > int(gw_end):
            continue
        hist = merged[merged["GW"] < gw]
        week = merged[merged["GW"] == gw]
        ag, aa = _asof_alphas(hist)
        pair = score_gameweek_arms(
            hist, week, fd, season=season, alpha_g=ag, alpha_a=aa
        )
        if not pair.empty:
            parts.append(pair)
    if not parts:
        raise RuntimeError(f"no rows for {season}")
    rows = pd.concat(parts, ignore_index=True)
    rows["element"] = rows["element"].astype(int)
    rows["season"] = season
    y = rows["total_points"].to_numpy(float)
    rows["log_recompute"] = gaussian_log_score(
        y, rows["xp_recompute"].to_numpy(float), SIGMA
    )
    rows["log_freeze"] = gaussian_log_score(
        y, rows["xp_freeze"].to_numpy(float), SIGMA
    )
    rows["d_log"] = rows["log_freeze"] - rows["log_recompute"]
    return rows


def summarise(rows: pd.DataFrame) -> dict[str, Any]:
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        sub = rows.loc[rows["season"] == season]
        if sub.empty:
            continue
        # Pair by GW: mean d_log within GW, then cluster across GWs.
        by_gw = sub.groupby("gw", sort=True)["d_log"].mean()
        arr = by_gw.to_numpy(float)
        if arr.size:
            deltas[season] = arr
    if set(deltas) != set(CLOSED_SEASONS):
        raise RuntimeError(f"missing seasons in deltas: {sorted(deltas)}")
    for season, arr in deltas.items():
        if arr.size < 20:
            raise RuntimeError(f"{season}: fewer than 20 GWs ({arr.size})")
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
    key = comparison_key(TREAT, BASE)
    return {
        "comparisons": {
            key: {
                **{k: interval[k] for k in ("mean", "lo", "hi", "n_gws")},
                "covers_zero": interval["covers_zero"],
                "by_season_mean": interval["by_season_mean"],
                "verdict": verdict,
                "treatment": TREAT,
                "baseline": BASE,
            }
        },
        "n_rows": int(len(rows)),
        "mean_xp_recompute": float(rows["xp_recompute"].mean()),
        "mean_xp_freeze": float(rows["xp_freeze"].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-end", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("score_xp_v2_scout_m_share_policy") or {}
    if not screen:
        raise RuntimeError("protocol.score_xp_v2_scout_m_share_policy missing")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    gw_end = int(args.gw_end) if args.gw_end else None
    if args.smoke:
        seasons = ["2025-26"]
        gw_end = 12
        print("[scout-m-share] SMOKE 2025-26 GW≤12", flush=True)

    print("[scout-m-share] as-of audit…", flush=True)
    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}

    t0 = time.time()
    parts = []
    for season in seasons:
        print(f"scoring {season} …", flush=True)
        parts.append(run_season(season, gw_end=gw_end))
    rows = pd.concat(parts, ignore_index=True)
    elapsed = time.time() - t0

    if args.smoke:
        print(rows.head(20).to_string(index=False), flush=True)
        print(
            json.dumps(
                {
                    "smoke": True,
                    "n_rows": len(rows),
                    "mean_d_log": float(rows["d_log"].mean()),
                    "elapsed_s": elapsed,
                },
                indent=2,
            )
        )
        STATE.write_text(
            json.dumps(
                {
                    "task": "score_xp_v2_scout_m_share_policy",
                    "status": "smoke_ok",
                    "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mean_d_log": float(rows["d_log"].mean()),
                    "elapsed_s": elapsed,
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    if set(seasons) != set(CLOSED_SEASONS):
        raise RuntimeError(f"full gate needs {CLOSED_SEASONS}; got {seasons}")

    doc = summarise(rows)
    key = comparison_key(TREAT, BASE)
    iv = doc["comparisons"][key]
    comparisons = {**_certified_protocol_comparison(), **doc["comparisons"]}
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "score_xp_v2_scout_m_share_policy",
    }

    lines = [
        "# Scout-m share policy — freeze vs recompute (likelihood)",
        "",
        "Pre-registered: `protocol.score_xp_v2_scout_m_share_policy`.",
        "",
        "Same as-of scout `m` = rolling-3. Freeze = shares from expanding `m`;",
        "recompute = shares from scout `m`. Appear/CS/GC use scout `m` both arms.",
        "",
        f"- Pool: minutes>{int(MIN_MINUTES)}, FD AH+OU, GW≥{GW_EVAL_START}, σ={SIGMA}.",
        f"- Rows: {doc['n_rows']}. Wall {elapsed:.0f}s.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Results",
        "",
        f"### `{TREAT}` − `{BASE}`",
        "",
        f"- Mean Δ log-score/GW: **{iv['mean']:+.6f}** [{iv['lo']:+.6f}, {iv['hi']:+.6f}]",
        f"- Covers 0: **{iv['covers_zero']}**",
        f"- Verdict: **{iv['verdict']}**",
        f"- Mean xp recompute/freeze: {doc['mean_xp_recompute']:.3f} / {doc['mean_xp_freeze']:.3f}",
        "",
        "| season | mean Δ log/GW |",
        "| --- | ---: |",
    ]
    for s in CLOSED_SEASONS:
        lines.append(f"| {s} | {iv['by_season_mean'][s]:+.6f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Positive = freeze shares beat recompute under scout rolling-3 m.",
        "- Climb follows only if this likelihood does not clearly lose.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.markets.scout_m_share_policy_likelihood_gate",
        "```",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    # Update protocol result + climb gate flag
    protocol = load_protocol()
    screen = protocol["score_xp_v2_scout_m_share_policy"]
    screen["status"] = (
        "promoted"
        if iv["verdict"] == "promote"
        else ("parked" if iv["verdict"] == "park_clear_lose" else "reported")
    )
    screen["result"] = {
        key: {
            "mean": iv["mean"],
            "lo": iv["lo"],
            "hi": iv["hi"],
            "covers_zero": iv["covers_zero"],
            "verdict": iv["verdict"],
            "by_season_mean": iv["by_season_mean"],
        }
    }
    climb = protocol.get("l4_opta_joint_path_scout_m_share_policy") or {}
    if iv["verdict"] == "park_clear_lose":
        climb["status"] = "skipped_likelihood_clear_lose"
        climb["note"] = "Likelihood clear lose — do not run path climb."
    else:
        climb["status"] = "pre_registered_run"
        climb["note"] = "Likelihood did not clearly lose — climb authorised."
    protocol["l4_opta_joint_path_scout_m_share_policy"] = climb
    protocol["score_xp_v2_scout_m_share_policy"] = screen
    (ROOT / "experiments" / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    STATE.write_text(
        json.dumps(
            {
                "task": "score_xp_v2_scout_m_share_policy",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "verdict": iv["verdict"],
                "mean": iv["mean"],
                "lo": iv["lo"],
                "hi": iv["hi"],
                "climb_authorised": iv["verdict"] != "park_clear_lose",
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
                "verdict": iv["verdict"],
                "mean": iv["mean"],
                "lo": iv["lo"],
                "hi": iv["hi"],
                "climb_authorised": iv["verdict"] != "park_clear_lose",
                "report": str(REPORT),
                "elapsed_s": elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
