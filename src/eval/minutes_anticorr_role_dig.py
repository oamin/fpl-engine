"""Within-season anti-correlated teammate roles → blurb → minutes rating.

**Scope:** one season only (test on 2025-26). Live must not mix prior seasons.
All features are as-of: at decision GW ``t`` use only rows with ``GW < t``.

Pipeline
--------
1. Within each club, pairwise Pearson corr of as-of minutes series.
2. Strongest anti-correlated teammate per player (ρ ≤ −threshold).
3. Role from start rate + median minutes | start (tactical starter ~60, etc.).
4. Contextual blurb + generalised minutes rating.
5. Score next-GW minutes vs expanding-mean baseline; score replacement hits.

  PYTHONPATH=. python3 -m src.eval.minutes_anticorr_role_dig
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
STATE = ROOT / "REPORT_STATE.json"

SEASON_DEFAULT = "2025-26"
HIST_DEFAULT = CACHE / "merged_gw_2025_26.csv"
MIN_PRIOR_GWS_DEFAULT = 8
MIN_PAIR_GWS = 6
MIN_EXPAND = 25.0
ANTICORR_MAX = -0.35
TACTICAL_MED_LO = 50.0
TACTICAL_MED_HI = 75.0
TACTICAL_START_RATE = 0.35

# Runtime knobs set in main() for live vs closed-season trials.
MIN_PRIOR_GWS = MIN_PRIOR_GWS_DEFAULT


@dataclass
class RoleSnapshot:
    element: int
    name: str
    team: str
    position: str
    as_of_gw: int
    expand_m: float
    start_rate: float
    n_starts: int
    med_m_start: float | None
    role: str
    rating_m: float
    anti_element: int | None
    anti_name: str | None
    anti_corr: float | None
    blurb: str


def _load_hist(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path)
    # Live cache uses player_id / player_name / gw.
    if "element" not in h.columns and "player_id" in h.columns:
        h["element"] = h["player_id"]
    if "name" not in h.columns and "player_name" in h.columns:
        h["name"] = h["player_name"]
    if "GW" not in h.columns and "gw" in h.columns:
        h["GW"] = h["gw"]
    if "position" not in h.columns:
        h["position"] = "MID"
    h["GW"] = pd.to_numeric(h["GW"], errors="coerce")
    h["element"] = pd.to_numeric(h["element"], errors="coerce")
    h["minutes"] = pd.to_numeric(h["minutes"], errors="coerce").fillna(0.0)
    if "starts" not in h.columns:
        h["starts"] = (h["minutes"] >= 60.0).astype(int)
    else:
        h["starts"] = pd.to_numeric(h["starts"], errors="coerce").fillna(0).astype(int)
    h = h.dropna(subset=["GW", "element", "team"])
    # One row per element-GW (double GWs: sum minutes / max start)
    g = (
        h.groupby(["element", "GW", "team", "name", "position"], as_index=False)
        .agg(minutes=("minutes", "sum"), starts=("starts", "max"))
    )
    return g


def _series_maps(
    hist: pd.DataFrame, *, before_gw: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    """GW-aligned minutes/starts arrays for GW < before_gw, plus meta."""
    sub = hist.loc[hist["GW"] < int(before_gw)].copy()
    meta: dict[int, dict[str, Any]] = {}
    for eid, g in sub.groupby("element", sort=False):
        last = g.sort_values("GW").iloc[-1]
        meta[int(eid)] = {
            "name": str(last["name"]),
            "team": str(last["team"]),
            "position": str(last["position"]),
        }
    # Club → elements present
    gws = sorted(int(x) for x in sub["GW"].unique())
    gw_index = {g: i for i, g in enumerate(gws)}
    n = len(gws)
    minutes: dict[int, np.ndarray] = {}
    starts: dict[int, np.ndarray] = {}
    for eid, g in sub.groupby("element", sort=False):
        m = np.zeros(n, dtype=float)
        s = np.zeros(n, dtype=float)
        for row in g.itertuples(index=False):
            i = gw_index.get(int(row.GW))
            if i is None:
                continue
            m[i] = float(row.minutes)
            s[i] = float(row.starts)
        minutes[int(eid)] = m
        starts[int(eid)] = s
    return minutes, starts, meta


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    need = max(4, min(int(MIN_PAIR_GWS), int(MIN_PRIOR_GWS)))
    if len(a) < need:
        return None
    if float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(r):
        return None
    return r


def _role_and_rating(
    *,
    expand_m: float,
    start_rate: float,
    n_starts: int,
    med_m_start: float | None,
    anti_corr: float | None,
) -> tuple[str, float]:
    """Map as-of shape → role label + generalised minutes rating."""
    if n_starts >= 3 and med_m_start is not None and med_m_start >= 80.0 and start_rate >= 0.5:
        return "nailed_starter", float(min(90.0, max(expand_m, med_m_start * 0.95)))
    if (
        n_starts >= 3
        and med_m_start is not None
        and TACTICAL_MED_LO <= med_m_start <= TACTICAL_MED_HI
        and start_rate >= TACTICAL_START_RATE
    ):
        # Mixture: P(start)·median|start + P(bench)·small bench residual.
        # Do NOT shrink expand toward 60 unconditionally — that overshoots on
        # non-start GWs (injury / full rotation).
        bench = 8.0
        rating = float(start_rate) * float(med_m_start) + (1.0 - float(start_rate)) * bench
        return "tactical_starter", float(rating)
    if expand_m < 20.0 and start_rate < 0.15:
        return "fringe", float(expand_m)
    if start_rate < 0.25 and expand_m >= 15.0:
        return "impact_sub", float(min(35.0, max(expand_m, 20.0)))
    if anti_corr is not None and anti_corr <= ANTICORR_MAX and expand_m >= MIN_EXPAND:
        return "rotate_pair", float(expand_m)
    return "expanding_default", float(expand_m)


def _blurb(snap: RoleSnapshot) -> str:
    bits: list[str] = []
    if snap.role == "tactical_starter" and snap.med_m_start is not None:
        bits.append(
            f"Often starts ({snap.start_rate:.0%} as-of) and plays ~{snap.med_m_start:.0f}' "
            f"— tactical starter, not a full-90 nailed profile"
        )
    elif snap.role == "nailed_starter" and snap.med_m_start is not None:
        bits.append(
            f"Regular starter ({snap.start_rate:.0%}); median ~{snap.med_m_start:.0f}' when starting"
        )
    elif snap.role == "impact_sub":
        bits.append(
            f"Low start rate ({snap.start_rate:.0%}); typically an impact / rotation option "
            f"(expand ~{snap.expand_m:.0f}')"
        )
    elif snap.role == "rotate_pair":
        bits.append(
            f"Minutes compete with a teammate (expand ~{snap.expand_m:.0f}')"
        )
    else:
        bits.append(f"Expanding mean ~{snap.expand_m:.0f}' (no strong role tag)")

    if snap.anti_name and snap.anti_corr is not None:
        bits.append(
            f"anti-correlated with {snap.anti_name} (ρ={snap.anti_corr:.2f}) — "
            f"when one sits, the other tends to take minutes"
        )
    return "; ".join(bits) + "."


def snapshots_for_gw(hist: pd.DataFrame, as_of_gw: int) -> list[RoleSnapshot]:
    minutes, starts, meta = _series_maps(hist, before_gw=as_of_gw)
    # Team → element ids
    by_team: dict[str, list[int]] = {}
    for eid, info in meta.items():
        by_team.setdefault(info["team"], []).append(eid)

    out: list[RoleSnapshot] = []
    for team, eids in by_team.items():
        # Precompute pairwise anti-corr within club
        best_anti: dict[int, tuple[float, int]] = {}
        for i, a in enumerate(eids):
            ma = minutes.get(a)
            if ma is None or len(ma) < MIN_PRIOR_GWS:
                continue
            pos_a = str(meta[a]["position"])
            if pos_a == "GK":
                continue
            for b in eids[i + 1 :]:
                mb = minutes.get(b)
                if mb is None or len(mb) < MIN_PRIOR_GWS:
                    continue
                pos_b = str(meta[b]["position"])
                # Same broad role only — avoid GK↔FWD artefacts.
                if pos_b == "GK" or pos_a != pos_b:
                    continue
                r = _pearson(ma, mb)
                if r is None or r > ANTICORR_MAX:
                    continue
                for src, dst in ((a, b), (b, a)):
                    prev = best_anti.get(src)
                    if prev is None or r < prev[0]:
                        best_anti[src] = (r, dst)

        for eid in eids:
            m = minutes.get(eid)
            s = starts.get(eid)
            if m is None or s is None or len(m) < MIN_PRIOR_GWS:
                continue
            expand = float(m.mean())
            if expand < MIN_EXPAND and float(s.sum()) < 3:
                continue
            n_starts = int(s.sum())
            start_rate = float(n_starts / len(m)) if len(m) else 0.0
            if n_starts > 0:
                med_m_start = float(np.median(m[s > 0.5]))
            else:
                med_m_start = None
            anti = best_anti.get(eid)
            anti_corr = anti[0] if anti else None
            anti_eid = anti[1] if anti else None
            anti_name = meta[anti_eid]["name"] if anti_eid is not None else None
            role, rating = _role_and_rating(
                expand_m=expand,
                start_rate=start_rate,
                n_starts=n_starts,
                med_m_start=med_m_start,
                anti_corr=anti_corr,
            )
            info = meta[eid]
            snap = RoleSnapshot(
                element=int(eid),
                name=str(info["name"]),
                team=str(info["team"]),
                position=str(info["position"]),
                as_of_gw=int(as_of_gw),
                expand_m=expand,
                start_rate=start_rate,
                n_starts=n_starts,
                med_m_start=med_m_start,
                role=role,
                rating_m=rating,
                anti_element=anti_eid,
                anti_name=anti_name,
                anti_corr=anti_corr,
                blurb="",
            )
            snap.blurb = _blurb(snap)
            out.append(snap)
    return out


def score_season(hist: pd.DataFrame, *, gw_from: int, gw_to: int) -> dict[str, Any]:
    """Walk as-of GWs; score next-GW minutes + anti-corr replacement hits."""
    actual = {
        (int(r.element), int(r.GW)): float(r.minutes)
        for r in hist.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    repl_hits = 0
    repl_n = 0
    example_blurbs: list[dict[str, Any]] = []

    for t in range(int(gw_from), int(gw_to) + 1):
        snaps = snapshots_for_gw(hist, t)
        # keep a few tactical_starter blurbs mid-season
        if t in {12, 20, 28}:
            for s in snaps:
                if s.role == "tactical_starter" and s.anti_name:
                    example_blurbs.append(asdict(s))
                    if sum(1 for e in example_blurbs if e["as_of_gw"] == t) >= 5:
                        break
        by_id = {s.element: s for s in snaps}
        for s in snaps:
            y = actual.get((s.element, t))
            if y is None:
                continue
            rows.append(
                {
                    "gw": t,
                    "element": s.element,
                    "role": s.role,
                    "y": y,
                    "expand": s.expand_m,
                    "rating": s.rating_m,
                    "err_expand": abs(y - s.expand_m),
                    "err_rating": abs(y - s.rating_m),
                }
            )
            # Replacement: if A plays ≤30 next GW and has anti-partner, does B play ≥45?
            if (
                s.anti_element is not None
                and s.anti_corr is not None
                and s.anti_corr <= ANTICORR_MAX
            ):
                ya = y
                yb = actual.get((int(s.anti_element), t))
                if yb is None:
                    continue
                if ya <= 30.0:
                    repl_n += 1
                    if yb >= 45.0:
                        repl_hits += 1

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n": 0}

    def _mae(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if len(df) else float("nan")

    by_role: dict[str, Any] = {}
    for role, g in frame.groupby("role"):
        by_role[str(role)] = {
            "n": int(len(g)),
            "mae_expand": _mae(g, "err_expand"),
            "mae_rating": _mae(g, "err_rating"),
            "delta_mae": _mae(g, "err_expand") - _mae(g, "err_rating"),
        }

    tactical = frame.loc[frame["role"] == "tactical_starter"]
    return {
        "n_pred_rows": int(len(frame)),
        "gw_from": int(gw_from),
        "gw_to": int(gw_to),
        "mae_expand_all": _mae(frame, "err_expand"),
        "mae_rating_all": _mae(frame, "err_rating"),
        "delta_mae_all": _mae(frame, "err_expand") - _mae(frame, "err_rating"),
        "by_role": by_role,
        "tactical_starter": {
            "n": int(len(tactical)),
            "mae_expand": _mae(tactical, "err_expand"),
            "mae_rating": _mae(tactical, "err_rating"),
            "delta_mae": _mae(tactical, "err_expand") - _mae(tactical, "err_rating"),
            "mean_y": float(tactical["y"].mean()) if len(tactical) else None,
            "mean_rating": float(tactical["rating"].mean()) if len(tactical) else None,
            "mean_expand": float(tactical["expand"].mean()) if len(tactical) else None,
        },
        "replacement": {
            "n_events_a_le_30": int(repl_n),
            "n_partner_ge_45": int(repl_hits),
            "hit_rate": (repl_hits / repl_n) if repl_n else None,
            "rule": "when A ≤30' next GW, anti-corr B ≥45'",
        },
        "example_blurbs": example_blurbs[:20],
    }


def write_report(
    doc: dict[str, Any],
    mid_snaps: list[RoleSnapshot],
    *,
    season: str,
    report_path: Path,
    reproduce: str,
) -> Path:
    ts = doc.get("tactical_starter") or {}
    rep = doc.get("replacement") or {}
    example_gw = doc.get("example_as_of_gw")
    lines = [
        f"# Within-season anti-corr roles → blurb → minutes rating ({season})",
        "",
        "As-of only (`GW < t`). **No cross-season history.** Live seasons must "
        "fit roles from the current season's completed GWs only.",
        "",
        "## Method",
        "",
        f"- Prior window: expanding from GW1 through `t-1` (min {MIN_PRIOR_GWS} GWs).",
        f"- Anti-corr: within-club Pearson on minutes, keep ρ ≤ {ANTICORR_MAX}.",
        "- Roles: `nailed_starter`, `tactical_starter` (starts often, median | start "
        f"in [{TACTICAL_MED_LO:.0f},{TACTICAL_MED_HI:.0f}]), `impact_sub`, "
        "`rotate_pair`, else expanding default.",
        "- Rating: tactical starters use "
        "`P(start)·median|start + P(bench)·8′`; nailed uses max(expand, 0.95·med|start); "
        "others keep expanding mean.",
        "- Anti-corr pairs restricted to the **same position** (no GK↔outfield).",
        "- Within-season only (no prior-season bleed into live).",
        "",
        "## Next-GW minutes accuracy",
        "",
        f"- GWs scored: {doc.get('gw_from')}–{doc.get('gw_to')} "
        f"({doc.get('n_pred_rows')} player-GW rows).",
        f"- MAE expand (all): **{doc.get('mae_expand_all'):.2f}′**",
        f"- MAE role rating (all): **{doc.get('mae_rating_all'):.2f}′**",
        f"- Δ (expand − rating): **{doc.get('delta_mae_all'):.2f}′** "
        f"(positive ⇒ rating better).",
        "",
        "### Tactical starters only",
        "",
    ]
    if ts.get("n"):
        lines += [
            f"- n={ts.get('n')}; mean actual={ts.get('mean_y')}; "
            f"mean expand={ts.get('mean_expand')}; mean rating={ts.get('mean_rating')}",
            f"- MAE expand **{ts.get('mae_expand'):.2f}′** vs rating "
            f"**{ts.get('mae_rating'):.2f}′** (Δ={ts.get('delta_mae'):.2f}′).",
        ]
    else:
        lines.append("- _No tactical-starter rows in this window._")
    lines += [
        "",
        "### By role",
        "",
        "| role | n | MAE expand | MAE rating | Δ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for role, b in sorted((doc.get("by_role") or {}).items()):
        lines.append(
            f"| `{role}` | {b['n']} | {b['mae_expand']:.2f} | "
            f"{b['mae_rating']:.2f} | {b['delta_mae']:.2f} |"
        )
    hit = rep.get("hit_rate")
    hit_s = f"**{100.0 * hit:.1f}%**" if hit is not None else "_n/a_"
    lines += [
        "",
        "## Anti-corr replacement check",
        "",
        f"- Rule: {rep.get('rule')}",
        f"- Events: {rep.get('n_events_a_le_30')}; partner ≥45′: "
        f"{rep.get('n_partner_ge_45')}; hit rate {hit_s}",
        "",
        f"## Example blurbs (as-of GW{example_gw})",
        "",
    ]
    mid = [s for s in mid_snaps if s.role == "tactical_starter" and s.anti_name]
    if len(mid) < 6:
        mid = [s for s in mid_snaps if s.anti_name]
    mid = sorted(
        mid,
        key=lambda s: (
            s.anti_corr is None,
            s.anti_corr if s.anti_corr is not None else 0.0,
            s.name,
        ),
    )[:12]
    for s in mid:
        lines.append(f"- **{s.name}** ({s.team}): {s.blurb}")
        lines.append(
            f"  - rating={s.rating_m:.1f}′ expand={s.expand_m:.1f}′ role=`{s.role}`"
        )
    if not mid:
        lines.append("_No anti-corr partners at this as-of GW (short history)._")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        reproduce,
        "```",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    global MIN_PRIOR_GWS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season-csv", type=Path, default=HIST_DEFAULT)
    ap.add_argument("--season", default=SEASON_DEFAULT)
    ap.add_argument("--gw-from", type=int, default=10)
    ap.add_argument("--gw-to", type=int, default=37)
    ap.add_argument("--example-as-of", type=int, default=20)
    ap.add_argument("--min-prior-gws", type=int, default=MIN_PRIOR_GWS_DEFAULT)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    MIN_PRIOR_GWS = int(args.min_prior_gws)
    season = str(args.season)
    slug = season.replace("/", "-").replace(" ", "_")
    file_slug = slug.replace("-", "_")

    t0 = time.time()
    hist = _load_hist(Path(args.season_csv))
    print(
        f"loaded {season}: rows={len(hist)} players={hist['element'].nunique()} "
        f"GW {hist['GW'].min()}–{hist['GW'].max()} min_prior={MIN_PRIOR_GWS}",
        flush=True,
    )
    doc = score_season(hist, gw_from=int(args.gw_from), gw_to=int(args.gw_to))
    mid_snaps = snapshots_for_gw(hist, int(args.example_as_of))
    doc["example_as_of_gw"] = int(args.example_as_of)
    doc["elapsed_s"] = time.time() - t0
    doc["season"] = season
    doc["within_season_only"] = True
    doc["min_prior_gws"] = MIN_PRIOR_GWS

    PROCESSED.mkdir(parents=True, exist_ok=True)
    snaps_path = (
        PROCESSED
        / f"minutes_anticorr_roles_asof_gw{int(args.example_as_of):02d}_{file_slug}.csv"
    )
    pd.DataFrame([asdict(s) for s in mid_snaps]).to_csv(snaps_path, index=False)
    json_path = PROCESSED / f"minutes_anticorr_role_dig_{file_slug}.json"
    json_path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")

    report_path = Path(args.report) if args.report else (
        REPORTS / f"minutes_anticorr_role_dig_{file_slug}.md"
    )
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    reproduce = (
        f"PYTHONPATH=. python3 -m src.eval.minutes_anticorr_role_dig \\\n"
        f"  --season {season} --season-csv {args.season_csv} \\\n"
        f"  --gw-from {args.gw_from} --gw-to {args.gw_to} \\\n"
        f"  --example-as-of {args.example_as_of} --min-prior-gws {MIN_PRIOR_GWS}"
    )
    report = write_report(
        doc, mid_snaps, season=season, report_path=report_path, reproduce=reproduce
    )

    STATE.write_text(
        json.dumps(
            {
                "task": f"minutes_anticorr_role_dig_{file_slug}",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "season": season,
                "report": str(report.relative_to(ROOT)),
                "delta_mae_all": doc.get("delta_mae_all"),
                "tactical_delta_mae": (doc.get("tactical_starter") or {}).get(
                    "delta_mae"
                ),
                "replacement_hit_rate": (doc.get("replacement") or {}).get("hit_rate"),
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
                "season": season,
                "mae_expand_all": doc.get("mae_expand_all"),
                "mae_rating_all": doc.get("mae_rating_all"),
                "delta_mae_all": doc.get("delta_mae_all"),
                "tactical": doc.get("tactical_starter"),
                "replacement": doc.get("replacement"),
                "report": str(report),
                "snaps": str(snaps_path),
                "elapsed_s": doc["elapsed_s"],
            },
            indent=2,
        )
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
