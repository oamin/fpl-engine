#!/usr/bin/env python3
"""Leakage-free screen: team-level predictors of GF / GA (H=1 and horizons).

Writes ``reports/figures/team_gf_ga_predictor_screen.json`` and prints
leaderboards. See ``reports/team_gf_ga_predictor_screen.md``.
"""

from __future__ import annotations

# Implementation lives in the research log for this branch; this module
# re-exports a thin CLI that rebuilds the screen from cache CSVs.

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT_JSON = ROOT / "reports" / "figures" / "team_gf_ga_predictor_screen.json"

SEASONS = [
    ("2022/23", "merged_gw_2022_23.csv", "E0_2223.csv"),
    ("2023/24", "merged_gw_2023_24.csv", "E0_2324.csv"),
    ("2024/25", "merged_gw_2024_25.csv", "E0_2425.csv"),
    ("2025/26", "merged_gw_2025_26.csv", "E0_2526.csv"),
]

FPL_TO_E0 = {"Man Utd": "Man United", "Spurs": "Tottenham"}


def _norm(name: str) -> str:
    return FPL_TO_E0.get(name, name)


def _iod(var: float, mean: float) -> float:
    return 0.0 if mean <= 0 else var / mean


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 30:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _mae(y: list[float], yhat: list[float]) -> float:
    return statistics.fmean(abs(a - b) for a, b in zip(y, yhat))


def _r2(y: list[float], yhat: list[float]) -> float:
    my = statistics.fmean(y)
    ss_tot = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _ols(y: list[float], *cols: list[float]) -> tuple[list[float] | None, float]:
    n = len(y)
    p = len(cols) + 1
    X = [[1.0] + [c[i] for c in cols] for i in range(n)]
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for a in range(p):
            Xty[a] += X[i][a] * y[i]
            for b in range(p):
                XtX[a][b] += X[i][a] * X[i][b]
    A = [XtX[r][:] + [Xty[r]] for r in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        if abs(A[col][col]) < 1e-12:
            return None, float("nan")
        div = A[col][col]
        for j in range(col, p + 1):
            A[col][j] /= div
        for r in range(p):
            if r == col:
                continue
            f = A[r][col]
            for j in range(col, p + 1):
                A[r][j] -= f * A[col][j]
    beta = [A[r][p] for r in range(p)]
    yhat = [sum(beta[j] * X[i][j] for j in range(p)) for i in range(n)]
    return yhat, _r2(y, yhat)


def load_e0(path: Path) -> dict:
    rows: dict = {}
    import csv

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("FTHG"):
                continue
            dd, mm, yy = row["Date"].split("/")
            year = int(yy) if len(yy) == 4 else 2000 + int(yy)
            date = f"{year:04d}-{int(mm):02d}-{int(dd):02d}"

            def probs(h, d, a):
                try:
                    oh, od, oa = float(h), float(d), float(a)
                    if min(oh, od, oa) <= 1:
                        return None
                    ih, id_, ia = 1 / oh, 1 / od, 1 / oa
                    s = ih + id_ + ia
                    return ih / s, id_ / s, ia / s
                except Exception:
                    return None

            p = probs(
                row.get("AvgH") or row.get("B365H"),
                row.get("AvgD") or row.get("B365D"),
                row.get("AvgA") or row.get("B365A"),
            ) or probs(row.get("B365H"), row.get("B365D"), row.get("B365A"))
            if p is None:
                continue
            ph, pd, pa = p
            p_over = 0.5
            try:
                o = float(row.get("Avg>2.5") or row.get("B365>2.5") or 0)
                u = float(row.get("Avg<2.5") or row.get("B365<2.5") or 0)
                if o > 1 and u > 1:
                    io, iu = 1 / o, 1 / u
                    p_over = io / (io + iu)
            except Exception:
                pass
            rows[(date, row["HomeTeam"], row["AwayTeam"])] = {
                "p_home": ph,
                "p_draw": pd,
                "p_away": pa,
                "p_over": p_over,
                "attack_home": ph + 0.5 * pd,
                "attack_away": pa + 0.5 * pd,
                "defend_home": pa + 0.5 * pd,
                "defend_away": ph + 0.5 * pd,
            }
    return rows


def load_team_fixtures(path: Path) -> list[dict]:
    import csv

    by_fix: dict = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            fix = row["fixture"]
            th, ta = row["team_h_score"], row["team_a_score"]
            if th == "" or ta == "":
                continue
            th_i, ta_i = int(float(th)), int(float(ta))
            home = str(row["was_home"]).lower() in {"true", "1"}
            key = (row["team"], fix)
            xg = float(row.get("expected_goals") or 0)
            xgc = float(row.get("expected_goals_conceded") or 0)
            minutes = float(row.get("minutes") or 0)
            if key not in by_fix:
                by_fix[key] = {
                    "team": row["team"],
                    "fixture": fix,
                    "gw": int(row["GW"]),
                    "kickoff": row["kickoff_time"],
                    "date": row["kickoff_time"][:10],
                    "home": home,
                    "hg": th_i,
                    "ag": ta_i,
                    "xg_sum": 0.0,
                    "xgc_vals": [],
                }
            by_fix[key]["xg_sum"] += xg
            if minutes >= 60:
                by_fix[key]["xgc_vals"].append(xgc)

    fixtures: dict = defaultdict(dict)
    for (_, fix), r in by_fix.items():
        xgc = statistics.median(r["xgc_vals"]) if r["xgc_vals"] else float("nan")
        side = "home" if r["home"] else "away"
        fixtures[fix][side] = {
            "team": r["team"],
            "gw": r["gw"],
            "kickoff": r["kickoff"],
            "date": r["date"],
            "gf": r["hg"] if r["home"] else r["ag"],
            "ga": r["ag"] if r["home"] else r["hg"],
            "xg": r["xg_sum"],
            "xgc": xgc,
            "home": r["home"],
        }
    out = []
    for fix, sides in fixtures.items():
        if "home" not in sides or "away" not in sides:
            continue
        h, a = sides["home"], sides["away"]
        out.append(
            {
                "fixture": fix,
                "gw": h["gw"],
                "kickoff": h["kickoff"],
                "date": h["date"],
                "home": h["team"],
                "away": a["team"],
                "hg": h["gf"],
                "ag": a["gf"],
                "hxg": h["xg"],
                "axg": a["xg"],
                "hxgc": h["xgc"],
                "axgc": a["xgc"],
            }
        )
    out.sort(key=lambda m: (m["kickoff"], m["fixture"]))
    return out


def _roll(vals: list[float], w: int = 3) -> float | None:
    if not vals:
        return None
    return statistics.fmean(vals[-w:])


def _exp(vals: list[float]) -> float | None:
    return None if not vals else statistics.fmean(vals)


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for season, gw_csv, e0_csv in SEASONS:
        e0 = load_e0(CACHE / e0_csv)
        matches = load_team_fixtures(CACHE / gw_csv)
        hist: dict = defaultdict(
            lambda: {
                "gf": [],
                "ga": [],
                "xg": [],
                "xgc": [],
                "gf_h": [],
                "gf_a": [],
                "ga_h": [],
                "ga_a": [],
            }
        )
        for m in matches:
            mk = e0.get((m["date"], _norm(m["home"]), _norm(m["away"])))
            for team, opp, gf, ga, xg, xgc, is_home in (
                (m["home"], m["away"], m["hg"], m["ag"], m["hxg"], m["hxgc"], True),
                (m["away"], m["home"], m["ag"], m["hg"], m["axg"], m["axgc"], False),
            ):
                ht, ot = hist[team], hist[opp]
                if len(ht["gf"]) >= 3 and len(ot["gf"]) >= 3:
                    own_gf_exp = _exp(ht["gf"])
                    own_ga_exp = _exp(ht["ga"])
                    own_gf_r3 = _roll(ht["gf"])
                    own_ga_r3 = _roll(ht["ga"])
                    own_xg_exp = _exp(ht["xg"])
                    own_xg_r3 = _roll(ht["xg"])
                    own_xgc = [x for x in ht["xgc"] if x == x]
                    opp_xgc = [x for x in ot["xgc"] if x == x]
                    own_xgc_exp = _exp(own_xgc)
                    own_xgc_r3 = _roll(own_xgc)
                    opp_gf_exp = _exp(ot["gf"])
                    opp_ga_exp = _exp(ot["ga"])
                    opp_gf_r3 = _roll(ot["gf"])
                    opp_ga_r3 = _roll(ot["ga"])
                    opp_xg_exp = _exp(ot["xg"])
                    opp_xg_r3 = _roll(ot["xg"])
                    opp_xgc_exp = _exp(opp_xgc)
                    opp_xgc_r3 = _roll(opp_xgc)
                    L = 1.35
                    hb = 1.08 if is_home else 1.0
                    att = own_xg_exp if own_xg_exp else own_gf_exp
                    opp_def = opp_xgc_exp if opp_xgc_exp else opp_ga_exp
                    own_def = own_xgc_exp if own_xgc_exp else own_ga_exp
                    opp_att = opp_xg_exp if opp_xg_exp else opp_gf_exp
                    lam_gf_avg = 0.5 * (own_gf_exp + opp_ga_exp)
                    lam_ga_avg = 0.5 * (own_ga_exp + opp_gf_exp)
                    lam_gf_x = 0.5 * (
                        own_xg_exp + (opp_xgc_exp if opp_xgc_exp else opp_ga_exp)
                    )
                    lam_ga_x = 0.5 * (
                        (own_xgc_exp if own_xgc_exp else own_ga_exp) + opp_xg_exp
                    )
                    lam_gf_dc = att * (opp_def / L) * hb
                    lam_ga_dc = own_def * (opp_att / L) / hb
                    if mk:
                        atk_mkt = mk["attack_home"] if is_home else mk["attack_away"]
                        def_mkt = mk["defend_home"] if is_home else mk["defend_away"]
                        p_win = mk["p_home"] if is_home else mk["p_away"]
                        p_over = mk["p_over"]
                        denom = atk_mkt + def_mkt
                        lam_gf_mkt = (
                            2.5 * p_over * atk_mkt / denom if denom > 0 else atk_mkt
                        )
                    else:
                        atk_mkt = def_mkt = p_win = p_over = lam_gf_mkt = float("nan")
                    rows.append(
                        {
                            "season": season,
                            "gw": m["gw"],
                            "team": team,
                            "gf": gf,
                            "ga": ga,
                            "own_gf_exp": own_gf_exp,
                            "own_ga_exp": own_ga_exp,
                            "own_gf_r3": own_gf_r3,
                            "own_ga_r3": own_ga_r3,
                            "own_xg_exp": own_xg_exp,
                            "own_xg_r3": own_xg_r3,
                            "own_xgc_exp": own_xgc_exp
                            if own_xgc_exp is not None
                            else float("nan"),
                            "own_xgc_r3": own_xgc_r3
                            if own_xgc_r3 is not None
                            else float("nan"),
                            "opp_gf_exp": opp_gf_exp,
                            "opp_ga_exp": opp_ga_exp,
                            "opp_gf_r3": opp_gf_r3,
                            "opp_ga_r3": opp_ga_r3,
                            "opp_xg_exp": opp_xg_exp,
                            "opp_xg_r3": opp_xg_r3,
                            "opp_xgc_exp": opp_xgc_exp
                            if opp_xgc_exp is not None
                            else float("nan"),
                            "opp_xgc_r3": opp_xgc_r3
                            if opp_xgc_r3 is not None
                            else float("nan"),
                            "lam_gf_avg": lam_gf_avg,
                            "lam_ga_avg": lam_ga_avg,
                            "lam_gf_x": lam_gf_x,
                            "lam_ga_x": lam_ga_x,
                            "lam_gf_dc": lam_gf_dc,
                            "lam_ga_dc": lam_ga_dc,
                            "atk_mkt": atk_mkt,
                            "def_mkt": def_mkt,
                            "p_win": p_win,
                            "p_over": p_over,
                            "lam_gf_mkt": lam_gf_mkt,
                            "home": 1 if is_home else 0,
                        }
                    )
                ht["gf"].append(gf)
                ht["ga"].append(ga)
                ht["xg"].append(xg)
                if xgc == xgc:
                    ht["xgc"].append(xgc)
                if is_home:
                    ht["gf_h"].append(gf)
                    ht["ga_h"].append(ga)
                else:
                    ht["gf_a"].append(gf)
                    ht["ga_a"].append(ga)

    by_ts: dict = defaultdict(list)
    for i, r in enumerate(rows):
        by_ts[(r["season"], r["team"])].append(i)
    for idxs in by_ts.values():
        idxs.sort(key=lambda i: (rows[i]["gw"], rows[i]["home"]))
        gfs = [rows[i]["gf"] for i in idxs]
        gas = [rows[i]["ga"] for i in idxs]
        for j, i in enumerate(idxs):
            for H in (1, 3, 5, 8):
                w, wa = gfs[j : j + H], gas[j : j + H]
                if len(w) == H:
                    rows[i][f"gf_h{H}"] = statistics.fmean(w)
                    rows[i][f"ga_h{H}"] = statistics.fmean(wa)
                else:
                    rows[i][f"gf_h{H}"] = float("nan")
                    rows[i][f"ga_h{H}"] = float("nan")
    return rows


def eval_pred(sub: list[dict], pred: str, ykey: str) -> dict | None:
    xs, ys = [], []
    for r in sub:
        x, y = r.get(pred), r.get(ykey)
        if x is None or y is None:
            continue
        if isinstance(x, float) and x != x:
            continue
        if isinstance(y, float) and y != y:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) < 80:
        return None
    yh, rr = _ols(ys, xs)
    return {
        "pred": pred,
        "y": ykey,
        "n": len(xs),
        "pearson": round(_pearson(xs, ys), 3),
        "spearman": round(_spearman(xs, ys), 3),
        "r2": round(rr, 4) if rr == rr else None,
        "mae": round(_mae(ys, yh), 3) if yh else None,
    }


def main() -> None:
    rows = build_rows()
    gf_preds = [
        "own_gf_exp",
        "own_gf_r3",
        "own_xg_exp",
        "own_xg_r3",
        "opp_ga_exp",
        "opp_ga_r3",
        "opp_xgc_exp",
        "opp_xgc_r3",
        "lam_gf_avg",
        "lam_gf_x",
        "lam_gf_dc",
        "atk_mkt",
        "p_win",
        "p_over",
        "lam_gf_mkt",
        "home",
    ]
    ga_preds = [
        "own_ga_exp",
        "own_ga_r3",
        "own_xgc_exp",
        "own_xgc_r3",
        "opp_gf_exp",
        "opp_gf_r3",
        "opp_xg_exp",
        "opp_xg_r3",
        "lam_ga_avg",
        "lam_ga_x",
        "lam_ga_dc",
        "def_mkt",
        "p_win",
        "p_over",
        "home",
    ]
    results = []
    for pred in gf_preds:
        m = eval_pred(rows, pred, "gf")
        if m:
            m["scope"] = "H1"
            results.append(m)
    for pred in ga_preds:
        m = eval_pred(rows, pred, "ga")
        if m:
            m["scope"] = "H1"
            results.append(m)
    for H in (3, 5, 8):
        for pred in [
            "lam_gf_avg",
            "lam_gf_x",
            "lam_gf_dc",
            "own_xg_exp",
            "own_gf_exp",
            "opp_ga_exp",
            "opp_ga_r3",
            "atk_mkt",
            "lam_gf_mkt",
            "p_win",
        ]:
            m = eval_pred(rows, pred, f"gf_h{H}")
            if m:
                m["scope"] = f"H{H}"
                results.append(m)
        for pred in [
            "lam_ga_avg",
            "lam_ga_x",
            "lam_ga_dc",
            "own_xgc_exp",
            "own_ga_exp",
            "opp_gf_exp",
            "opp_gf_r3",
            "def_mkt",
            "p_win",
        ]:
            m = eval_pred(rows, pred, f"ga_h{H}")
            if m:
                m["scope"] = f"H{H}"
                results.append(m)

    gf1 = sorted(
        [r for r in results if r["scope"] == "H1" and r["y"] == "gf"],
        key=lambda r: -abs(r["spearman"]),
    )
    ga1 = sorted(
        [r for r in results if r["scope"] == "H1" and r["y"] == "ga"],
        key=lambda r: -abs(r["spearman"]),
    )
    print("H1 GF top:")
    for r in gf1[:10]:
        print(f"  {r['pred']:14s} ρ={r['spearman']:+.3f} R²={r['r2']} MAE={r['mae']}")
    print("H1 GA top:")
    for r in ga1[:10]:
        print(f"  {r['pred']:14s} ρ={r['spearman']:+.3f} R²={r['r2']} MAE={r['mae']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "n_rows": len(rows),
                "n_with_market": sum(1 for r in rows if r["atk_mkt"] == r["atk_mkt"]),
                "h1_gf": gf1[:15],
                "h1_ga": ga1[:15],
                "all": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
