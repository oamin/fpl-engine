"""Trial: fit scoreline M from football-data opening AH + OU (no BTTS).

Example: 2025/26 GW24 Liverpool vs Newcastle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.markets.scoreline import (
    GRID_MAX,
    fair_pair_from_decimals,
    fit_scoreline,
    team_moments,
)

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "data" / "cache" / "E0_2526.csv"
OUT = ROOT / "data" / "scratch" / "scoreline_fd_gw24_liverpool_newcastle.json"


def _matrix_table(mat: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            p = float(mat[i, j])
            if p < 1e-4:
                continue
            rows.append(
                {
                    "home_goals": i if i < GRID_MAX else GRID_MAX,
                    "away_goals": j if j < GRID_MAX else GRID_MAX,
                    "tail_home": i >= GRID_MAX,
                    "tail_away": j >= GRID_MAX,
                    "p": round(p, 5),
                }
            )
    rows.sort(key=lambda r: -float(r["p"]))
    return rows


def run(
    home: str = "Liverpool",
    away: str = "Newcastle",
    date: str = "2026-01-31",
) -> dict:
    fd = pd.read_csv(FD)
    fd["Date"] = pd.to_datetime(fd["Date"], dayfirst=True)
    row = fd[(fd["Date"] == date) & (fd["HomeTeam"] == home) & (fd["AwayTeam"] == away)]
    if row.empty:
        raise SystemExit(f"No football-data row for {home} vs {away} on {date}")
    r = row.iloc[0]

    ah_line = float(r["AHh"])
    p_ah, _ = fair_pair_from_decimals(float(r["AvgAHH"]), float(r["AvgAHA"]))
    p_over, _ = fair_pair_from_decimals(float(r["Avg>2.5"]), float(r["Avg<2.5"]))

    # football-data has no BTTS column — fit AH + OU only.
    fit = fit_scoreline(
        p_ah=p_ah,
        ah_line=ah_line,
        p_over_25=p_over,
        p_btts=0.5,
        weights=(1.0, 1.0, 0.0),
    )
    moments = team_moments(fit.matrix)
    top = _matrix_table(fit.matrix)[:12]

    payload = {
        "season": "2025-26",
        "gw": 24,
        "match": f"{home} vs {away}",
        "date": date,
        "result": {"home": int(r["FTHG"]), "away": int(r["FTAG"])},
        "source": "football-data opening Avg* (E0_2526)",
        "markets_available": ["AH", "OU2.5"],
        "markets_missing": ["BTTS"],
        "opening": {
            "ah_line": ah_line,
            "AvgAHH": float(r["AvgAHH"]),
            "AvgAHA": float(r["AvgAHA"]),
            "Avg>2.5": float(r["Avg>2.5"]),
            "Avg<2.5": float(r["Avg<2.5"]),
            "AvgH": float(r["AvgH"]),
            "AvgD": float(r["AvgD"]),
            "AvgA": float(r["AvgA"]),
        },
        "fair_targets": {"p_ah": p_ah, "p_over_25": p_over, "p_btts": None},
        "fit": fit.summary(),
        "moments": moments,
        "top_scorelines": top,
        "matrix": np.round(fit.matrix, 5).tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = run()
    m = p["moments"]
    f = p["fit"]
    print(f"{p['match']}  ({p['date']})  result {p['result']['home']}-{p['result']['away']}")
    print(f"AH line {p['opening']['ah_line']:+.2f}  fair p_AH={p['fair_targets']['p_ah']:.3f}  "
          f"p_O2.5={p['fair_targets']['p_over_25']:.3f}  (BTTS missing → weight 0)")
    print(f"λH={f['lam_h']:.3f}  λA={f['lam_a']:.3f}  ρ={f['rho']:+.3f}  "
          f"SSE(AH,OU)={f['sse']:.2e}  ok={f['success']}")
    print(f"fitted  p_AH={f['fitted']['p_ah']:.3f}  p_O2.5={f['fitted']['p_over_25']:.3f}  "
          f"p_BTTS(model)={f['fitted']['p_btts']:.3f}")
    print(f"moments xG {m['xg_home']:.2f}/{m['xg_away']:.2f}  "
          f"xCS {m['xcs_home']:.3f}/{m['xcs_away']:.3f}  "
          f"xGC {m['xgc_home']:.3f}/{m['xgc_away']:.3f}")
    print("top scorelines:")
    for row in p["top_scorelines"][:8]:
        h = f"{row['home_goals']}{'+' if row['tail_home'] else ''}"
        a = f"{row['away_goals']}{'+' if row['tail_away'] else ''}"
        print(f"  {h}-{a}: {row['p']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
