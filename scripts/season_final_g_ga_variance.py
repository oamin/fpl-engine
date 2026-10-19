#!/usr/bin/env python3
"""Final league rankings with GF / GA and match-level variance.

Closed seasons only (2022/23–2025/26) from football-data E0 sheets.
For each club: points table rank, total GF/GA, sample variance of per-match
GF and GA, and Fano dispersion (var/mean). Also ranks clubs by each metric.

Writes:
  data/processed/season_final_g_ga_variance.csv
  reports/figures/season_final_g_ga_variance.json
  reports/season_final_g_ga_variance.md
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Closed seasons only — do not include live 2026/27 holdout.
SEASONS = [
    ("2022/23", "E0_2223.csv"),
    ("2023/24", "E0_2324.csv"),
    ("2024/25", "E0_2425.csv"),
    ("2025/26", "E0_2526.csv"),
]


def iod(var: float, mean: float) -> float:
    return 0.0 if mean <= 0 else var / mean


def load_matches(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("FTHG") or not row.get("FTAG"):
                continue
            rows.append(
                {
                    "home": row["HomeTeam"],
                    "away": row["AwayTeam"],
                    "hg": int(row["FTHG"]),
                    "ag": int(row["FTAG"]),
                }
            )
    return rows


def season_table(matches: list[dict]) -> list[dict]:
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "gf_list": [],
            "ga_list": [],
            "w": 0,
            "d": 0,
            "l": 0,
            "pts": 0,
            "played": 0,
        }
    )
    for m in matches:
        for team, gf, ga, result in (
            (m["home"], m["hg"], m["ag"], "H"),
            (m["away"], m["ag"], m["hg"], "A"),
        ):
            s = stats[team]
            s["gf_list"].append(gf)
            s["ga_list"].append(ga)
            s["played"] += 1
            if gf > ga:
                s["w"] += 1
                s["pts"] += 3
            elif gf == ga:
                s["d"] += 1
                s["pts"] += 1
            else:
                s["l"] += 1

    rows = []
    for team, s in stats.items():
        gf_list = s["gf_list"]
        ga_list = s["ga_list"]
        gf = sum(gf_list)
        ga = sum(ga_list)
        mean_gf = statistics.fmean(gf_list)
        mean_ga = statistics.fmean(ga_list)
        var_gf = statistics.variance(gf_list) if len(gf_list) > 1 else 0.0
        var_ga = statistics.variance(ga_list) if len(ga_list) > 1 else 0.0
        rows.append(
            {
                "team": team,
                "played": s["played"],
                "w": s["w"],
                "d": s["d"],
                "l": s["l"],
                "pts": s["pts"],
                "gf": gf,
                "ga": ga,
                "gd": gf - ga,
                "mean_gf": round(mean_gf, 4),
                "mean_ga": round(mean_ga, 4),
                "var_gf": round(var_gf, 4),
                "var_ga": round(var_ga, 4),
                "sd_gf": round(math.sqrt(var_gf), 4),
                "sd_ga": round(math.sqrt(var_ga), 4),
                "disp_gf": round(iod(var_gf, mean_gf), 4),
                "disp_ga": round(iod(var_ga, mean_ga), 4),
            }
        )

    # League table order: pts, GD, GF
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    def add_metric_rank(key: str, rank_key: str, *, reverse: bool = True) -> None:
        ordered = sorted(rows, key=lambda r: (r[key], r["team"]), reverse=reverse)
        for i, r in enumerate(ordered, start=1):
            r[rank_key] = i

    add_metric_rank("gf", "rank_gf", reverse=True)
    add_metric_rank("ga", "rank_ga", reverse=False)  # fewest GA = rank 1
    add_metric_rank("var_gf", "rank_var_gf", reverse=True)
    add_metric_rank("var_ga", "rank_var_ga", reverse=True)
    add_metric_rank("disp_gf", "rank_disp_gf", reverse=True)
    add_metric_rank("disp_ga", "rank_disp_ga", reverse=True)
    return rows


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 5:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main() -> None:
    all_rows: list[dict] = []
    by_season: dict[str, list[dict]] = {}
    associations: list[dict] = []

    for label, fname in SEASONS:
        matches = load_matches(CACHE / fname)
        table = season_table(matches)
        for r in table:
            row = {"season": label, **r}
            all_rows.append(row)
        by_season[label] = table

        rank = [r["rank"] for r in table]
        associations.append(
            {
                "season": label,
                "n_teams": len(table),
                "n_matches": len(matches),
                "spearman_rank_vs_gf": round(
                    spearman(rank, [r["gf"] for r in table]), 4
                ),
                "spearman_rank_vs_ga": round(
                    spearman(rank, [r["ga"] for r in table]), 4
                ),
                "spearman_rank_vs_var_gf": round(
                    spearman(rank, [r["var_gf"] for r in table]), 4
                ),
                "spearman_rank_vs_var_ga": round(
                    spearman(rank, [r["var_ga"] for r in table]), 4
                ),
                "spearman_rank_vs_disp_gf": round(
                    spearman(rank, [r["disp_gf"] for r in table]), 4
                ),
                "spearman_rank_vs_disp_ga": round(
                    spearman(rank, [r["disp_ga"] for r in table]), 4
                ),
                "spearman_pts_vs_gf": round(
                    spearman([r["pts"] for r in table], [r["gf"] for r in table]), 4
                ),
                "spearman_pts_vs_ga": round(
                    spearman([r["pts"] for r in table], [r["ga"] for r in table]), 4
                ),
            }
        )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED / "season_final_g_ga_variance.csv"
    fields = list(all_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # Pooled association across seasons (rank within season already table order)
    pooled = {
        "spearman_rank_vs_gf": round(
            spearman([r["rank"] for r in all_rows], [r["gf"] for r in all_rows]), 4
        ),
        "spearman_rank_vs_ga": round(
            spearman([r["rank"] for r in all_rows], [r["ga"] for r in all_rows]), 4
        ),
        "spearman_rank_vs_var_gf": round(
            spearman([r["rank"] for r in all_rows], [r["var_gf"] for r in all_rows]), 4
        ),
        "spearman_rank_vs_var_ga": round(
            spearman([r["rank"] for r in all_rows], [r["var_ga"] for r in all_rows]), 4
        ),
        "spearman_rank_vs_disp_gf": round(
            spearman([r["rank"] for r in all_rows], [r["disp_gf"] for r in all_rows]), 4
        ),
        "spearman_rank_vs_disp_ga": round(
            spearman([r["rank"] for r in all_rows], [r["disp_ga"] for r in all_rows]), 4
        ),
    }

    payload = {
        "seasons": [s[0] for s in SEASONS],
        "n_team_seasons": len(all_rows),
        "associations_by_season": associations,
        "associations_pooled": pooled,
        "tables": {k: v for k, v in by_season.items()},
    }
    json_path = FIGURES / "season_final_g_ga_variance.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Final rankings → GF / GA / variance (closed seasons)",
        "",
        "Source: football-data E0 match sheets. Live 2026/27 holdout excluded.",
        "",
        "Per club, after 38 matches: league **rank** (Pts → GD → GF), totals **GF/GA**, "
        "sample **variance** of per-match GF and GA, and Fano **dispersion** (var/mean).",
        "",
        "`rank_ga` = 1 is fewest goals conceded. Higher `rank_var_*` / `rank_disp_*` = more variable.",
        "",
        "## Association with final table rank (Spearman ρ)",
        "",
        "| season | rank↔GF | rank↔GA | rank↔var_GF | rank↔var_GA | rank↔disp_GF | rank↔disp_GA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in associations:
        lines.append(
            f"| {a['season']} | {a['spearman_rank_vs_gf']} | {a['spearman_rank_vs_ga']} | "
            f"{a['spearman_rank_vs_var_gf']} | {a['spearman_rank_vs_var_ga']} | "
            f"{a['spearman_rank_vs_disp_gf']} | {a['spearman_rank_vs_disp_ga']} |"
        )
    lines.append(
        f"| **pooled** | {pooled['spearman_rank_vs_gf']} | {pooled['spearman_rank_vs_ga']} | "
        f"{pooled['spearman_rank_vs_var_gf']} | {pooled['spearman_rank_vs_var_ga']} | "
        f"{pooled['spearman_rank_vs_disp_gf']} | {pooled['spearman_rank_vs_disp_ga']} |"
    )
    lines += [
        "",
        "Negative ρ for GF means better (lower) table rank ↔ more goals scored. "
        "Positive ρ for GA means better rank ↔ fewer goals conceded.",
        "",
    ]

    for label, table in by_season.items():
        lines += [
            f"## {label}",
            "",
            "| Rk | Team | Pts | GF | GA | GD | var_GF | var_GA | disp_GF | disp_GA | "
            "Rk_GF | Rk_GA | Rk_varGF | Rk_varGA |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in table:
            lines.append(
                f"| {r['rank']} | {r['team']} | {r['pts']} | {r['gf']} | {r['ga']} | {r['gd']} | "
                f"{r['var_gf']:.2f} | {r['var_ga']:.2f} | {r['disp_gf']:.2f} | {r['disp_ga']:.2f} | "
                f"{r['rank_gf']} | {r['rank_ga']} | {r['rank_var_gf']} | {r['rank_var_ga']} |"
            )
        lines.append("")

    lines += [
        "## Artefacts",
        "",
        f"- `{csv_path.relative_to(ROOT)}`",
        f"- `{json_path.relative_to(ROOT)}`",
        "",
        "Reproduce: `python3 scripts/season_final_g_ga_variance.py`",
        "",
    ]
    report_path = REPORTS / "season_final_g_ga_variance.md"
    report_path.write_text("\n".join(lines))

    print(json.dumps({"associations_pooled": pooled, "seasons": associations}, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
