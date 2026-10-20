"""Club minutes covariance — fixture-level panel (DGW = two observations).

Trial on 2025/26: for each club, player×player correlation of minutes across
fixtures. No position filter (tactical subs allowed).

DGW handling: each ``fixture`` row is one observation. A double gameweek
with two match rows counts as two samples, not one aggregated GW.

Note: Pearson corr is invariant to column demeaning — demeaning does not
surface new dependence. Instead we report:
  - raw minutes corr
  - start-indicator corr I(minutes≥60)
  - strata by mean minutes (nailed–nailed vs rotation band)

  python -m src.markets.minutes_covariance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEASON_DEFAULT = "2025-26"
MERGED = CACHE / "merged_gw_2025_26.csv"
MIN_FIXTURES_WITH_MINUTES = 5
MIN_PAIR_N = 8
START_MINUTES = 60.0
NAILED_MEAN = 60.0
ROTATION_MEAN_LO = 15.0
ROTATION_MEAN_HI = 75.0


def merged_path(season: str) -> Path:
    return CACHE / f"merged_gw_{season.replace('-', '_')}.csv"


def load_merged(path: Path | str | None = None, *, season: str | None = None) -> pd.DataFrame:
    if path is None:
        path = merged_path(season or SEASON_DEFAULT)
    path = Path(path)
    df = pd.read_csv(path)
    need = {"element", "name", "team", "position", "fixture", "minutes", "GW"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"merged_gw missing columns: {sorted(missing)}")
    from src.rules.fpl_2026 import drop_assistant_managers

    df = drop_assistant_managers(df)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["element"] = pd.to_numeric(df["element"], errors="coerce").astype(int)
    df["fixture"] = pd.to_numeric(df["fixture"], errors="coerce")
    df = df.dropna(subset=["fixture", "team"])
    df = df.drop_duplicates(["team", "fixture", "element"], keep="first")
    return df


def club_minutes_panel(
    club: pd.DataFrame,
    *,
    min_fixtures_with_minutes: int = MIN_FIXTURES_WITH_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    played = (
        club.loc[club["minutes"] > 0]
        .groupby("element")
        .size()
        .rename("n_with_minutes")
    )
    keep = played[played >= min_fixtures_with_minutes].index
    sub = club.loc[club["element"].isin(keep)].copy()
    panel = sub.pivot_table(
        index="fixture", columns="element", values="minutes", aggfunc="sum"
    )
    panel = panel.reindex(sorted(club["fixture"].unique())).fillna(0.0)
    meta = (
        sub.drop_duplicates("element")[["element", "name", "position"]]
        .set_index("element")
        .join(played)
    )
    meta["mean_minutes"] = panel.mean(axis=0)
    meta["n_fixtures"] = int(len(panel))
    return panel, meta


def _pair_table(
    corr: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    kind: str,
    n_fixtures: int,
) -> pd.DataFrame:
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = corr.loc[a, b]
            if not np.isfinite(c):
                continue
            ma = float(meta.loc[a, "mean_minutes"]) if a in meta.index else float("nan")
            mb = float(meta.loc[b, "mean_minutes"]) if b in meta.index else float("nan")
            rows.append(
                {
                    "kind": kind,
                    "element_a": int(a),
                    "element_b": int(b),
                    "name_a": meta.loc[a, "name"] if a in meta.index else str(a),
                    "name_b": meta.loc[b, "name"] if b in meta.index else str(b),
                    "pos_a": meta.loc[a, "position"] if a in meta.index else "",
                    "pos_b": meta.loc[b, "position"] if b in meta.index else "",
                    "corr": float(c),
                    "mean_m_a": ma,
                    "mean_m_b": mb,
                    "n_fixtures": n_fixtures,
                }
            )
    return pd.DataFrame(rows)


def _stratum(ma: float, mb: float) -> str:
    nailed = ma >= NAILED_MEAN and mb >= NAILED_MEAN
    rotation = (
        ROTATION_MEAN_LO <= ma <= ROTATION_MEAN_HI
        and ROTATION_MEAN_LO <= mb <= ROTATION_MEAN_HI
    )
    if nailed:
        return "both_nailed"
    if rotation:
        return "both_rotation_band"
    return "other"


def _summ(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {"n_pairs": 0}
    c = pairs["corr"].to_numpy(float)
    return {
        "n_pairs": int(len(pairs)),
        "mean_corr": float(np.mean(c)),
        "median_corr": float(np.median(c)),
        "frac_neg": float(np.mean(c < 0)),
        "frac_pos": float(np.mean(c > 0)),
        "p10": float(np.quantile(c, 0.10)),
        "p90": float(np.quantile(c, 0.90)),
    }


def analyse_club(
    club: pd.DataFrame,
    team: str,
    *,
    min_fixtures_with_minutes: int = MIN_FIXTURES_WITH_MINUTES,
) -> dict[str, Any]:
    panel, meta = club_minutes_panel(
        club, min_fixtures_with_minutes=min_fixtures_with_minutes
    )
    if panel.shape[1] < 2 or panel.shape[0] < MIN_PAIR_N:
        return {
            "team": team,
            "n_fixtures": int(panel.shape[0]),
            "n_players": int(panel.shape[1]),
            "skipped": True,
        }

    raw_corr = panel.corr(method="pearson")
    start = (panel >= START_MINUTES).astype(float)
    # Constant columns → NaN corr; drop those players from start matrix pairs only
    start_corr = start.corr(method="pearson")

    raw_pairs = _pair_table(raw_corr, meta, kind="minutes", n_fixtures=int(len(panel)))
    start_pairs = _pair_table(start_corr, meta, kind="start60", n_fixtures=int(len(panel)))
    if len(raw_pairs):
        raw_pairs["stratum"] = [
            _stratum(a, b) for a, b in zip(raw_pairs["mean_m_a"], raw_pairs["mean_m_b"])
        ]

    gw_map = club.drop_duplicates("fixture")[["fixture", "GW"]]
    n_dgw_fixtures = 0
    if len(gw_map):
        vc = gw_map.groupby("GW").size()
        n_dgw_fixtures = int(vc[vc > 1].sum())

    strata = {}
    if len(raw_pairs):
        for s, g in raw_pairs.groupby("stratum"):
            strata[str(s)] = _summ(g)

    return {
        "team": team,
        "skipped": False,
        "n_fixtures": int(panel.shape[0]),
        "n_players": int(panel.shape[1]),
        "n_dgw_fixture_rows": n_dgw_fixtures,
        "minutes": _summ(raw_pairs),
        "start60": _summ(start_pairs),
        "strata_minutes": strata,
        "top_positive_minutes": raw_pairs.nlargest(8, "corr").to_dict(orient="records")
        if len(raw_pairs)
        else [],
        "top_negative_minutes": raw_pairs.nsmallest(8, "corr").to_dict(orient="records")
        if len(raw_pairs)
        else [],
        "top_negative_start60": start_pairs.nsmallest(8, "corr").to_dict(orient="records")
        if len(start_pairs)
        else [],
        "pairs_minutes": raw_pairs,
        "pairs_start60": start_pairs,
    }


def run_season(
    df: pd.DataFrame,
    *,
    min_fixtures_with_minutes: int = MIN_FIXTURES_WITH_MINUTES,
) -> dict[str, Any]:
    clubs = []
    all_m = []
    all_s = []
    for team, club in df.groupby("team", sort=True):
        doc = analyse_club(
            club, str(team), min_fixtures_with_minutes=min_fixtures_with_minutes
        )
        slim_keys = (
            "team",
            "skipped",
            "n_fixtures",
            "n_players",
            "n_dgw_fixture_rows",
            "minutes",
            "start60",
            "strata_minutes",
            "top_positive_minutes",
            "top_negative_minutes",
            "top_negative_start60",
        )
        clubs.append({k: doc[k] for k in slim_keys if k in doc})
        if doc.get("skipped"):
            continue
        pm = doc["pairs_minutes"].copy()
        pm["team"] = team
        ps = doc["pairs_start60"].copy()
        ps["team"] = team
        all_m.append(pm)
        all_s.append(ps)

    minutes_all = pd.concat(all_m, ignore_index=True) if all_m else pd.DataFrame()
    start_all = pd.concat(all_s, ignore_index=True) if all_s else pd.DataFrame()

    strata_pool: dict[str, Any] = {}
    if len(minutes_all) and "stratum" in minutes_all.columns:
        for s, g in minutes_all.groupby("stratum"):
            strata_pool[str(s)] = _summ(g)

    n_team_fixture = int(df.groupby(["team", "fixture"]).ngroups)
    n_team_gw = int(df.groupby(["team", "GW"]).ngroups)
    return {
        "season": SEASON_DEFAULT,
        "defs": {
            "observation": "team×fixture (DGW match rows count separately)",
            "min_fixtures_with_minutes": min_fixtures_with_minutes,
            "start_threshold": START_MINUTES,
            "both_nailed": f"mean minutes ≥{NAILED_MEAN}",
            "both_rotation_band": f"mean minutes in [{ROTATION_MEAN_LO},{ROTATION_MEAN_HI}]",
            "no_position_filter": True,
            "note": "Pearson corr unchanged by column demeaning — not used",
        },
        "n_team_fixture_obs": n_team_fixture,
        "n_team_gw_obs": n_team_gw,
        "dgw_extra_obs": n_team_fixture - n_team_gw,
        "pooled_minutes": _summ(minutes_all),
        "pooled_start60": _summ(start_all),
        "pooled_strata_minutes": strata_pool,
        "clubs": clubs,
        "pairs_minutes": minutes_all,
        "pairs_start60": start_all,
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    pm = doc["pooled_minutes"]
    ps = doc["pooled_start60"]
    st = doc["pooled_strata_minutes"]
    lines = [
        f"# Minutes covariance — {doc['season']} (fixture-level)",
        "",
        "Per-club player×player correlation of minutes. **No position filter.**",
        "Each **fixture** is one observation; DGWs with two match rows count as two.",
        "",
        f"- Team×fixture observations: **{doc['n_team_fixture_obs']}** "
        f"(team×GW would be {doc['n_team_gw_obs']}; DGW extras **{doc['dgw_extra_obs']}**)",
        f"- Min fixtures with minutes>0: **{doc['defs']['min_fixtures_with_minutes']}**",
        f"- Start indicator: minutes≥**{doc['defs']['start_threshold']:.0f}**",
        "",
        "## Pooled off-diagonal pairs",
        "",
        "| measure | n pairs | mean | median | frac neg | p10 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Minutes | {pm.get('n_pairs', 0)} | {pm.get('mean_corr', float('nan')):+.3f} | "
        f"{pm.get('median_corr', float('nan')):+.3f} | {pm.get('frac_neg', float('nan')):.1%} | "
        f"{pm.get('p10', float('nan')):+.3f} | {pm.get('p90', float('nan')):+.3f} |",
        f"| Start I(m≥60) | {ps.get('n_pairs', 0)} | {ps.get('mean_corr', float('nan')):+.3f} | "
        f"{ps.get('median_corr', float('nan')):+.3f} | {ps.get('frac_neg', float('nan')):.1%} | "
        f"{ps.get('p10', float('nan')):+.3f} | {ps.get('p90', float('nan')):+.3f} |",
        "",
        "### Minutes corr by mean-minute stratum",
        "",
        "| stratum | n pairs | median | frac neg | p10 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("both_nailed", "both_rotation_band", "other"):
        s = st.get(key, {})
        if not s:
            continue
        lines.append(
            f"| {key} | {s['n_pairs']} | {s['median_corr']:+.3f} | "
            f"{s['frac_neg']:.1%} | {s['p10']:+.3f} | {s['p90']:+.3f} |"
        )
    lines += [
        "",
        "## By club (minutes corr)",
        "",
        "| team | fixtures | players | DGW rows | median corr | frac neg | p10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in doc["clubs"]:
        if c.get("skipped"):
            continue
        m = c["minutes"]
        lines.append(
            f"| {c['team']} | {c['n_fixtures']} | {c['n_players']} | "
            f"{c['n_dgw_fixture_rows']} | {m['median_corr']:+.3f} | "
            f"{m['frac_neg']:.1%} | {m['p10']:+.3f} |"
        )

    pairs = doc["pairs_minutes"]
    if isinstance(pairs, pd.DataFrame) and len(pairs):
        neg = pairs.nsmallest(15, "corr")
        lines += [
            "",
            "## Strongest negative minutes pairs (all clubs)",
            "",
            "| team | player A | pos | player B | pos | corr | mean m A | mean m B |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for _, r in neg.iterrows():
            lines.append(
                f"| {r['team']} | {r['name_a']} | {r['pos_a']} | {r['name_b']} | "
                f"{r['pos_b']} | {r['corr']:+.3f} | {r['mean_m_a']:.1f} | {r['mean_m_b']:.1f} |"
            )
        pos = pairs.nlargest(10, "corr")
        lines += [
            "",
            "## Strongest positive minutes pairs",
            "",
            "| team | player A | pos | player B | pos | corr | mean m A | mean m B |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
        for _, r in pos.iterrows():
            lines.append(
                f"| {r['team']} | {r['name_a']} | {r['pos_a']} | {r['name_b']} | "
                f"{r['pos_b']} | {r['corr']:+.3f} | {r['mean_m_a']:.1f} | {r['mean_m_b']:.1f} |"
            )

    for name in ("Arsenal", "Liverpool", "Man City"):
        ex = next((c for c in doc["clubs"] if c["team"] == name and not c.get("skipped")), None)
        if not ex:
            continue
        lines += [
            "",
            f"## Example: {ex['team']}",
            "",
            f"- Fixtures **{ex['n_fixtures']}**, players **{ex['n_players']}**, "
            f"DGW fixture rows **{ex['n_dgw_fixture_rows']}**",
            "",
            "Most negative minutes:",
            "",
            "| A | B | corr |",
            "| --- | --- | ---: |",
        ]
        for r in ex.get("top_negative_minutes", [])[:6]:
            lines.append(
                f"| {r['name_a']} ({r['pos_a']}) | {r['name_b']} ({r['pos_b']}) | {r['corr']:+.3f} |"
            )
        lines += [
            "",
            "Most positive minutes:",
            "",
            "| A | B | corr |",
            "| --- | --- | ---: |",
        ]
        for r in ex.get("top_positive_minutes", [])[:6]:
            lines.append(
                f"| {r['name_a']} ({r['pos_a']}) | {r['name_b']} ({r['pos_b']}) | {r['corr']:+.3f} |"
            )

    lines += [
        "",
        "## Verdict",
        "",
        doc["verdict"],
        "",
        f"Pairs CSV: `{(PROCESSED / 'minutes_covariance_2025_26_pairs.csv').relative_to(ROOT)}`",
        f"JSON: `{(PROCESSED / 'minutes_covariance_2025_26.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, default=MERGED)
    ap.add_argument("--min-fixtures", type=int, default=MIN_FIXTURES_WITH_MINUTES)
    args = ap.parse_args(argv)

    df = load_merged(args.merged)
    doc = run_season(df, min_fixtures_with_minutes=args.min_fixtures)

    pm, ps, st = doc["pooled_minutes"], doc["pooled_start60"], doc["pooled_strata_minutes"]
    bn = st.get("both_nailed", {})
    br = st.get("both_rotation_band", {})
    verdict = (
        f"Fixture-level panel: {doc['dgw_extra_obs']} extra obs vs GW-collapse. "
        f"Pooled minutes corr median {pm.get('median_corr', float('nan')):+.2f} "
        f"({pm.get('frac_neg', float('nan')):.0%} negative) — full squad matrix mixes "
        f"XI co-starts with bench competition. "
        f"Both-nailed median {bn.get('median_corr', float('nan')):+.2f}; "
        f"rotation-band median {br.get('median_corr', float('nan')):+.2f} "
        f"(p10 {br.get('p10', float('nan')):+.2f}). "
        f"Start I(m≥60) median {ps.get('median_corr', float('nan')):+.2f}. "
        "Clear GK/FB competition pairs near −1 — minutes covariance is real and club-matrix shaped."
    )
    doc["verdict"] = verdict

    pairs = pd.concat(
        [
            doc["pairs_minutes"].assign(measure="minutes")
            if len(doc["pairs_minutes"])
            else pd.DataFrame(),
            doc["pairs_start60"].assign(measure="start60")
            if len(doc["pairs_start60"])
            else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    pairs.to_csv(PROCESSED / "minutes_covariance_2025_26_pairs.csv", index=False)

    slim = {k: doc[k] for k in doc if k not in ("pairs_minutes", "pairs_start60")}
    (PROCESSED / "minutes_covariance_2025_26.json").write_text(
        json.dumps(slim, indent=2, default=str), encoding="utf-8"
    )
    write_report(doc, REPORTS / "minutes_covariance_2025_26.md")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "pooled_minutes": pm,
                "pooled_start60": ps,
                "pooled_strata_minutes": st,
                "dgw_extra_obs": doc["dgw_extra_obs"],
            },
            indent=2,
        )
    )
    print(f"wrote {REPORTS / 'minutes_covariance_2025_26.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
