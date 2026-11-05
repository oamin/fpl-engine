"""Team goals/assists covariance dig — fixture-level, 2025-26 trial.

Per club, player×player correlations of goals and assists across fixtures
(DGW match rows count separately). Companion to ``minutes_covariance``;
this is realised event co-movement, not a scorer law.

Measures
--------
- G×G, A×A (same-team off-diagonal)
- Cross G_i × A_j (i≠j) — teammate goal→assist coupling
- Same-player G×A
- Binary I(G≥1) / I(A≥1) versions (zero-inflated counts)
- Strata: all players; MID+FWD only; both mean-m≥60; pairwise both-played
  fixtures only (minutes>0 for both on that fixture)

  PYTHONPATH=. python3 -m src.eval.ga_team_covariance
  PYTHONPATH=. python3 -m src.eval.ga_team_covariance --season 2025-26
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
MIN_FIXTURES_APPEAR = 5
MIN_PAIR_N = 8
START_MINUTES = 60.0
ATTACK_POS = {"MID", "FWD"}
# Drop sparse-event perfect-corr artifacts (one shared non-zero week).
ACTIVE_MIN_GOALS = 3.0
ACTIVE_MIN_ASSISTS = 3.0


def merged_path(season: str) -> Path:
    return CACHE / f"merged_gw_{season.replace('-', '_')}.csv"


def load_merged(path: Path | str | None = None, *, season: str | None = None) -> pd.DataFrame:
    if path is None:
        path = merged_path(season or SEASON_DEFAULT)
    path = Path(path)
    df = pd.read_csv(path)
    need = {
        "element",
        "name",
        "team",
        "position",
        "fixture",
        "minutes",
        "goals_scored",
        "assists",
        "GW",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"merged_gw missing columns: {sorted(missing)}")
    from src.rules.fpl_2026 import drop_assistant_managers

    df = drop_assistant_managers(df)
    for col in ("minutes", "goals_scored", "assists"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["element"] = pd.to_numeric(df["element"], errors="coerce").astype(int)
    df["fixture"] = pd.to_numeric(df["fixture"], errors="coerce")
    df = df.dropna(subset=["fixture", "team"])
    df = df.drop_duplicates(["team", "fixture", "element"], keep="first")
    return df


def _summ(series: pd.Series | np.ndarray) -> dict[str, Any]:
    c = np.asarray(series, dtype=float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(c.size),
        "mean_corr": float(np.mean(c)),
        "median_corr": float(np.median(c)),
        "frac_neg": float(np.mean(c < 0)),
        "frac_pos": float(np.mean(c > 0)),
        "p10": float(np.quantile(c, 0.10)),
        "p90": float(np.quantile(c, 0.90)),
    }


def _pair_corrs(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    kind: str,
    same_player: bool = False,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pairwise Pearson corr. If mask given, corr only on fixtures where mask[a]&mask[b]."""
    cols = list(panel_a.columns)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        js = [a] if same_player else cols[i + 1 :]
        for b in js:
            if not same_player and a == b:
                continue
            xa = panel_a[a].to_numpy(float)
            xb = panel_b[b].to_numpy(float)
            n_obs = int(len(xa))
            if mask is not None:
                m = mask[a].to_numpy(bool) & mask[b].to_numpy(bool)
                xa, xb = xa[m], xb[m]
                n_obs = int(m.sum())
            if n_obs < MIN_PAIR_N:
                continue
            if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
                continue
            c = float(np.corrcoef(xa, xb)[0, 1])
            if not np.isfinite(c):
                continue
            rows.append(
                {
                    "kind": kind,
                    "element_a": int(a),
                    "element_b": int(b),
                    "name_a": meta.loc[a, "name"] if a in meta.index else str(a),
                    "name_b": meta.loc[b, "name"] if b in meta.index else str(b),
                    "pos_a": meta.loc[a, "position"] if a in meta.index else "",
                    "pos_b": meta.loc[b, "position"] if b in meta.index else "",
                    "corr": c,
                    "n_obs": n_obs,
                    "mean_g_a": float(meta.loc[a, "mean_g"]) if a in meta.index else float("nan"),
                    "mean_a_a": float(meta.loc[a, "mean_a"]) if a in meta.index else float("nan"),
                    "mean_g_b": float(meta.loc[b, "mean_g"]) if b in meta.index else float("nan"),
                    "mean_a_b": float(meta.loc[b, "mean_a"]) if b in meta.index else float("nan"),
                    "sum_g_a": float(meta.loc[a, "sum_g"]) if a in meta.index else float("nan"),
                    "sum_a_a": float(meta.loc[a, "sum_a"]) if a in meta.index else float("nan"),
                    "sum_g_b": float(meta.loc[b, "sum_g"]) if b in meta.index else float("nan"),
                    "sum_a_b": float(meta.loc[b, "sum_a"]) if b in meta.index else float("nan"),
                    "mean_m_a": float(meta.loc[a, "mean_m"]) if a in meta.index else float("nan"),
                    "mean_m_b": float(meta.loc[b, "mean_m"]) if b in meta.index else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def club_meta(club: pd.DataFrame) -> pd.DataFrame:
    played = (
        club.loc[club["minutes"] > 0]
        .groupby("element")
        .size()
        .rename("n_appear")
    )
    keep = played[played >= MIN_FIXTURES_APPEAR].index
    sub = club.loc[club["element"].isin(keep)].copy()
    g_panel = sub.pivot_table(
        index="fixture", columns="element", values="goals_scored", aggfunc="sum"
    ).reindex(sorted(club["fixture"].unique())).fillna(0.0)
    a_panel = sub.pivot_table(
        index="fixture", columns="element", values="assists", aggfunc="sum"
    ).reindex(sorted(club["fixture"].unique())).fillna(0.0)
    m_panel = sub.pivot_table(
        index="fixture", columns="element", values="minutes", aggfunc="sum"
    ).reindex(sorted(club["fixture"].unique())).fillna(0.0)
    meta = (
        sub.drop_duplicates("element")[["element", "name", "position"]]
        .set_index("element")
        .join(played)
    )
    meta["mean_g"] = g_panel.mean(axis=0)
    meta["mean_a"] = a_panel.mean(axis=0)
    meta["mean_m"] = m_panel.mean(axis=0)
    meta["sum_g"] = g_panel.sum(axis=0)
    meta["sum_a"] = a_panel.sum(axis=0)
    meta["n_fixtures"] = int(len(g_panel))
    return meta, g_panel, a_panel, m_panel


def analyse_club(club: pd.DataFrame, team: str) -> dict[str, Any]:
    meta, g_panel, a_panel, m_panel = club_meta(club)
    if meta.shape[0] < 2 or g_panel.shape[0] < MIN_PAIR_N:
        return {"team": team, "skipped": True, "n_players": int(meta.shape[0])}

    elements = [int(e) for e in meta.index.tolist()]
    g_panel = g_panel.reindex(columns=elements, fill_value=0.0)
    a_panel = a_panel.reindex(columns=elements, fill_value=0.0)
    m_panel = m_panel.reindex(columns=elements, fill_value=0.0)
    appear = (m_panel > 0).astype(bool)
    g_bin = (g_panel >= 1).astype(float)
    a_bin = (a_panel >= 1).astype(float)

    attack_els = [
        e for e in elements if str(meta.loc[e, "position"]) in ATTACK_POS
    ]
    nailed_els = [e for e in elements if float(meta.loc[e, "mean_m"]) >= START_MINUTES]

    blocks: dict[str, pd.DataFrame] = {}

    def add(name: str, els: list[int], *, both_played: bool = False) -> None:
        if len(els) < 2:
            blocks[name + "_gg"] = pd.DataFrame()
            blocks[name + "_aa"] = pd.DataFrame()
            blocks[name + "_cross_ga"] = pd.DataFrame()
            blocks[name + "_gg_bin"] = pd.DataFrame()
            blocks[name + "_aa_bin"] = pd.DataFrame()
            blocks[name + "_cross_ga_bin"] = pd.DataFrame()
            return
        gp = g_panel[els]
        ap = a_panel[els]
        gb = g_bin[els]
        ab = a_bin[els]
        mask = appear[els] if both_played else None
        blocks[name + "_gg"] = _pair_corrs(gp, gp, meta, kind=name + "_gg", mask=mask)
        blocks[name + "_aa"] = _pair_corrs(ap, ap, meta, kind=name + "_aa", mask=mask)
        # Full directed cross i≠j: G_i vs A_j and G_j vs A_i as separate unordered? 
        # Use unordered unique pairs: for each unordered {i,j}, store corr(G_i,A_j)
        # and corr(G_j,A_i) as two rows (asymmetric coupling).
        cross_rows = []
        for i, a in enumerate(els):
            for b in els[i + 1 :]:
                for src, dst, tag in ((a, b, "g_a_vs_a_b"), (b, a, "g_b_vs_a_a")):
                    xa = g_panel[src].to_numpy(float)
                    xb = a_panel[dst].to_numpy(float)
                    n_obs = int(len(xa))
                    if both_played:
                        m = appear[src].to_numpy(bool) & appear[dst].to_numpy(bool)
                        xa, xb = xa[m], xb[m]
                        n_obs = int(m.sum())
                    if n_obs < MIN_PAIR_N:
                        continue
                    if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
                        continue
                    c = float(np.corrcoef(xa, xb)[0, 1])
                    if not np.isfinite(c):
                        continue
                    cross_rows.append(
                        {
                            "kind": name + "_cross_ga",
                            "tag": tag,
                            "element_a": int(src),
                            "element_b": int(dst),
                            "name_a": meta.loc[src, "name"],
                            "name_b": meta.loc[dst, "name"],
                            "pos_a": meta.loc[src, "position"],
                            "pos_b": meta.loc[dst, "position"],
                            "corr": c,
                            "n_obs": n_obs,
                            "mean_g_a": float(meta.loc[src, "mean_g"]),
                            "mean_a_b": float(meta.loc[dst, "mean_a"]),
                            "sum_g_a": float(meta.loc[src, "sum_g"]),
                            "sum_a_b": float(meta.loc[dst, "sum_a"]),
                            "mean_m_a": float(meta.loc[src, "mean_m"]),
                            "mean_m_b": float(meta.loc[dst, "mean_m"]),
                        }
                    )
                # binary cross
                for src, dst, tag in ((a, b, "g_a_vs_a_b"), (b, a, "g_b_vs_a_a")):
                    xa = g_bin[src].to_numpy(float)
                    xb = a_bin[dst].to_numpy(float)
                    n_obs = int(len(xa))
                    if both_played:
                        m = appear[src].to_numpy(bool) & appear[dst].to_numpy(bool)
                        xa, xb = xa[m], xb[m]
                        n_obs = int(m.sum())
                    if n_obs < MIN_PAIR_N:
                        continue
                    if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
                        continue
                    c = float(np.corrcoef(xa, xb)[0, 1])
                    if not np.isfinite(c):
                        continue
                    cross_rows.append(
                        {
                            "kind": name + "_cross_ga_bin",
                            "tag": tag,
                            "element_a": int(src),
                            "element_b": int(dst),
                            "name_a": meta.loc[src, "name"],
                            "name_b": meta.loc[dst, "name"],
                            "pos_a": meta.loc[src, "position"],
                            "pos_b": meta.loc[dst, "position"],
                            "corr": c,
                            "n_obs": n_obs,
                        }
                    )
        cross = pd.DataFrame(cross_rows)
        blocks[name + "_cross_ga"] = cross.loc[cross["kind"] == name + "_cross_ga"].copy() if len(cross) else pd.DataFrame()
        blocks[name + "_cross_ga_bin"] = (
            cross.loc[cross["kind"] == name + "_cross_ga_bin"].copy() if len(cross) else pd.DataFrame()
        )
        blocks[name + "_gg_bin"] = _pair_corrs(gb, gb, meta, kind=name + "_gg_bin", mask=mask)
        blocks[name + "_aa_bin"] = _pair_corrs(ab, ab, meta, kind=name + "_aa_bin", mask=mask)

    add("all", elements)
    add("attack", attack_els)
    add("nailed", nailed_els)
    add("all_both_played", elements, both_played=True)
    add("attack_both_played", attack_els, both_played=True)

    # Same-player G×A
    same = _pair_corrs(
        g_panel, a_panel, meta, kind="same_player_ga", same_player=True
    )
    same_bin = _pair_corrs(
        g_bin, a_bin, meta, kind="same_player_ga_bin", same_player=True
    )
    blocks["same_player_ga"] = same
    blocks["same_player_ga_bin"] = same_bin

    # Team totals per fixture
    team_g = g_panel.sum(axis=1).to_numpy(float)
    team_a = a_panel.sum(axis=1).to_numpy(float)
    team_corr = (
        float(np.corrcoef(team_g, team_a)[0, 1])
        if np.std(team_g) > 1e-12 and np.std(team_a) > 1e-12
        else float("nan")
    )

    gw_map = club.drop_duplicates("fixture")[["fixture", "GW"]]
    n_dgw = 0
    if len(gw_map):
        vc = gw_map.groupby("GW").size()
        n_dgw = int(vc[vc > 1].sum())

    summaries = {k: _summ(v["corr"]) if len(v) and "corr" in v.columns else {"n_pairs": 0} for k, v in blocks.items()}

    def top(df: pd.DataFrame, n: int = 8, ascending: bool = False) -> list[dict[str, Any]]:
        if df is None or len(df) == 0 or "corr" not in df.columns:
            return []
        out = df.nsmallest(n, "corr") if ascending else df.nlargest(n, "corr")
        return out.to_dict(orient="records")

    return {
        "team": team,
        "skipped": False,
        "n_fixtures": int(g_panel.shape[0]),
        "n_players": int(len(elements)),
        "n_attack": int(len(attack_els)),
        "n_dgw_fixture_rows": n_dgw,
        "team_g_vs_a_corr": team_corr,
        "team_mean_g": float(np.mean(team_g)),
        "team_mean_a": float(np.mean(team_a)),
        "summaries": summaries,
        "top_cross_ga_pos": top(blocks.get("attack_cross_ga", pd.DataFrame())),
        "top_cross_ga_neg": top(
            blocks.get("attack_cross_ga", pd.DataFrame()), ascending=True
        ),
        "top_gg_pos": top(blocks.get("attack_gg", pd.DataFrame())),
        "top_gg_neg": top(blocks.get("attack_gg", pd.DataFrame()), ascending=True),
        "blocks": blocks,
    }


def run_season(df: pd.DataFrame, *, season: str) -> dict[str, Any]:
    clubs_out = []
    pool: dict[str, list[pd.DataFrame]] = {}
    team_corrs = []

    for team, club in df.groupby("team", sort=True):
        doc = analyse_club(club, str(team))
        slim = {k: doc[k] for k in doc if k != "blocks"}
        clubs_out.append(slim)
        if doc.get("skipped"):
            continue
        team_corrs.append(
            {
                "team": team,
                "team_g_vs_a_corr": doc["team_g_vs_a_corr"],
                "team_mean_g": doc["team_mean_g"],
                "team_mean_a": doc["team_mean_a"],
                "n_fixtures": doc["n_fixtures"],
            }
        )
        for k, block in doc["blocks"].items():
            if len(block):
                b = block.copy()
                b["team"] = team
                pool.setdefault(k, []).append(b)

    pooled_frames = {
        k: pd.concat(vs, ignore_index=True) if vs else pd.DataFrame()
        for k, vs in pool.items()
    }
    pooled_summ = {
        k: _summ(v["corr"]) if len(v) and "corr" in v.columns else {"n_pairs": 0}
        for k, v in pooled_frames.items()
    }

    n_team_fixture = int(df.groupby(["team", "fixture"]).ngroups)
    n_team_gw = int(df.groupby(["team", "GW"]).ngroups)

    # Active-volume filters (drop one-hit perfect-corr artifacts).
    def _active_gg(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or len(frame) == 0:
            return pd.DataFrame()
        if "sum_g_a" not in frame.columns or "sum_g_b" not in frame.columns:
            return pd.DataFrame()
        keep = (frame["sum_g_a"] >= ACTIVE_MIN_GOALS) & (
            frame["sum_g_b"] >= ACTIVE_MIN_GOALS
        )
        return frame.loc[keep].copy()

    def _active_cross(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or len(frame) == 0:
            return pd.DataFrame()
        if "sum_g_a" not in frame.columns or "sum_a_b" not in frame.columns:
            return pd.DataFrame()
        keep = (frame["sum_g_a"] >= ACTIVE_MIN_GOALS) & (
            frame["sum_a_b"] >= ACTIVE_MIN_ASSISTS
        )
        return frame.loc[keep].copy()

    active_gg = _active_gg(pooled_frames.get("attack_gg", pd.DataFrame()))
    active_cross = _active_cross(pooled_frames.get("attack_cross_ga", pd.DataFrame()))
    active_cross_bp = _active_cross(
        pooled_frames.get("attack_both_played_cross_ga", pd.DataFrame())
    )
    pooled_frames["attack_active_gg"] = active_gg
    pooled_frames["attack_active_cross_ga"] = active_cross
    pooled_frames["attack_active_both_played_cross_ga"] = active_cross_bp
    pooled_summ["attack_active_gg"] = _summ(active_gg["corr"]) if len(active_gg) else {"n_pairs": 0}
    pooled_summ["attack_active_cross_ga"] = (
        _summ(active_cross["corr"]) if len(active_cross) else {"n_pairs": 0}
    )
    pooled_summ["attack_active_both_played_cross_ga"] = (
        _summ(active_cross_bp["corr"]) if len(active_cross_bp) else {"n_pairs": 0}
    )

    return {
        "season": season,
        "defs": {
            "observation": "team×fixture (DGW match rows count separately)",
            "min_appear_fixtures": MIN_FIXTURES_APPEAR,
            "min_pair_n": MIN_PAIR_N,
            "attack_pos": sorted(ATTACK_POS),
            "nailed_mean_m": START_MINUTES,
            "both_played": "pairwise fixtures where both minutes>0",
            "cross_ga": "corr(goals_i, assists_j) for i≠j (directed; both directions stored)",
            "bin": "I(count≥1)",
            "active": (
                f"scorer season goals≥{ACTIVE_MIN_GOALS:g}, "
                f"assister season assists≥{ACTIVE_MIN_ASSISTS:g}"
            ),
            "note": "diagnostic only — not a μ-board scorer law",
        },
        "n_team_fixture_obs": n_team_fixture,
        "n_team_gw_obs": n_team_gw,
        "dgw_extra_obs": n_team_fixture - n_team_gw,
        "pooled": pooled_summ,
        "team_totals": team_corrs,
        "team_totals_summary": _summ(
            pd.Series([t["team_g_vs_a_corr"] for t in team_corrs])
        ),
        "clubs": clubs_out,
        "pairs": pooled_frames,
    }


def _fmt_row(label: str, s: dict[str, Any]) -> str:
    if not s or s.get("n_pairs", 0) == 0:
        return f"| {label} | 0 | — | — | — | — | — |"
    return (
        f"| {label} | {s['n_pairs']} | {s['mean_corr']:+.3f} | {s['median_corr']:+.3f} | "
        f"{s['frac_pos']:.1%} | {s['p10']:+.3f} | {s['p90']:+.3f} |"
    )


def write_report(doc: dict[str, Any], path: Path) -> None:
    p = doc["pooled"]
    lines = [
        f"# Team G/A covariance — {doc['season']} (fixture-level)",
        "",
        "Per-club player×player correlation of **goals** and **assists**.",
        "Each **fixture** is one observation; DGWs with two match rows count as two.",
        "Diagnostic only — not a scorer law.",
        "",
        f"- Team×fixture observations: **{doc['n_team_fixture_obs']}** "
        f"(team×GW {doc['n_team_gw_obs']}; DGW extras **{doc['dgw_extra_obs']}**)",
        f"- Min appear fixtures: **{doc['defs']['min_appear_fixtures']}**; "
        f"min pair n: **{doc['defs']['min_pair_n']}**",
        f"- Attack positions: {', '.join(doc['defs']['attack_pos'])}",
        "",
        "## Team totals (ΣG vs ΣA per fixture)",
        "",
        f"Across clubs, corr(team goals, team assists) median "
        f"**{doc['team_totals_summary'].get('median_corr', float('nan')):+.3f}** "
        f"(mean {doc['team_totals_summary'].get('mean_corr', float('nan')):+.3f}; "
        f"n={doc['team_totals_summary'].get('n_pairs', 0)}).",
        "",
        "| team | corr(ΣG,ΣA) | mean ΣG | mean ΣA | fixtures |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for t in sorted(doc["team_totals"], key=lambda x: -abs(x.get("team_g_vs_a_corr") or 0)):
        lines.append(
            f"| {t['team']} | {t['team_g_vs_a_corr']:+.3f} | {t['team_mean_g']:.2f} | "
            f"{t['team_mean_a']:.2f} | {t['n_fixtures']} |"
        )

    lines += [
        "",
        "## Pooled player pairs",
        "",
        "Positive frac = share of pairs with corr > 0.",
        "",
        "| measure | n pairs | mean | median | frac pos | p10 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _fmt_row("Attack G×G", p.get("attack_gg", {})),
        _fmt_row("Attack A×A", p.get("attack_aa", {})),
        _fmt_row("Attack cross G_i×A_j", p.get("attack_cross_ga", {})),
        _fmt_row(
            f"Attack active cross (≥{ACTIVE_MIN_GOALS:g}G/≥{ACTIVE_MIN_ASSISTS:g}A)",
            p.get("attack_active_cross_ga", {}),
        ),
        _fmt_row("Attack active G×G", p.get("attack_active_gg", {})),
        _fmt_row("Attack G×G (bin)", p.get("attack_gg_bin", {})),
        _fmt_row("Attack A×A (bin)", p.get("attack_aa_bin", {})),
        _fmt_row("Attack cross bin", p.get("attack_cross_ga_bin", {})),
        _fmt_row("Attack G×G both-played", p.get("attack_both_played_gg", {})),
        _fmt_row("Attack A×A both-played", p.get("attack_both_played_aa", {})),
        _fmt_row("Attack cross both-played", p.get("attack_both_played_cross_ga", {})),
        _fmt_row(
            "Attack active cross both-played",
            p.get("attack_active_both_played_cross_ga", {}),
        ),
        _fmt_row("All G×G", p.get("all_gg", {})),
        _fmt_row("All cross G_i×A_j", p.get("all_cross_ga", {})),
        _fmt_row("Nailed G×G", p.get("nailed_gg", {})),
        _fmt_row("Nailed cross G_i×A_j", p.get("nailed_cross_ga", {})),
        _fmt_row("Same-player G×A", p.get("same_player_ga", {})),
        _fmt_row("Same-player G×A bin", p.get("same_player_ga_bin", {})),
        "",
        f"## Strongest *active* attack cross G→A (scorer≥{ACTIVE_MIN_GOALS:g}G, "
        f"assister≥{ACTIVE_MIN_ASSISTS:g}A)",
        "",
        "| team | scorer | pos | assister | pos | corr | n |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    cross = doc["pairs"].get("attack_active_cross_ga", pd.DataFrame())
    if len(cross):
        for _, r in cross.nlargest(15, "corr").iterrows():
            lines.append(
                f"| {r['team']} | {r['name_a']} | {r['pos_a']} | {r['name_b']} | "
                f"{r['pos_b']} | {r['corr']:+.3f} | {int(r['n_obs'])} |"
            )
    lines += [
        "",
        "## Strongest *active* attack cross G→A (negative)",
        "",
        "| team | scorer | pos | assister | pos | corr | n |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    if len(cross):
        for _, r in cross.nsmallest(15, "corr").iterrows():
            lines.append(
                f"| {r['team']} | {r['name_a']} | {r['pos_a']} | {r['name_b']} | "
                f"{r['pos_b']} | {r['corr']:+.3f} | {int(r['n_obs'])} |"
            )

    gg = doc["pairs"].get("attack_active_gg", pd.DataFrame())
    lines += [
        "",
        f"## Strongest *active* attack G×G (both ≥{ACTIVE_MIN_GOALS:g}G)",
        "",
        "| team | A | pos | B | pos | corr | n |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    if len(gg):
        for _, r in gg.nlargest(12, "corr").iterrows():
            lines.append(
                f"| {r['team']} | {r['name_a']} | {r['pos_a']} | {r['name_b']} | "
                f"{r['pos_b']} | {r['corr']:+.3f} | {int(r['n_obs'])} |"
            )
    lines += [
        "",
        "## Strongest *active* attack G×G negative",
        "",
        "| team | A | pos | B | pos | corr | n |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    if len(gg):
        for _, r in gg.nsmallest(12, "corr").iterrows():
            lines.append(
                f"| {r['team']} | {r['name_a']} | {r['pos_a']} | {r['name_b']} | "
                f"{r['pos_b']} | {r['corr']:+.3f} | {int(r['n_obs'])} |"
            )

    # Reading
    att_cross = p.get("attack_cross_ga", {})
    att_active = p.get("attack_active_cross_ga", {})
    att_gg = p.get("attack_gg", {})
    att_gg_a = p.get("attack_active_gg", {})
    att_bp = p.get("attack_both_played_cross_ga", {})
    lines += [
        "",
        "## Reading",
        "",
        f"- Club **ΣG vs ΣA** is tightly coupled (median "
        f"**{doc['team_totals_summary'].get('median_corr', float('nan')):+.3f}**).",
        f"- Unfiltered attack **cross G_i×A_j** median "
        f"**{att_cross.get('median_corr', float('nan')):+.3f}** "
        f"(frac pos {att_cross.get('frac_pos', float('nan')):.1%}) — "
        "most MID/FWD pairs are not linked; sparse zeros dominate.",
        f"- **Active** cross (volume filter) median "
        f"**{att_active.get('median_corr', float('nan')):+.3f}** "
        f"(n={att_active.get('n_pairs', 0)}, "
        f"frac pos {att_active.get('frac_pos', float('nan')):.1%}).",
        f"- Attack **G×G** median "
        f"**{att_gg.get('median_corr', float('nan')):+.3f}**; active "
        f"**{att_gg_a.get('median_corr', float('nan')):+.3f}** "
        f"(co-scoring is weak / slightly negative — pot sharing, not joint spikes).",
        f"- Both-played cross median "
        f"**{att_bp.get('median_corr', float('nan')):+.3f}**.",
        "- Unfiltered top ±1.0 pairs are one-hit artifacts; prefer the active tables.",
        "- Implication for the scorer debate: team attack volume is shared, but "
        "player-level G/A co-movement is weak once sparse pairs are dropped — "
        "not strong evidence for a dense teammate covariance matrix on μ boards.",
        "",
        "Reproduce:",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.ga_team_covariance --season {doc['season']}",
        "```",
        "",
        f"Pairs CSV: `data/processed/ga_team_covariance_{doc['season'].replace('-', '_')}_pairs.csv`",
        f"JSON: `data/processed/ga_team_covariance_{doc['season'].replace('-', '_')}.json`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "pairs"}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, pd.DataFrame):
        return None
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=SEASON_DEFAULT)
    ap.add_argument("--merged", type=Path, default=None)
    args = ap.parse_args(argv)

    season = str(args.season)
    tag = season.replace("-", "_")
    df = load_merged(args.merged, season=season)
    doc = run_season(df, season=season)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Flatten pairs for CSV
    frames = []
    for kind, frame in doc["pairs"].items():
        if len(frame):
            frames.append(frame)
    pairs_path = PROCESSED / f"ga_team_covariance_{tag}_pairs.csv"
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(pairs_path, index=False)
    else:
        pd.DataFrame().to_csv(pairs_path, index=False)

    json_path = PROCESSED / f"ga_team_covariance_{tag}.json"
    json_path.write_text(
        json.dumps(_jsonable(doc), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    report_path = REPORTS / f"ga_team_covariance_{tag}.md"
    write_report(doc, report_path)

    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(f"wrote {pairs_path}")
    att = doc["pooled"].get("attack_cross_ga", {})
    print(
        "attack cross G_i×A_j: "
        f"n={att.get('n_pairs', 0)} median={att.get('median_corr', float('nan')):+.3f} "
        f"frac_pos={att.get('frac_pos', float('nan')):.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
