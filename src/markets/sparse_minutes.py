"""Sparse minutes coupler from competition pairs (not a dense Σ).

Identify club pairs with minutes corr ≤ ``CORR_MAX`` (default −0.7), merge
into competition groups, and draw minutes:

- Competition group: Dirichlet on mean-minute weights × pot (≈90 for
  exclusive pairs; else sum of means). Captures A-up/B-down.
- Everyone else: independent N(μ_m, σ_m) clipped to [0, 90].
- Soft team budget: if Σ outfield minutes > 11×90, scale outfield down.
- Exactly one GK preferred: if multiple GKs outside groups, keep the
  highest-μ GK draw and zero others (groups already handle GK–GK).

  python -m src.markets.sparse_minutes
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets.minutes_covariance import (
    MERGED,
    MIN_FIXTURES_WITH_MINUTES,
    load_merged,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CORR_MAX = -0.7
DIRICHLET_STRENGTH = 3.0  # lower → more volatile within a competition group
OUTFIELD_BUDGET = 11.0 * 90.0
INDEP_SIGMA_FLOOR = 8.0
INDEP_SIGMA_FRAC = 0.25


@dataclass(frozen=True)
class CompetitionGroup:
    team: str
    elements: tuple[int, ...]
    names: tuple[str, ...]
    positions: tuple[str, ...]
    mean_minutes: tuple[float, ...]
    min_pair_corr: float


def competition_groups_from_pairs(
    pairs: pd.DataFrame,
    meta: pd.DataFrame,
    team: str,
    *,
    corr_max: float = CORR_MAX,
) -> list[CompetitionGroup]:
    """Greedy strongest pairs with corr ≤ ``corr_max`` (each player ≤ one pair).

    Avoids union-find over-merge (e.g. GK chained into midfield cliques).
    """
    if pairs.empty:
        return []
    sub = pairs.loc[pairs["corr"] <= float(corr_max)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("corr", ascending=True)  # most negative first
    meta_i = meta.set_index("element") if "element" in meta.columns else meta
    used: set[int] = set()
    out: list[CompetitionGroup] = []
    for _, r in sub.iterrows():
        a, b = int(r["element_a"]), int(r["element_b"])
        if a in used or b in used:
            continue
        if a not in meta_i.index or b not in meta_i.index:
            continue
        used.add(a)
        used.add(b)
        out.append(
            CompetitionGroup(
                team=team,
                elements=(a, b),
                names=(str(meta_i.loc[a, "name"]), str(meta_i.loc[b, "name"])),
                positions=(
                    str(meta_i.loc[a, "position"]),
                    str(meta_i.loc[b, "position"]),
                ),
                mean_minutes=(
                    float(meta_i.loc[a, "mean_minutes"]),
                    float(meta_i.loc[b, "mean_minutes"]),
                ),
                min_pair_corr=float(r["corr"]),
            )
        )
    return out


def find_club_competition_groups(
    club: pd.DataFrame,
    team: str,
    *,
    corr_max: float = CORR_MAX,
    min_fixtures_with_minutes: int = MIN_FIXTURES_WITH_MINUTES,
) -> tuple[list[CompetitionGroup], pd.DataFrame]:
    from src.markets.minutes_covariance import _pair_table, club_minutes_panel

    panel, meta = club_minutes_panel(
        club, min_fixtures_with_minutes=min_fixtures_with_minutes
    )
    if panel.shape[1] < 2 or panel.shape[0] < 8:
        return [], meta.reset_index()
    raw = panel.corr(method="pearson")
    pairs = _pair_table(raw, meta, kind="minutes", n_fixtures=int(len(panel)))
    groups = competition_groups_from_pairs(
        pairs, meta.reset_index(), team, corr_max=corr_max
    )
    return groups, meta.reset_index()


def draw_competition_minutes(
    means: np.ndarray,
    *,
    positions: list[str] | None = None,
    strength: float = DIRICHLET_STRENGTH,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw minutes inside one competition group (sum-constrained)."""
    mu = np.asarray(means, dtype=float)
    mu = np.where(np.isfinite(mu) & (mu > 0), mu, 1.0)
    pos = positions or ["MID"] * len(mu)
    all_gk = all(str(p) in ("GKP", "GK") for p in pos)
    if all_gk or (len(mu) == 2 and float(mu.min()) + float(mu.max()) <= 100):
        # Near-exclusive slot (~90′)
        pot = 90.0
    else:
        pot = float(np.clip(mu.sum(), 60.0, 90.0 * len(mu)))
    alpha = strength * (mu / mu.sum())
    alpha = np.maximum(alpha, 1e-3)
    shares = rng.dirichlet(alpha)
    m = shares * pot
    return np.clip(m, 0.0, 90.0)


def draw_club_minutes(
    players: pd.DataFrame,
    groups: list[CompetitionGroup],
    *,
    rng: np.random.Generator | None = None,
    strength: float = DIRICHLET_STRENGTH,
    indep_sigma_frac: float = INDEP_SIGMA_FRAC,
) -> pd.DataFrame:
    """Return a copy of ``players`` with drawn ``m`` (and ``m_source``).

    Requires columns: ``element`` (or ``player_id``), ``position``, ``m`` (prior mean).
    Optional ``sigma_m``.
    """
    rng = rng or np.random.default_rng(0)
    out = players.copy()
    if "element" not in out.columns and "player_id" in out.columns:
        out["element"] = pd.to_numeric(out["player_id"], errors="coerce")
    out["element"] = pd.to_numeric(out["element"], errors="coerce").astype(int)
    out["m"] = pd.to_numeric(out["m"], errors="coerce").fillna(0.0)
    if "sigma_m" not in out.columns:
        out["sigma_m"] = np.maximum(INDEP_SIGMA_FLOOR, indep_sigma_frac * out["m"])
    else:
        out["sigma_m"] = pd.to_numeric(out["sigma_m"], errors="coerce").fillna(
            INDEP_SIGMA_FLOOR
        )
    out["m_source"] = "independent"
    drawn = out["m"].to_numpy(float).copy()
    source = np.array(["independent"] * len(out), dtype=object)
    grouped: set[int] = set()

    el_to_ix = {int(e): i for i, e in enumerate(out["element"].tolist())}
    for g in groups:
        idxs = [el_to_ix[e] for e in g.elements if e in el_to_ix]
        if len(idxs) < 2:
            continue
        means = out["m"].iloc[idxs].to_numpy(float)
        # Prefer stored group means if lengths match
        if len(g.mean_minutes) == len(idxs):
            means = np.asarray(g.mean_minutes, dtype=float)
        pos = out["position"].iloc[idxs].astype(str).tolist()
        m_g = draw_competition_minutes(means, positions=pos, strength=strength, rng=rng)
        for j, ix in enumerate(idxs):
            drawn[ix] = float(m_g[j])
            source[ix] = "competition"
            grouped.add(int(out["element"].iloc[ix]))

    # Independent draws for non-grouped
    for i, e in enumerate(out["element"].tolist()):
        if int(e) in grouped:
            continue
        mu = float(out["m"].iloc[i])
        sig = float(max(out["sigma_m"].iloc[i], INDEP_SIGMA_FLOOR))
        drawn[i] = float(np.clip(rng.normal(mu, sig), 0.0, 90.0))

    # Prefer single GK among non-competition GKs if multiple high draws
    gk_mask = out["position"].astype(str).isin(["GKP", "GK"]).to_numpy()
    gk_ix = np.where(gk_mask)[0]
    if len(gk_ix) > 1:
        # If any GK already in a competition group, zero independent GKs with less draw
        in_comp = np.array([int(out["element"].iloc[i]) in grouped for i in gk_ix])
        if not in_comp.any():
            best = gk_ix[int(np.argmax(drawn[gk_ix]))]
            for i in gk_ix:
                if i != best:
                    drawn[i] = 0.0
                    source[i] = "gk_zeroed"

    # Soft outfield budget
    outfield = ~gk_mask
    total = float(drawn[outfield].sum())
    if total > OUTFIELD_BUDGET and total > 0:
        drawn[outfield] *= OUTFIELD_BUDGET / total
        drawn = np.clip(drawn, 0.0, 90.0)

    out["m"] = drawn
    out["m_source"] = source
    return out


def build_season_groups(
    merged: pd.DataFrame | None = None,
    *,
    corr_max: float = CORR_MAX,
) -> dict[str, Any]:
    df = merged if merged is not None else load_merged()
    all_groups: list[dict[str, Any]] = []
    for team, club in df.groupby("team", sort=True):
        groups, _meta = find_club_competition_groups(
            club, str(team), corr_max=corr_max
        )
        for g in groups:
            all_groups.append(
                {
                    "team": g.team,
                    "elements": list(g.elements),
                    "names": list(g.names),
                    "positions": list(g.positions),
                    "mean_minutes": list(g.mean_minutes),
                    "min_pair_corr": g.min_pair_corr,
                }
            )
    return {
        "season": "2025-26",
        "corr_max": corr_max,
        "n_groups": len(all_groups),
        "groups": all_groups,
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    lines = [
        "# Sparse minutes coupler — competition groups (2025/26)",
        "",
        f"Pairs with minutes corr ≤ **{doc['corr_max']}** merged into groups. "
        "Draws use Dirichlet×pot inside the group; others independent.",
        "",
        f"- Groups: **{doc['n_groups']}**",
        "",
        "| team | players | positions | min corr | mean minutes |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for g in sorted(doc["groups"], key=lambda x: x["min_pair_corr"]):
        lines.append(
            f"| {g['team']} | {', '.join(g['names'])} | "
            f"{'/'.join(g['positions'])} | {g['min_pair_corr']:+.3f} | "
            f"{', '.join(f'{m:.0f}' for m in g['mean_minutes'])} |"
        )
    lines += [
        "",
        f"JSON: `{(PROCESSED / 'sparse_minutes_groups_2025_26.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corr-max", type=float, default=CORR_MAX)
    ap.add_argument("--merged", type=Path, default=MERGED)
    args = ap.parse_args(argv)
    df = load_merged(args.merged)
    doc = build_season_groups(df, corr_max=args.corr_max)
    out_json = PROCESSED / "sparse_minutes_groups_2025_26.json"
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    write_report(doc, REPORTS / "sparse_minutes_groups_2025_26.md")
    print(json.dumps({"n_groups": doc["n_groups"], "corr_max": doc["corr_max"]}, indent=2))
    print(f"wrote {REPORTS / 'sparse_minutes_groups_2025_26.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
