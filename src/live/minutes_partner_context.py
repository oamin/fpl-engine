"""Within-season anti-correlated teammate blurbs for scout context.

As-of only (``GW < decision_gw``). **No cross-season history.** Does **not**
emit a minutes rating — partner text only (see dig
``reports/minutes_anticorr_role_dig_2025_26.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ANTICORR_MAX = -0.35
MIN_PRIOR_DEFAULT = 5
MIN_PAIR_DEFAULT = 5


@dataclass(frozen=True)
class PartnerBlurb:
    element: int
    name: str
    team: str
    position: str
    anti_element: int
    anti_name: str
    anti_corr: float
    blurb: str


def _hist_frame(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
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
    h = h.dropna(subset=["GW", "element", "team"])
    return (
        h.groupby(["element", "GW", "team", "name", "position"], as_index=False)
        .agg(minutes=("minutes", "sum"))
    )


def _pearson(a: np.ndarray, b: np.ndarray, *, min_n: int) -> float | None:
    if len(a) < min_n:
        return None
    if float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(r):
        return None
    return r


def partner_blurbs_as_of(
    hist: pd.DataFrame,
    *,
    decision_gw: int,
    min_prior_gws: int = MIN_PRIOR_DEFAULT,
    anticorr_max: float = ANTICORR_MAX,
) -> dict[int, PartnerBlurb]:
    """Map element → strongest same-position anti-corr partner blurb (as-of)."""
    h = _hist_frame(hist)
    sub = h.loc[h["GW"] < int(decision_gw)].copy()
    if sub.empty:
        return {}
    gws = sorted(int(x) for x in sub["GW"].unique())
    if len(gws) < int(min_prior_gws):
        return {}
    gw_index = {g: i for i, g in enumerate(gws)}
    n = len(gws)
    min_pair = max(4, min(int(MIN_PAIR_DEFAULT), int(min_prior_gws)))

    meta: dict[int, dict[str, str]] = {}
    minutes: dict[int, np.ndarray] = {}
    for eid, g in sub.groupby("element", sort=False):
        last = g.sort_values("GW").iloc[-1]
        meta[int(eid)] = {
            "name": str(last["name"]),
            "team": str(last["team"]),
            "position": str(last["position"]),
        }
        arr = np.zeros(n, dtype=float)
        for row in g.itertuples(index=False):
            i = gw_index.get(int(row.GW))
            if i is not None:
                arr[i] = float(row.minutes)
        minutes[int(eid)] = arr

    by_team: dict[str, list[int]] = {}
    for eid, info in meta.items():
        by_team.setdefault(info["team"], []).append(eid)

    best: dict[int, tuple[float, int]] = {}
    for _team, eids in by_team.items():
        for i, a in enumerate(eids):
            ma = minutes.get(a)
            if ma is None or len(ma) < int(min_prior_gws):
                continue
            pos_a = meta[a]["position"]
            if pos_a == "GK":
                continue
            for b in eids[i + 1 :]:
                mb = minutes.get(b)
                if mb is None or len(mb) < int(min_prior_gws):
                    continue
                pos_b = meta[b]["position"]
                if pos_b == "GK" or pos_a != pos_b:
                    continue
                r = _pearson(ma, mb, min_n=min_pair)
                if r is None or r > float(anticorr_max):
                    continue
                for src, dst in ((a, b), (b, a)):
                    prev = best.get(src)
                    if prev is None or r < prev[0]:
                        best[src] = (r, dst)

    out: dict[int, PartnerBlurb] = {}
    for eid, (corr, anti_eid) in best.items():
        info = meta[eid]
        anti = meta[anti_eid]
        blurb = (
            f"Minutes anti-correlated with {anti['name']} (ρ={corr:.2f}) — "
            f"when one sits, the other tends to take minutes"
        )
        out[int(eid)] = PartnerBlurb(
            element=int(eid),
            name=info["name"],
            team=info["team"],
            position=info["position"],
            anti_element=int(anti_eid),
            anti_name=anti["name"],
            anti_corr=float(corr),
            blurb=blurb,
        )
    return out


def attach_partner_blurbs_to_overlays(
    overlays: Mapping[int, Any],
    hist: pd.DataFrame,
    *,
    decision_gw: int,
    min_prior_gws: int = MIN_PRIOR_DEFAULT,
) -> dict[str, Any]:
    """Mutate scout overlays: set partner fields; do **not** change m0/rating.

    Appends partner text onto ``context_label`` / ``evidence`` when present.
    """
    partners = partner_blurbs_as_of(
        hist, decision_gw=int(decision_gw), min_prior_gws=int(min_prior_gws)
    )
    n_hit = 0
    for eid, ov in overlays.items():
        pb = partners.get(int(eid))
        if pb is None:
            # clear optional attrs if re-running
            if hasattr(ov, "partner_blurb"):
                ov.partner_blurb = ""
                ov.anti_element = None
                ov.anti_name = None
                ov.anti_corr = None
            continue
        n_hit += 1
        ov.partner_blurb = pb.blurb
        ov.anti_element = pb.anti_element
        ov.anti_name = pb.anti_name
        ov.anti_corr = pb.anti_corr
        evidence = list(getattr(ov, "evidence", None) or [])
        tag = f"partner:{pb.blurb}"
        if tag not in evidence:
            evidence.append(tag)
        ov.evidence = evidence
        label = str(getattr(ov, "context_label", "") or "")
        if pb.blurb and pb.blurb not in label:
            ov.context_label = f"{label}; {pb.blurb}" if label else pb.blurb
    return {
        "n_partner_blurbs_available": len(partners),
        "n_overlays_with_partner": n_hit,
        "min_prior_gws": int(min_prior_gws),
        "anticorr_max": float(ANTICORR_MAX),
        "applies_minutes_rating": False,
    }
