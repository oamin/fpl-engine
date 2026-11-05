"""Live scout minutes packet — expanding baseline + sparse FFScout overlays.

Product default (everyone): expanding as-of mean ``m``, flat across the
horizon. Scout treatment is **sparse**: only keyed players (FFScout Out /
Banned / Doubts / XI-omit) get a minutes pin for the **known horizon**,
then **snap** back to expanding mean (no ρ-decay):

  m(h) = force_steps[h] if pinned
       = m0             if h == 0 or h < return_gws
       = expand         otherwise

Structured ``out_for_n_gws`` / return parses hard-pin early steps via
``force_steps``; after that horizon xmi is expanding ``m``.

Goal/assist shares stay frozen from expanding m
(``reports/SCOUT_M_SHARE_FREEZE_LOCK.md``).

  PYTHONPATH=. python3 -m src.live.scout_minutes --gw 9 --from-json \\
    data/scratch/ffscout_gw09/ffscout_team_news.json --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.live.ffscout_team_news import match_ffs_name_to_fpl, parse_team_news, pull_team_news
from src.live.news_tags import minutes_for_tag, prior_minutes
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "predictions" / "2026-27"
LIVE = ROOT / "data" / "live"
CACHE = ROOT / "data" / "cache"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class ScoutOverlay:
    """Scout minutes anchor at decision week (step 0), with optional hard pins."""

    element: int
    name: str
    club: str
    tag: str
    m0: float
    reason: str
    source: str = "ffscout"
    chance: float | None = None
    # Hard minutes for known-horizon steps (e.g. out_for_n_gws); else snap to expand.
    force_steps: dict[int, float] = field(default_factory=dict)
    # Combined FFScout + FPL + blurb context (see ``scout_context``).
    context: str = ""
    context_label: str = ""
    context_detail: str = ""
    return_gws: int | None = None
    return_date: str | None = None
    evidence: list[str] = field(default_factory=list)
    # Within-season anti-corr teammate blurb only (no minutes rating).
    partner_blurb: str = ""
    anti_element: int | None = None
    anti_name: str | None = None
    anti_corr: float | None = None
    # Closed role tag (see ``scout_role.ROLE_TAGS``).
    role: str = ""
    role_reason: str = ""


@dataclass
class ScoutMinutesPacket:
    """Sparse scout minutes on an expanding-mean baseline.

    ``baseline_m[element]`` = expanding as-of mean (GW ``< decision_gw``).
    ``overlays[element]`` pins minutes for the known horizon (``force_steps`` /
    ``return_gws`` / step-0 ``m0``); after that horizon ``m_at`` snaps to
    expanding baseline — no ρ-decay.
    ``roles`` covers the full as-of pool (firm starters included, not only overlays).
    """

    season: str
    decision_gw: int
    horizon: int
    source: str
    fetched_at: str
    baseline_m: dict[int, float] = field(default_factory=dict)
    overlays: dict[int, ScoutOverlay] = field(default_factory=dict)
    roles: dict[int, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def m_at(self, element: int, step: int = 0) -> float:
        eid = int(element)
        base = float(self.baseline_m.get(eid, 0.0))
        ov = self.overlays.get(eid)
        if ov is None:
            return base
        h = max(0, int(step))
        if h in ov.force_steps:
            return float(ov.force_steps[h])
        # Known return horizon without an explicit pin for this step.
        if ov.return_gws is not None and h < int(ov.return_gws):
            return float(ov.m0)
        # Step-0 scout anchor (doubts / unknown-return Out with empty pins).
        if h == 0:
            return float(ov.m0)
        # Past known horizon → snap to expanding mean.
        return base

    def step_map(self, step: int = 0) -> dict[int, float]:
        """Full element→m map for one horizon step (baseline ⊕ scout pins)."""
        out = {int(e): float(m) for e, m in self.baseline_m.items()}
        for eid in self.overlays:
            out[int(eid)] = self.m_at(int(eid), int(step))
        return out

    def week0_xmi_map(self) -> dict[int, float]:
        """Step-0 map for live pick / probe CSV."""
        return self.step_map(0)

    def overlay_elements(self) -> list[int]:
        return sorted(int(e) for e in self.overlays)

    def minutes_path(self, element: int) -> list[float]:
        """Minutes path length ``horizon`` for one element (pin then snap)."""
        return [self.m_at(int(element), h) for h in range(max(1, int(self.horizon)))]

    def decayed_path(self, element: int) -> list[float]:
        """Alias for ``minutes_path`` (legacy name; no ρ-decay)."""
        return self.minutes_path(element)

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "decision_gw": int(self.decision_gw),
            "horizon": int(self.horizon),
            "source": self.source,
            "fetched_at": self.fetched_at,
            "protocol": "scout_minutes_ffscout_v1",
            "share_policy": "share_freeze",
            "lock": "reports/SCOUT_M_SHARE_FREEZE_LOCK.md",
            "horizon_law": "pin_then_snap_expand",
            "horizon_law_detail": (
                "m(h)=force_steps[h] | m0 if h<return_gws or h==0 | else expand"
            ),
            "baseline_n": len(self.baseline_m),
            "n_overlays": len(self.overlays),
            "n_roles": len(self.roles),
            "baseline_m": {str(k): float(v) for k, v in sorted(self.baseline_m.items())},
            "roles": {
                str(eid): dict(row) for eid, row in sorted(self.roles.items())
            },
            "overlays": {
                str(eid): {
                    "element": int(eid),
                    "name": ov.name,
                    "club": ov.club,
                    "tag": ov.tag,
                    "role": ov.role,
                    "role_reason": ov.role_reason,
                    "m0": float(ov.m0),
                    "reason": ov.reason,
                    "source": ov.source,
                    "chance": ov.chance,
                    "force_steps": {
                        str(s): float(m) for s, m in sorted(ov.force_steps.items())
                    },
                    "context": ov.context,
                    "context_label": ov.context_label,
                    "context_detail": ov.context_detail,
                    "return_gws": ov.return_gws,
                    "return_date": ov.return_date,
                    "evidence": list(ov.evidence),
                    "partner_blurb": ov.partner_blurb,
                    "anti_element": ov.anti_element,
                    "anti_name": ov.anti_name,
                    "anti_corr": ov.anti_corr,
                    "path": self.minutes_path(eid),
                }
                for eid, ov in sorted(self.overlays.items())
            },
            "meta": self.meta,
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def write_week0_xmi_csv(self, path: Path) -> Path:
        """Write step-0 xmi CSV compatible with ``numagent_live_pick`` / ``load_xmi``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"player_id": int(e), "gw": int(self.decision_gw), "xmi": float(m)}
            for e, m in sorted(self.week0_xmi_map().items())
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


def load_scout_packet(path: Path) -> ScoutMinutesPacket:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    overlays: dict[int, ScoutOverlay] = {}
    for key, raw in (doc.get("overlays") or {}).items():
        eid = int(raw.get("element") or key)
        force = {
            int(s): float(m) for s, m in (raw.get("force_steps") or {}).items()
        }
        # Backward compat: old packets stored steps={0: m0} only.
        legacy_steps = {
            int(s): float(m) for s, m in (raw.get("steps") or {}).items()
        }
        if "m0" in raw and raw["m0"] is not None:
            m0 = float(raw["m0"])
        elif 0 in legacy_steps:
            m0 = float(legacy_steps[0])
            force = {**legacy_steps, **force}
        else:
            m0 = 0.0
        overlays[eid] = ScoutOverlay(
            element=eid,
            name=str(raw.get("name") or ""),
            club=str(raw.get("club") or ""),
            tag=str(raw.get("tag") or ""),
            m0=m0,
            reason=str(raw.get("reason") or ""),
            source=str(raw.get("source") or "ffscout"),
            chance=(
                float(raw["chance"])
                if raw.get("chance") is not None and raw.get("chance") != ""
                else None
            ),
            force_steps=force,
            context=str(raw.get("context") or ""),
            context_label=str(raw.get("context_label") or ""),
            context_detail=str(raw.get("context_detail") or ""),
            return_gws=(
                int(raw["return_gws"])
                if raw.get("return_gws") is not None and raw.get("return_gws") != ""
                else None
            ),
            return_date=(
                str(raw.get("return_date"))
                if raw.get("return_date") not in (None, "")
                else None
            ),
            evidence=[str(x) for x in (raw.get("evidence") or [])],
            partner_blurb=str(raw.get("partner_blurb") or ""),
            anti_element=(
                int(raw["anti_element"])
                if raw.get("anti_element") is not None and raw.get("anti_element") != ""
                else None
            ),
            anti_name=(
                str(raw.get("anti_name"))
                if raw.get("anti_name") not in (None, "")
                else None
            ),
            anti_corr=(
                float(raw["anti_corr"])
                if raw.get("anti_corr") is not None and raw.get("anti_corr") != ""
                else None
            ),
            role=str(raw.get("role") or ""),
            role_reason=str(raw.get("role_reason") or ""),
        )
    baseline = {int(k): float(v) for k, v in (doc.get("baseline_m") or {}).items()}
    roles_raw = doc.get("roles") or {}
    roles: dict[int, dict[str, Any]] = {}
    for key, raw in roles_raw.items():
        eid = int(raw.get("element") or key)
        roles[eid] = dict(raw)
        roles[eid]["element"] = eid
    meta = dict(doc.get("meta") or {})
    # Legacy packets may still carry decay_rho; ignored under pin-then-snap.
    meta.pop("decay_rho", None)
    meta.pop("decay_law", None)
    return ScoutMinutesPacket(
        season=str(doc.get("season") or "2026-27"),
        decision_gw=int(doc["decision_gw"]),
        horizon=int(doc.get("horizon") or 2),
        source=str(doc.get("source") or "ffscout"),
        fetched_at=str(doc.get("fetched_at") or ""),
        baseline_m=baseline,
        overlays=overlays,
        roles=roles,
        meta=meta,
    )


def _hist_frame(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    if "GW" not in h.columns and "gw" in h.columns:
        h["GW"] = h["gw"]
    if "element" not in h.columns and "player_id" in h.columns:
        h["element"] = h["player_id"]
    h["GW"] = pd.to_numeric(h["GW"], errors="coerce")
    h["element"] = pd.to_numeric(h["element"], errors="coerce")
    h["minutes"] = pd.to_numeric(h.get("minutes"), errors="coerce").fillna(0.0)
    return h.dropna(subset=["GW", "element"])


def _expanding_minutes_by_element(hist: pd.DataFrame) -> dict[int, float]:
    """As-of expanding mean minutes (zeros included). Local to avoid strategy import."""
    out: dict[int, float] = {}
    for eid, g in hist.groupby("element", sort=False):
        by_gw = g.groupby("GW", sort=True)["minutes"].sum()
        if len(by_gw) == 0:
            continue
        out[int(eid)] = float(by_gw.mean())
    return out


def baseline_expanding_from_hist(
    hist: pd.DataFrame, *, before_gw: int
) -> dict[int, float]:
    """Expanding as-of mean minutes for GW ``< before_gw`` (product default)."""
    h = _hist_frame(hist)
    h = h.loc[h["GW"] < int(before_gw)]
    return _expanding_minutes_by_element(h)


def _series_by_element(hist: pd.DataFrame, *, before_gw: int) -> dict[int, list[float]]:
    h = _hist_frame(hist)
    h = h.loc[h["GW"] < int(before_gw)]
    out: dict[int, list[float]] = {}
    for eid, g in h.groupby("element", sort=False):
        by_gw = g.sort_values("GW").groupby("GW", sort=True)["minutes"].sum()
        out[int(eid)] = [float(v) for v in by_gw.tolist()]
    return out


def _bootstrap_by_club(bootstrap: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    by: dict[str, list[dict[str, Any]]] = {}
    for el in bootstrap["elements"]:
        club = teams.get(int(el["team"]), "")
        key = norm_team(club)
        by.setdefault(key, []).append(el)
    return by


def _parse_chance(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def overlays_from_ffscout(
    ffs_payload: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
    hist: pd.DataFrame,
    decision_gw: int,
) -> tuple[dict[int, ScoutOverlay], dict[str, Any]]:
    """Build sparse step-0 overlays from FFScout Out / Banned / Doubts only.

    Predicted XI / firm starters stay on expanding baseline (not scout-keyed
    unless also listed Out/Doubt/Banned).
    """
    by_club = _bootstrap_by_club(bootstrap)
    series = _series_by_element(hist, before_gw=int(decision_gw))
    els = {int(e["id"]): e for e in bootstrap["elements"]}
    overlays: dict[int, ScoutOverlay] = {}
    unmatched: list[dict[str, Any]] = []
    n_out = n_banned = n_doubt = 0

    def _resolve(name: str, club_norm: str) -> dict[str, Any] | None:
        # Common FFScout ↔ FPL spelling gaps (public page transliteration).
        aliases = {
            "yarmolyuk": "Yarmoliuk",
            "miley": "L.Miley",  # Lewis; Mason is rare
        }
        key = str(name).strip().lower()
        resolve_name = aliases.get(key, name)
        cands = by_club.get(club_norm) or []
        hit = match_ffs_name_to_fpl(resolve_name, cands)
        if hit is not None:
            return hit
        if resolve_name != name:
            hit = match_ffs_name_to_fpl(name, cands)
            if hit is not None:
                return hit
        # Fallback: all clubs (loan / name clash rare)
        flat = [e for xs in by_club.values() for e in xs]
        return match_ffs_name_to_fpl(resolve_name, flat)

    for club in ffs_payload.get("clubs") or []:
        club_name = str(club.get("club") or "")
        club_norm = str(club.get("club_norm") or norm_team(club_name))

        for name in club.get("out") or []:
            el = _resolve(str(name), club_norm)
            if el is None:
                unmatched.append({"club": club_name, "name": name, "bucket": "out"})
                continue
            eid = int(el["id"])
            overlays[eid] = ScoutOverlay(
                element=eid,
                name=str(el.get("web_name") or name),
                club=club_name,
                tag="injured",
                m0=0.0,
                reason=f"FFScout Out: {name}",
                source="ffscout",
                chance=0.0,
            )
            n_out += 1

        for name in club.get("banned") or []:
            el = _resolve(str(name), club_norm)
            if el is None:
                unmatched.append({"club": club_name, "name": name, "bucket": "banned"})
                continue
            eid = int(el["id"])
            # Banned wins over Out if both listed.
            overlays[eid] = ScoutOverlay(
                element=eid,
                name=str(el.get("web_name") or name),
                club=club_name,
                tag="injured",
                m0=0.0,
                reason=f"FFScout Banned: {name}",
                source="ffscout",
                chance=0.0,
            )
            n_banned += 1

        for d in club.get("doubts") or []:
            if isinstance(d, dict):
                name = str(d.get("name") or "")
                chance = _parse_chance(d.get("chance"))  # type: ignore[arg-type]
            else:
                name = str(d)
                chance = None
            if not name:
                continue
            el = _resolve(name, club_norm)
            if el is None:
                unmatched.append({"club": club_name, "name": name, "bucket": "doubt"})
                continue
            eid = int(el["id"])
            if eid in overlays and float(overlays[eid].m0) <= 0.0:
                # Already Out/Banned — keep zero.
                continue
            pos = POS.get(int(el.get("element_type") or 3), "MID")
            prior = prior_minutes(series.get(eid) or [])
            # Chance-scaled injury tag; missing chance → treat as 50/50.
            if chance is None:
                m_val = minutes_for_tag("50/50", pos, prior, None, "d")
            else:
                m_val = minutes_for_tag("injured", pos, prior, float(chance), "d")
            if m_val is None:
                continue
            overlays[eid] = ScoutOverlay(
                element=eid,
                name=str(el.get("web_name") or name),
                club=club_name,
                tag="injured" if chance is not None else "50/50",
                m0=float(m_val),
                reason=f"FFScout Doubt: {name}"
                + (f" ({chance:.0f}%)" if chance is not None else ""),
                source="ffscout",
                chance=chance,
            )
            n_doubt += 1

    diag = {
        "n_out": n_out,
        "n_banned": n_banned,
        "n_doubt": n_doubt,
        "n_overlays": len(overlays),
        "n_unmatched": len(unmatched),
        "unmatched": unmatched[:40],
        "els_known": len(els),
    }
    return overlays, diag


def build_scout_packet_from_ffscout(
    *,
    decision_gw: int,
    hist: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    ffs_payload: Mapping[str, Any] | None = None,
    ffs_json: Path | None = None,
    ffs_html: Path | None = None,
    fetch: bool = False,
    horizon: int = 2,
    season: str = "2026-27",
    fetch_out_dir: Path | None = None,
) -> ScoutMinutesPacket:
    """Assemble expanding baseline + FFScout sparse overlays for a live GW."""
    if ffs_payload is None:
        if ffs_json is not None and Path(ffs_json).is_file():
            ffs_payload = json.loads(Path(ffs_json).read_text(encoding="utf-8"))
        elif ffs_html is not None and Path(ffs_html).is_file():
            clubs = parse_team_news(Path(ffs_html).read_text(encoding="utf-8"))
            from dataclasses import asdict as _asdict

            ffs_payload = {
                "source": "ffscout_html",
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "clubs": [_asdict(c) for c in clubs],
            }
        elif fetch:
            ffs_payload = pull_team_news(out_dir=fetch_out_dir)
        else:
            raise FileNotFoundError(
                "need --from-json / --from-html / --fetch for FFScout payload"
            )

    baseline = baseline_expanding_from_hist(hist, before_gw=int(decision_gw))
    overlays, diag = overlays_from_ffscout(
        ffs_payload,
        bootstrap=bootstrap,
        hist=hist,
        decision_gw=int(decision_gw),
    )
    from src.live.minutes_partner_context import attach_partner_blurbs_to_overlays
    from src.live.scout_context import enrich_overlays_with_context

    series = _series_by_element(hist, before_gw=int(decision_gw))
    overlays, context_rows, ctx_diag = enrich_overlays_with_context(
        overlays,
        ffs_payload=ffs_payload,
        bootstrap=bootstrap,
        decision_gw=int(decision_gw),
        baseline_m=baseline,
        series_by_element=series,
        include_xi_omissions=True,
    )
    # Partner blurbs: within-season anti-corr text only — never a minutes rating.
    min_prior = max(5, min(8, int(decision_gw) - 1))
    partner_diag = attach_partner_blurbs_to_overlays(
        overlays,
        hist,
        decision_gw=int(decision_gw),
        min_prior_gws=min_prior,
    )
    from src.live.scout_role import ROLE_TAGS, attach_roles

    club_blurbs = {
        str(c.get("club_norm") or norm_team(str(c.get("club") or ""))): str(
            c.get("news") or ""
        )
        for c in (ffs_payload.get("clubs") or [])
    }
    role_rows, role_counts = attach_roles(
        bootstrap=bootstrap,
        series_by_element=series,
        baseline_m=baseline,
        overlays=overlays,
        club_blurbs=club_blurbs,
    )
    roles_out = {
        eid: {
            "element": int(eid),
            "name": row.name,
            "club": row.club,
            "role": row.role,
            "role_reason": row.reason,
            "expand_m": round(row.features.expand_m, 2),
            "start60_rate": round(row.features.start60_rate, 3),
            "early_sub_rate": round(row.features.early_sub_rate, 3),
        }
        for eid, row in role_rows.items()
    }
    return ScoutMinutesPacket(
        season=season,
        decision_gw=int(decision_gw),
        horizon=int(horizon),
        source=str(ffs_payload.get("source") or "ffscout"),
        fetched_at=str(
            ffs_payload.get("fetched_at")
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
        baseline_m=baseline,
        overlays=overlays,
        roles=roles_out,
        meta={
            "ffscout": diag,
            "context": ctx_diag,
            "partner_context": partner_diag,
            "role_counts": role_counts,
            "role_tags": list(ROLE_TAGS),
            "context_samples": [
                {
                    "element": r.element,
                    "name": r.name,
                    "club": r.club,
                    "bucket": r.bucket,
                    "context_label": r.human_label(),
                    "return_gws": r.return_gws,
                    "return_date": r.return_date,
                    "m0": r.m0,
                    "evidence": r.evidence[:4],
                }
                for r in sorted(
                    context_rows,
                    key=lambda x: (
                        0 if x.return_gws is not None else 1,
                        x.bucket,
                        x.name,
                    ),
                )[:40]
            ],
            "default": "expanding_mean_flat_horizon",
            "overlay_policy": (
                "ffscout_out_banned_doubts_plus_xi_omit_rotation;"
                "context=fpl_news+blurb+xi;"
                "partner_blurb=within_season_anticorr_no_rating;"
                "role=scout_role_v1"
            ),
            "horizon_law": "pin_then_snap_expand",
            "share_policy": "share_freeze",
        },
    )


def load_live_hist(decision_gw: int) -> pd.DataFrame:
    """Prefer cache hist; require rows with GW < decision_gw."""
    path = CACHE / "player_gw_2026_27.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    hist = pd.read_csv(path)
    played = int(pd.to_numeric(hist["gw"], errors="coerce").max())
    if played < int(decision_gw) - 1:
        raise RuntimeError(
            f"hist only through GW{played}; need GW < {decision_gw} for scout baseline"
        )
    return hist


def load_bootstrap(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else LIVE / "bootstrap.json"
    return json.loads(p.read_text(encoding="utf-8"))


def default_packet_path(gw: int) -> Path:
    return PRED / f"gw{int(gw):02d}" / "scout_minutes.json"


def _write_example_md(packet: ScoutMinutesPacket, path: Path) -> None:
    """Human-readable inspect table for context labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        packet.overlays.values(),
        key=lambda ov: (
            0 if ov.return_gws is not None else 1,
            ov.context,
            ov.club,
            ov.name,
        ),
    )
    lines = [
        f"# Scout context example — GW{packet.decision_gw}",
        "",
        f"Source: `{packet.source}` · fetched `{packet.fetched_at}` · "
        f"law=`pin_then_snap_expand` · overlays **{len(packet.overlays)}** · "
        f"roles **{len(packet.roles)}**",
        "",
        "Combined signals: FFScout Out/Banned/Doubts + predicted-XI omissions "
        "+ FPL `news`/status + club Latest News blurb parse + role tags.",
        "",
        "## Role tags (full pool)",
        "",
    ]
    for k, v in sorted(((packet.meta or {}).get("role_counts") or {}).items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Context counts", ""]
    ctx = (packet.meta or {}).get("context") or {}
    for k, v in (ctx.get("context_counts") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        f"- with `return_gws`: {ctx.get('n_with_return_gws')}",
        f"- with blurb evidence: {ctx.get('n_with_blurb_evidence')}",
        f"- with FPL news: {ctx.get('n_with_fpl_news')}",
        "",
        "## Overlay table (all)",
        "",
        "| player | club | role | bucket/tag | context | partner | return_gws | m0 | force |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for ov in rows:
        force = ",".join(f"{h}:{m:g}" for h, m in sorted(ov.force_steps.items())) or "—"
        partner = (ov.anti_name or "—") + (
            f" (ρ={ov.anti_corr:.2f})" if ov.anti_corr is not None else ""
        )
        ctx_l = (ov.context_label or ov.context or "").replace("|", "/")
        if len(ctx_l) > 80:
            ctx_l = ctx_l[:77] + "…"
        lines.append(
            f"| {ov.name} | {ov.club} | {ov.role or '—'} | {ov.tag}/{ov.source} | "
            f"{ctx_l} | {partner} | "
            f"{'' if ov.return_gws is None else ov.return_gws} | "
            f"{ov.m0:.0f} | {force} |"
        )
    lines += ["", "## Reproduce", "", "```bash"]
    lines.append(
        f"PYTHONPATH=. python3 -m src.live.scout_minutes --gw {packet.decision_gw} "
        f"--horizon {packet.horizon} --example-md {path}"
    )
    lines += ["```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--from-json", type=Path, default=None)
    ap.add_argument("--from-html", type=Path, default=None)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--bootstrap", type=Path, default=None)
    ap.add_argument("--hist", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--xmi-out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--example-md",
        type=Path,
        default=None,
        help="Write a short markdown example table for inspection",
    )
    args = ap.parse_args(argv)

    gw = int(args.gw)
    hist = pd.read_csv(args.hist) if args.hist else load_live_hist(gw)
    boot = load_bootstrap(args.bootstrap)
    fetch_dir = PRED / f"gw{gw:02d}" / "ffscout" if args.fetch else None
    packet = build_scout_packet_from_ffscout(
        decision_gw=gw,
        hist=hist,
        bootstrap=boot,
        ffs_json=args.from_json,
        ffs_html=args.from_html,
        fetch=bool(args.fetch),
        horizon=int(args.horizon),
        season=str(args.season),
        fetch_out_dir=fetch_dir,
    )
    out = args.out or default_packet_path(gw)
    packet.write_json(out)
    xmi_path = args.xmi_out
    if xmi_path is None and not args.smoke:
        xmi_path = PRED / f"gw{gw:02d}" / "scout_xmi_week0.csv"
    if xmi_path is not None:
        packet.write_week0_xmi_csv(xmi_path)

    if args.example_md is not None:
        _write_example_md(packet, Path(args.example_md))

    summary = {
        "gw": gw,
        "horizon": packet.horizon,
        "horizon_law": "pin_then_snap_expand",
        "baseline_n": len(packet.baseline_m),
        "n_overlays": len(packet.overlays),
        "ffscout": packet.meta.get("ffscout"),
        "context": packet.meta.get("context"),
        "out": str(out),
        "xmi_out": str(xmi_path) if xmi_path else None,
        "sample_overlays": [
            {
                "element": ov.element,
                "name": ov.name,
                "tag": ov.tag,
                "m0": ov.m0,
                "context_label": ov.context_label,
                "return_gws": ov.return_gws,
                "force_steps": ov.force_steps,
                "path": packet.minutes_path(ov.element)[:6],
                "reason": ov.reason,
            }
            for ov in list(packet.overlays.values())[:12]
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.smoke:
        zeros = [
            e
            for e, ov in packet.overlays.items()
            if float(ov.m0) <= 0.0 and not ov.force_steps
        ]
        if not zeros:
            zeros = [e for e, ov in packet.overlays.items() if float(ov.m0) <= 0.0]
        if not zeros:
            raise SystemExit("smoke: expected some Out/Banned zeros")
        e0 = zeros[0]
        base = float(packet.baseline_m.get(e0, 0.0))
        m0 = packet.m_at(e0, 0)
        m1 = packet.m_at(e0, 1)
        m2 = packet.m_at(e0, 2)
        if abs(m0) > 1e-9:
            raise SystemExit("smoke: Out m0 should be 0")
        ov0 = packet.overlays[e0]
        # Pin then snap: after known horizon (or step 0 only), m → expand.
        if ov0.force_steps:
            last = max(ov0.force_steps)
            after = packet.m_at(e0, last + 1)
            if abs(after - base) > 1e-6:
                raise SystemExit(
                    f"smoke: expected snap to expand after pin; "
                    f"got m({last + 1})={after:.2f} base={base:.2f}"
                )
        elif base > 1.0:
            if abs(m1 - base) > 1e-6 or abs(m2 - base) > 1e-6:
                raise SystemExit(
                    f"smoke: expected snap to expand after step 0; "
                    f"got m=[{m0:.2f},{m1:.2f},{m2:.2f}] base={base:.2f}"
                )
        # Cherki-style single-week pin: m0 then expand.
        single = [
            e
            for e, ov in packet.overlays.items()
            if ov.return_gws == 1 and 0 in ov.force_steps
        ]
        if single:
            e1 = single[0]
            b1 = float(packet.baseline_m.get(e1, 0.0))
            p0 = packet.m_at(e1, 0)
            p1 = packet.m_at(e1, 1)
            if abs(p0 - float(packet.overlays[e1].m0)) > 1e-6:
                raise SystemExit("smoke: step-0 pin mismatch")
            if abs(p1 - b1) > 1e-6:
                raise SystemExit(
                    f"smoke: single-week should snap at step 1; "
                    f"got {p1:.2f} vs expand {b1:.2f}"
                )
        if not packet.roles:
            raise SystemExit("smoke: expected full-pool roles map")
        firm = [r for r in packet.roles.values() if r.get("role") == "firm_starter"]
        if len(firm) < 5:
            raise SystemExit(f"smoke: too few firm_starters ({len(firm)})")
        print("smoke_ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
