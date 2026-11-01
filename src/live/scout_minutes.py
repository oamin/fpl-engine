"""Live scout minutes packet — expanding baseline + sparse FFScout overlays.

Product default (everyone): expanding as-of mean ``m``, flat across the
horizon. Scout treatment is **sparse**: only keyed players (FFScout Out /
Banned / Doubts first) get step overlays. Goal/assist shares stay frozen from
expanding m when scout minutes are applied
(``reports/SCOUT_M_SHARE_FREEZE_LOCK.md``).

FFScout is the first live port of call. It does not yet encode return
horizons — Out/Banned force ``m=0`` at step 0 only; later steps revert to
expanding until structured ``out_for_n_gws`` exists.

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
from src.strategy.layer4_mupost_priced import expanding_minutes_by_element
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "predictions" / "2026-27"
LIVE = ROOT / "data" / "live"
CACHE = ROOT / "data" / "cache"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class ScoutOverlay:
    """One scout override for a player at one or more horizon steps."""

    element: int
    name: str
    club: str
    tag: str
    steps: dict[int, float]
    reason: str
    source: str = "ffscout"
    chance: float | None = None


@dataclass
class ScoutMinutesPacket:
    """Sparse scout minutes on an expanding-mean baseline.

    ``baseline_m[element]`` = expanding as-of mean (GW ``< decision_gw``).
    ``overlays`` keyed by element; each maps horizon step → minutes.
    Missing overlay steps fall back to baseline.
    """

    season: str
    decision_gw: int
    horizon: int
    source: str
    fetched_at: str
    baseline_m: dict[int, float] = field(default_factory=dict)
    overlays: dict[int, ScoutOverlay] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def m_at(self, element: int, step: int = 0) -> float:
        eid = int(element)
        ov = self.overlays.get(eid)
        if ov is not None and int(step) in ov.steps:
            return float(ov.steps[int(step)])
        return float(self.baseline_m.get(eid, 0.0))

    def step_map(self, step: int = 0) -> dict[int, float]:
        """Full element→m map for one horizon step (baseline ⊕ overlays)."""
        out = {int(e): float(m) for e, m in self.baseline_m.items()}
        for eid, ov in self.overlays.items():
            if int(step) in ov.steps:
                out[int(eid)] = float(ov.steps[int(step)])
        return out

    def week0_xmi_map(self) -> dict[int, float]:
        """Step-0 map for live pick / probe CSV (overlay elements only + baseline)."""
        return self.step_map(0)

    def overlay_elements(self) -> list[int]:
        return sorted(int(e) for e in self.overlays)

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
            "baseline_n": len(self.baseline_m),
            "n_overlays": len(self.overlays),
            "baseline_m": {str(k): float(v) for k, v in sorted(self.baseline_m.items())},
            "overlays": {
                str(eid): {
                    **{k: v for k, v in asdict(ov).items() if k != "element"},
                    "element": int(eid),
                    "steps": {str(s): float(m) for s, m in sorted(ov.steps.items())},
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
        steps = {int(s): float(m) for s, m in (raw.get("steps") or {}).items()}
        overlays[eid] = ScoutOverlay(
            element=eid,
            name=str(raw.get("name") or ""),
            club=str(raw.get("club") or ""),
            tag=str(raw.get("tag") or ""),
            steps=steps,
            reason=str(raw.get("reason") or ""),
            source=str(raw.get("source") or "ffscout"),
            chance=(
                float(raw["chance"])
                if raw.get("chance") is not None and raw.get("chance") != ""
                else None
            ),
        )
    baseline = {int(k): float(v) for k, v in (doc.get("baseline_m") or {}).items()}
    return ScoutMinutesPacket(
        season=str(doc.get("season") or "2026-27"),
        decision_gw=int(doc["decision_gw"]),
        horizon=int(doc.get("horizon") or 2),
        source=str(doc.get("source") or "ffscout"),
        fetched_at=str(doc.get("fetched_at") or ""),
        baseline_m=baseline,
        overlays=overlays,
        meta=dict(doc.get("meta") or {}),
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


def baseline_expanding_from_hist(
    hist: pd.DataFrame, *, before_gw: int
) -> dict[int, float]:
    """Expanding as-of mean minutes for GW ``< before_gw`` (product default)."""
    h = _hist_frame(hist)
    h = h.loc[h["GW"] < int(before_gw)]
    return expanding_minutes_by_element(h)


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
                steps={0: 0.0},
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
                steps={0: 0.0},
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
            if eid in overlays and overlays[eid].steps.get(0, 1.0) <= 0.0:
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
                steps={0: float(m_val)},
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
        meta={
            "ffscout": diag,
            "default": "expanding_mean_flat_horizon",
            "overlay_policy": "ffscout_out_banned_doubts_step0_only",
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--from-json", type=Path, default=None)
    ap.add_argument("--from-html", type=Path, default=None)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--bootstrap", type=Path, default=None)
    ap.add_argument("--hist", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--xmi-out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
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

    summary = {
        "gw": gw,
        "horizon": packet.horizon,
        "baseline_n": len(packet.baseline_m),
        "n_overlays": len(packet.overlays),
        "ffscout": packet.meta.get("ffscout"),
        "out": str(out),
        "xmi_out": str(xmi_path) if xmi_path else None,
        "sample_overlays": [
            {
                "element": ov.element,
                "name": ov.name,
                "tag": ov.tag,
                "m0": ov.steps.get(0),
                "reason": ov.reason,
            }
            for ov in list(packet.overlays.values())[:12]
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.smoke:
        # Sanity: Out overlays are 0; baseline has mass; step 1 equals baseline for Out.
        zeros = [
            e
            for e, ov in packet.overlays.items()
            if ov.steps.get(0) == 0.0
        ]
        if not zeros:
            raise SystemExit("smoke: expected some Out/Banned zeros")
        e0 = zeros[0]
        if abs(packet.m_at(e0, 1) - packet.baseline_m.get(e0, 0.0)) > 1e-9:
            raise SystemExit("smoke: step1 should revert to baseline for Out")
        print("smoke_ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
