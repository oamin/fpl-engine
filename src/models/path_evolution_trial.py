"""Quick trial: Opta-blend vs natural path-only ClubPath evolution.

No MILP. Compares cross-strand λ / rate fan-out by horizon step.

Modes
-----
- ``opta_blend`` — current ``sample_opta_scenario`` (0.7 Opta + 0.3 path;
  Opta MAE bands forever).
- ``path_only`` — Opta seeds ClubPath with prior strength ``n0``; each
  fixture uses path λ only (lo=hi=centre), rolls ``k ∼ M``, observes.
  No per-fixture Opta re-anchor, no ±15% / z(h).

Reproduce
---------
::

    PYTHONPATH=. python3 -m src.models.path_evolution_trial
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.models.multistage_mc_milp import (
    ClubPath,
    fixtures_by_gw,
    init_paths_from_opta,
    path_match_pots,
    sample_opta_scenario,
    sample_scoreline,
)
from src.models.opta_gf_ga_horizon import league_avg_gf, rank_gf_ga_profile
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

Mode = Literal["opta_blend", "path_only"]


def _init_paths(*, n0: float) -> dict[str, ClubPath]:
    paths = init_paths_from_opta()
    if n0 <= 0:
        return paths
    # Opta rates stay; give them prior strength so match-1 does not wipe them.
    return {
        c: ClubPath(att=p.att, dfn=p.dfn, n=float(n0)) for c, p in paths.items()
    }


def _sample_path_only(
    home: str,
    away: str,
    *,
    paths: dict[str, ClubPath],
    league_gf: float,
    rng: random.Random,
) -> dict[str, Any]:
    pred = path_match_pots(paths, home, away, league_gf=league_gf)
    # Natural: no artificial band — centreline λ, uncertainty via k∼M + path.
    gh, ga, sh, sa = sample_scoreline(
        pred["lam_home"],
        pred["lam_away"],
        lam_h_lo=pred["lam_home"],
        lam_h_hi=pred["lam_home"],
        lam_a_lo=pred["lam_away"],
        lam_a_hi=pred["lam_away"],
        rng=rng,
        centre_blend=1.0,
    )
    return {
        "home": home,
        "away": away,
        "hg": gh,
        "ag": ga,
        "lam_h": sh,
        "lam_a": sa,
    }


def run_strands(
    fixtures_gw: dict[int, list[dict[str, str]]],
    horizon_gws: list[int],
    *,
    mode: Mode,
    n_strands: int,
    seed: int,
    league_gf: float,
    n0: float = 4.0,
    track_club: str = "Arsenal",
) -> dict[str, Any]:
    track = norm_team(track_club)
    # offset → list of (lam_h mean that GW, track att after GW)
    by_offset: dict[int, list[dict[str, float]]] = defaultdict(list)
    rng0 = random.Random(int(seed))

    for s in range(int(n_strands)):
        rng = random.Random(rng0.randint(0, 2**31 - 1))
        paths = _init_paths(n0=n0 if mode == "path_only" else 0.0)
        for oi, gw in enumerate(horizon_gws):
            lams_h: list[float] = []
            for fx in fixtures_gw.get(int(gw), []):
                if fx["home"] not in paths or fx["away"] not in paths:
                    continue
                if mode == "opta_blend":
                    sc = sample_opta_scenario(
                        fx["home"],
                        fx["away"],
                        paths=paths,
                        league_gf=league_gf,
                        rng=rng,
                    )
                else:
                    sc = _sample_path_only(
                        fx["home"],
                        fx["away"],
                        paths=paths,
                        league_gf=league_gf,
                        rng=rng,
                    )
                paths[fx["home"]].observe(sc["hg"], sc["ag"])
                paths[fx["away"]].observe(sc["ag"], sc["hg"])
                lams_h.append(float(sc["lam_h"]))
            if not lams_h:
                continue
            att = float(paths[track].att) if track in paths else float("nan")
            by_offset[oi].append(
                {
                    "mean_lam_h": float(np.mean(lams_h)),
                    "track_att": att,
                    "track_n": float(paths[track].n) if track in paths else 0.0,
                }
            )

    rows = []
    for oi in sorted(by_offset):
        block = by_offset[oi]
        lam = np.array([r["mean_lam_h"] for r in block], dtype=float)
        att = np.array([r["track_att"] for r in block], dtype=float)
        rows.append(
            {
                "offset": oi,
                "gw": int(horizon_gws[oi]),
                "n_strands": len(block),
                "lam_h_mean": float(np.mean(lam)),
                "lam_h_sd": float(np.std(lam, ddof=1)) if len(lam) > 1 else 0.0,
                "track_att_mean": float(np.nanmean(att)),
                "track_att_sd": float(np.nanstd(att, ddof=1)) if len(att) > 1 else 0.0,
                "track_n_mean": float(np.mean([r["track_n"] for r in block])),
            }
        )
    return {
        "mode": mode,
        "n0": float(n0) if mode == "path_only" else 0.0,
        "track_club": track,
        "n_strands": int(n_strands),
        "by_offset": rows,
    }


def write_report(payload: dict[str, Any], path: Path) -> Path:
    blend = payload["opta_blend"]["by_offset"]
    path_only = payload["path_only"]["by_offset"]
    lines = [
        "# Path evolution trial — Opta blend vs path-only",
        "",
        "Diagnostic (no MILP, no likelihood gate). Compares cross-strand fan-out "
        "of match λ and a tracked club's attack rate under:",
        "",
        "1. **opta_blend** — current cascade_long (`0.7 Opta + 0.3 path`, Opta bands).",
        "2. **path_only** — Opta seeds `ClubPath` with prior `n0`; fixtures use path λ "
        "only; `k ∼ M`; observe. No fixture-level Opta re-anchor.",
        "",
        f"- decision_gw={payload['decision_gw']} · horizon "
        f"GW{payload['horizon']['start']}–{payload['horizon']['end']}",
        f"- n_strands={payload['n_strands']} · seed={payload['seed']} · "
        f"n0={payload['path_only']['n0']} · track={payload['path_only']['track_club']}",
        "",
        "## Fan-out by horizon offset",
        "",
        "| off | GW | blend λ SD | path λ SD | blend att SD | path att SD | path n |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b, p in zip(blend, path_only):
        lines.append(
            f"| {b['offset']} | {b['gw']} | {b['lam_h_sd']:.4f} | {p['lam_h_sd']:.4f} | "
            f"{b['track_att_sd']:.4f} | {p['track_att_sd']:.4f} | {p['track_n_mean']:.1f} |"
        )

    def growth(rows: list[dict[str, Any]], key: str) -> str:
        """Last vs first positive value (path λ SD is 0 at offset 0 by design)."""
        first = next((r[key] for r in rows if r[key] > 1e-12), None)
        last = rows[-1][key] if rows else None
        if first is None or last is None or first <= 1e-12:
            return f"{rows[0][key]:.4f} → {rows[-1][key]:.4f}"
        return f"{first:.4f} → {last:.4f} ({last / first:.2f}×)"

    lines += [
        "",
        "## Growth (early → late)",
        "",
        f"| metric | opta_blend | path_only |",
        f"| --- | --- | --- |",
        f"| λ SD | {growth(blend, 'lam_h_sd')} | {growth(path_only, 'lam_h_sd')} |",
        f"| {payload['path_only']['track_club']} att SD | "
        f"{growth(blend, 'track_att_sd')} | {growth(path_only, 'track_att_sd')} |",
        "",
        "## Read",
        "",
        "- **path_only:** λ SD and att SD **rise** with horizon — strand outcomes "
        "compound; no `z(h)` needed.",
        "- **opta_blend:** λ SD stays tiny; att SD **falls** — Opta re-anchor "
        "pulls strands back toward the prior and suppresses fan-out.",
        "- `n0` only protects the Opta seed from a one-match wipe; it is not a "
        "fixture blend weight. `n0=0` wipes att to the first rolled scoreline.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.models.path_evolution_trial --n 40 --n0 4",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision-gw", type=int, default=6)
    p.add_argument("--horizon-end", type=int, default=19)
    p.add_argument("--start-offset", type=int, default=2)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n0", type=float, default=4.0)
    p.add_argument("--track-club", type=str, default="Arsenal")
    p.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT
        / "data/predictions/2026-27/gw06/betfair_20261008/fixtures.json",
    )
    p.add_argument(
        "--bootstrap",
        type=Path,
        default=ROOT
        / "data/predictions/2026-27/gw06/betfair_20261008/bootstrap.json",
    )
    args = p.parse_args()

    bootstrap = json.loads(args.bootstrap.read_text())
    fixtures = json.loads(args.fixtures.read_text())
    team_names = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    start = int(args.decision_gw) + int(args.start_offset)
    end = int(args.horizon_end)
    by_gw = fixtures_by_gw(fixtures, team_names, start=start, end=end)
    horizon_gws = sorted(g for g in by_gw if start <= g <= end)
    if not horizon_gws:
        raise RuntimeError("no fixtures in horizon")

    profile = rank_gf_ga_profile()
    league_gf = league_avg_gf(profile)

    common = dict(
        fixtures_gw=by_gw,
        horizon_gws=horizon_gws,
        n_strands=args.n,
        seed=args.seed,
        league_gf=league_gf,
        n0=args.n0,
        track_club=args.track_club,
    )
    blend = run_strands(**common, mode="opta_blend")
    path_only = run_strands(**common, mode="path_only")

    payload = {
        "task": "path_evolution_trial",
        "decision_gw": int(args.decision_gw),
        "horizon": {"start": start, "end": end, "gws": horizon_gws},
        "n_strands": int(args.n),
        "seed": int(args.seed),
        "opta_blend": blend,
        "path_only": path_only,
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = PROCESSED / "path_evolution_trial.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    report = write_report(payload, REPORTS / "path_evolution_trial.md")

    # Compact stdout summary
    b0, b1 = blend["by_offset"][0], blend["by_offset"][-1]
    p0, p1 = path_only["by_offset"][0], path_only["by_offset"][-1]
    print(
        json.dumps(
            {
                "n_strands": args.n,
                "offsets": len(horizon_gws),
                "n0": args.n0,
                "lam_sd_first_last": {
                    "opta_blend": [round(b0["lam_h_sd"], 4), round(b1["lam_h_sd"], 4)],
                    "path_only": [round(p0["lam_h_sd"], 4), round(p1["lam_h_sd"], 4)],
                },
                "att_sd_first_last": {
                    "opta_blend": [
                        round(b0["track_att_sd"], 4),
                        round(b1["track_att_sd"], 4),
                    ],
                    "path_only": [
                        round(p0["track_att_sd"], 4),
                        round(p1["track_att_sd"], 4),
                    ],
                },
                "report": str(report),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
