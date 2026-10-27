"""OOS club-λ screen for candidate ``s=0.768`` on prior closed seasons.

``s=0.768`` was chosen from the 2025-26 XI resid G/CS crossing, so that
season is **selection-contaminated** for XI. The shared L3 XI path also
only exists for 2025-26. This diagnostic checks club λ (Opta − actual) for
``s=0.768`` vs ``s=1.0`` on **2022-23, 2023-24, 2024-25** GW8–38 — seasons
that did not pick ``s``.

Result (2026-10-27): **0/3 PASS** → park fixed ``s``; default ``AWAY_SHRINK``
stays **1.0**. Formula (away-only shrink, no λ clip) kept.

  PYTHONPATH=. python3 -m src.eval.cascade_away_shrink_oos_seasons
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.eval.cascade_long_away_shrink_lam_screen import (
    AWAY_IMPROVE_MIN,
    GW_END,
    GW_START,
    HOME_WORSEN_MAX,
    MAE_WORSEN_MAX,
    N_DRAWS,
    SEED,
    _promote,
    _season_team_sides,
    _summarise,
)
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_away_shrink_oos_seasons.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_away_shrink_oos_seasons.json")

OOS_SEASONS = ("2022-23", "2023-24", "2024-25")
# Locked candidate from 2025-26 XI resid crossing — not the live default.
CANDIDATE = 0.768
BASELINE = 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--candidate", type=float, default=CANDIDATE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    n_draws = int(args.n_draws)
    seed0 = int(args.seed)
    cand = float(args.candidate)
    t0 = time.time()

    print(
        f"[oos] seasons={OOS_SEASONS} GW{GW_START}–{GW_END} "
        f"baseline={BASELINE} candidate={cand} N={n_draws}",
        flush=True,
    )

    by_season: dict[str, Any] = {}
    for i, season in enumerate(OOS_SEASONS):
        print(f"[oos] === {season} ===", flush=True)
        arms: dict[str, Any] = {}
        for name, s in (("baseline_1_00", BASELINE), ("candidate", cand)):
            print(f"[oos] {season} arm={name} s={s}", flush=True)
            df = _season_team_sides(
                season,
                away_shrink=s,
                n_draws=n_draws,
                seed0=seed0 + 1000 * i + (0 if name == "baseline_1_00" else 1),
                with_fd=True,
            )
            opta = _summarise(df, lam_col="opta")
            fd = _summarise(df, lam_col="fd")
            arms[name] = {
                "away_shrink": float(s),
                "opta": opta,
                "fd": fd,
                "n_team_sides": int(opta["n"]),
            }
            print(
                f"  opta bias_A={opta['bias_A']:+.3f} bias_H={opta['bias_H']:+.3f} "
                f"MAE={opta['mae']:.3f} | fd bias_A={fd['bias_A']:+.3f}",
                flush=True,
            )
        dec = _promote(arms["baseline_1_00"]["opta"], arms["candidate"]["opta"])
        by_season[season] = {"arms": arms, "decision": dec}
        print(
            f"[oos] {season} promote={'PASS' if dec['pass'] else 'FAIL'} "
            f"away_improve={dec['away_improve']:+.3f} "
            f"home_worsen={dec['home_worsen']:+.3f} "
            f"mae_worsen={dec['mae_worsen']:+.3f}",
            flush=True,
        )

    # Pooled OOS (concat train seasons) — informative, not a second gate.
    print("[oos] === pooled 2022-23…2024-25 ===", flush=True)
    pooled_arms: dict[str, Any] = {}
    for name, s in (("baseline_1_00", BASELINE), ("candidate", cand)):
        parts = [
            _season_team_sides(
                season,
                away_shrink=s,
                n_draws=n_draws,
                seed0=seed0 + 2000 + 10 * i,
                with_fd=False,
            )
            for i, season in enumerate(OOS_SEASONS)
        ]
        import pandas as pd

        df = pd.concat(parts, ignore_index=True)
        opta = _summarise(df, lam_col="opta")
        pooled_arms[name] = {"away_shrink": float(s), "opta": opta}
        print(
            f"  pooled {name} bias_A={opta['bias_A']:+.3f} "
            f"bias_H={opta['bias_H']:+.3f} MAE={opta['mae']:.3f}",
            flush=True,
        )
    pooled_dec = _promote(
        pooled_arms["baseline_1_00"]["opta"], pooled_arms["candidate"]["opta"]
    )

    n_pass = sum(1 for b in by_season.values() if b["decision"]["pass"])
    state = {
        "task": "cascade_away_shrink_oos_seasons",
        "candidate": cand,
        "baseline": BASELINE,
        "default_away_shrink": 1.0,
        "verdict": "reject_fixed_s_keep_default_1_0",
        "selection_note": (
            "s=0.768 chosen from 2025-26 XI resid crossing; "
            "OOS club-λ on 2022-23…2024-25 only (no shared L3 XI on those seasons)"
        ),
        "oos_seasons": list(OOS_SEASONS),
        "gw_range": [GW_START, GW_END],
        "n_draws": n_draws,
        "seed": seed0,
        "by_season": by_season,
        "pooled": {"arms": pooled_arms, "decision": pooled_dec},
        "n_pass": n_pass,
        "n_seasons": len(OOS_SEASONS),
        "elapsed_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# OOS club-λ screen: away_shrink=0.768 on prior seasons",
        "",
        f"Candidate **s={cand}** (PI default from 2025-26 XI resid crossing) "
        f"vs **s=1.0** · GW{GW_START}–{GW_END} · N={n_draws} · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## Why these seasons",
        "",
        "- `s=0.768` was read from **2025-26** XI resid G/CS → that season is "
        "selection-contaminated for XI.",
        "- Shared L3 XI seed/path exists only for 2025-26.",
        "- Club λ on **2022-23 / 2023-24 / 2024-25** is the clean OOS check.",
        "",
        f"Promote (per season vs that season's s=1.0): away |bias| ↓ ≥"
        f"{AWAY_IMPROVE_MIN}, home |bias| ↑ ≤{HOME_WORSEN_MAX}, "
        f"MAE ↑ ≤{MAE_WORSEN_MAX}.",
        "",
        "## Per-season holdout (Opta λ − actual)",
        "",
        "| season | s=1.0 bias A | s=1.0 bias H | s=0.768 bias A | "
        "s=0.768 bias H | MAE Δ | promote |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for season, block in by_season.items():
        b = block["arms"]["baseline_1_00"]["opta"]
        c = block["arms"]["candidate"]["opta"]
        d = block["decision"]
        lines.append(
            f"| {season} | {b['bias_A']:+.3f} | {b['bias_H']:+.3f} | "
            f"{c['bias_A']:+.3f} | {c['bias_H']:+.3f} | "
            f"{d['mae_worsen']:+.3f} | "
            f"{'PASS' if d['pass'] else 'FAIL'} |"
        )
    pb = pooled_arms["baseline_1_00"]["opta"]
    pc = pooled_arms["candidate"]["opta"]
    lines += [
        "",
        f"**Per-season PASS count:** {n_pass}/{len(OOS_SEASONS)}",
        "",
        "## Pooled 2022-23…2024-25",
        "",
        f"- baseline bias A/H/MAE = {pb['bias_A']:+.3f} / {pb['bias_H']:+.3f} / "
        f"{pb['mae']:.3f}",
        f"- s={cand} bias A/H/MAE = {pc['bias_A']:+.3f} / {pc['bias_H']:+.3f} / "
        f"{pc['mae']:.3f}",
        f"- promote: **{'PASS' if pooled_dec['pass'] else 'FAIL'}** "
        f"(away_improve={pooled_dec['away_improve']:+.3f}, "
        f"home_worsen={pooled_dec['home_worsen']:+.3f}, "
        f"mae_worsen={pooled_dec['mae_worsen']:+.3f})",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_away_shrink_oos_seasons",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_away_shrink_oos_seasons",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": f"OOS club-λ screen s={cand} on 2022-23…2024-25",
                "verdict": {
                    "candidate": cand,
                    "n_pass": n_pass,
                    "n_seasons": len(OOS_SEASONS),
                    "pooled_pass": pooled_dec["pass"],
                    "by_season_pass": {
                        s: by_season[s]["decision"]["pass"] for s in OOS_SEASONS
                    },
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_pass": n_pass,
                "pooled_pass": pooled_dec["pass"],
                "by_season": {
                    s: by_season[s]["decision"]["pass"] for s in OOS_SEASONS
                },
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
