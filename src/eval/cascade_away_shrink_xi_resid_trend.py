"""Opta XI resid G/CS vs ``away_shrink`` s ∈ [0.50, 1.00].

Same shared-L3-XI decision-GW screen as
``cascade_vs_l3_component_xp`` (π∝r, Opta λ, GW8–38, N=40). Runs one
``s`` per worker (default 4), then plots season Σ(casc−P) for G and CS.

  PYTHONPATH=. python3 -m src.eval.cascade_away_shrink_xi_resid_trend
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_away_shrink_xi_resid_trend.md"
STATE = ROOT / "REPORT_STATE.json"
ART = Path("/opt/cursor/artifacts")
OUT = ART / "cascade_away_shrink_xi_resid_trend.json"
PLOT = ART / "cascade_away_shrink_xi_resid_trend.png"

STEPS = ",".join(str(g) for g in range(8, 39))
N_STRANDS = 40
SEED = 7
GRID_DEFAULT = sorted(
    {
        *[round(x, 2) for x in np.arange(0.50, 1.01, 0.05)],
        0.86,  # train-fit from away_shrink λ screen
    }
)


def _row_from_payload(s: float, payload: dict[str, Any], *, elapsed: float, artifact: str, log: str | None = None) -> dict[str, Any]:
    summ = payload["summary"]
    return {
        "away_shrink": float(s),
        "resid_G": float(summ["sum_casc_xP_G"] - summ["sum_P_G"]),
        "resid_CS": float(summ["sum_casc_xP_CS"] - summ["sum_P_CS"]),
        "resid_GA": float(summ["sum_casc_xP_GA"] - summ["sum_P_GA"]),
        "sum_casc_xP_G": float(summ["sum_casc_xP_G"]),
        "sum_casc_xP_CS": float(summ["sum_casc_xP_CS"]),
        "sum_P_G": float(summ["sum_P_G"]),
        "sum_P_CS": float(summ["sum_P_CS"]),
        "mae_casc_G": float(summ["mae_casc_G"]),
        "mae_casc_CS": float(summ["mae_casc_CS"]),
        "elapsed_s": float(elapsed),
        "artifact": artifact,
        "log": log,
    }


def _reuse_paths(s: float, out_dir: Path) -> Path | None:
    tag = f"as{str(s).replace('.', '')}"
    cand = [
        out_dir / f"cascade_vs_l3_component_xp_pi_r_opta_{tag}_gw8-38.json",
    ]
    # Baseline s=1.0 published without an as* tag.
    if abs(s - 1.0) < 1e-9:
        cand.append(out_dir / "cascade_vs_l3_component_xp_pi_r_opta_gw8-38.json")
    for p in cand:
        if p.exists():
            return p
    return None


def _run_one(s: float, *, n: int, seed: int, out_dir: Path) -> dict[str, Any]:
    tag = f"as{str(s).replace('.', '')}"
    out = out_dir / f"cascade_vs_l3_component_xp_pi_r_opta_{tag}_gw8-38.json"
    reused = _reuse_paths(s, out_dir)
    if reused is not None:
        payload = json.loads(reused.read_text(encoding="utf-8"))
        # Only reuse when N matches (or baseline has same n).
        if int(payload.get("n_strands") or 0) == int(n):
            row = _row_from_payload(s, payload, elapsed=0.0, artifact=str(reused))
            print(
                f"[trend] reuse s={s:.2f} resid_G={row['resid_G']:+.1f} "
                f"resid_CS={row['resid_CS']:+.1f} ← {reused.name}",
                flush=True,
            )
            return row
    log = ROOT / "LOGS" / f"cascade_away_shrink_xi_resid_trend_{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.eval.cascade_vs_l3_component_xp",
        "--n",
        str(int(n)),
        "--seed",
        str(int(seed)),
        "--goal-weight",
        "rate",
        "--match-source",
        "opta",
        "--away-shrink",
        f"{s:.2f}",
        "--steps",
        STEPS,
        "--decision-gw-only",
        "--out",
        str(out),
    ]
    t0 = time.time()
    print(f"[trend] start s={s:.2f} → {out.name}", flush=True)
    env = {**__import__("os").environ, "PYTHONPATH": "."}
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - t0
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"s={s:.2f} failed rc={proc.returncode} log={log} elapsed={elapsed:.0f}s"
        )
    payload = json.loads(out.read_text(encoding="utf-8"))
    row = _row_from_payload(
        s, payload, elapsed=elapsed, artifact=str(out), log=str(log)
    )
    print(
        f"[trend] done s={s:.2f} resid_G={row['resid_G']:+.1f} "
        f"resid_CS={row['resid_CS']:+.1f} wall={elapsed:.0f}s",
        flush=True,
    )
    return row


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [r["away_shrink"] for r in rows]
    g = [r["resid_G"] for r in rows]
    cs = [r["resid_CS"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.plot(xs, g, "o-", color="#c0392b", label="resid G (casc−P)")
    ax.plot(xs, cs, "s-", color="#2980b9", label="resid CS (casc−P)")
    for s_mark, lab, col in (
        (0.86, "fit 0.86", "#27ae60"),
        (1.0, "default 1.0", "#666"),
    ):
        if any(abs(x - s_mark) < 1e-9 for x in xs):
            ax.axvline(s_mark, color=col, ls="--", lw=0.9, alpha=0.8, label=lab)
    ax.set_xlabel("away_shrink s")
    ax.set_ylabel("season XI Σ resid (points)")
    ax.set_title("2025-26 GW8–38 Opta π∝r: XI resid G/CS vs away_shrink")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--grid",
        type=str,
        default=",".join(f"{x:.2f}" for x in GRID_DEFAULT),
        help="Comma-separated away_shrink values",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--plot", type=Path, default=PLOT)
    args = ap.parse_args(argv)

    grid = [float(x) for x in str(args.grid).split(",") if x.strip()]
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(
        f"[trend] grid={grid} N={args.n} workers={args.workers} steps=8…38",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    # Thread pool of subprocesses — each s is its own process via _run_one.
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futs = {
            pool.submit(
                _run_one, s, n=int(args.n), seed=int(args.seed), out_dir=ART
            ): s
            for s in grid
        }
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["away_shrink"])
    _plot(rows, args.plot)

    state = {
        "task": "cascade_away_shrink_xi_resid_trend",
        "season": "2025-26",
        "gw_range": [8, 38],
        "n_strands": int(args.n),
        "seed": int(args.seed),
        "goal_weight": "rate",
        "match_source": "opta",
        "grid": grid,
        "rows": rows,
        "plot": str(args.plot),
        "elapsed_s": time.time() - t0,
        "note": (
            "Shared L3 XI, no captain, decision-GW Opta λ; "
            "default AWAY_SHRINK still 1.0"
        ),
    }
    args.out.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Opta XI resid G/CS vs away_shrink",
        "",
        f"2025-26 GW8–38 · π∝r · Opta λ · N={args.n} · shared L3 XI · "
        f"wall {state['elapsed_s']:.0f}s",
        "",
        "Formula: `λ_H` unchanged; `λ_A ← λ_A · s`. Default `s=1.0` unchanged.",
        "",
        f"![trend]({args.plot})",
        "",
        "| s | resid G | resid CS | resid GA | MAE_G | MAE_CS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['away_shrink']:.2f} | {r['resid_G']:+.1f} | "
            f"{r['resid_CS']:+.1f} | {r['resid_GA']:+.1f} | "
            f"{r['mae_casc_G']:.3f} | {r['mae_casc_CS']:.3f} |"
        )
    # highlight nearest to zero G
    best_g = min(rows, key=lambda r: abs(r["resid_G"]))
    best_cs = min(rows, key=lambda r: abs(r["resid_CS"]))
    lines += [
        "",
        f"- Closest resid G to 0: **s={best_g['away_shrink']:.2f}** "
        f"({best_g['resid_G']:+.1f}).",
        f"- Closest resid CS to 0: **s={best_cs['away_shrink']:.2f}** "
        f"({best_cs['resid_CS']:+.1f}).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_away_shrink_xi_resid_trend",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_away_shrink_xi_resid_trend",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Opta XI resid G/CS vs away_shrink s=0.5…1.0",
                "verdict": {
                    "closest_resid_G": {
                        "s": best_g["away_shrink"],
                        "resid_G": best_g["resid_G"],
                    },
                    "closest_resid_CS": {
                        "s": best_cs["away_shrink"],
                        "resid_CS": best_cs["resid_CS"],
                    },
                    "at_s_1_00": next(
                        (r for r in rows if abs(r["away_shrink"] - 1.0) < 1e-9), None
                    ),
                    "at_s_0_86": next(
                        (
                            r
                            for r in rows
                            if abs(r["away_shrink"] - 0.86) < 1e-9
                        ),
                        None,
                    ),
                },
                "artifact": str(args.out),
                "plot": str(args.plot),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": rows}, indent=2, default=str))
    print(f"wrote {args.out} {args.plot} {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
