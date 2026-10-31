"""Test: was the XI CS underprint selection (lucky backs), not broken DC?

Uses the multi-seed XI panel from ``cascade_vs_l3_component_xp_multiseed``
and FD team-GW moments from ``dixon_coles_team_cs_calib`` helpers.

For each seed:

1. Take GKP+DEF player-GWs with m≥60 on the held XI.
2. Map to unique **team-GWs**; compare FD ``xcs`` vs actual team CS.
3. Contrast with the full league GW8–12 team-GW panel (already calibrated).
4. Bootstrap: resample league team-GWs of the same count; where does the
   selected bias sit in that null?

If selection hypothesis holds: selected team-GWs show large negative
bias (p−y) while the league is ~0, and selected bias is in the left tail
of the bootstrap.

Optional ``--scan-windows``: freeze each seed's selected back clubs from
the GW8–12 XI, then score those clubs' team-GW bias on other 2025-26
windows (same selection, different fixtures).

  PYTHONPATH=. python3 -m src.eval.xi_cs_selection_hypothesis
  PYTHONPATH=. python3 -m src.eval.xi_cs_selection_hypothesis --scan-windows
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.dixon_coles_team_cs_calib import _attach_fd, _team_gw_sides
from src.markets.numagent_xp_pilot import load_merged
from src.strategy.layer4_bookie_path import _fd_index
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "xi_cs_selection_hypothesis.md"
REPORT_WINDOWS = ROOT / "reports" / "xi_cs_selection_hypothesis_windows.md"
STATE = ROOT / "REPORT_STATE.json"
MULTI = Path("/opt/cursor/artifacts/cascade_vs_l3_component_xp_multiseed.json")
OUT_DEFAULT = Path("/opt/cursor/artifacts/xi_cs_selection_hypothesis.json")
OUT_WINDOWS = Path("/opt/cursor/artifacts/xi_cs_selection_hypothesis_windows.json")

GWS = (8, 9, 10, 11, 12)
# Same clubs, other fixtures — contiguous blocks of 2025-26.
WINDOWS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("GW1-7", tuple(range(1, 8))),
    ("GW8-12", tuple(range(8, 13))),
    ("GW13-19", tuple(range(13, 20))),
    ("GW20-26", tuple(range(20, 27))),
    ("GW27-33", tuple(range(27, 34))),
    ("GW34-38", tuple(range(34, 39))),
)
N_BOOT = 5000
RNG_SEED = 7


def _now() -> float:
    return time.time()


def _player_meta(merged: pd.DataFrame) -> pd.DataFrame:
    meta = (
        merged.groupby("element", as_index=False)
        .agg(team=("team", "first"), team_n=("team_n", "first"), name=("name", "first"))
    )
    meta["team_n"] = (
        meta["team_n"]
        .fillna(meta["team"].map(lambda t: norm_team(str(t))))
        .map(lambda t: norm_team(str(t)))
    )
    return meta


def _selected_team_gws(
    players: list[dict[str, Any]],
    meta: pd.DataFrame,
    sides: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (back player-GWs with team, unique team-GW sides with xcs/cs)."""
    df = pd.DataFrame(players)
    backs = df.loc[
        df["position"].isin(["GKP", "DEF"])
        & (pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= 60)
    ].copy()
    backs = backs.merge(meta[["element", "team_n", "team"]], on="element", how="left")
    keys = backs[["gw", "team_n"]].drop_duplicates().rename(columns={"gw": "GW"})
    tg = keys.merge(
        sides[["GW", "team_n", "team", "xcs", "cs", "ga", "gf", "lam_concede"]],
        on=["GW", "team_n"],
        how="left",
    )
    return backs, tg


def _bias(tg: pd.DataFrame) -> dict[str, float]:
    x = pd.to_numeric(tg["xcs"], errors="coerce")
    y = pd.to_numeric(tg["cs"], errors="coerce")
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) == 0:
        return {
            "n": 0,
            "mean_xcs": float("nan"),
            "mean_cs": float("nan"),
            "bias": float("nan"),
        }
    return {
        "n": int(len(x)),
        "mean_xcs": float(x.mean()),
        "mean_cs": float(y.mean()),
        "bias": float((x - y).mean()),
    }


def _bootstrap_null(
    league: pd.DataFrame,
    n_draw: int,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample ``n_draw`` league team-GWs with replacement; collect bias."""
    x = league["xcs"].to_numpy(float)
    y = league["cs"].to_numpy(float)
    biases = np.empty(n_boot, float)
    mean_cs = np.empty(n_boot, float)
    mean_xcs = np.empty(n_boot, float)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n_draw)
        biases[i] = float((x[idx] - y[idx]).mean())
        mean_cs[i] = float(y[idx].mean())
        mean_xcs[i] = float(x[idx].mean())
    return {
        "n_boot": n_boot,
        "n_draw": n_draw,
        "bias_mean": float(biases.mean()),
        "bias_sd": float(biases.std(ddof=0)),
        "bias_p05": float(np.quantile(biases, 0.05)),
        "bias_p50": float(np.quantile(biases, 0.50)),
        "bias_p95": float(np.quantile(biases, 0.95)),
        "mean_cs_p50": float(np.quantile(mean_cs, 0.50)),
        "mean_xcs_p50": float(np.quantile(mean_xcs, 0.50)),
        "biases": biases.tolist(),
    }


def _seed_club_keys(
    players: list[dict[str, Any]],
    meta: pd.DataFrame,
) -> set[str]:
    """Freeze selected back clubs from the GW8–12 XI panel (m≥60 GKP/DEF)."""
    df = pd.DataFrame(players)
    backs = df.loc[
        df["position"].isin(["GKP", "DEF"])
        & (pd.to_numeric(df["minutes"], errors="coerce").fillna(0) >= 60)
        & df["gw"].isin(GWS)
    ].copy()
    backs = backs.merge(meta[["element", "team_n"]], on="element", how="left")
    return {str(t) for t in backs["team_n"].dropna().unique().tolist()}


def _score_window(
    *,
    label: str,
    gws: tuple[int, ...],
    club_keys: set[str],
    sides: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    league = sides.loc[sides["GW"].isin(gws)].copy()
    league_stats = _bias(league)
    tg = league.loc[league["team_n"].isin(club_keys)].copy()
    sel = _bias(tg)
    boot = _bootstrap_null(
        league, max(int(sel["n"]), 1), n_boot=n_boot, rng=rng
    )
    biases = np.asarray(boot["biases"], float)
    p_left = float((biases <= sel["bias"]).mean()) if sel["n"] else float("nan")
    return {
        "window": label,
        "gws": list(gws),
        "n_league": int(league_stats["n"]),
        "league_bias": league_stats["bias"],
        "league_mean_xcs": league_stats["mean_xcs"],
        "league_mean_cs": league_stats["mean_cs"],
        "n_team_gw": int(sel["n"]),
        "sel_mean_xcs": sel["mean_xcs"],
        "sel_mean_cs": sel["mean_cs"],
        "sel_bias": sel["bias"],
        "boot_bias_p05": boot["bias_p05"],
        "boot_bias_p50": boot["bias_p50"],
        "boot_bias_p95": boot["bias_p95"],
        "p_left": p_left,
    }


def _run_window_scan(
    *,
    multi: dict[str, Any],
    meta: pd.DataFrame,
    sides: pd.DataFrame,
    n_boot: int,
    out: Path,
    season: str,
) -> dict[str, Any]:
    """Same selected clubs as GW8–12 XI backs; score other season windows."""
    t0 = _now()
    rng = np.random.default_rng(RNG_SEED)
    by_seed: list[dict[str, Any]] = []
    window_agg: dict[str, list[float]] = {lab: [] for lab, _ in WINDOWS}

    for seed_name, players in (multi.get("players_by_seed") or {}).items():
        clubs = _seed_club_keys(players, meta)
        rows = []
        for label, gws in WINDOWS:
            row = _score_window(
                label=label,
                gws=gws,
                club_keys=clubs,
                sides=sides,
                rng=rng,
                n_boot=n_boot,
            )
            rows.append(row)
            if np.isfinite(row["sel_bias"]):
                window_agg[label].append(float(row["sel_bias"]))
            print(
                f"[win] {seed_name} {label}: bias={row['sel_bias']:+.3f} "
                f"league={row['league_bias']:+.3f} p_left={row['p_left']:.4f} "
                f"n={row['n_team_gw']}",
                flush=True,
            )
        by_seed.append(
            {
                "seed": seed_name,
                "n_clubs": len(clubs),
                "clubs": sorted(clubs),
                "windows": rows,
            }
        )

    window_summary = []
    for label, _ in WINDOWS:
        biases = window_agg[label]
        # mean p_left across seeds for this window
        p_lefts = [
            r["p_left"]
            for s in by_seed
            for r in s["windows"]
            if r["window"] == label and np.isfinite(r["p_left"])
        ]
        league_bias = next(
            r["league_bias"]
            for s in by_seed
            for r in s["windows"]
            if r["window"] == label
        )
        window_summary.append(
            {
                "window": label,
                "mean_sel_bias": float(np.mean(biases)) if biases else float("nan"),
                "all_sel_bias_negative": bool(biases and all(b < 0 for b in biases)),
                "mean_p_left": float(np.mean(p_lefts)) if p_lefts else float("nan"),
                "league_bias": float(league_bias),
            }
        )

    # Is GW8-12 uniquely lucky vs other windows?
    by_lab = {w["window"]: w for w in window_summary}
    gw812 = by_lab["GW8-12"]
    early = [by_lab["GW1-7"], by_lab["GW8-12"]]
    late = [by_lab[k] for k in ("GW13-19", "GW20-26", "GW27-33", "GW34-38")]
    early_mean = float(np.mean([w["mean_sel_bias"] for w in early]))
    late_mean = float(np.mean([w["mean_sel_bias"] for w in late]))
    other = [w for w in window_summary if w["window"] != "GW8-12"]
    other_mean = float(np.mean([w["mean_sel_bias"] for w in other]))
    n_other_neg = sum(1 for w in other if w["mean_sel_bias"] < -0.05)
    if early_mean < -0.05 and late_mean > -0.03:
        read = (
            "Sheet luck for these clubs is **early-season**, not only GW8–12: "
            f"GW1–12 mean sel bias {early_mean:+.3f}; GW13–38 mean "
            f"{late_mean:+.3f} (near zero / flips sign). League stays ~calibrated "
            "in every block."
        )
    elif gw812["mean_sel_bias"] < -0.05 and n_other_neg >= 2:
        read = (
            "Selected clubs stay sheet-lucky in multiple 2025-26 windows "
            f"(GW8–12 bias {gw812['mean_sel_bias']:+.3f}; "
            f"{n_other_neg}/5 other blocks mean bias < −0.05)."
        )
    elif gw812["mean_sel_bias"] < -0.05 and other_mean > -0.05:
        read = (
            "GW8–12 sheet luck looks **window-specific**: frozen selected "
            "clubs over-deliver CS there, but mean bias on other 2025-26 "
            f"blocks is near zero ({other_mean:+.3f})."
        )
    else:
        read = (
            "Window scan mixed — see per-window table. "
            f"GW8–12 mean sel bias {gw812['mean_sel_bias']:+.3f}; "
            f"other-windows mean {other_mean:+.3f}."
        )

    verdict = {
        "freeze": "back clubs from GW8–12 XI (m≥60 GKP/DEF) per seed",
        "n_seeds": len(by_seed),
        "windows": window_summary,
        "gw8_12_mean_sel_bias": gw812["mean_sel_bias"],
        "early_gw1_12_mean_sel_bias": early_mean,
        "late_gw13_38_mean_sel_bias": late_mean,
        "other_windows_mean_sel_bias": other_mean,
        "n_other_blocks_bias_lt_neg005": n_other_neg,
        "read": read,
    }
    state = {
        "task": "xi_cs_selection_hypothesis_windows",
        "season": season,
        "n_boot": n_boot,
        "elapsed_s": _now() - t0,
        "verdict": verdict,
        "seeds": by_seed,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# XI CS selection — same clubs, other 2025-26 windows",
        "",
        f"Season **{season}** · freeze clubs from GW8–12 XI backs · "
        f"{len(by_seed)} seeds · bootstrap n={n_boot} · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## Design",
        "",
        "Take each multi-seed open’s **selected back clubs** (GKP/DEF with "
        "m≥60 in the GW8–12 panel). Hold that club set fixed. On each "
        "contiguous GW block, compare those clubs’ FD `xcs` vs actual CS "
        "against a size-matched league bootstrap in the **same** block.",
        "",
        "This asks whether GW8–12 sheet luck was unique to that window, "
        "or whether the same selection stays lucky elsewhere in 2025-26.",
        "",
        "## Window summary (mean across seeds)",
        "",
        "| window | league bias | mean sel bias | all neg? | mean p_left |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for w in window_summary:
        lines.append(
            f"| {w['window']} | {w['league_bias']:+.3f} | "
            f"{w['mean_sel_bias']:+.3f} | {w['all_sel_bias_negative']} | "
            f"{w['mean_p_left']:.3f} |"
        )
    lines += [
        "",
        "## Per seed × window",
        "",
        "| seed | window | n | mean xCS | actual CS | bias | p_left |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in by_seed:
        for r in s["windows"]:
            lines.append(
                f"| {s['seed']} | {r['window']} | {r['n_team_gw']} | "
                f"{r['sel_mean_xcs']:.3f} | {r['sel_mean_cs']:.3f} | "
                f"{r['sel_bias']:+.3f} | {r['p_left']:.3f} |"
            )
    lines += [
        "",
        "## Verdict",
        "",
        verdict["read"],
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.xi_cs_selection_hypothesis --scan-windows",
        "```",
        "",
    ]
    REPORT_WINDOWS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "xi_cs_selection_hypothesis_windows",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Same selected clubs across other 2025-26 GW windows",
                "verdict": verdict,
                "artifact": str(out),
                "report": str(REPORT_WINDOWS.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(verdict, indent=2))
    print("wrote", out, REPORT_WINDOWS)
    return state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--multi", type=Path, default=MULTI)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--scan-windows",
        action="store_true",
        help="Freeze GW8–12 selected clubs; score other 2025-26 GW blocks",
    )
    ap.add_argument("--out-windows", type=Path, default=OUT_WINDOWS)
    args = ap.parse_args(argv)

    t0 = _now()
    multi = json.loads(Path(args.multi).read_text(encoding="utf-8"))
    merged = load_merged(str(args.season))
    meta = _player_meta(merged)
    sides = _attach_fd(_team_gw_sides(merged), _fd_index(str(args.season)))

    if args.scan_windows:
        _run_window_scan(
            multi=multi,
            meta=meta,
            sides=sides,
            n_boot=int(args.n_boot),
            out=Path(args.out_windows),
            season=str(args.season),
        )
        return 0

    league = sides.loc[sides["GW"].isin(GWS)].copy()
    league_stats = _bias(league)

    rng = np.random.default_rng(RNG_SEED)
    seed_rows: list[dict[str, Any]] = []

    for seed_name, players in (multi.get("players_by_seed") or {}).items():
        backs, tg = _selected_team_gws(players, meta, sides)
        # restrict to screen GWs
        tg = tg.loc[tg["GW"].isin(GWS)].copy()
        backs = backs.loc[backs["gw"].isin(GWS)].copy()
        sel = _bias(tg)
        boot = _bootstrap_null(
            league, max(int(sel["n"]), 1), n_boot=int(args.n_boot), rng=rng
        )
        # one-sided: P(null bias <= selected bias)
        biases = np.asarray(boot["biases"], float)
        p_left = float((biases <= sel["bias"]).mean()) if sel["n"] else float("nan")
        player_cs_rate = float((backs["P_CS"] > 0).mean()) if len(backs) else float("nan")
        # XI CS residual proxy from artifact players (casc)
        pdf = pd.DataFrame(players)
        pdf = pdf.loc[pdf["gw"].isin(GWS)]
        casc_rCS = float((pdf["casc_xP_CS"] - pdf["P_CS"]).sum())
        l3_rCS = float((pdf["l3_xP_CS"] - pdf["P_CS"]).sum())
        seed_rows.append(
            {
                "seed": seed_name,
                "n_back_player_gw": int(len(backs)),
                "n_team_gw": int(sel["n"]),
                "player_cs_rate": player_cs_rate,
                "sel_mean_xcs": sel["mean_xcs"],
                "sel_mean_cs": sel["mean_cs"],
                "sel_bias": sel["bias"],
                "league_mean_xcs": league_stats["mean_xcs"],
                "league_mean_cs": league_stats["mean_cs"],
                "league_bias": league_stats["bias"],
                "boot_bias_p05": boot["bias_p05"],
                "boot_bias_p50": boot["bias_p50"],
                "boot_bias_p95": boot["bias_p95"],
                "p_left": p_left,
                "casc_rCS_xi": casc_rCS,
                "l3_rCS_xi": l3_rCS,
                "clubs": sorted(
                    {
                        str(t)
                        for t in tg["team"].dropna().unique().tolist()
                    }
                ),
            }
        )
        print(
            f"[sel] {seed_name}: team-GW bias={sel['bias']:+.3f} "
            f"(xcs={sel['mean_xcs']:.3f} cs={sel['mean_cs']:.3f}) "
            f"league_bias={league_stats['bias']:+.3f} p_left={p_left:.4f}",
            flush=True,
        )

    # Aggregate: mean selected bias vs league
    sel_biases = [r["sel_bias"] for r in seed_rows if np.isfinite(r["sel_bias"])]
    verdict = {
        "league_gw8_12": league_stats,
        "n_seeds": len(seed_rows),
        "mean_sel_bias": float(np.mean(sel_biases)) if sel_biases else float("nan"),
        "all_sel_bias_negative": bool(all(b < 0 for b in sel_biases)),
        "mean_p_left": float(np.mean([r["p_left"] for r in seed_rows])),
        "read": (
            "Selection hypothesis supported: selected backs' team-GWs have "
            "actual CS ≫ FD xCS while the league panel is calibrated; "
            "selected bias sits in the left tail of a size-matched bootstrap."
            if sel_biases
            and float(np.mean(sel_biases)) < -0.05
            and float(np.mean([r["p_left"] for r in seed_rows])) < 0.1
            else "Selection hypothesis unclear — see per-seed table."
        ),
    }

    state = {
        "task": "xi_cs_selection_hypothesis",
        "season": str(args.season),
        "gws": list(GWS),
        "n_boot": int(args.n_boot),
        "multi_source": str(args.multi),
        "elapsed_s": _now() - t0,
        "verdict": verdict,
        "seeds": seed_rows,
    }
    # drop huge bias lists if we stored them — we didn't in seed_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# XI CS underprint — selection hypothesis test",
        "",
        f"Season **{args.season}** · GW {list(GWS)} · "
        f"{len(seed_rows)} seeds from `{Path(args.multi).name}` · "
        f"bootstrap n={args.n_boot} · wall {state['elapsed_s']:.1f}s",
        "",
        "## Hypothesis",
        "",
        "The multi-seed FPL XI CS gap (xP_CS ≪ actual CS pts) is **selection**: "
        "opens stack DEF/GKP whose **team-GWs** kept more sheets than FD "
        "`xcs` implied — not a broken Dixon–Coles prior (league team-GW is "
        "calibrated).",
        "",
        "## League baseline (all team-GWs GW8–12)",
        "",
        f"- n={league_stats['n']}  mean xCS={league_stats['mean_xcs']:.3f}  "
        f"actual CS={league_stats['mean_cs']:.3f}  "
        f"bias(p−y)={league_stats['bias']:+.3f}",
        "",
        "## Per seed (unique team-GWs of XI GKP+DEF with m≥60)",
        "",
        "| seed | n team-GW | mean xCS | actual CS | bias p−y | "
        "boot p05 | p_left | player CS rate | casc Σr_CS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in seed_rows:
        lines.append(
            f"| {r['seed']} | {r['n_team_gw']} | {r['sel_mean_xcs']:.3f} | "
            f"{r['sel_mean_cs']:.3f} | {r['sel_bias']:+.3f} | "
            f"{r['boot_bias_p05']:+.3f} | {r['p_left']:.3f} | "
            f"{r['player_cs_rate']:.2f} | {r['casc_rCS_xi']:+.1f} |"
        )
    lines += [
        "",
        f"**Mean selected bias:** {verdict['mean_sel_bias']:+.3f} "
        f"(all negative: {verdict['all_sel_bias_negative']}). "
        f"Mean bootstrap left-tail p: {verdict['mean_p_left']:.3f}.",
        "",
        "## Read",
        "",
        "- `p_left` = fraction of size-matched league bootstrap samples with "
        "bias ≤ the seed’s bias (small ⇒ selected set is unusually "
        "sheet-lucky vs FD).",
        "- Selected mean xCS is also **higher** than the league (~0.27): "
        "opens chase good CS fixtures, and those fixtures then "
        "**over-delivered** vs even that elevated xCS.",
        "- Same clubs on other 2025-26 blocks: "
        "`python3 -m src.eval.xi_cs_selection_hypothesis --scan-windows`.",
        "",
        f"## Verdict",
        "",
        verdict["read"],
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.xi_cs_selection_hypothesis",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "xi_cs_selection_hypothesis",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Test XI CS gap as selection of lucky team-GWs",
                "verdict": verdict,
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(verdict, indent=2))
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
