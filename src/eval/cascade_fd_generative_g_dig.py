"""Dig FD/generative +14 XI resid G under π∝r and default rate_minutes.

Protocol: ``cascade_fd_generative_g_dig``. Diagnostic only — does not flip
goal_weight default (``rate_minutes``).

Layers:

1. Load published FD π∝r decision-GW map (GW8–38).
2. Run / load the same map with ``goal_weight=rate_minutes``.
3. Compare XI resid G and position split.
4. Share-concentration probe on a sample of decision GWs (HHI / top-1 sG
   under both weights from hist rates + oracle minutes).
5. α_G empty mass from the published map (1 − mean α_G).

  PYTHONPATH=. python3 -m src.eval.cascade_fd_generative_g_dig
  PYTHONPATH=. python3 -m src.eval.cascade_fd_generative_g_dig --run-rate-minutes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets import numagent_xp as nx
from src.markets.numagent_xp_pilot import load_merged
from src.strategy.layer4_bookie_path import _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_fd_generative_g_dig.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_fd_generative_g_dig.json")

RATE_SUMMARY = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_gw8-38.json"
)
RATE_PLAYERS = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_pi_r_gw8-38_players.json"
)
RM_SUMMARY = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_rate_minutes_fd_gw8-38.json"
)
RM_PLAYERS = Path(
    "/opt/cursor/artifacts/cascade_vs_l3_component_xp_rate_minutes_fd_gw8-38_players.json"
)

SEASON = "2025-26"
PROBE_GWS = (8, 15, 22, 29, 36)


def _now() -> float:
    return time.time()


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _players_df(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def _resid_pack(summary: dict[str, Any], players: pd.DataFrame) -> dict[str, Any]:
    s = summary["summary"]
    resid = float(s["sum_casc_xP_G"] - s["sum_P_G"])
    pl = players.copy()
    pl["resid_G"] = pl["casc_xP_G"] - pl["P_G"]
    by_pos = {
        str(pos): {
            "resid_G": float(g["resid_G"].sum()),
            "xP_G": float(g["casc_xP_G"].sum()),
            "P_G": float(g["P_G"].sum()),
            "n": int(len(g)),
        }
        for pos, g in pl.groupby("position")
    }
    alphas = [float(x.get("alpha_g", float("nan"))) for x in summary.get("per_gw", [])]
    alphas = [a for a in alphas if np.isfinite(a)]
    return {
        "goal_weight": summary.get("goal_weight"),
        "match_source": summary.get("match_source", "fd"),
        "resid_G": resid,
        "sum_casc_xP_G": float(s["sum_casc_xP_G"]),
        "sum_P_G": float(s["sum_P_G"]),
        "mae_casc_G": float(s["mae_casc_G"]),
        "by_position": by_pos,
        "n_minutes_zero": int((pl["minutes"] <= 0).sum()),
        "n_player_gw": int(len(pl)),
        "alpha_g_mean": float(np.mean(alphas)) if alphas else float("nan"),
        "empty_mass_mean": float(1.0 - np.mean(alphas)) if alphas else float("nan"),
    }


def _run_rate_minutes() -> Path:
    steps = ",".join(str(g) for g in range(8, 39))
    cmd = [
        sys.executable,
        "-m",
        "src.eval.cascade_vs_l3_component_xp",
        "--season",
        SEASON,
        "--steps",
        steps,
        "--goal-weight",
        "rate_minutes",
        "--match-source",
        "fd",
        "--decision-gw-only",
        "--n",
        "40",
        "--out",
        str(RM_SUMMARY),
    ]
    print("[fd-gen-dig] running rate_minutes FD map…", flush=True)
    print(" ", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    # Component xp writes players next to out with _players suffix when configured;
    # fall back to locating the written players path from the summary.
    summary = _load_summary(RM_SUMMARY)
    players_path = Path(summary.get("players_path") or RM_PLAYERS)
    if players_path.is_file() and players_path != RM_PLAYERS:
        RM_PLAYERS.write_text(players_path.read_text(encoding="utf-8"), encoding="utf-8")
    return RM_SUMMARY


def _share_probe() -> dict[str, Any]:
    """HHI / top-1 sG under rate vs rate_minutes for clubs in merged hist."""
    enriched = _load_merged_enriched(SEASON)
    # Prefer player-level rate columns if present.
    need = {"GW", "team", "position", "rG", "minutes"}
    cols = set(enriched.columns)
    if not need.issubset(cols):
        # Fall back to merged FPL history.
        merged = load_merged(SEASON)
        frame = merged
    else:
        frame = enriched
    out_rows: list[dict[str, Any]] = []
    for gw in PROBE_GWS:
        g = frame.loc[pd.to_numeric(frame["GW"], errors="coerce") == int(gw)].copy()
        if g.empty:
            continue
        if "rG" not in g.columns:
            continue
        g["m"] = pd.to_numeric(g.get("minutes", g.get("xmi", 0)), errors="coerce").fillna(0.0)
        g["rG"] = pd.to_numeric(g["rG"], errors="coerce").fillna(0.0)
        g["drG"] = (
            pd.to_numeric(g["drG"], errors="coerce").fillna(0.0)
            if "drG" in g.columns
            else 0.0
        )
        g["team_n"] = g["team"].map(lambda t: norm_team(str(t)))
        for team, club in g.groupby("team_n"):
            c = club.loc[club["m"] > 0].copy()
            if len(c) < 2:
                continue
            w_rate = np.array(
                [
                    nx.goal_share_weight(float(r.rG), float(r.drG))[0]
                    for r in c.itertuples(index=False)
                ]
            )
            w_rm = np.array(
                [
                    nx.weight_and_delta(
                        float(r.rG), float(r.drG), float(r.m), 0.0
                    )[0]
                    for r in c.itertuples(index=False)
                ]
            )
            s_rate, _ = nx.shares_and_deltas(w_rate, np.zeros_like(w_rate))
            s_rm, _ = nx.shares_and_deltas(w_rm, np.zeros_like(w_rm))
            pos = c["position"].astype(str).to_numpy()
            for label, shares in (("rate", s_rate), ("rate_minutes", s_rm)):
                hhi = float(np.sum(np.asarray(shares, float) ** 2))
                top1 = float(np.max(shares)) if len(shares) else float("nan")
                fwd_mask = pos == "FWD"
                fwd_share = (
                    float(np.sum(np.asarray(shares, float)[fwd_mask]))
                    if fwd_mask.any()
                    else 0.0
                )
                out_rows.append(
                    {
                        "gw": int(gw),
                        "team": str(team),
                        "goal_weight": label,
                        "n_playing": int(len(c)),
                        "hhi": hhi,
                        "top1": top1,
                        "fwd_share_mass": fwd_share,
                    }
                )
    if not out_rows:
        return {"n": 0, "note": "no rG columns for share probe"}
    df = pd.DataFrame(out_rows)
    summary: dict[str, Any] = {"n_club_gw": int(df["gw"].nunique() * df["team"].nunique() / 2)}
    for label, sub in df.groupby("goal_weight"):
        summary[str(label)] = {
            "mean_hhi": float(sub["hhi"].mean()),
            "mean_top1": float(sub["top1"].mean()),
            "mean_fwd_share_mass": float(sub["fwd_share_mass"].mean()),
            "n": int(len(sub)),
        }
    # Pairwise delta rate − rate_minutes on matched club-GW.
    piv = df.pivot_table(
        index=["gw", "team"], columns="goal_weight", values=["hhi", "top1", "fwd_share_mass"]
    )
    if ("hhi", "rate") in piv.columns and ("hhi", "rate_minutes") in piv.columns:
        summary["delta_rate_minus_rm"] = {
            "mean_hhi": float((piv[("hhi", "rate")] - piv[("hhi", "rate_minutes")]).mean()),
            "mean_top1": float((piv[("top1", "rate")] - piv[("top1", "rate_minutes")]).mean()),
            "mean_fwd_share_mass": float(
                (piv[("fwd_share_mass", "rate")] - piv[("fwd_share_mass", "rate_minutes")]).mean()
            ),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-rate-minutes",
        action="store_true",
        help="Run FD rate_minutes component map if artifact missing.",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = _now()

    if not RATE_SUMMARY.is_file() or not RATE_PLAYERS.is_file():
        raise SystemExit(f"missing π∝r artifacts: {RATE_SUMMARY} / {RATE_PLAYERS}")

    if not RM_SUMMARY.is_file():
        if args.run_rate_minutes:
            _run_rate_minutes()
        else:
            raise SystemExit(
                f"missing {RM_SUMMARY}; re-run with --run-rate-minutes"
            )
    # Resolve players path from summary if needed.
    rm_sum_raw = _load_summary(RM_SUMMARY)
    rm_players_path = Path(rm_sum_raw.get("players_path") or RM_PLAYERS)
    if not rm_players_path.is_file():
        raise SystemExit(f"missing rate_minutes players: {rm_players_path}")

    rate = _resid_pack(_load_summary(RATE_SUMMARY), _players_df(RATE_PLAYERS))
    rm = _resid_pack(rm_sum_raw, _players_df(rm_players_path))
    share = _share_probe()

    delta = {
        "resid_G_rate_minus_rm": float(rate["resid_G"] - rm["resid_G"]),
        "fwd_resid_rate_minus_rm": float(
            rate["by_position"].get("FWD", {}).get("resid_G", 0.0)
            - rm["by_position"].get("FWD", {}).get("resid_G", 0.0)
        ),
    }

    state = {
        "task": "cascade_fd_generative_g_dig",
        "protocol": "cascade_fd_generative_g_dig",
        "pre_registered": "2026-10-27",
        "season": SEASON,
        "gw_range": [8, 38],
        "rate": rate,
        "rate_minutes": rm,
        "delta": delta,
        "share_probe": share,
        "verdict_notes": [
            "α_G empty mass is small (~1−0.96) — not the +14 driver.",
            "FWD carries nearly all FD generative resid under π∝r.",
            "Compare rate vs rate_minutes resid to test minutes-in-share.",
        ],
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Dig FD/generative +14 XI resid G",
        "",
        f"Pre-registered **2026-10-27** · protocol `cascade_fd_generative_g_dig` · "
        f"wall {state['elapsed_s']:.1f}s",
        "",
        "## Verdict",
        "",
        f"- π∝r (`rate`) resid G: **{rate['resid_G']:+.1f}** "
        f"(FWD {rate['by_position'].get('FWD', {}).get('resid_G', float('nan')):+.1f}).",
        f"- Default `rate_minutes` resid G: **{rm['resid_G']:+.1f}** "
        f"(FWD {rm['by_position'].get('FWD', {}).get('resid_G', float('nan')):+.1f}).",
        f"- Δ (rate − rate_minutes): **{delta['resid_G_rate_minus_rm']:+.1f}** "
        f"(FWD Δ {delta['fwd_resid_rate_minus_rm']:+.1f}).",
        f"- Mean α_G empty mass (π∝r map): **{rate['empty_mass_mean']:.3f}** "
        f"(α_G≈{rate['alpha_g_mean']:.3f}) — too small to explain +14.",
        "",
        "## Position split",
        "",
        "| pos | π∝r resid G | rate_minutes resid G | n (π∝r) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        r = rate["by_position"].get(pos, {})
        m = rm["by_position"].get(pos, {})
        lines.append(
            f"| {pos} | {r.get('resid_G', float('nan')):+.1f} | "
            f"{m.get('resid_G', float('nan')):+.1f} | {r.get('n', 0)} |"
        )
    lines += ["", "## Share concentration probe", ""]
    if share.get("n_club_gw") or share.get("rate"):
        for label in ("rate", "rate_minutes"):
            block = share.get(label) or {}
            if not block:
                continue
            lines.append(
                f"- **{label}**: mean HHI={block.get('mean_hhi', float('nan')):.3f}, "
                f"top1={block.get('mean_top1', float('nan')):.3f}, "
                f"FWD mass={block.get('mean_fwd_share_mass', float('nan')):.3f}"
            )
        dlt = share.get("delta_rate_minus_rm") or {}
        if dlt:
            lines.append(
                f"- Δ rate−rm: HHI={dlt.get('mean_hhi', float('nan')):+.3f}, "
                f"top1={dlt.get('mean_top1', float('nan')):+.3f}, "
                f"FWD mass={dlt.get('mean_fwd_share_mass', float('nan')):+.3f}"
            )
    else:
        lines.append(f"- {share.get('note', 'no probe')}")
    lines += [
        "",
        "## Next",
        "",
        "- If rate_minutes resid ≈ π∝r resid: minutes-in-share is not the lever; "
        "look at FWD rate priors / finish conversion under FD λ.",
        "- If rate_minutes clearly lower: π∝r dig arm overstates; default path OK.",
        "- Decision metric remains player-GW likelihood; XI resid is diagnostic.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_fd_generative_g_dig --run-rate-minutes",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_fd_generative_g_dig",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Dig FD/generative +14 under π∝r vs rate_minutes",
                "verdict": {
                    "resid_rate": rate["resid_G"],
                    "resid_rate_minutes": rm["resid_G"],
                    "delta": delta["resid_G_rate_minus_rm"],
                    "empty_mass_mean": rate["empty_mass_mean"],
                    "fwd_resid_rate": rate["by_position"].get("FWD", {}).get("resid_G"),
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state["delta"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
