"""Quick path-V diagnostic: attack dice vs share-conditional E[xP|k].

Protocol: ``cascade_pathv_share_cond_xp``. Same horizon/seeds; toggles
``--player-attack``. Diagnostic only — not a gated four-season promote.

  PYTHONPATH=. python3 -m src.eval.cascade_pathv_share_cond
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets.short_horizon_generative import score_fixture_generative

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_pathv_share_cond.md"
OUT_DICE = ROOT / "data/predictions/2026-27/cascade_pathv_dice"
OUT_SHARE = ROOT / "data/predictions/2026-27/cascade_pathv_share_mean"
STATE_JSON = ROOT / "data/processed/cascade_pathv_share_cond" / "state.json"

N = 8
SEED = 7
DECISION_GW = 6
HORIZON_END = 19
PRICED_GW = 8
SNAPSHOT_EVERY = 1


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return float(len(a & b) / len(u)) if u else 0.0


def _micro_stability() -> dict[str, Any]:
    """Fixed scoreline: dice xp variance vs share_mean (deterministic)."""
    home = pd.DataFrame(
        {
            "element": [1, 2, 3],
            "name": ["Star", "Support", "Fringe"],
            "position": ["FWD", "MID", "MID"],
            "team": ["Home"] * 3,
            "team_n": ["Home"] * 3,
            "m": [90.0, 80.0, 60.0],
            "sG": [0.55, 0.30, 0.15],
            "sA": [0.20, 0.50, 0.30],
            "bonus_mu": [0.2, 0.2, 0.1],
            "card_mu": [-0.1, -0.1, -0.1],
        }
    )
    away = home.copy()
    away["element"] = [4, 5, 6]
    away["name"] = ["Opp1", "Opp2", "Opp3"]
    away["team"] = ["Away"] * 3
    away["team_n"] = ["Away"] * 3

    dice_xp = []
    for seed in range(200):
        h_xp, _, _ = score_fixture_generative(
            home,
            away,
            lam_h=1.6,
            lam_a=1.1,
            rho=0.0,
            alpha_g=0.96,
            alpha_a=0.90,
            rng=np.random.default_rng(seed),
            lineup_cov=False,
            goals_home=2,
            goals_away=1,
            player_attack="dice",
        )
        dice_xp.append(float(h_xp.loc[h_xp["name"] == "Star", "xp"].iloc[0]))
    share_vals = []
    for seed in (0, 1, 2):
        h_xp, _, _ = score_fixture_generative(
            home,
            away,
            lam_h=1.6,
            lam_a=1.1,
            rho=0.0,
            alpha_g=0.96,
            alpha_a=0.90,
            rng=np.random.default_rng(seed),
            lineup_cov=False,
            goals_home=2,
            goals_away=1,
            player_attack="share_mean",
        )
        share_vals.append(float(h_xp.loc[h_xp["name"] == "Star", "xp"].iloc[0]))
    return {
        "n_dice": 200,
        "star_dice_mean": float(np.mean(dice_xp)),
        "star_dice_var": float(np.var(dice_xp)),
        "star_share_mean": float(np.mean(share_vals)),
        "star_share_nunique": int(len(set(share_vals))),
        "star_share_var": float(np.var(share_vals)),
    }


def _run_arm(*, player_attack: str, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.models.multistage_mc_milp",
        "--decision-gw",
        str(DECISION_GW),
        "--horizon-end",
        str(HORIZON_END),
        "--n",
        str(N),
        "--seed",
        str(SEED),
        "--snapshot-every",
        str(SNAPSHOT_EVERY),
        "--objective",
        "xi_plus_captain_ft",
        "--no-minutes-overlay",
        "--priced-gw",
        str(PRICED_GW),
        "--generative-attack",
        "--player-attack",
        player_attack,
        "--out",
        str(out),
    ]
    print(f"[share_cond] running player_attack={player_attack} N={N}…", flush=True)
    t0 = _now()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    elapsed = _now() - t0
    cands = sorted(out.glob("*trial.json"))
    trial: dict[str, Any] = {}
    if cands:
        trial = json.loads(cands[0].read_text())
    freqs = trial.get("frequencies") or []
    top = [
        {
            "name": r.get("name"),
            "freq": r.get("freq"),
            "freq_xi": r.get("freq_xi"),
            "freq_captain": r.get("freq_captain"),
            "team": r.get("team_norm"),
        }
        for r in freqs[:15]
    ]
    if freqs:
        by_cap = sorted(
            freqs, key=lambda r: float(r.get("freq_captain") or 0), reverse=True
        )
        cap_top = {
            "name": by_cap[0].get("name"),
            "freq_captain": by_cap[0].get("freq_captain"),
        }
    else:
        cap_top = {}
    return {
        "player_attack": player_attack,
        "returncode": proc.returncode,
        "stderr_tail": (proc.stderr or "")[-800:],
        "elapsed_s": elapsed,
        "n_squads": trial.get("n_squads_solved"),
        "strands_with_squads": trial.get("strands_with_squads"),
        "mean_transfers": trial.get("mean_transfers"),
        "mean_hits": trial.get("mean_hits"),
        "scoring": trial.get("scoring"),
        "player_attack_meta": trial.get("player_attack"),
        "top15": top,
        "top15_names": {str(r.get("name")) for r in top if r.get("name")},
        "cap_top": cap_top,
    }


def _write_report(state: dict[str, Any]) -> None:
    dice = state["dice"]
    share = state["share_mean"]
    micro = state["micro"]
    jac = state["top15_jaccard"]
    lines = [
        "# Path-V diagnostic — dice vs share-conditional E[xP|k]",
        "",
        f"Protocol: `cascade_pathv_share_cond_xp` · N={N} · seed={SEED} · "
        f"decision GW{DECISION_GW} → horizon {HORIZON_END} · priced GW{PRICED_GW} · "
        "objective `xi_plus_captain_ft`",
        "",
        "**Diagnostic only** — not a gated four-season path-V promote.",
        "",
        f"- Smoke passed: **{state['smoke_ok']}**",
        f"- Top-15 freq Jaccard (dice ∩ share_mean): **{jac:.3f}**",
        f"- Wall: {state['wall_s']:.1f}s",
        "",
        "## Micro (fixed k=2–1, Star FWD)",
        "",
        f"- Dice xp mean/var over 200 seeds: "
        f"**{micro['star_dice_mean']:.3f}** / **{micro['star_dice_var']:.3f}**",
        f"- Share_mean xp (nunique across seeds): "
        f"**{micro['star_share_mean']:.3f}** / nunique={micro['star_share_nunique']}",
        "- Share_mean is deterministic given the scoreline (player RNG off).",
        "",
        "## Dice strands (player_attack=dice)",
        "",
        f"- Squads: {dice.get('n_squads')} · "
        f"strands_with_squads={dice.get('strands_with_squads')}",
        f"- Mean transfers / hits: {dice.get('mean_transfers')} / "
        f"{dice.get('mean_hits')}",
        f"- Top captain: {dice.get('cap_top')}",
        f"- Elapsed: {dice.get('elapsed_s'):.1f}s",
        "",
        "Top freq:",
        "",
    ]
    for r in dice.get("top15") or []:
        lines.append(
            f"- {r.get('name')} ({r.get('team')}): freq={r.get('freq')} "
            f"xi={r.get('freq_xi')} cap={r.get('freq_captain')}"
        )
    lines += [
        "",
        "## Share-mean strands (player_attack=share_mean)",
        "",
        f"- Squads: {share.get('n_squads')} · "
        f"strands_with_squads={share.get('strands_with_squads')}",
        f"- Mean transfers / hits: {share.get('mean_transfers')} / "
        f"{share.get('mean_hits')}",
        f"- Top captain: {share.get('cap_top')}",
        f"- Elapsed: {share.get('elapsed_s'):.1f}s",
        "",
        "Top freq:",
        "",
    ]
    for r in share.get("top15") or []:
        lines.append(
            f"- {r.get('name')} ({r.get('team')}): freq={r.get('freq')} "
            f"xi={r.get('freq_xi')} cap={r.get('freq_captain')}"
        )
    lines += [
        "",
        "## Read",
        "",
        "- Same rolled scorelines / Opta path; only the player layer changes.",
        "- Lower hit churn and higher top-15 overlap vs dice would support the "
        "stability claim; this is not a likelihood promote.",
        "",
        "Artifacts:",
        f"- `{OUT_DICE}`",
        f"- `{OUT_SHARE}`",
        "",
        "Reproduce:",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_pathv_share_cond",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = _now()
    micro = _micro_stability()
    dice = _run_arm(player_attack="dice", out=OUT_DICE)
    share = _run_arm(player_attack="share_mean", out=OUT_SHARE)
    jac = _jaccard(dice.get("top15_names") or set(), share.get("top15_names") or set())
    smoke = (
        dice.get("returncode") == 0
        and share.get("returncode") == 0
        and int(dice.get("n_squads") or 0) > 0
        and int(share.get("n_squads") or 0) > 0
    )
    state = {
        "protocol": "cascade_pathv_share_cond_xp",
        "n": N,
        "seed": SEED,
        "micro": micro,
        "dice": {k: v for k, v in dice.items() if k != "top15_names"},
        "share_mean": {k: v for k, v in share.items() if k != "top15_names"},
        "top15_jaccard": jac,
        "smoke_ok": smoke,
        "wall_s": _now() - t0,
    }
    # keep name sets out of json (sets); already stripped
    _save(state)
    _write_report(state)
    print(json.dumps({
        "smoke_ok": smoke,
        "top15_jaccard": jac,
        "dice_hits": dice.get("mean_hits"),
        "share_hits": share.get("mean_hits"),
        "micro_dice_var": micro["star_dice_var"],
        "wall_s": state["wall_s"],
        "report": str(REPORT_MD),
    }, indent=2))
    if not smoke:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
