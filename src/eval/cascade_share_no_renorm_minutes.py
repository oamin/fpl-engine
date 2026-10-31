"""Screen: drop minutes from the attack-share weights before club renorm.

Locked dice: ``w = r·m/90``, ``π = w/Σw`` on the full club (m>0).
Treatment: ``π = r/Σr`` among m>0 — minutes only gate who is on the dice,
not the share weights.

Same OL XI (GW8–19), generative ``k∼M`` path, no share-SE shrink.

  PYTHONPATH=. python3 -m src.eval.cascade_share_no_renorm_minutes --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cascade_g_overprint_dig import _pid_key, _player_actual_g, _xi_by_gw
from src.eval.cascade_mc_vs_mean_hist import (
    SEASON_E0,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.markets import numagent_xp
from src.markets.attack_roll import build_side_dice, roll_side_attack
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import trivial_lineup
from src.models.multistage_mc_milp import (
    ClubPath,
    _priced_index,
    init_paths_from_opta,
    sample_week_scenario,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_share_no_renorm_minutes.md"
STATE = ROOT / "REPORT_STATE.json"
OUT_DEFAULT = Path("/opt/cursor/artifacts/cascade_share_no_renorm_minutes.json")
ROLLING = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json")

ARMS = ("comp_rm", "comp_r")


def _now() -> float:
    return time.time()


def _element(pid: str) -> int | None:
    try:
        return int(float(_pid_key(pid)))
    except (TypeError, ValueError):
        return None


def _name_col(df: pd.DataFrame) -> str:
    return "name" if "name" in df.columns else "player_name"


def _id_col(df: pd.DataFrame) -> str:
    return "player_id" if "player_id" in df.columns else "element"


def _club_weights(club: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    c = club.copy()
    if "m" not in c.columns and "xmi" in c.columns:
        c["m"] = pd.to_numeric(c["xmi"], errors="coerce").fillna(0.0)
    c["m"] = pd.to_numeric(c["m"], errors="coerce").fillna(0.0)
    c = c.loc[c["m"] > 0].reset_index(drop=True)
    if c.empty:
        return c, np.zeros(0), np.zeros(0)
    for col in ("rG", "drG", "rA", "drA", "sigma_m"):
        if col not in c.columns:
            c[col] = 0.0
    wG = np.array(
        [
            numagent_xp.weight_and_delta(
                float(c.iloc[i]["rG"] or 0.0),
                float(c.iloc[i]["drG"] or 0.0),
                float(c.iloc[i]["m"]),
                float(c.iloc[i]["sigma_m"] or 0.0),
            )[0]
            for i in range(len(c))
        ],
        dtype=float,
    )
    rG = np.maximum(
        pd.to_numeric(c["rG"], errors="coerce").fillna(0.0).to_numpy(float), 0.0
    )
    return c, wG, rG


def _pi_from_weights(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    z = float(w.sum())
    if z <= 0 or w.size == 0:
        n = int(w.size)
        return np.full(n, 1.0 / max(n, 1))
    return w / z


def _roll_arm(
    club: pd.DataFrame,
    *,
    arm: str,
    goals_for: int,
    alpha_g: float,
    alpha_a: float,
    rng: Any,
) -> dict[str, float]:
    c, wG, rG = _club_weights(club)
    out: dict[str, float] = {}
    if c.empty or int(goals_for) <= 0:
        return out
    if arm == "comp_rm":
        pi = _pi_from_weights(wG)
    elif arm == "comp_r":
        pi = _pi_from_weights(rG)
    else:
        raise ValueError(arm)
    sA, _ = numagent_xp.shares_and_deltas(np.maximum(wG, 0.0), np.zeros_like(wG))
    names = c[_name_col(c)].astype(str).tolist()
    dice = build_side_dice(
        names,
        pi,
        sA,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        G_team=1.0,
    )
    lineup = trivial_lineup(c["m"].to_numpy(float))
    rolled = roll_side_attack(dice, int(goals_for), rng, lineup=lineup)
    idc = _id_col(c)
    name_to_pid = {
        str(c.iloc[i][_name_col(c)]): _pid_key(str(c.iloc[i][idc]))
        for i in range(len(c))
    }
    for scorer in rolled.scorers:
        pid = name_to_pid.get(str(scorer))
        if pid is None:
            continue
        out[pid] = out.get(pid, 0.0) + 1.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=ROLLING)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)

    t0 = _now()
    src = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    season = str(src["season"])
    e0 = SEASON_E0[season]
    seed0 = int(src["seed"])
    start_gw = int(src["horizon_start"])
    end_gw = int(src["horizon_end"])
    xi_by_gw = _xi_by_gw(src)

    print(f"[no-renorm] season={season} N={args.n} arms={list(ARMS)}", flush=True)
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)
    hist = numagent_xp.asof_share_history(merged, before_gw=start_gw, season=season)
    alpha_g, alpha_a = _asof_alphas(hist)
    share = _build_hist_share_pool(hist, prices, decision_gw=start_gw, season=season)
    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start_gw, end=end_gw)
    plan_gws = sorted(fixtures_gw)
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    priced_index = _priced_index(priced_by_gw)
    pact = _player_actual_g(merged)

    acc: dict[str, dict[tuple[int, str], float]] = {
        a: defaultdict(float) for a in ARMS
    }
    pot_mass = {a: 0.0 for a in ARMS}
    n_xi_sides = 0

    paths0 = init_paths_from_opta(ranks, profile=profile)
    for s in range(int(args.n)):
        if s == 0 or (s + 1) % 20 == 0 or s + 1 == int(args.n):
            print(f"  [no-renorm] strand {s + 1}/{args.n}", flush=True)
        rng = random.Random(seed0 + s)
        paths = {c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()}
        arm_rngs = {
            a: random.Random(seed0 + 20_000 + s + 1000 * i)
            for i, a in enumerate(ARMS)
        }
        for gw in plan_gws:
            xi = xi_by_gw.get(int(gw), set())
            for fx in fixtures_gw.get(int(gw), []):
                if fx["home"] not in paths or fx["away"] not in paths:
                    continue
                sc = sample_week_scenario(
                    fx["home"],
                    fx["away"],
                    gw=int(gw),
                    priced_index=priced_index,
                    paths=paths,
                    league_gf=league_gf,
                    rng=rng,
                    ranks=ranks,
                    profile=profile,
                    sd_league_gf=0.0,
                )
                paths[fx["home"]].observe(sc["hg"], sc["ag"])
                paths[fx["away"]].observe(sc["ag"], sc["hg"])
                hn = norm_team(str(sc["home"]))
                an = norm_team(str(sc["away"]))
                home = share.loc[share["team_norm"].map(norm_team) == hn].copy()
                away = share.loc[share["team_norm"].map(norm_team) == an].copy()
                for club, k in ((home, int(sc["hg"])), (away, int(sc["ag"]))):
                    if club.empty:
                        continue
                    c_w, wG, rG = _club_weights(club)
                    on_xi = [
                        _pid_key(str(c_w.iloc[i][_id_col(c_w)])) in xi
                        for i in range(len(c_w))
                    ]
                    if any(on_xi) and len(wG):
                        n_xi_sides += 1
                        pi_rm = _pi_from_weights(wG)
                        pi_r = _pi_from_weights(rG)
                        pot_mass["comp_rm"] += float(
                            sum(pi_rm[i] for i, hit in enumerate(on_xi) if hit)
                        )
                        pot_mass["comp_r"] += float(
                            sum(pi_r[i] for i, hit in enumerate(on_xi) if hit)
                        )
                    for arm in ARMS:
                        got = _roll_arm(
                            club,
                            arm=arm,
                            goals_for=k,
                            alpha_g=alpha_g,
                            alpha_a=alpha_a,
                            rng=arm_rngs[arm],
                        )
                        for pid, g in got.items():
                            if pid in xi:
                                acc[arm][(int(gw), pid)] += float(g)

    n = float(args.n)
    for arm in ARMS:
        for key in list(acc[arm]):
            acc[arm][key] /= n

    rows: list[dict[str, Any]] = []
    for gw, xi in xi_by_gw.items():
        act_gw = pact.loc[pact["GW"] == int(gw)].set_index("element")
        for pid in xi:
            eid = _element(pid)
            if eid is None:
                continue
            if eid in act_gw.index:
                a = act_gw.loc[eid]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[0]
                g = float(a["goals"])
                name = str(a["name"])
                pos = str(a["position"])
            else:
                g, name, pos = 0.0, pid, ""
            rec: dict[str, Any] = {
                "gw": int(gw),
                "player_id": str(pid),
                "name": name,
                "position": pos,
                "G": g,
            }
            for arm in ARMS:
                e = float(acc[arm].get((int(gw), str(pid)), 0.0))
                rec[f"E_{arm}"] = e
                rec[f"resid_{arm}"] = e - g
            rows.append(rec)
    df = pd.DataFrame(rows)

    def _sum(col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

    summary: dict[str, Any] = {
        "n_xi_player_gw": int(len(df)),
        "sum_G": _sum("G"),
        "alpha_g": float(alpha_g),
        "arms": {},
        "xi_pot_mass_mean": {
            a: pot_mass[a] / max(n_xi_sides, 1) for a in ARMS
        },
        "n_xi_sides": int(n_xi_sides),
    }
    for arm in ARMS:
        summary["arms"][arm] = {
            "sum_E_G": _sum(f"E_{arm}"),
            "sum_resid": _sum(f"resid_{arm}"),
            "mae": float(df[f"resid_{arm}"].abs().mean()) if len(df) else float("nan"),
        }
    base = summary["arms"]["comp_rm"]["sum_resid"]
    summary["delta_resid_vs_comp_rm"] = {
        a: float(summary["arms"][a]["sum_resid"] - base) for a in ARMS
    }

    by_player = (
        df.groupby(["name", "position"], as_index=False)
        .agg(
            n=("G", "size"),
            G=("G", "sum"),
            E_comp_rm=("E_comp_rm", "sum"),
            E_comp_r=("E_comp_r", "sum"),
            r_comp_rm=("resid_comp_rm", "sum"),
            r_comp_r=("resid_comp_r", "sum"),
        )
        .sort_values("r_comp_rm", ascending=False)
    )

    d = summary["delta_resid_vs_comp_rm"]["comp_r"]
    read = (
        f"Locked **comp_rm** (π∝r·m) XI resid "
        f"**{summary['arms']['comp_rm']['sum_resid']:+.1f}**. "
        f"**comp_r** (π∝r, minutes gate only) "
        f"{summary['arms']['comp_r']['sum_resid']:+.1f} "
        f"(Δ {d:+.1f}). "
        "Dropping minutes from share weights "
        + (
            "cuts"
            if d < -0.5
            else ("worsens" if d > 0.5 else "barely moves")
        )
        + " the G overprint. Club pot still renormalises; only the weight changes."
    )

    state = {
        "task": "cascade_share_no_renorm_minutes",
        "season": season,
        "gws": plan_gws,
        "n_strands": int(args.n),
        "seed": seed0,
        "share_pool": "full_m_gt_0",
        "share_se_shrink": False,
        "arms": {
            "comp_rm": "π ∝ r·m/90, club renorm (locked)",
            "comp_r": "π ∝ r among m>0, club renorm (no minutes in weights)",
        },
        "elapsed_s": _now() - t0,
        "summary": summary,
        "read": read,
        "top_players": by_player.head(10).to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    lines = [
        "# Attack shares without minutes in the weights",
        "",
        f"Season **{season}** · OL XI GW{start_gw}–{end_gw} · full club m>0 · "
        f"N={args.n} · wall {state['elapsed_s']:.1f}s · no share-SE shrink",
        "",
        "## Arms",
        "",
        "| arm | rule |",
        "| --- | --- |",
        "| **comp_rm** | locked: ``w=r·m/90``, ``π=w/Σw`` |",
        "| **comp_r** | ``π=r/Σr`` among m>0 (minutes = on-dice gate only) |",
        "",
        "## OL XI Σ E[G] vs actual",
        "",
        f"Σ G = **{summary['sum_G']:.1f}** · α_G={alpha_g:.3f}",
        "",
        "| arm | Σ E[G] | Σ resid | Δ vs comp_rm | MAE | XI π mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        a = summary["arms"][arm]
        lines.append(
            f"| {arm} | {a['sum_E_G']:.2f} | {a['sum_resid']:+.2f} | "
            f"{summary['delta_resid_vs_comp_rm'][arm]:+.2f} | {a['mae']:.3f} | "
            f"{summary['xi_pot_mass_mean'][arm]:.3f} |"
        )
    lines += [
        "",
        "## Top resid players (comp_rm)",
        "",
        "| name | pos | G | E_rm | E_r | r_rm | r_r |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in by_player.head(10).itertuples(index=False):
        lines.append(
            f"| {r.name} | {r.position} | {r.G:.0f} | {r.E_comp_rm:.2f} | "
            f"{r.E_comp_r:.2f} | {r.r_comp_rm:+.2f} | {r.r_comp_r:+.2f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        read,
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_share_no_renorm_minutes --n {args.n}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_share_no_renorm_minutes",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Screen π∝r (no minutes in weights) vs locked π∝r·m",
                "verdict": {
                    "resid": {a: summary["arms"][a]["sum_resid"] for a in ARMS},
                    "delta_vs_comp_rm": summary["delta_resid_vs_comp_rm"],
                    "read": read,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(read)
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
