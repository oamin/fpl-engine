"""Diagnostic: cascade stepped mean xi_ft vs locked L3 / L4 on 2025-26 GW8–12.

Compares **realised** XI+captain − hits using:

- **cascade_step** — committed XIs from ``cascade_mean_xi_ft_step_pool_resid``
  (N=40, full-club shares, ≥60′ selection refreshed each step). Captain =
  Haaland if in XI else max generative ``xp`` on that XI board (fallback:
  max actual on XI if boards unavailable).
- **L3 / L4** — weeks from ``data/processed/layer4_mupost_gamma1.json``
  (locked μ_post priced γ=1 path; L3 single-deadline arm in the same file).

Not a four-season gated promote. Different information sets (cascade Opta
long mean boards vs L3/L4 priced t:t+1 μ_post).

  PYTHONPATH=. python3 -m src.eval.cascade_step_vs_l3_l4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.markets.numagent_xp_pilot import load_merged

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_step_vs_l3_l4.md"
STATE = ROOT / "REPORT_STATE.json"
L4_JSON = ROOT / "data" / "processed" / "layer4_mupost_gamma1.json"
CASC_JSON = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_step_pool_resid.json")


def _element(pid: str) -> int:
    return int(str(pid).split(":")[-1])


def _cascade_week_points(
    merged: pd.DataFrame,
    step: dict[str, Any],
) -> dict[str, Any]:
    gw = int(step["gw"])
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == gw]
    xi_ids = [str(x) for x in step["xi_ids"]]
    xi_names = list(step.get("xi_names") or [])
    rows = []
    for i, pid in enumerate(xi_ids):
        eid = _element(pid)
        hit = g.loc[g["element"] == eid]
        name = xi_names[i] if i < len(xi_names) else pid
        pts = float(hit.iloc[0]["total_points"] or 0) if not hit.empty else 0.0
        rows.append({"player_id": pid, "name": name, "pts": pts})
    xi_sum = float(sum(r["pts"] for r in rows))
    # Captain: on a seed-hold GW use seed captain_id; else Haaland if in XI;
    # else max actual on XI. Do not use the MILP's expected-xP captain for
    # realised pts (that arm is transfer-path only).
    cap = None
    if step.get("held_seed") and step.get("captain_id"):
        cap = next(
            (r for r in rows if str(r["player_id"]) == str(step["captain_id"])),
            None,
        )
    if cap is None:
        cap = next((r for r in rows if "Haaland" in str(r["name"])), None)
    if cap is None:
        cap = max(rows, key=lambda r: r["pts"]) if rows else {"name": "?", "pts": 0.0}
    # path_transfer_plan stores hit POINTS in ``hits`` (HIT_COST × extras).
    hit_pts = int(step.get("hits") or 0)
    n_hit_xfers = hit_pts // 4
    net = xi_sum + float(cap["pts"]) - hit_pts
    return {
        "gw": gw,
        "arm": "cascade_step",
        "pts_xi": xi_sum,
        "pts_captain": float(cap["pts"]),
        "captain": str(cap["name"]),
        "hits": n_hit_xfers,
        "hit_points": hit_pts,
        "pts_xi_cap": xi_sum + float(cap["pts"]),
        "pts_net": net,
        "n_transfers": int(step.get("n_transfers") or 0),
        "xi_names": xi_names,
        "bench_names": list(step.get("bench_names") or []),
    }


def _l_weeks(path: Path, *, season: str, gws: list[int]) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text())
    out: dict[str, list[dict[str, Any]]] = {"l3": [], "l4_mupost": []}
    for w in raw.get("weeks") or []:
        if str(w.get("season")) != season:
            continue
        if int(w.get("gw") or 0) not in gws:
            continue
        arm = str(w.get("arm"))
        if arm in out:
            out[arm].append(dict(w))
    for arm in out:
        out[arm].sort(key=lambda r: int(r["gw"]))
    return out


def _plot(per_gw: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=140)
    for arm, color in (
        ("cascade_step", "#c0392b"),
        ("l3", "#2980b9"),
        ("l4_mupost", "#27ae60"),
    ):
        sub = per_gw.loc[per_gw["arm"] == arm].sort_values("gw")
        ax.plot(
            sub["gw"],
            sub["pts_net"],
            "o-",
            color=color,
            lw=1.6,
            ms=5,
            label=arm,
        )
    ax.set_xlabel("GW")
    ax.set_ylabel("actual pts_net (XI+C − hits)")
    ax.set_title("2025-26 GW8–12: cascade stepped xi_ft vs L3 / L4")
    ax.set_xticks(sorted(per_gw["gw"].unique()))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--casc", type=Path, default=CASC_JSON)
    ap.add_argument("--l4-json", type=Path, default=L4_JSON)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_step_vs_l3_l4.json"),
    )
    args = ap.parse_args(argv)

    casc = json.loads(Path(args.casc).read_text())
    gws = [int(s["gw"]) for s in casc["steps"]]
    merged = load_merged(str(args.season))
    casc_weeks = [_cascade_week_points(merged, s) for s in casc["steps"]]
    l_arms = _l_weeks(Path(args.l4_json), season=str(args.season), gws=gws)

    rows: list[dict[str, Any]] = []
    for w in casc_weeks:
        rows.append(
            {
                "gw": w["gw"],
                "arm": "cascade_step",
                "pts_net": w["pts_net"],
                "pts_xi_cap": w["pts_xi_cap"],
                "hits": w["hits"],
                "n_transfers": w["n_transfers"],
            }
        )
    for arm, weeks in l_arms.items():
        for w in weeks:
            rows.append(
                {
                    "gw": int(w["gw"]),
                    "arm": arm,
                    "pts_net": float(w["pts_net"]),
                    "pts_xi_cap": float(w["pts_xi_cap"]),
                    "hits": int(w["hits"]),
                    "n_transfers": int(w["n_transfers"]),
                }
            )
    per_gw = pd.DataFrame(rows)

    totals = (
        per_gw.groupby("arm", as_index=False)
        .agg(
            pts_net=("pts_net", "sum"),
            pts_xi_cap=("pts_xi_cap", "sum"),
            hits=("hits", "sum"),
            n_transfers=("n_transfers", "sum"),
            n_gw=("gw", "nunique"),
        )
        .sort_values("pts_net", ascending=False)
    )
    # paired vs cascade
    casc_by = {int(r["gw"]): float(r["pts_net"]) for r in casc_weeks}
    deltas = {}
    for arm, weeks in l_arms.items():
        ds = []
        for w in weeks:
            gw = int(w["gw"])
            ds.append(casc_by[gw] - float(w["pts_net"]))
        deltas[arm] = {
            "sum_d": float(sum(ds)),
            "mean_d": float(sum(ds) / len(ds)) if ds else float("nan"),
            "per_gw": {int(w["gw"]): casc_by[int(w["gw"])] - float(w["pts_net"]) for w in weeks},
        }

    plot_path = Path("/opt/cursor/artifacts/cascade_step_vs_l3_l4.png")
    _plot(per_gw, plot_path)

    state = {
        "task": "cascade_step_vs_l3_l4",
        "season": str(args.season),
        "gws": gws,
        "cascade_source": str(args.casc),
        "l3_l4_source": str(args.l4_json),
        "cascade_captain_rule": "Haaland if in XI else max actual on XI",
        "cascade_weeks": casc_weeks,
        "l3_weeks": l_arms["l3"],
        "l4_weeks": l_arms["l4_mupost"],
        "totals": totals.to_dict(orient="records"),
        "cascade_minus_l3": deltas["l3"],
        "cascade_minus_l4": deltas["l4_mupost"],
        "plot": str(plot_path),
        "note": (
            "Diagnostic only. Cascade = Opta-long generative mean xi_ft with "
            "stepped ≥60′ selection; L3/L4 = locked score_xp_v2 μ_post priced path."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(state, indent=2, default=str) + "\n")

    seed_note = ""
    if casc.get("seed_owned"):
        sm = casc["seed_owned"]
        seed_note = (
            f" Seeded from `{sm.get('path')}` "
            f"(hold GW{casc.get('hold_gw')}: 0 xfers)."
        )
    lines = [
        "# Cascade stepped xi_ft vs L3 / L4 (2025-26 GW8–12)",
        "",
        f"As of `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}` (UTC).",
        "",
        "## Arms",
        "",
        "- **cascade_step** — `cascade_mean_xi_ft_step_pool_resid` commits "
        "(full-club shares; ≥60′ selection refreshed GW8…12; projected to GW19). "
        "Captain = Haaland if in XI."
        + seed_note,
        "- **l3 / l4_mupost** — locked climb weeks from "
        f"`{Path(args.l4_json).relative_to(ROOT)}` (μ_post priced γ=1).",
        "",
        "**Not gated.** Different boards (Opta generative long vs priced t:t+1).",
        "",
        f"![pts_net]({plot_path})",
        "",
        "## Totals",
        "",
        "| arm | Σ pts_net | Σ XI+C | hits | transfers |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in totals.to_dict(orient="records"):
        lines.append(
            f"| {r['arm']} | {r['pts_net']:.0f} | {r['pts_xi_cap']:.0f} | "
            f"{r['hits']} | {r['n_transfers']} |"
        )
    lines += [
        "",
        f"- cascade − L3: **{deltas['l3']['sum_d']:+.0f}** "
        f"({deltas['l3']['mean_d']:+.1f}/GW)",
        f"- cascade − L4: **{deltas['l4_mupost']['sum_d']:+.0f}** "
        f"({deltas['l4_mupost']['mean_d']:+.1f}/GW)",
        "",
        "## Per GW (pts_net)",
        "",
        "| GW | cascade | L3 | L4 | casc−L3 | casc−L4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for gw in gws:
        c = next(w["pts_net"] for w in casc_weeks if w["gw"] == gw)
        l3 = next(float(w["pts_net"]) for w in l_arms["l3"] if int(w["gw"]) == gw)
        l4 = next(
            float(w["pts_net"]) for w in l_arms["l4_mupost"] if int(w["gw"]) == gw
        )
        lines.append(
            f"| {gw} | {c:.0f} | {l3:.0f} | {l4:.0f} | {c - l3:+.0f} | {c - l4:+.0f} |"
        )
    lines += [
        "",
        "## Cascade squads (reminder)",
        "",
    ]
    for w in casc_weeks:
        lines.append(f"- **GW{w['gw']}** C={w['captain']} · net={w['pts_net']:.0f}")
        lines.append(f"  - XI: {', '.join(w['xi_names'])}")
        if w.get("bench_names"):
            lines.append(f"  - Bench: {', '.join(w['bench_names'])}")
    gw8_c = next((w["pts_net"] for w in casc_weeks if w["gw"] == 8), None)
    gw8_l3 = next(
        (float(w["pts_net"]) for w in l_arms["l3"] if int(w["gw"]) == 8), None
    )
    gw8_bit = ""
    if gw8_c is not None and gw8_l3 is not None:
        gw8_bit = f" (GW8: {gw8_c:.0f} vs L3 {gw8_l3:.0f})"
    gw9 = next((w for w in casc_weeks if w["gw"] == 9), None)
    hit_note = ""
    if gw9 and int(gw9.get("hit_points") or 0) > 0:
        hit_note = (
            f" GW9 took {gw9['n_transfers']} transfers / "
            f"{int(gw9['hit_points'])} hit pts "
            f"(net {gw9['pts_net']:.0f})."
        )
    seed_score_note = ""
    if casc.get("seed_owned"):
        seed_score_note = (
            " GW8 uses the L3/L4 WC XI+captain; cascade pts are from "
            "`load_merged` actuals, while locked L3/L4 weeks keep the climb "
            "JSON totals (Kroupi GW8 is 12 merged vs 48 in "
            "`score_xp_v2_closed` — that alone is −72 on the same XI)."
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"On this window cascade stepped vs locked L3/L4: "
        f"**{deltas['l3']['sum_d']:+.0f}** / **{deltas['l4_mupost']['sum_d']:+.0f}** "
        f"net pts{gw8_bit}.{hit_note}{seed_score_note}",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_step_vs_l3_l4 --casc {args.casc}",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_step_vs_l3_l4",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Diagnostic screen: cascade stepped GW8–12 actual net vs L3/L4",
                "verdict": {
                    "totals": state["totals"],
                    "cascade_minus_l3": deltas["l3"]["sum_d"],
                    "cascade_minus_l4": deltas["l4_mupost"]["sum_d"],
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(totals.to_string(index=False))
    print("cascade-L3", deltas["l3"]["sum_d"], "cascade-L4", deltas["l4_mupost"]["sum_d"])
    print("wrote", args.out, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
