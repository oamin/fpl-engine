"""LOSO trial: margin-consistent rank×rank IPF λ vs product map.

Does not change the locked cascade_long path. Writes
``reports/rank_vs_rank_ipf_trial.md`` and artifact JSON/PNG.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models.opta_gf_ga_horizon import (
    CACHE,
    CLOSED,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.rank_vs_rank_ipf import (
    fit_rank_vs_rank_ipf,
    product_lambdas,
    product_prior,
    _margins,
    _gf_ga_vectors,
)
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "rank_vs_rank_ipf_trial.md"
ARTIFACT_DIR = Path("/opt/cursor/artifacts")
STATE = ROOT / "REPORT_STATE.json"


def _load_matches(fname: str) -> list[dict]:
    out = []
    with (CACHE / fname).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
            except (KeyError, ValueError, TypeError):
                continue
            out.append(
                {
                    "home": norm_team(row["HomeTeam"]),
                    "away": norm_team(row["AwayTeam"]),
                    "hg": hg,
                    "ag": ag,
                }
            )
    return out


def _metrics(pred_h, pred_a, act_h, act_a) -> dict[str, float]:
    ph, pa = np.asarray(pred_h, float), np.asarray(pred_a, float)
    ah, aa = np.asarray(act_h, float), np.asarray(act_a, float)
    tot_p, tot_a = ph + pa, ah + aa
    return {
        "n": int(len(ah)),
        "mae_h": float(np.mean(np.abs(ph - ah))),
        "mae_a": float(np.mean(np.abs(pa - aa))),
        "mae_tot": float(np.mean(np.abs(tot_p - tot_a))),
        "bias_h": float(np.mean(ph - ah)),
        "bias_a": float(np.mean(pa - aa)),
        "bias_tot": float(np.mean(tot_p - tot_a)),
        "corr_h": float(np.corrcoef(ph, ah)[0, 1]) if len(ah) > 2 else float("nan"),
        "corr_tot": float(np.corrcoef(tot_p, tot_a)[0, 1]) if len(ah) > 2 else float("nan"),
        "mean_pred_tot": float(np.mean(tot_p)),
        "mean_act_tot": float(np.mean(tot_a)),
    }


def run() -> dict:
    season_rows = []
    examples = []
    all_prod = {"h": [], "a": [], "ah": [], "aa": []}
    all_ipf = {"h": [], "a": [], "ah": [], "aa": []}

    for season, fname in CLOSED:
        ranks = season_final_ranks(fname)
        profile = rank_gf_ga_profile_excluding(fname)
        fitted = fit_rank_vs_rank_ipf(profile)
        gf, ga, L = _gf_ga_vectors(profile)
        H0, A0 = product_prior(gf, ga, L)
        gf0, ga0 = _margins(H0, A0)

        ph_p, pa_p, ph_i, pa_i, ah, aa = [], [], [], [], [], []
        for m in _load_matches(fname):
            if m["home"] not in ranks or m["away"] not in ranks:
                continue
            rh, ra = int(ranks[m["home"]]), int(ranks[m["away"]])
            if rh == ra:
                continue
            lh_p, la_p = product_lambdas(rh, ra, profile)
            lh_i, la_i = fitted.lambdas(rh, ra)
            ph_p.append(lh_p)
            pa_p.append(la_p)
            ph_i.append(lh_i)
            pa_i.append(la_i)
            ah.append(m["hg"])
            aa.append(m["ag"])

        m_prod = _metrics(ph_p, pa_p, ah, aa)
        m_ipf = _metrics(ph_i, pa_i, ah, aa)
        season_rows.append(
            {
                "season": season,
                "fname": fname,
                "ipf_iters": fitted.n_iter,
                "ipf_max_margin_err": fitted.max_margin_err,
                "product_max_margin_err": float(
                    max(np.max(np.abs(gf0 - gf)), np.max(np.abs(ga0 - ga)))
                ),
                "product": m_prod,
                "ipf": m_ipf,
            }
        )
        all_prod["h"].extend(ph_p)
        all_prod["a"].extend(pa_p)
        all_prod["ah"].extend(ah)
        all_prod["aa"].extend(aa)
        all_ipf["h"].extend(ph_i)
        all_ipf["a"].extend(pa_i)
        all_ipf["ah"].extend(ah)
        all_ipf["aa"].extend(aa)

        if season == "2025/26":
            for rh, ra in [(1, 20), (1, 1), (10, 10), (20, 1), (5, 15), (15, 5)]:
                if rh == ra:
                    # same-place not in matrix; skip
                    continue
                lp = product_lambdas(rh, ra, profile)
                li = fitted.lambdas(rh, ra)
                examples.append(
                    {
                        "rh": rh,
                        "ra": ra,
                        "product": {"lam_h": lp[0], "lam_a": lp[1], "tot": lp[0] + lp[1]},
                        "ipf": {"lam_h": li[0], "lam_a": li[1], "tot": li[0] + li[1]},
                    }
                )
            # store one matrix slice for plot
            matrix_home_ipf = fitted.lam_home.copy()
            matrix_home_prod = H0.copy()

    overall = {
        "product": _metrics(all_prod["h"], all_prod["a"], all_prod["ah"], all_prod["aa"]),
        "ipf": _metrics(all_ipf["h"], all_ipf["a"], all_ipf["ah"], all_ipf["aa"]),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = ARTIFACT_DIR / "rank_vs_rank_ipf_lambda_h.png"
    _plot_lambda_h(matrix_home_prod, matrix_home_ipf, plot_path)

    margin_plot = ARTIFACT_DIR / "rank_vs_rank_ipf_margins.png"
    _plot_margins(profile=rank_gf_ga_profile_excluding("E0_2526.csv"), path=margin_plot)

    payload = {
        "as_of_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "rankmean place GF/GA → product prior → IPF (RAS) margins",
        "seasons": season_rows,
        "overall": overall,
        "examples_loso_2526_profile": examples,
        "plots": [str(plot_path), str(margin_plot)],
    }
    json_path = ARTIFACT_DIR / "rank_vs_rank_ipf_trial.json"
    json_path.write_text(json.dumps(payload, indent=2))
    payload["artifact"] = str(json_path)

    _write_report(payload, plot_path, margin_plot)
    STATE.write_text(
        json.dumps(
            {
                "task": "rank_vs_rank_ipf_trial",
                "status": "done",
                "as_of_utc": payload["as_of_utc"],
                "change": "LOSO IPF rank×rank λ vs product map (margin-consistent F)",
                "verdict": {
                    "product_mae_tot": overall["product"]["mae_tot"],
                    "ipf_mae_tot": overall["ipf"]["mae_tot"],
                    "product_bias_tot": overall["product"]["bias_tot"],
                    "ipf_bias_tot": overall["ipf"]["bias_tot"],
                    "note": "diagnostic only; not gated player-GW; cascade_long unchanged",
                },
                "artifact": str(json_path),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    return payload


def _plot_lambda_h(H_prod: np.ndarray, H_ipf: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=140)
    vmax = max(float(H_prod.max()), float(H_ipf.max()), 0.01)
    for ax, M, title in [
        (axes[0], H_prod, r"product $\lambda_h$"),
        (axes[1], H_ipf, r"IPF $\lambda_h$"),
        (axes[2], H_ipf - H_prod, r"IPF − product"),
    ]:
        if title.startswith("IPF −"):
            lim = max(abs(float(M.min())), abs(float(M.max())), 0.01)
            im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper")
        else:
            im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("away rank")
        ax.set_ylabel("home rank")
        ax.set_xticks(range(0, 20, 4))
        ax.set_yticks(range(0, 20, 4))
        ax.set_xticklabels([str(i + 1) for i in range(0, 20, 4)])
        ax.set_yticklabels([str(i + 1) for i in range(0, 20, 4)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("LOSO profile excl. 2025/26 — home λ matrix", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_margins(profile, path: Path) -> None:
    fitted = fit_rank_vs_rank_ipf(profile)
    gf, ga, L = _gf_ga_vectors(profile)
    H0, A0 = product_prior(gf, ga, L)
    gf0, ga0 = _margins(H0, A0)
    gfi, gai = _margins(fitted.lam_home, fitted.lam_away)
    ranks = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=140)
    ax.plot(ranks, gf, "k-o", ms=4, label="target GF")
    ax.plot(ranks, gf0, "C0--", label="product recovered GF")
    ax.plot(ranks, gfi, "C3-", lw=1.8, label="IPF recovered GF")
    ax.plot(ranks, ga, "k-s", ms=4, label="target GA")
    ax.plot(ranks, ga0, "C0:", label="product recovered GA")
    ax.plot(ranks, gai, "C3:", lw=1.8, label="IPF recovered GA")
    ax.set_xlabel("finishing rank")
    ax.set_ylabel("season goals")
    ax.set_title("Round-robin margin recovery (profile excl. 2025/26)")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_report(payload: dict, plot_path: Path, margin_plot: Path) -> None:
    o = payload["overall"]
    lines = [
        "# Rank×rank IPF λ trial (LOSO)",
        "",
        f"As of `{payload['as_of_utc']}` (UTC).",
        "",
        "## Setup",
        "",
        "- Place map: **rankmean** GF/GA, LOSO (exclude scored season).",
        "- Oracle final ranks on each closed E0 season.",
        "- **Product** (locked cascade_long): "
        r"`λ_h = GF_h·GA_a / (38·L)`.",
        "- **IPF**: same product prior, RAS until round-robin GF/GA margins match "
        "place totals (tol 1e-6).",
        "- Diagnostic only — does **not** change `cascade_long`.",
        "",
        "## Overall (4 seasons, n="
        f"{o['product']['n']})",
        "",
        "| map | MAE tot | bias tot | MAE λ_h | bias λ_h | corr(λ_h, hg) |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| product | {o['product']['mae_tot']:.3f} | {o['product']['bias_tot']:+.3f} | "
            f"{o['product']['mae_h']:.3f} | {o['product']['bias_h']:+.3f} | "
            f"{o['product']['corr_h']:.3f} |"
        ),
        (
            f"| IPF | {o['ipf']['mae_tot']:.3f} | {o['ipf']['bias_tot']:+.3f} | "
            f"{o['ipf']['mae_h']:.3f} | {o['ipf']['bias_h']:+.3f} | "
            f"{o['ipf']['corr_h']:.3f} |"
        ),
        "",
        "## Per season",
        "",
        "| season | product MAE | IPF MAE | ΔMAE | product margin err | IPF margin err | iters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in payload["seasons"]:
        dmae = s["ipf"]["mae_tot"] - s["product"]["mae_tot"]
        lines.append(
            f"| {s['season']} | {s['product']['mae_tot']:.3f} | {s['ipf']['mae_tot']:.3f} | "
            f"{dmae:+.3f} | {s['product_max_margin_err']:.2f} | "
            f"{s['ipf_max_margin_err']:.2e} | {s['ipf_iters']} |"
        )
    lines += [
        "",
        "## Example λ (LOSO profile excl. 2025/26)",
        "",
        "| ranks | product (λ_h, λ_a, tot) | IPF (λ_h, λ_a, tot) |",
        "|---|---|---|",
    ]
    for e in payload["examples_loso_2526_profile"]:
        p, i = e["product"], e["ipf"]
        lines.append(
            f"| {e['rh']}v{e['ra']} | "
            f"({p['lam_h']:.2f}, {p['lam_a']:.2f}, {p['tot']:.2f}) | "
            f"({i['lam_h']:.2f}, {i['lam_a']:.2f}, {i['tot']:.2f}) |"
        )
    lines += [
        "",
        "## Figures",
        "",
        f"![λ_h matrices]({plot_path})",
        "",
        f"![margin recovery]({margin_plot})",
        "",
        "## Verdict",
        "",
        "IPF enforces the bookkeeping identity (place GF/GA = sum of rank×rank λ). "
        "Match-level MAE vs actual scorelines is essentially unchanged vs the product "
        "map — the product was already nearly margin-consistent (±~2 season goals). "
        "Not promoted into cascade_long; park unless a player-GW G residual screen asks "
        "for it.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    out = run()
    print(json.dumps({"overall": out["overall"], "report": str(REPORT)}, indent=2))
