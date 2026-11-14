"""Do match odds and minutes carry precision/recall weight on the Baudot gate?

Same mixer, Tversky, FOV, seed. Three arms:
  intact   — locked card
  no_odds  — 1X2 odds bits zeroed on every GW row (past and remainder)
  no_mins  — minutes bits zeroed on every player slot

Primary: holdout precision and recall at 20% by channel and by position.
Also a role-pool readout: actual vs painted event rate for the average
GKP/DEF/MID/FWD appearance (point 1 of the investigation agenda).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_baudot_row_trial import (
    ART,
    CLUB_BITS,
    CUT_FRACS,
    EVENT_BITS,
    GATE,
    GATE_META,
    GOAL_BITS,
    HEADER_BITS,
    MAX_PLAYERS,
    MINS_BITS,
    OPP_BITS,
    PLAYER_BITS,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    WHO_BITS,
    BaudotCard,
    BlockTokenMixer,
    _channel_slice,
    _player_pad,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    event_target,
    fit_clubs,
    load_odds,
    remainder_view,
    train_mixer,
)
from src.models.flashcard_card_score import balanced_score

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_odds_weight.md"
POS = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}
ODDS_LO = CLUB_BITS + OPP_BITS


def blank_card(card: BaudotCard, kind: str) -> BaudotCard:
    bits = card.bits.copy()
    if kind == "no_odds":
        bits[:, ODDS_LO:HEADER_BITS] = 0.0
    elif kind == "no_mins":
        for row in range(MAX_PLAYERS):
            base = HEADER_BITS + row * PLAYER_BITS + WHO_BITS
            bits[:, base : base + MINS_BITS] = 0.0
    elif kind != "intact":
        raise ValueError(kind)
    return replace(card, bits=bits)


def pr_tables(model, cards, device, frac: float = 0.2) -> dict:
    """Presence-level precision/recall at one cut, overall and by position."""
    model.eval()
    out = {}
    for channel in ("goals", "assists"):
        sl = _channel_slice(channel)
        buckets = {
            name: {"hit": 0, "fp": 0, "miss": 0, "tn": 0}
            for name in ("ALL", *POS.values())
        }
        for card in cards:
            cut = cut_at_fraction(card, frac)
            view = remainder_view(card, cut)
            pad = _player_pad(card)
            pred = model(
                torch.from_numpy(view["bits"][None]).to(device),
                torch.from_numpy(view["row_on"][None]).to(device),
                torch.from_numpy(pad[None]).to(device),
            )[0].cpu().numpy()
            for row in range(card.n_players):
                if pad[row]:
                    continue
                pos = POS.get(int(card.position[row]))
                if pos is None:
                    continue
                for gw in range(cut, card.n_fixtures):
                    if not view["present"][row, gw]:
                        continue
                    actual = int(event_target(card, gw)[row, sl].sum() > 0)
                    painted = int((pred[row, gw, sl] >= 0.5).sum() > 0)
                    key = (
                        "hit"
                        if actual and painted
                        else "fp"
                        if painted and not actual
                        else "miss"
                        if actual and not painted
                        else "tn"
                    )
                    buckets["ALL"][key] += 1
                    buckets[pos][key] += 1
        def pack(b):
            actual = b["hit"] + b["miss"]
            painted = b["hit"] + b["fp"]
            prec = b["hit"] / painted if painted else None
            rec = b["hit"] / actual if actual else None
            return {
                **b,
                "actual": actual,
                "painted": painted,
                "precision": prec,
                "recall": rec,
            }
        out[channel] = {name: pack(b) for name, b in buckets.items()}
    return out


def role_rates(model, cards, device, frac: float = 0.2) -> dict:
    """Average positional appearance: event rate truth vs paint."""
    model.eval()
    acc = {
        name: {"n": 0, "g_act": 0, "g_paint": 0, "a_act": 0, "a_paint": 0, "mins": 0.0}
        for name in POS.values()
    }
    for card in cards:
        cut = cut_at_fraction(card, frac)
        view = remainder_view(card, cut)
        pad = _player_pad(card)
        pred = model(
            torch.from_numpy(view["bits"][None]).to(device),
            torch.from_numpy(view["row_on"][None]).to(device),
            torch.from_numpy(pad[None]).to(device),
        )[0].cpu().numpy()
        for row in range(card.n_players):
            if pad[row]:
                continue
            pos = POS.get(int(card.position[row]))
            if pos is None:
                continue
            for gw in range(cut, card.n_fixtures):
                if not card.present[row, gw]:
                    continue
                acc[pos]["n"] += 1
                acc[pos]["mins"] += float(card.minutes[row, gw])
                g_act = int(card.goals[row, gw] > 0)
                a_act = int(card.assists[row, gw] > 0)
                g_p = int((pred[row, gw, :GOAL_BITS] >= 0.5).sum() > 0)
                a_p = int((pred[row, gw, GOAL_BITS:] >= 0.5).sum() > 0)
                acc[pos]["g_act"] += g_act
                acc[pos]["g_paint"] += g_p
                acc[pos]["a_act"] += a_act
                acc[pos]["a_paint"] += a_p
    rows = {}
    for name, b in acc.items():
        n = max(b["n"], 1)
        rows[name] = {
            "n": b["n"],
            "mins": b["mins"] / n,
            "goal_rate_actual": b["g_act"] / n,
            "goal_rate_painted": b["g_paint"] / n,
            "assist_rate_actual": b["a_act"] / n,
            "assist_rate_painted": b["a_paint"] / n,
        }
    return rows


def _fmt_pr(b: dict) -> str:
    p = "—" if b["precision"] is None else f"{b['precision']:.3f}"
    r = "—" if b["recall"] is None else f"{b['recall']:.3f}"
    return f"{b['actual']}/{b['painted']}/{b['hit']} · P {p} · R {r}"


def write_report(arms: list[dict]) -> Path:
    intact = next(a for a in arms if a["kind"] == "intact")
    lines = [
        "# Baudot odds / minutes weight (precision & recall)",
        "",
        "Same mixer, learnable Tversky, FOV, seed. Odds bits or minutes bits "
        "zeroed on every row. Judged on holdout **precision and recall** at 20%, "
        "not volume-matched balanced_score alone.",
        "",
        f"Gate locked mean is {float(np.mean([balanced_score(GATE[k])['score'] for k in GATE])):.3f}. "
        "This replay's intact arm is the within-run control.",
        "",
        "## League presence at 20%",
        "",
        "| arm | goals a/p/h | goals P | goals R | assists a/p/h | assists P | assists R | mean score |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for a in arms:
        g = a["pr20"]["goals"]["ALL"]
        s = a["pr20"]["assists"]["ALL"]
        gp = "—" if g["precision"] is None else f"{g['precision']:.3f}"
        gr = "—" if g["recall"] is None else f"{g['recall']:.3f}"
        sp = "—" if s["precision"] is None else f"{s['precision']:.3f}"
        sr = "—" if s["recall"] is None else f"{s['recall']:.3f}"
        lines.append(
            f"| {a['kind']} | {g['actual']}/{g['painted']}/{g['hit']} | {gp} | {gr} | "
            f"{s['actual']}/{s['painted']}/{s['hit']} | {sp} | {sr} | {a['mean_score']:.3f} |"
        )
    lines.extend(["", "## Goals at 20% by position (a/p/h · P · R)", ""])
    lines.append("| arm | GKP | DEF | MID | FWD |")
    lines.append("|---|---|---|---|---|")
    for a in arms:
        g = a["pr20"]["goals"]
        lines.append(
            "| "
            + a["kind"]
            + " | "
            + " | ".join(_fmt_pr(g[p]) for p in POS.values())
            + " |"
        )
    lines.extend(["", "## Average positional appearance (20% remainder)", ""])
    lines.append(
        "Rate = share of present player-GW rows with ≥1 event. "
        "Painted rate is the model's thr-0.5 positive rate on that pool."
    )
    lines.append("")
    lines.append("| arm | pos | n | mean mins | goal rate act | goal rate paint | assist rate act | assist rate paint |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for a in arms:
        for pos, row in a["roles"].items():
            lines.append(
                f"| {a['kind']} | {pos} | {row['n']} | {row['mins']:.1f} | "
                f"{row['goal_rate_actual']:.3f} | {row['goal_rate_painted']:.3f} | "
                f"{row['assist_rate_actual']:.3f} | {row['assist_rate_painted']:.3f} |"
            )
    # Call
    def pr(arm, channel):
        b = arm["pr20"][channel]["ALL"]
        return b["precision"] or 0.0, b["recall"] or 0.0

    no_odds = next(a for a in arms if a["kind"] == "no_odds")
    no_mins = next(a for a in arms if a["kind"] == "no_mins")
    igp, igr = pr(intact, "goals")
    ogp, ogr = pr(no_odds, "goals")
    mgp, mgr = pr(no_mins, "goals")
    odds_drop = (igp - ogp) + (igr - ogr)
    mins_drop = (igp - mgp) + (igr - mgr)
    if odds_drop < 0.02 and mins_drop >= 0.04:
        call = (
            "**Odds are not carrying goal precision/recall.** Blanking the 1X2 strip "
            f"changes goals P/R by {ogp - igp:+.3f} / {ogr - igr:+.3f}. "
            f"Blanking minutes moves them by {mgp - igp:+.3f} / {mgr - igr:+.3f}. "
            "Match-up dynamics are not in the odds bits; a later opponent-card or "
            "conceded strip has to add information the 1X2 fill does not."
        )
    elif odds_drop >= 0.04:
        call = (
            "**Odds carry measurable goal placement weight.** "
            f"Blanking them moves goals P/R by {ogp - igp:+.3f} / {ogr - igr:+.3f} "
            f"(minutes blank {mgp - igp:+.3f} / {mgr - igr:+.3f}). "
            "Keep the strip; coupling work should add to it, not replace it."
        )
    else:
        call = (
            "**Both strips are weak on goal placement relative to the intact card.** "
            f"Odds ΔP/ΔR {ogp - igp:+.3f}/{ogr - igr:+.3f}; "
            f"minutes {mgp - igp:+.3f}/{mgr - igr:+.3f}. "
            "The painter is mostly using role and identity, not the two headline influencers."
        )
    lines.extend(["", "## Call", "", call, ""])
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def _mean_from_eval(model, cards, device) -> float:
    from src.models.flashcard_baudot_row_trial import evaluate_mixer

    rows = evaluate_mixer(model, cards, device)["rows"]
    return float(np.mean([r["score"] for r in rows]))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    print("loading", flush=True)
    league = build_league(seasons)
    odds = load_odds()
    codes = build_name_codes(seasons)
    squad = build_cards(seasons, league)
    clubs = fit_clubs([c for c in squad if c.season in TRAIN_SEASONS])
    for card in squad:
        clubs.setdefault(card.team, len(clubs) + 1)
        for opp in card.opponent:
            if opp:
                clubs.setdefault(opp, len(clubs) + 1)
    baudot = build_baudot_cards(squad, clubs, codes, odds)
    train = [c for c in baudot if c.season in TRAIN_SEASONS]
    test = [c for c in baudot if c.season == TEST_SEASON]
    width = train[0].width
    arms_out = []
    t0 = time.time()
    for kind in ("intact", "no_odds", "no_mins"):
        print(f"\n=== {kind} ===", flush=True)
        tr = [blank_card(c, kind) for c in train]
        te = [blank_card(c, kind) for c in test]
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = BlockTokenMixer(width=width).to(device)
        meta = train_mixer(model, tr, rng, device, test=te)
        pr = pr_tables(model, te, device, 0.2)
        roles = role_rates(model, te, device, 0.2)
        mean = _mean_from_eval(model, te, device)
        g = pr["goals"]["ALL"]
        print(
            f"{kind} mean {mean:.3f} goals P {g['precision']} R {g['recall']} "
            f"a/p/h {g['actual']}/{g['painted']}/{g['hit']}",
            flush=True,
        )
        arms_out.append(
            {
                "kind": kind,
                "epochs_run": meta["epochs_run"],
                "mean_score": mean,
                "final_alpha": meta["final_alpha"],
                "final_beta": meta["final_beta"],
                "pr20": pr,
                "roles": roles,
            }
        )
    path = write_report(arms_out)
    state = {
        "trial": "baudot_odds_weight",
        "gate": GATE_META,
        "seconds": round(time.time() - t0, 1),
        "arms": arms_out,
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (ART / "baudot_odds_weight.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
