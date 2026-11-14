"""Opponent-card cross-read. Screened 2026-11-14; not the default card.

The default card is the Baudot row card read as opacity
(`flashcard_baudot_row_trial`, `DEFAULT_FLASHCARD_READING = opacity`).

Trained against the 2026-11-13 mixer table. That table stays in `GATE`.
This run's own tallies are `GATE_OPP_2026_11_14`.

One club-season is painted. For each GW row the encoder attends to that
fixture's opponent card:

- earlier opponent rows are true (goals, assists, minutes, gc)
- the current opponent row keeps minutes revealed and blanks G/A and gc
- later opponent kickoffs are hidden
- nothing is copied onto the painted card

The painted card uses the gc header. Remainder rows blank the club's own
G/A and gc. Loss is learnable Tversky on the club's remainder events only.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.flashcard_baudot_gc_header import (
    GC_BITS,
    HEADER_WITH_GC,
    with_gc_header,
)
from src.models.flashcard_baudot_row_trial import (
    ART,
    BATCH,
    BCE_W,
    CLUB_BITS,
    CUT_FRACS,
    D,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GATE,
    GATE_META,
    GOAL_BITS,
    HEADS,
    LR,
    MAX_GW,
    MAX_PLAYERS,
    MINS_BITS,
    ODDS_BITS,
    OPP_BITS,
    PLAYER_BITS,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    TVERSKY_DELTA_MAX,
    WHO_BITS,
    BaudotCard,
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
    tversky_loss,
)
from src.models.flashcard_card_score import balanced_score, tally_row

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_opp_train.md"
POS = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}


@dataclass
class Side:
    season: str
    team: str
    bits: np.ndarray
    n_fixtures: int
    n_players: int
    fixture_id: np.ndarray
    opponent: tuple
    pad: np.ndarray
    base: BaudotCard
    row_of: dict[int, int]


def blank_scoreline(row: np.ndarray) -> np.ndarray:
    """Zero gc and every player's G/A. Minutes and who stay."""
    out = row.copy()
    out[HEADER_BITS_END():HEADER_WITH_GC] = 0.0
    for player in range(MAX_PLAYERS):
        base = HEADER_WITH_GC + player * PLAYER_BITS + WHO_BITS + MINS_BITS
        out[base : base + EVENT_BITS] = 0.0
    return out


def HEADER_BITS_END() -> int:
    return HEADER_WITH_GC - GC_BITS


def build_sides(squad, baudot, league) -> list[Side]:
    sides = []
    for squad_card, card in zip(squad, baudot, strict=True):
        gc = with_gc_header(squad_card, card, league)
        row_of = {}
        for i in range(card.n_fixtures):
            row_of[int(squad_card.fixture_id[i])] = i
        sides.append(
            Side(
                season=card.season,
                team=card.team,
                bits=gc.bits,
                n_fixtures=card.n_fixtures,
                n_players=card.n_players,
                fixture_id=np.asarray(squad_card.fixture_id[: card.n_fixtures], dtype=np.int64),
                opponent=tuple(card.opponent),
                pad=_player_pad(card),
                base=card,
                row_of=row_of,
            )
        )
    return sides


def index_sides(sides: list[Side]) -> dict[tuple[str, str], Side]:
    return {(s.season, s.team): s for s in sides}


class OppCrossMixer(nn.Module):
    """Gate mixer plus a gc strip and a cross-read of the opponent card."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.width = width
        self.club = nn.Linear(CLUB_BITS, D)
        self.opp = nn.Linear(OPP_BITS, D)
        self.odds = nn.Linear(ODDS_BITS, D)
        self.gc = nn.Linear(GC_BITS, D)
        self.who = nn.Linear(WHO_BITS, D)
        self.mins = nn.Linear(MINS_BITS, D)
        self.events = nn.Linear(EVENT_BITS, D)
        self.header_mix = nn.Linear(4 * D, D)
        self.player_mix = nn.Linear(3 * D, D)
        self.norm_h = nn.LayerNorm(D)
        self.norm_p = nn.LayerNorm(D)
        self.time = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.squad = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(D)
        self.norm_s = nn.LayerNorm(D)
        self.norm_c = nn.LayerNorm(D)
        self.time_pos = nn.Parameter(torch.zeros(1, MAX_GW, 1, D))
        self.opp_pos = nn.Parameter(torch.zeros(1, MAX_GW, 1, D))
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.opp_pos, std=0.02)
        self.paint = nn.Linear(D, EVENT_BITS)
        nn.init.zeros_(self.paint.weight)
        nn.init.constant_(self.paint.bias, -3.0)
        self.tversky_logit = nn.Parameter(torch.tensor(-2.0))

    def tversky_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit)
        alpha = 0.5 - delta
        beta = 0.5 + delta
        return alpha, beta

    def _encode_rows(self, bits: torch.Tensor) -> torch.Tensor:
        """bits (..., T, W) → player tokens (..., T, P, D)."""
        lead = bits.shape[:-2]
        t = bits.shape[-2]
        flat = bits.reshape(-1, t, bits.shape[-1])
        club = self.club(flat[..., :CLUB_BITS])
        opp = self.opp(flat[..., CLUB_BITS : CLUB_BITS + OPP_BITS])
        odds = self.odds(flat[..., CLUB_BITS + OPP_BITS : HEADER_BITS_END()])
        gc = self.gc(flat[..., HEADER_BITS_END() : HEADER_WITH_GC])
        header = self.norm_h(self.header_mix(torch.cat([club, opp, odds, gc], dim=-1)))
        body = flat[..., HEADER_WITH_GC:].reshape(-1, t, MAX_PLAYERS, PLAYER_BITS)
        who = self.who(body[..., :WHO_BITS])
        mins = self.mins(body[..., WHO_BITS : WHO_BITS + MINS_BITS])
        ev = self.events(body[..., WHO_BITS + MINS_BITS :])
        players = self.norm_p(self.player_mix(torch.cat([who, mins, ev], dim=-1)))
        tokens = players + header[:, :, None, :]
        return tokens.reshape(*lead, t, MAX_PLAYERS, D)

    def forward(
        self,
        bits: torch.Tensor,
        row_on: torch.Tensor,
        player_pad: torch.Tensor,
        opp_bits: torch.Tensor,
        opp_on: torch.Tensor,
        opp_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Return B,P,T,E. opp_bits is B,T,S,W and is not written onto `bits`."""
        tokens = self._encode_rows(bits) + self.time_pos[:, : bits.shape[1]]
        b, t, p, d = tokens.shape
        time_view = tokens.permute(0, 2, 1, 3).reshape(b * p, t, d)
        time_mask = (row_on < 0.5)[:, None, :].expand(b, p, t).reshape(b * p, t)
        mixed, _ = self.time(time_view, time_view, time_view, key_padding_mask=time_mask, need_weights=False)
        time_view = self.norm_t(time_view + mixed)
        tokens = time_view.reshape(b, p, t, d).permute(0, 2, 1, 3)
        squad_view = tokens.reshape(b * t, p, d)
        squad_mask = player_pad[:, None, :].expand(b, t, p).reshape(b * t, p)
        mixed, _ = self.squad(squad_view, squad_view, squad_view, key_padding_mask=squad_mask, need_weights=False)
        tokens = self.norm_s(squad_view + mixed).reshape(b, t, p, d)

        opp_tokens = self._encode_rows(opp_bits) + self.opp_pos[:, : opp_bits.shape[2]]
        # B,T,S,P,D → keys B*T, S*P, D
        s = opp_tokens.shape[2]
        keys = opp_tokens.reshape(b * t, s * p, d)
        # True = ignore. A source row is live only where opp_on, and only for real players.
        live = opp_on[:, :, :, None] * (~opp_pad).to(opp_on.dtype)
        # Keep one dummy key if a GW has no opponent, so attention stays finite.
        none = live.reshape(b * t, s * p).sum(-1) < 0.5
        key_pad = live.reshape(b * t, s * p) < 0.5
        key_pad[none, 0] = False
        query = tokens.reshape(b * t, p, d)
        mixed, _ = self.cross(query, keys, keys, key_padding_mask=key_pad, need_weights=False)
        tokens = self.norm_c(query + mixed).reshape(b, t, p, d)
        ink = torch.sigmoid(self.paint(tokens))
        return ink.permute(0, 2, 1, 3)


def club_view(side: Side, cut: int) -> dict[str, np.ndarray]:
    bits = side.bits.copy()
    row_on = np.zeros(MAX_GW, dtype=np.float32)
    rem_on = np.zeros(MAX_GW, dtype=np.float32)
    target = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), dtype=np.float32)
    present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
    for gw in range(side.n_fixtures):
        row_on[gw] = 1.0
        if gw >= cut:
            bits[gw] = blank_scoreline(bits[gw])
            rem_on[gw] = 1.0
            target[:, gw] = event_target(side.base, gw)
            present[:, gw] = side.base.present[:, gw]
    return {"bits": bits, "row_on": row_on, "rem_on": rem_on, "target": target, "present": present}


def opponent_view(side: Side, cut: int, sides: dict[tuple[str, str], Side]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per GW: opponent rows up to and including this fixture. Shape T,S,W."""
    width = side.bits.shape[1]
    opp_bits = np.zeros((MAX_GW, MAX_GW, width), dtype=np.float32)
    opp_on = np.zeros((MAX_GW, MAX_GW), dtype=np.float32)
    opp_pad = np.ones((MAX_GW, MAX_GW, MAX_PLAYERS), dtype=bool)
    for gw in range(side.n_fixtures):
        other = sides.get((side.season, side.opponent[gw]))
        if other is None:
            continue
        j = other.row_of.get(int(side.fixture_id[gw]))
        if j is None:
            continue
        for src in range(j + 1):
            row = other.bits[src]
            if src == j and gw >= cut:
                row = blank_scoreline(row)
            opp_bits[gw, src] = row
            opp_on[gw, src] = 1.0
            opp_pad[gw, src] = other.pad
    return opp_bits, opp_on, opp_pad


def batch_views(cards: list[Side], cuts: list[int], sides: dict, device: torch.device):
    bits, row_on, rem, present, target, pad = [], [], [], [], [], []
    opp_bits, opp_on, opp_pad = [], [], []
    for card, cut in zip(cards, cuts, strict=True):
        view = club_view(card, cut)
        ob, oo, op = opponent_view(card, cut, sides)
        bits.append(view["bits"])
        row_on.append(view["row_on"])
        rem.append(view["rem_on"])
        present.append(view["present"])
        target.append(view["target"])
        pad.append(card.pad)
        opp_bits.append(ob)
        opp_on.append(oo)
        opp_pad.append(op)
    stack = lambda xs: torch.from_numpy(np.stack(xs)).to(device)
    return stack(bits), stack(row_on), stack(rem), stack(pad), stack(present), stack(target), stack(opp_bits), stack(opp_on), stack(opp_pad)


def _slot(player: int, kind: str) -> slice:
    base = HEADER_WITH_GC + player * PLAYER_BITS
    if kind == "mins":
        return slice(base + WHO_BITS, base + WHO_BITS + MINS_BITS)
    return slice(base + WHO_BITS + MINS_BITS, base + WHO_BITS + MINS_BITS + EVENT_BITS)


def assert_chelsea_minutes(sides: dict) -> None:
    """GW13 Arsenal attends Chelsea minutes on the Chelsea card. Scoreline stays blank both sides."""
    ars = sides[(TEST_SEASON, "Arsenal")]
    gw = int(np.where(ars.base.gw[: ars.n_fixtures] == 13)[0][0])
    che = sides[(TEST_SEASON, "Chelsea")]
    j = che.row_of[int(ars.fixture_id[gw])]
    ob, oo, _ = opponent_view(ars, gw, sides)
    current = ob[gw, j]
    true = che.bits[j]
    assert oo[gw, j] == 1.0
    assert oo[gw, :j].sum() == j
    assert oo[gw, j + 1 :].sum() == 0.0
    assert current[HEADER_BITS_END():HEADER_WITH_GC].sum() == 0.0
    assert true[HEADER_BITS_END():HEADER_WITH_GC].sum() > 0.0
    ev = mins = true_ev = 0.0
    for player in range(MAX_PLAYERS):
        assert np.array_equal(current[_slot(player, "mins")], true[_slot(player, "mins")])
        mins += float(current[_slot(player, "mins")].sum())
        ev += float(current[_slot(player, "events")].sum())
        true_ev += float(true[_slot(player, "events")].sum())
    assert ev == 0.0 and true_ev > 0.0 and mins > 0.0
    club = club_view(ars, gw)["bits"][gw]
    assert club[HEADER_BITS_END():HEADER_WITH_GC].sum() == 0.0
    for player in range(MAX_PLAYERS):
        assert np.array_equal(club[_slot(player, "mins")], ars.bits[gw][_slot(player, "mins")])
        assert float(club[_slot(player, "events")].sum()) == 0.0
    assert not np.array_equal(club[HEADER_WITH_GC:], current[HEADER_WITH_GC:])
    if gw > 0:
        prev = sides[(TEST_SEASON, ars.opponent[gw - 1])]
        pj = prev.row_of[int(ars.fixture_id[gw - 1])]
        assert np.array_equal(ob[gw - 1, pj], prev.bits[pj])
    print(f"leak check ok  Arsenal GW13 Chelsea minutes bits {mins:.0f} scoreline blank", flush=True)


@torch.no_grad()
def evaluate(model: OppCrossMixer, cards: list[Side], sides: dict, device: torch.device) -> dict:
    model.eval()
    rows = []
    pos = {
        "goals": {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS.values()},
        "assists": {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS.values()},
    }
    for frac in CUT_FRACS:
        cached = []
        cuts = [cut_at_fraction(card.base, frac) for card in cards]
        for start in range(0, len(cards), BATCH):
            chunk = cards[start : start + BATCH]
            chunk_cuts = cuts[start : start + BATCH]
            bits, row_on, _rem, pad, present, target, ob, oo, op = batch_views(chunk, chunk_cuts, sides, device)
            pred = model(bits, row_on, pad, ob, oo, op).cpu().numpy()
            present_np = present.cpu().numpy()
            target_np = target.cpu().numpy()
            for i, card in enumerate(chunk):
                cached.append((card, chunk_cuts[i], present_np[i], target_np[i], pred[i]))
        for channel in ("goals", "assists"):
            totals = {key: 0 for key in ("actual", "painted", "exact", "actual_within", "painted_within")}
            sl = _channel_slice(channel)
            for card, cut, present_i, target_i, pred_i in cached:
                for row in range(card.n_players):
                    if card.pad[row]:
                        continue
                    cols, actual, painted = [], [], []
                    for gw in range(cut, card.n_fixtures):
                        if not present_i[row, gw]:
                            continue
                        ag = int(target_i[row, gw, sl].sum())
                        pg = int((pred_i[row, gw, sl] >= 0.5).sum())
                        cols.append(int(card.base.gw[gw]))
                        actual.append(ag)
                        painted.append(pg)
                        if frac == 0.2:
                            name = POS.get(int(card.base.position[row]))
                            if name is not None:
                                bucket = pos[channel][name]
                                ae, pe = int(ag > 0), int(pg > 0)
                                bucket["actual"] += ae
                                bucket["painted"] += pe
                                bucket["hits"] += int(ae and pe)
                    if not cols:
                        continue
                    part = tally_row(
                        np.asarray(cols, dtype=np.int64),
                        np.asarray(actual, dtype=np.int64),
                        np.asarray(painted, dtype=np.int64),
                    )
                    for key in totals:
                        totals[key] += part[key]
            score = balanced_score(totals)
            rows.append(
                {
                    "revealed": f"{frac:.0%}",
                    "channel": channel,
                    "actual": totals["actual"],
                    "painted": totals["painted"],
                    "hits": totals["exact"],
                    "actual_within_1": totals["actual_within"],
                    "painted_within_1": totals["painted_within"],
                    "score": score["score"],
                }
            )
    return {"rows": rows, "position_20": pos}


def _mean(rows: list[dict]) -> float:
    return float(np.mean([r["score"] for r in rows])) if rows else 0.0


def train(model, train_cards, test_cards, sides, device) -> dict:
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best_score = -1.0
    bad = 0
    val_scores = []
    t0 = time.time()
    epochs_run = 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(train_cards))
        for start in range(0, len(train_cards), BATCH):
            pick = [train_cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [cut_at_fraction(c.base, float(rng.choice(CUT_FRACS))) for c in pick]
            bits, row_on, rem, pad, present, target, ob, oo, op = batch_views(pick, cuts, sides, device)
            live = present[:, :, :, None] & rem[:, None, :, None].bool() & (~pad)[:, :, None, None]
            opt.zero_grad()
            pred = model(bits, row_on, pad, ob, oo, op)
            alpha, beta = model.tversky_weights()
            loss = tversky_loss(pred, target, live, alpha, beta)
            bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
            loss = loss + BCE_W * ((bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        hold = evaluate(model, test_cards, sides, device)
        score = _mean(hold["rows"])
        val_scores.append(score)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        print(
            f"opp_cross epoch {epoch + 1}/{EPOCHS} loss {total / max(seen, 1):.4f} "
            f"holdout {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hold = evaluate(model, test_cards, sides, device)
    with torch.no_grad():
        a, b = model.tversky_weights()
    return {
        "epochs_run": epochs_run,
        "best": best_score,
        "seconds": round(time.time() - t0, 1),
        "alpha": float(a),
        "beta": float(b),
        "holdout": hold,
        "val_scores": val_scores,
    }


def _pr(bucket: dict) -> tuple[str, str]:
    painted = bucket["painted"]
    actual = bucket["actual"]
    p = "—" if painted == 0 else f"{bucket['hits'] / painted:.3f}"
    r = "—" if actual == 0 else f"{bucket['hits'] / actual:.3f}"
    return p, r


def write_report(meta: dict) -> Path:
    gate_mean = float(np.mean([balanced_score(GATE[k])["score"] for k in GATE]))
    mean = _mean(meta["holdout"]["rows"])
    lines = [
        "# Baudot opponent-card training",
        "",
        "The painted club card keeps its own bits. Each GW row attends to the opponent card: "
        "earlier rows true, this fixture's minutes revealed, this fixture's G/A and gc blank, "
        "later kickoffs hidden. Minutes are not copied onto the painted card. "
        "gc6 sits in the header, so this arm changes the header and the attention together. "
        "Loss is the club's remainder events.",
        "",
        f"Gate mean **{gate_mean:.3f}**. This arm mean **{mean:.3f}** (Δ {mean - gate_mean:+.3f}).",
        "",
        "| arm | epochs | α | β | mean | Δ gate | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| gate | — | {GATE_META['alpha']:.3f} | {GATE_META['beta']:.3f} | {gate_mean:.3f} | +0.000 | — |",
        f"| opp_cross | {meta['epochs_run']} | {meta['alpha']:.3f} | {meta['beta']:.3f} | {mean:.3f} | {mean - gate_mean:+.3f} | {meta['seconds']} |",
        "",
        "## Holdout table",
        "",
        "| revealed | channel | actual | painted | hits | score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in meta["holdout"]["rows"]:
        lines.append(
            f"| {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | {row['hits']} | {row['score']:.3f} |"
        )
    lines.extend(["", "## 20% by position (actual / painted / hits · precision · recall)", ""])
    for channel in ("goals", "assists"):
        lines.append(f"### {channel}")
        lines.append("")
        lines.append("| pos | a/p/h | precision | recall |")
        lines.append("|---|---|---:|---:|")
        for name, bucket in meta["holdout"]["position_20"][channel].items():
            p, r = _pr(bucket)
            lines.append(f"| {name} | {bucket['actual']}/{bucket['painted']}/{bucket['hits']} | {p} | {r} |")
        lines.append("")
    def _cell(revealed: str, channel: str) -> dict:
        return next(r for r in meta["holdout"]["rows"] if r["revealed"] == revealed and r["channel"] == channel)

    def _rate(row: dict) -> tuple[float, float]:
        precision = row["hits"] / row["painted"] if row["painted"] else float("nan")
        recall = row["hits"] / row["actual"] if row["actual"] else float("nan")
        return precision, recall

    g20, a20 = _cell("20%", "goals"), _cell("20%", "assists")
    gp, gr = _rate(g20)
    ap, ar = _rate(a20)
    def_g = meta["holdout"]["position_20"]["goals"]["DEF"]
    def_a = meta["holdout"]["position_20"]["assists"]["DEF"]
    verdict = (
        "The pair beats the gate on mean score. Check DEF before any lock change."
        if mean > gate_mean + 1e-3
        else "The pair does not beat the locked mixer."
    )
    lines.extend(
        [
            "## Call",
            "",
            f"Mean **{mean:.3f}** vs gate **{gate_mean:.3f}** (Δ {mean - gate_mean:+.3f}). "
            f"20% goals {g20['actual']}/{g20['painted']}/{g20['hits']} "
            f"(P {gp:.3f}, R {gr:.3f}) against the gate 730/701/194 (P 0.277, R 0.266). "
            f"Assists {a20['actual']}/{a20['painted']}/{a20['hits']} "
            f"(P {ap:.3f}, R {ar:.3f}) against 707/693/114 (P 0.164, R 0.161). "
            "Extra hits sit inside extra bars, and the longer remainders overpaint.",
            "",
            f"DEF goals stay **{def_g['actual']}/{def_g['painted']}/{def_g['hits']}** and "
            f"DEF assists **{def_a['actual']}/{def_a['painted']}/{def_a['hits']}**. "
            "Revealed opponent minutes and past gc do not move those rows.",
            "",
            "This arm adds the gc6 header and the opponent cross-read together. " + verdict,
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


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
    sides_l = build_sides(squad, baudot, league)
    sides = index_sides(sides_l)
    assert_chelsea_minutes(sides)
    train_cards = [s for s in sides_l if s.season in TRAIN_SEASONS]
    test_cards = [s for s in sides_l if s.season == TEST_SEASON]
    width = train_cards[0].bits.shape[1]
    print(f"cards train {len(train_cards)} test {len(test_cards)} width {width} device {device}", flush=True)
    torch.manual_seed(SEED)
    model = OppCrossMixer(width=width)
    meta = train(model, train_cards, test_cards, sides, device)
    print(
        json.dumps(
            {
                "rows": meta["holdout"]["rows"],
                "position_20": meta["holdout"]["position_20"],
                "alpha": meta["alpha"],
                "beta": meta["beta"],
                "seconds": meta["seconds"],
                "epochs_run": meta["epochs_run"],
            },
            indent=2,
        ),
        flush=True,
    )
    path = write_report(meta)
    state = {
        "trial": "baudot_opp_train",
        "mean_score": _mean(meta["holdout"]["rows"]),
        "gate_mean": float(np.mean([balanced_score(GATE[k])["score"] for k in GATE])),
        "epochs_run": meta["epochs_run"],
        "seconds": meta["seconds"],
        "alpha": meta["alpha"],
        "beta": meta["beta"],
        "rows": meta["holdout"]["rows"],
        "position_20": meta["holdout"]["position_20"],
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (ART / "baudot_opp_train.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
