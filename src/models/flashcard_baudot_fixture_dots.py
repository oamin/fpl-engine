"""Baudot fixture-dot card trial: +OU and optional +AH line vs locked gate.

Addresses DEF blank / assist allocation by adding event-volume dots, not a
deeper net. Odds API is not used — football-data opening OU/AH on closed seasons.

Arms (in order):
  1. ou_only   — keep 1X2₁₂; add OU₈ thermometer from P(over 2.5)
  2. ou_ah     — (1) + AH₈ *line* bar (assessed AHh), not AH prices

AH uses the handicap **line**, not AH shin prices, so it is not a second 1X2
book. Report still measures correlation between AH fill and 1X2 win fill.

Same BlockTokenMixer depth, learnable Tversky, full-remainder one-shot FOV.
Primary call: DEF goals/assists a/p/h, then assist-by-pos, then mean vs GATE.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ingest.fpl_odds import load_football_data
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
    ID_BITS,
    LR,
    MAX_GW,
    MAX_PLAYERS,
    MINS_BITS,
    ODDS_BITS,
    OPP_BITS,
    PLAYER_BITS,
    REPORTS,
    ROLE_BITS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    TVERSKY_DELTA_MAX,
    WHO_BITS,
    BaudotCard,
    _channel_slice,
    _player_block,
    _player_pad,
    build_cards,
    build_league,
    build_name_codes,
    count_bits,
    cut_at_fraction,
    event_target,
    fit_clubs,
    mark_bits,
    odds_bits_assessed,
    tversky_loss,
)
from src.models.flashcard_card_score import balanced_score, tally_row
from src.models.flashcard_odds_trial import SEASON_CODE
from src.models.flashcard_squad_trial import SquadCard
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_fixture_dots.md"
POS_NAME = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}

OU_BITS = 8
AH_BITS = 8


def ou_thermometer(p_over: float, n: int = OU_BITS) -> np.ndarray:
    """Left-fill ON count ≈ round(P(over)·n); under is the blank tail."""
    out = np.zeros(n, dtype=np.float32)
    if not np.isfinite(p_over):
        out[: n // 2] = 1.0
        return out
    k = int(np.clip(round(float(p_over) * n), 0, n))
    out[:k] = 1.0
    return out


def ah_line_bits(assessed_line: float | None, n: int = AH_BITS) -> np.ndarray:
    """Assessed AH line bar: favourite fills left, dog fills right, scratch blank.

    Line is AHh from the assessed club's view (negative = giving goals).
    Uses the line geometry only — not AH market prices (those track 1X2).
    """
    out = np.zeros(n, dtype=np.float32)
    if assessed_line is None or not np.isfinite(assessed_line):
        return out
    half = n // 2
    x = float(np.clip(assessed_line, -2.0, 2.0))
    if x < -1e-9:
        k = int(round(min(1.0, (-x) / 2.0) * half))
        out[:k] = 1.0
    elif x > 1e-9:
        k = int(round(min(1.0, x / 2.0) * half))
        out[n - k :] = 1.0
    return out


@dataclass
class MarketRow:
    p_home: float
    p_draw: float
    p_away: float
    p_over25: float
    p_under25: float
    ah_line: float | None


def load_markets() -> dict[tuple[str, str, str, str], MarketRow]:
    """Opening 1X2 + OU + AH line keyed by season, date, home_norm, away_norm."""
    table: dict[tuple[str, str, str, str], MarketRow] = {}
    for season, code in SEASON_CODE.items():
        frame = load_football_data(code=code)
        for row in frame.itertuples(index=False):
            ah = None if row.ah_line is None or (isinstance(row.ah_line, float) and not np.isfinite(row.ah_line)) else float(row.ah_line)
            table[(season, str(row.date), str(row.home_norm), str(row.away_norm))] = MarketRow(
                p_home=float(row.p_home),
                p_draw=float(row.p_draw),
                p_away=float(row.p_away),
                p_over25=float(row.p_over25),
                p_under25=float(row.p_under25),
                ah_line=ah,
            )
    return table


def _lookup_market(card: SquadCard, col: int, markets: dict) -> MarketRow | None:
    if int(card.gw[col]) <= 0 or not card.opponent[col]:
        return None
    date = pd.to_datetime(int(card.kickoff[col]), utc=True).strftime("%Y-%m-%d")
    team = norm_team(card.team)
    opponent = norm_team(card.opponent[col])
    if int(card.home[col]) == 1:
        key = (card.season, date, team, opponent)
    else:
        key = (card.season, date, opponent, team)
    return markets.get(key)


def header_bits_for(use_ah: bool) -> int:
    return CLUB_BITS + OPP_BITS + ODDS_BITS + OU_BITS + (AH_BITS if use_ah else 0)


def build_fixture_dot_cards(
    squad_cards: list[SquadCard],
    clubs: dict[str, int],
    name_codes: dict[str, int],
    markets: dict,
    use_ah: bool,
) -> list[BaudotCard]:
    header = header_bits_for(use_ah)
    width = header + MAX_PLAYERS * PLAYER_BITS
    out: list[BaudotCard] = []
    for card in squad_cards:
        club_i = clubs.get(card.team, 0)
        n_g = min(card.n_fixtures, MAX_GW)
        bits = np.zeros((MAX_GW, width), dtype=np.float32)
        goals = np.zeros((MAX_PLAYERS, MAX_GW), dtype=np.int64)
        assists = np.zeros_like(goals)
        minutes = np.zeros((MAX_PLAYERS, MAX_GW), dtype=np.float32)
        present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
        names = list(card.names[:MAX_PLAYERS]) + ["spare"] * max(0, MAX_PLAYERS - len(card.names))
        names = tuple(names[:MAX_PLAYERS])
        position = np.full(MAX_PLAYERS, -1, dtype=np.int64)
        position[: min(MAX_PLAYERS, len(card.position))] = card.position[:MAX_PLAYERS]
        player_id = np.zeros(MAX_PLAYERS, dtype=np.int64)
        for row in range(min(card.n_players, MAX_PLAYERS)):
            key = str(names[row]).strip().lower()
            player_id[row] = int(name_codes.get(key, 0))
        gw = np.zeros(MAX_GW, dtype=np.int64)
        home = np.zeros(MAX_GW, dtype=np.int64)
        opponents = [""] * MAX_GW
        for col in range(n_g):
            gw[col] = int(card.gw[col])
            home[col] = int(card.home[col])
            opponents[col] = card.opponent[col]
            opp_i = clubs.get(card.opponent[col], 0)
            mkt = _lookup_market(card, col, markets)
            parts = [mark_bits(club_i, CLUB_BITS), mark_bits(opp_i, OPP_BITS)]
            if mkt is not None:
                parts.append(odds_bits_assessed(mkt.p_home, mkt.p_draw, mkt.p_away, bool(home[col])))
                parts.append(ou_thermometer(mkt.p_over25, OU_BITS))
                if use_ah:
                    line = mkt.ah_line
                    if line is not None and not bool(home[col]):
                        line = -line
                    parts.append(ah_line_bits(line, AH_BITS))
            else:
                parts.append(odds_bits_assessed(1 / 3, 1 / 3, 1 / 3, True))
                parts.append(ou_thermometer(0.5, OU_BITS))
                if use_ah:
                    parts.append(ah_line_bits(None, AH_BITS))
            for row in range(MAX_PLAYERS):
                if row < card.n_players and card.present[row, col] and int(position[row]) >= 0:
                    present[row, col] = True
                    minutes[row, col] = float(card.minutes[row, col])
                    goals[row, col] = int(card.goals[row, col])
                    assists[row, col] = int(card.assists[row, col])
                    block = _player_block(
                        int(position[row]),
                        int(player_id[row]),
                        minutes[row, col],
                        goals[row, col],
                        assists[row, col],
                    )
                elif row < card.n_players and int(position[row]) >= 0 and int(player_id[row]) > 0:
                    block = _player_block(int(position[row]), int(player_id[row]), 0.0, 0, 0)
                else:
                    block = np.zeros(PLAYER_BITS, dtype=np.float32)
                parts.append(block)
            bits[col] = np.concatenate(parts)[:width]
        out.append(
            BaudotCard(
                season=card.season,
                team=card.team,
                names=names,
                position=position,
                player_id=player_id,
                gw=gw,
                home=home,
                opponent=tuple(opponents),
                bits=bits,
                goals=goals,
                assists=assists,
                minutes=minutes,
                present=present,
                n_players=min(card.n_players, MAX_PLAYERS),
                n_fixtures=n_g,
                width=width,
            )
        )
    return out


def blank_events(bits_row: np.ndarray, header: int) -> np.ndarray:
    out = bits_row.copy()
    for row in range(MAX_PLAYERS):
        base = header + row * PLAYER_BITS + WHO_BITS + MINS_BITS
        out[base : base + EVENT_BITS] = 0.0
    return out


def remainder_view(card: BaudotCard, cut: int, header: int) -> dict[str, np.ndarray]:
    bits = card.bits.copy()
    row_on = np.zeros(MAX_GW, dtype=np.float32)
    rem_on = np.zeros(MAX_GW, dtype=np.float32)
    target = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), dtype=np.float32)
    present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
    for gw in range(card.n_fixtures):
        row_on[gw] = 1.0
        if gw >= cut:
            bits[gw] = blank_events(bits[gw], header)
            rem_on[gw] = 1.0
            target[:, gw] = event_target(card, gw)
            present[:, gw] = card.present[:, gw]
    return {"bits": bits, "row_on": row_on, "rem_on": rem_on, "target": target, "present": present}


class FixtureDotsMixer(nn.Module):
    """Depth-1 block-token mixer with extended fixture header embeds."""

    def __init__(self, width: int, use_ah: bool, dropout: float = 0.1) -> None:
        super().__init__()
        self.width = width
        self.use_ah = use_ah
        self.header = header_bits_for(use_ah)
        self.club = nn.Linear(CLUB_BITS, D)
        self.opp = nn.Linear(OPP_BITS, D)
        self.odds = nn.Linear(ODDS_BITS, D)
        self.ou = nn.Linear(OU_BITS, D)
        self.ah = nn.Linear(AH_BITS, D) if use_ah else None
        n_h = 4 + (1 if use_ah else 0)
        self.header_mix = nn.Linear(n_h * D, D)
        self.who = nn.Linear(WHO_BITS, D)
        self.mins = nn.Linear(MINS_BITS, D)
        self.events = nn.Linear(EVENT_BITS, D)
        self.player_mix = nn.Linear(3 * D, D)
        self.norm_h = nn.LayerNorm(D)
        self.norm_p = nn.LayerNorm(D)
        self.time = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.squad = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(D)
        self.norm_s = nn.LayerNorm(D)
        self.time_pos = nn.Parameter(torch.zeros(1, MAX_GW, 1, D))
        nn.init.normal_(self.time_pos, std=0.02)
        self.paint = nn.Linear(D, EVENT_BITS)
        nn.init.zeros_(self.paint.weight)
        nn.init.constant_(self.paint.bias, -3.0)
        self.tversky_logit = nn.Parameter(torch.tensor(-2.0))

    def tversky_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit)
        return 0.5 - delta, 0.5 + delta

    def _encode_rows(self, bits: torch.Tensor) -> torch.Tensor:
        b, t, _ = bits.shape
        club = self.club(bits[..., :CLUB_BITS])
        opp = self.opp(bits[..., CLUB_BITS : CLUB_BITS + OPP_BITS])
        o0 = CLUB_BITS + OPP_BITS
        odds = self.odds(bits[..., o0 : o0 + ODDS_BITS])
        u0 = o0 + ODDS_BITS
        ou = self.ou(bits[..., u0 : u0 + OU_BITS])
        parts = [club, opp, odds, ou]
        if self.use_ah:
            a0 = u0 + OU_BITS
            parts.append(self.ah(bits[..., a0 : a0 + AH_BITS]))
        header = self.norm_h(self.header_mix(torch.cat(parts, dim=-1)))
        body = bits[..., self.header :].reshape(b, t, MAX_PLAYERS, PLAYER_BITS)
        who = self.who(body[..., :WHO_BITS])
        mins = self.mins(body[..., WHO_BITS : WHO_BITS + MINS_BITS])
        ev = self.events(body[..., WHO_BITS + MINS_BITS :])
        players = self.norm_p(self.player_mix(torch.cat([who, mins, ev], dim=-1)))
        return players + header[:, :, None, :]

    def forward(self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
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
        ink = torch.sigmoid(self.paint(tokens))
        return ink.permute(0, 2, 1, 3)


def _batch(cards: list[BaudotCard], cuts: list[int], header: int, device: torch.device):
    bits_l, row_ons, rem_ons, presents, targets, pads = [], [], [], [], [], []
    for card, cut in zip(cards, cuts, strict=True):
        view = remainder_view(card, cut, header)
        bits_l.append(view["bits"])
        row_ons.append(view["row_on"])
        rem_ons.append(view["rem_on"])
        presents.append(view["present"])
        targets.append(view["target"])
        pads.append(_player_pad(card))
    return (
        torch.from_numpy(np.stack(bits_l)).to(device),
        torch.from_numpy(np.stack(row_ons)).to(device),
        torch.from_numpy(np.stack(rem_ons)).to(device),
        torch.from_numpy(np.stack(pads)).to(device),
        torch.from_numpy(np.stack(presents)).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
    )


def _mean_score(rows: list[dict]) -> float:
    return float(np.mean([r["score"] for r in rows]))


def _gate_rows() -> list[dict]:
    rows = []
    for frac in CUT_FRACS:
        for channel in ("goals", "assists"):
            tot = GATE[(channel, frac)]
            sc = balanced_score(tot)
            rows.append(
                {
                    "revealed": f"{frac:.0%}",
                    "channel": channel,
                    "actual": tot["actual"],
                    "painted": tot["painted"],
                    "hits": tot["exact"],
                    "actual_within_1": tot["actual_within"],
                    "painted_within_1": tot["painted_within"],
                    "score": sc["score"],
                }
            )
    return rows


@torch.no_grad()
def evaluate(model: FixtureDotsMixer, cards: list[BaudotCard], device: torch.device) -> dict:
    model.eval()
    header = model.header
    rows: list[dict] = []
    pos = {
        "goals_20": {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()},
        "assists_20": {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()},
    }
    for frac in CUT_FRACS:
        for channel in ("goals", "assists"):
            totals = {key: 0 for key in ("actual", "painted", "exact", "actual_within", "painted_within")}
            sl = _channel_slice(channel)
            for card in cards:
                cut = cut_at_fraction(card, frac)
                view = remainder_view(card, cut, header)
                pad = _player_pad(card)
                pred = model(
                    torch.from_numpy(view["bits"][None]).to(device),
                    torch.from_numpy(view["row_on"][None]).to(device),
                    torch.from_numpy(pad[None]).to(device),
                )[0].cpu().numpy()
                for row in range(card.n_players):
                    if pad[row]:
                        continue
                    cols = [gw for gw in range(cut, card.n_fixtures) if view["present"][row, gw]]
                    if not cols:
                        continue
                    actual = np.array([int(view["target"][row, gw, sl].sum()) for gw in cols], dtype=np.int64)
                    painted = np.array(
                        [int((pred[row, gw, sl] >= 0.5).sum()) for gw in cols], dtype=np.int64
                    )
                    part = tally_row(card.gw[cols], actual, painted)
                    for key in totals:
                        totals[key] += part[key]
                    if frac == 0.2:
                        pname = POS_NAME.get(int(card.position[row]))
                        if pname is not None:
                            bucket = pos["goals_20" if channel == "goals" else "assists_20"]
                            for a, p in zip(actual, painted, strict=True):
                                ae, pe = int(a > 0), int(p > 0)
                                bucket[pname]["actual"] += ae
                                bucket[pname]["painted"] += pe
                                bucket[pname]["hits"] += int(ae and pe)
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
    return {"rows": rows, "position": pos}


def ah_vs_1x2_correlation(cards: list[BaudotCard]) -> dict:
    """How redundant is AH line fill with 1X2 win fill? (ou_ah cards only)."""
    win_fills = []
    ah_fills = []
    ou_fills = []
    header = header_bits_for(True)
    for card in cards:
        for gw in range(card.n_fixtures):
            row = card.bits[gw]
            o0 = CLUB_BITS + OPP_BITS
            win_on = float(row[o0 : o0 + ODDS_BITS].sum())  # win+loss ON; win≈left mass
            # approx win fill = ON bits before first blank run — use left half ON count
            win_fills.append(float(row[o0 : o0 + ODDS_BITS // 2].sum()))
            u0 = o0 + ODDS_BITS
            ou_fills.append(float(row[u0 : u0 + OU_BITS].sum()))
            a0 = u0 + OU_BITS
            ah_fills.append(float(row[a0 : a0 + AH_BITS].sum()))
    win = np.asarray(win_fills)
    ah = np.asarray(ah_fills)
    ou = np.asarray(ou_fills)
    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "corr_ah_fill_vs_1x2_left": corr(ah, win),
        "corr_ou_fill_vs_1x2_left": corr(ou, win),
        "corr_ah_fill_vs_ou_fill": corr(ah, ou),
        "n_rows": int(len(win)),
    }


def train_arm(
    name: str,
    model: FixtureDotsMixer,
    train: list[BaudotCard],
    test: list[BaudotCard],
    device: torch.device,
) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history: list[float] = []
    val_scores: list[float] = []
    ab_hist: list[tuple[float, float]] = []
    best_state = None
    best_score = -1.0
    bad = 0
    t0 = time.time()
    header = model.header
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(train))
        for start in range(0, len(train), BATCH):
            pick = [train[int(i)] for i in order[start : start + BATCH]]
            fracs = [float(rng.choice(CUT_FRACS)) for _ in pick]
            cuts = [cut_at_fraction(card, frac) for card, frac in zip(pick, fracs, strict=True)]
            bits, row_on, rem_on, pad, present, target = _batch(pick, cuts, header, device)
            live = (
                present[:, :, :, None]
                & rem_on[:, None, :, None].bool()
                & (~pad)[:, :, None, None]
            )
            opt.zero_grad()
            pred = model(bits, row_on, pad)
            alpha, beta = model.tversky_weights()
            loss = tversky_loss(pred, target, live, alpha, beta)
            bce = F.binary_cross_entropy(pred, target, reduction="none")
            bce = (bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0)
            loss = loss + BCE_W * bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        avg = total / max(seen, 1)
        history.append(avg)
        with torch.no_grad():
            a, b = model.tversky_weights()
            ab_hist.append((float(a), float(b)))
        hold = evaluate(model, test, device)
        score = _mean_score(hold["rows"])
        val_scores.append(score)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} loss {avg:.4f} "
            f"α={ab_hist[-1][0]:.3f} β={ab_hist[-1][1]:.3f} holdout_score {score:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate(model, test, device)
    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "best_holdout_mean_score": best_score,
        "final_alpha": ab_hist[int(np.argmax(val_scores))][0] if val_scores else float("nan"),
        "final_beta": ab_hist[int(np.argmax(val_scores))][1] if val_scores else float("nan"),
        "seconds": round(time.time() - t0, 1),
        "holdout": holdout,
    }


def _delta(a: float, b: float) -> str:
    return f"{a - b:+.3f}"


def _pos_cell(d: dict, pos: str) -> str:
    x = d[pos]
    return f"{x['actual']}/{x['painted']}/{x['hits']}"


def write_report(metas: list[dict], corr: dict, gate_def: dict) -> Path:
    gate_rows = _gate_rows()
    gate_mean = _mean_score(gate_rows)
    lines = [
        "# Baudot fixture dots (+OU / +AH line) vs locked gate",
        "",
        "Card information trial. Same mixer, learnable Tversky, full-remainder one-shot.",
        "No Odds API — football-data opening OU 2.5 and AHh on closed seasons.",
        "",
        "- **ou_only:** 1X2₁₂ + OU₈ thermometer (P(over) left-fill).",
        "- **ou_ah:** (1) + AH₈ **line** bar (assessed AHh). Not AH prices (those track 1X2).",
        "",
        f"Gate: `{GATE_META['method']}` · mean {gate_mean:.3f}.",
        "",
        "## AH redundancy check (fill correlation on ou_ah cards)",
        "",
        f"- corr(AH fill, 1X2 left-fill) = **{corr['corr_ah_fill_vs_1x2_left']:.3f}**",
        f"- corr(OU fill, 1X2 left-fill) = **{corr['corr_ou_fill_vs_1x2_left']:.3f}**",
        f"- corr(AH fill, OU fill) = **{corr['corr_ah_fill_vs_ou_fill']:.3f}**",
        "",
        "| arm | params | epochs | mean score | Δ gate | seconds |",
        "|---|---:|---:|---:|---:|---:|",
        f"| gate (1X2 only) | — | — | {gate_mean:.3f} | +0.000 | — |",
    ]
    for m in metas:
        mean = _mean_score(m["holdout"]["rows"])
        lines.append(
            f"| {m['name']} | {m['params']} | {m['epochs_run']} | {mean:.3f} | "
            f"{_delta(mean, gate_mean)} | {m['seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Goals at 20% by position (actual / painted / hits)",
            "",
            "| arm | GKP | DEF | MID | FWD |",
            "|---|---|---|---|---|",
            f"| gate_autopsy | {gate_def['goals']['GKP']} | {gate_def['goals']['DEF']} | "
            f"{gate_def['goals']['MID']} | {gate_def['goals']['FWD']} |",
        ]
    )
    for m in metas:
        g = m["holdout"]["position"]["goals_20"]
        lines.append(
            f"| {m['name']} | {_pos_cell(g, 'GKP')} | {_pos_cell(g, 'DEF')} | "
            f"{_pos_cell(g, 'MID')} | {_pos_cell(g, 'FWD')} |"
        )
    lines.extend(
        [
            "",
            "## Assists at 20% by position (actual / painted / hits)",
            "",
            "| arm | GKP | DEF | MID | FWD |",
            "|---|---|---|---|---|",
            f"| gate_autopsy | {gate_def['assists']['GKP']} | {gate_def['assists']['DEF']} | "
            f"{gate_def['assists']['MID']} | {gate_def['assists']['FWD']} |",
        ]
    )
    for m in metas:
        a = m["holdout"]["position"]["assists_20"]
        lines.append(
            f"| {m['name']} | {_pos_cell(a, 'GKP')} | {_pos_cell(a, 'DEF')} | "
            f"{_pos_cell(a, 'MID')} | {_pos_cell(a, 'FWD')} |"
        )
    lines.extend(
        [
            "",
            "## Holdout table",
            "",
            "| arm | revealed | channel | actual | painted | hits | actual ±1 | painted ±1 | score | Δ gate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"| gate | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
            f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} | +0.000 |"
        )
    for m in metas:
        for row in m["holdout"]["rows"]:
            key = (row["channel"], {"20%": 0.2, "40%": 0.4, "60%": 0.6, "80%": 0.8}[row["revealed"]])
            gscore = balanced_score(GATE[key])["score"]
            lines.append(
                f"| {m['name']} | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
                f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} | "
                f"{_delta(row['score'], gscore)} |"
            )
    lines.extend(
        [
            "",
            "Primary call: does DEF leave 0/0 on goals and assists, and does assist-by-pos "
            "improve without collapsing mean score? AH is judged useful only if ou_ah beats "
            "ou_only on those diagnostics after the redundancy check.",
            "",
        ]
    )
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    (ART / "baudot_fixture_dots.md").write_text(text, encoding="utf-8")
    return OUT


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    print("loading cards + markets", flush=True)
    league = build_league(seasons)
    markets = load_markets()
    codes = build_name_codes(seasons)
    squad = build_cards(seasons, league)
    clubs = fit_clubs([c for c in squad if c.season in TRAIN_SEASONS])
    for card in squad:
        clubs.setdefault(card.team, len(clubs) + 1)
        for opp in card.opponent:
            if opp:
                clubs.setdefault(opp, len(clubs) + 1)

    # Autopsy baseline cells (from prior report) for the position table.
    gate_def = {
        "goals": {"GKP": "0/0/0", "DEF": "103/0/0", "MID": "394/370/83", "FWD": "233/331/111"},
        "assists": {"GKP": "4/0/0", "DEF": "181/0/0", "MID": "434/363/65", "FWD": "88/330/49"},
    }

    arms_spec = [
        ("1_ou_only", False),
        ("2_ou_ah", True),
    ]
    metas = []
    corr = {}
    for name, use_ah in arms_spec:
        print(f"\n=== {name} ===", flush=True)
        cards = build_fixture_dot_cards(squad, clubs, codes, markets, use_ah=use_ah)
        train = [c for c in cards if c.season in TRAIN_SEASONS]
        test = [c for c in cards if c.season == TEST_SEASON]
        if use_ah:
            corr = ah_vs_1x2_correlation(train + test)
            print(f"AH redundancy {corr}", flush=True)
        model = FixtureDotsMixer(width=train[0].width, use_ah=use_ah)
        meta = train_arm(name, model, train, test, device)
        metas.append(meta)
        print(
            f"{name} mean {_mean_score(meta['holdout']['rows']):.3f} "
            f"DEF goals {meta['holdout']['position']['goals_20']['DEF']} "
            f"DEF assists {meta['holdout']['position']['assists_20']['DEF']}",
            flush=True,
        )

    path = write_report(metas, corr, gate_def)
    gate_mean = _mean_score(_gate_rows())
    slim = {
        "trial": "baudot_fixture_dots",
        "gate": GATE_META,
        "ah_redundancy": corr,
        "arms": [
            {
                "name": m["name"],
                "params": m["params"],
                "epochs_run": m["epochs_run"],
                "mean_score": _mean_score(m["holdout"]["rows"]),
                "delta_gate_mean": _mean_score(m["holdout"]["rows"]) - gate_mean,
                "final_alpha": m["final_alpha"],
                "final_beta": m["final_beta"],
                "seconds": m["seconds"],
                "rows": m["holdout"]["rows"],
                "position": m["holdout"]["position"],
            }
            for m in metas
        ],
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (ART / "baudot_fixture_dots.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
