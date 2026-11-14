"""Two losses on the locked opponent-card method.

Pre-registered before the holdout totals were read. Both arms keep the
opponent cross-read, the gc6 header, seed 0, and the cell Tversky. Judged
on the same tally as ``GATE_OPP_2026_11_14``.

1_margins
    Add a weight-1 loss on three pools of the remainder picture: role
    (GKP/DEF/MID/FWD), GW row sums, and player column sums. Each term is
    the relative error of that pool's soft count.

2_gc_scoreline
    Predict goals conceded for the assessed club and for the opponent on
    the same fixture. The assessed club's goals are that opponent gc,
    distributed across the revealed XI. On away fixtures the opponent XI
    is filled in the same step, from the assessed club's gc.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flashcard_baudot_gc_header import GC_BITS, HEADER_WITH_GC
from src.models.flashcard_baudot_opp_train import (
    HEADER_BITS_END,
    OppCrossMixer,
    _mean,
    _pr,
    assert_chelsea_minutes,
    batch_views,
    build_sides,
    evaluate,
    index_sides,
)
from src.models.flashcard_baudot_row_trial import (
    ART,
    BATCH,
    BCE_W,
    CUT_FRACS,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GATE,
    GOAL_BITS,
    LR,
    MAX_GW,
    MAX_PLAYERS,
    MINS_BITS,
    PLAYER_BITS,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    WHO_BITS,
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
from src.models.flashcard_card_score import GATE_OPP_2026_11_14, balanced_score

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_margin_gc.md"
MARGIN_W = 1.0
GC_W = 1.0
AWAY_W = 1.0
GOAL_BIT_TEMP = 4.0
ARMS = ("1_margins", "2_gc_scoreline")


def _table_mean(table: dict) -> float:
    return float(np.mean([balanced_score(table[k])["score"] for k in table]))


def margin_loss(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Relative error of role pools, row sums, and column sums. One scalar."""
    live_f = live.to(pred.dtype)
    pos = position[:, :, None, None]
    terms = []
    for sl in (slice(0, GOAL_BITS), slice(GOAL_BITS, EVENT_BITS)):
        p = pred[..., sl]
        y = target[..., sl]
        row_on = live_f.squeeze(-1).sum(1) > 0
        rp = (p * live_f).sum(1)
        ry = (y * live_f).sum(1)
        row_err = (rp - ry).abs() / ry.detach().clamp_min(1.0)
        terms.append((row_err * row_on[:, :, None]).sum() / row_on.sum().clamp_min(1) / p.shape[-1])
        col_on = live_f.squeeze(-1).sum(2) > 0
        cp = (p * live_f).sum(2)
        cy = (y * live_f).sum(2)
        col_err = (cp - cy).abs() / cy.detach().clamp_min(1.0)
        terms.append((col_err * col_on[:, :, None]).sum() / col_on.sum().clamp_min(1) / p.shape[-1])
        for role in range(4):
            rm = live_f * (pos == role).to(pred.dtype)
            n = rm.sum().clamp_min(1.0)
            err = (p * rm).sum() - (y * rm).sum()
            scale = (y * rm).sum().detach().clamp_min(1.0)
            terms.append(err.abs() / scale)
            del n
    return torch.stack(terms).mean()


def minutes_mask(bits: torch.Tensor) -> torch.Tensor:
    """True where a player slot has any minute bits. bits (..., W) → (..., P)."""
    body = bits[..., HEADER_WITH_GC:].reshape(*bits.shape[:-1], MAX_PLAYERS, PLAYER_BITS)
    return body[..., WHO_BITS : WHO_BITS + MINS_BITS].sum(-1) > 0


def _distribute(tokens: torch.Tensor, lam: torch.Tensor, xi: torch.Tensor, share: nn.Linear, levels: torch.Tensor) -> torch.Tensor:
    """Spread a team goal count across the revealed XI. Returns (..., P, GOAL_BITS)."""
    logit = share(tokens).squeeze(-1)
    none = ~xi.any(dim=-1, keepdim=True)
    logit = logit.masked_fill(~xi, -1e4)
    logit = torch.where(none, torch.zeros_like(logit), logit)
    weights = torch.softmax(logit, dim=-1)
    weights = torch.where(none, torch.zeros_like(weights), weights)
    expected = weights * lam.unsqueeze(-1)
    return torch.sigmoid((expected.unsqueeze(-1) - levels) * GOAL_BIT_TEMP)


class ScorelineMixer(OppCrossMixer):
    """Opponent cross-read plus both clubs' gc, and a scoreline shared onto the XI."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__(width, dropout=dropout)
        self.gc_club = nn.Linear(self.norm_c.normalized_shape[0], GC_BITS)
        self.gc_opp = nn.Linear(self.norm_c.normalized_shape[0], GC_BITS)
        self.share = nn.Linear(self.norm_c.normalized_shape[0], 1)
        nn.init.zeros_(self.gc_club.weight)
        nn.init.zeros_(self.gc_opp.weight)
        nn.init.zeros_(self.share.weight)
        nn.init.constant_(self.gc_club.bias, -3.0)
        nn.init.constant_(self.gc_opp.bias, -3.0)
        nn.init.zeros_(self.share.bias)
        self.register_buffer("goal_levels", torch.tensor([0.5, 1.5, 2.5]), persistent=False)

    def forward_train(self, bits, row_on, player_pad, opp_bits, opp_on, opp_pad):
        tokens, opp_tokens = self._tokens(bits, row_on, player_pad, opp_bits, opp_on, opp_pad)
        pooled = self._pool(tokens, player_pad)
        gc_club = torch.sigmoid(self.gc_club(pooled))
        gc_opp = torch.sigmoid(self.gc_opp(pooled))
        lam = gc_opp.sum(-1)
        xi = minutes_mask(bits) & ~player_pad[:, None, :]
        goal_bits = _distribute(tokens, lam, xi, self.share, self.goal_levels)
        assist = torch.sigmoid(self.paint(tokens)[..., GOAL_BITS:])
        paint = torch.cat([goal_bits, assist], dim=-1).permute(0, 2, 1, 3)

        idx = opp_on.long().sum(-1) - 1
        safe = idx.clamp_min(0)
        b, t, s, p, d = opp_tokens.shape
        take = safe[:, :, None, None, None].expand(b, t, 1, p, d)
        opp_now = opp_tokens.gather(2, take).squeeze(2)
        w = opp_bits.shape[-1]
        row = opp_bits.gather(2, safe[:, :, None, None].expand(b, t, 1, w)).squeeze(2)
        opp_pad_now = opp_pad.gather(2, safe[:, :, None, None].expand(b, t, 1, p)).squeeze(2)
        opp_xi = minutes_mask(row) & ~opp_pad_now & (idx >= 0)[:, :, None]
        opp_goals = _distribute(opp_now, gc_club.sum(-1), opp_xi, self.share, self.goal_levels)
        opp_assist = torch.sigmoid(self.paint(opp_now)[..., GOAL_BITS:])
        opp_paint = torch.cat([opp_goals, opp_assist], dim=-1).permute(0, 2, 1, 3)
        return paint, {"gc_club": gc_club, "gc_opp": gc_opp, "opp_paint": opp_paint}

    def forward(self, bits, row_on, player_pad, opp_bits, opp_on, opp_pad):
        paint, _ = self.forward_train(bits, row_on, player_pad, opp_bits, opp_on, opp_pad)
        return paint

    def _pool(self, tokens: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
        keep = (~player_pad)[:, None, :, None].to(tokens.dtype)
        total = (tokens * keep).sum(2)
        return total / keep.sum(2).clamp_min(1.0)

    def _tokens(self, bits, row_on, player_pad, opp_bits, opp_on, opp_pad):
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
        s = opp_tokens.shape[2]
        keys = opp_tokens.reshape(b * t, s * p, d)
        live = opp_on[:, :, :, None] * (~opp_pad).to(opp_on.dtype)
        none = live.reshape(b * t, s * p).sum(-1) < 0.5
        key_pad = live.reshape(b * t, s * p) < 0.5
        key_pad[none, 0] = False
        query = tokens.reshape(b * t, p, d)
        mixed, _ = self.cross(query, keys, keys, key_padding_mask=key_pad, need_weights=False)
        tokens = self.norm_c(query + mixed).reshape(b, t, p, d)
        return tokens, opp_tokens


def _cell_loss(pred, target, live, model) -> torch.Tensor:
    alpha, beta = model.tversky_weights()
    loss = tversky_loss(pred, target, live, alpha, beta)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    loss = loss + BCE_W * ((bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0))
    return loss


def gc_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(pred.dtype)[:, :, None]
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    return (bce * m).sum() / m.sum().clamp_min(1.0) / GC_BITS


def scoreline_extras(cards, sides):
    gc, ogc, away, ogoal, opresent = [], [], [], [], []
    lo, hi = HEADER_BITS_END(), HEADER_WITH_GC
    for card in cards:
        gc_i = np.zeros((MAX_GW, GC_BITS), np.float32)
        ogc_i = np.zeros_like(gc_i)
        away_i = np.zeros(MAX_GW, dtype=bool)
        ogoal_i = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), np.float32)
        opre_i = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
        for gw in range(card.n_fixtures):
            gc_i[gw] = card.bits[gw, lo:hi]
            away_i[gw] = int(card.base.home[gw]) == 0
            other = sides.get((card.season, card.opponent[gw]))
            if other is None:
                continue
            j = other.row_of.get(int(card.fixture_id[gw]))
            if j is None:
                continue
            ogc_i[gw] = other.bits[j, lo:hi]
            ogoal_i[:, gw] = event_target(other.base, j)
            opre_i[:, gw] = other.base.present[:, j]
        gc.append(gc_i)
        ogc.append(ogc_i)
        away.append(away_i)
        ogoal.append(ogoal_i)
        opresent.append(opre_i)
    stack = lambda xs: torch.from_numpy(np.stack(xs))
    return stack(gc), stack(ogc), stack(away), stack(ogoal), stack(opresent)


def assert_scoreline_mask(sides) -> None:
    """Remainder inputs stay blank. Labels still hold the true gc and the away XI."""
    ars = sides[(TEST_SEASON, "Arsenal")]
    gw = int(np.where(ars.base.gw[: ars.n_fixtures] == 13)[0][0])
    assert int(ars.base.home[gw]) == 0
    from src.models.flashcard_baudot_opp_train import club_view

    view = club_view(ars, gw)
    lo, hi = HEADER_BITS_END(), HEADER_WITH_GC
    assert view["bits"][gw, lo:hi].sum() == 0.0
    assert ars.bits[gw, lo:hi].sum() > 0.0
    che = sides[(TEST_SEASON, "Chelsea")]
    j = che.row_of[int(ars.fixture_id[gw])]
    assert che.bits[j, lo:hi].sum() >= 0.0
    print(
        f"scoreline mask ok  Arsenal GW13 away gc-bits {ars.bits[gw, lo:hi].sum():.0f} "
        f"Chelsea gc-bits {che.bits[j, lo:hi].sum():.0f}",
        flush=True,
    )


def train_arm(name: str, model, train_cards, test_cards, sides, device) -> dict:
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best_score = -1.0
    bad = 0
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
            if name == "1_margins":
                pred = model(bits, row_on, pad, ob, oo, op)
                loss = _cell_loss(pred, target, live, model)
                pos = torch.from_numpy(np.stack([c.base.position for c in pick])).to(device)
                loss = loss + MARGIN_W * margin_loss(pred, target, live, pos)
            else:
                pred, extra = model.forward_train(bits, row_on, pad, ob, oo, op)
                loss = _cell_loss(pred, target, live, model)
                gc, ogc, away, ogoal, opresent = scoreline_extras(pick, sides)
                gc, ogc, away, ogoal, opresent = (t.to(device) for t in (gc, ogc, away, ogoal, opresent))
                rem_b = rem.bool()
                loss = loss + GC_W * gc_bce(extra["gc_club"], gc, rem_b)
                loss = loss + GC_W * gc_bce(extra["gc_opp"], ogc, rem_b)
                away_live = opresent[:, :, :, None] & away[:, None, :, None] & rem_b[:, None, :, None]
                loss = loss + AWAY_W * tversky_loss(
                    extra["opp_paint"], ogoal, away_live, *model.tversky_weights()
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        hold = evaluate(model, test_cards, sides, device)
        score = _mean(hold["rows"])
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} loss {total / max(seen, 1):.4f} "
            f"holdout {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hold = evaluate(model, test_cards, sides, device)
    with torch.no_grad():
        alpha, beta = model.tversky_weights()
    return {
        "name": name,
        "epochs_run": epochs_run,
        "best": best_score,
        "seconds": round(time.time() - t0, 1),
        "alpha": float(alpha),
        "beta": float(beta),
        "holdout": hold,
    }


def write_report(arms: list[dict]) -> Path:
    opp = _table_mean(GATE_OPP_2026_11_14)
    gate = _table_mean(GATE)
    lines = [
        "# Margin pools and both-card gc",
        "",
        "Both arms start from the locked opponent-card method. Cell Tversky stays. "
        f"`1_margins` adds role pools and row/column sums at weight {MARGIN_W:g}. "
        f"`2_gc_scoreline` adds gc for the assessed club and the opponent at weight {GC_W:g}, "
        "distributes that scoreline across the revealed XI, and on away fixtures fills the opponent XI "
        f"in the same step (weight {AWAY_W:g}).",
        "",
        f"Opponent-card baseline **{opp:.3f}**. 2026-11-13 mixer **{gate:.3f}**.",
        "",
        "| arm | epochs | α | β | mean | Δ opp | Δ mixer | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| opp_cross | 20 | 0.447 | 0.553 | {opp:.3f} | +0.000 | {opp - gate:+.3f} | 358.8 |",
    ]
    for meta in arms:
        mean = _mean(meta["holdout"]["rows"])
        lines.append(
            f"| {meta['name']} | {meta['epochs_run']} | {meta['alpha']:.3f} | {meta['beta']:.3f} | "
            f"{mean:.3f} | {mean - opp:+.3f} | {mean - gate:+.3f} | {meta['seconds']} |"
        )
    for meta in arms:
        mean = _mean(meta["holdout"]["rows"])
        lines.extend(["", f"## {meta['name']}", "", "| revealed | channel | actual | painted | hits | score |", "|---|---|---:|---:|---:|---:|"])
        for row in meta["holdout"]["rows"]:
            lines.append(
                f"| {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | {row['hits']} | {row['score']:.3f} |"
            )
        lines.extend(["", "### 20% by position", ""])
        for channel in ("goals", "assists"):
            lines.append(f"**{channel}**")
            lines.append("")
            lines.append("| pos | a/p/h | precision | recall |")
            lines.append("|---|---|---:|---:|")
            for name, bucket in meta["holdout"]["position_20"][channel].items():
                p, r = _pr(bucket)
                lines.append(f"| {name} | {bucket['actual']}/{bucket['painted']}/{bucket['hits']} | {p} | {r} |")
            lines.append("")
        def_g = meta["holdout"]["position_20"]["goals"]["DEF"]
        def_a = meta["holdout"]["position_20"]["assists"]["DEF"]
        lines.append(
            f"DEF goals {def_g['actual']}/{def_g['painted']}/{def_g['hits']}. "
            f"DEF assists {def_a['actual']}/{def_a['painted']}/{def_a['hits']}. Mean {mean:.3f}."
        )
        lines.append("")
    lines.extend(["## Call", ""])
    best = max(arms, key=lambda m: _mean(m["holdout"]["rows"]))
    best_mean = _mean(best["holdout"]["rows"])
    if best_mean > opp + 1e-3:
        lines.append(
            f"{best['name']} is ahead of the opponent-card baseline ({best_mean:.3f} vs {opp:.3f}). "
            "The default stays the cell-Tversky opponent card until that arm is locked."
        )
    else:
        lines.append(
            "Neither added loss beats the opponent-card baseline. "
            "Role pools, row/column sums, and the joint gc scoreline stay off the default."
        )
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def _load():
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
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
    assert_scoreline_mask(sides)
    train_cards = [s for s in sides_l if s.season in TRAIN_SEASONS]
    test_cards = [s for s in sides_l if s.season == TEST_SEASON]
    return sides, train_cards, test_cards


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading", flush=True)
    sides, train_cards, test_cards = _load()
    width = train_cards[0].bits.shape[1]
    print(f"cards train {len(train_cards)} test {len(test_cards)} width {width} device {device}", flush=True)
    results = []
    for name in ARMS:
        torch.manual_seed(SEED)
        model = OppCrossMixer(width=width) if name == "1_margins" else ScorelineMixer(width=width)
        results.append(train_arm(name, model, train_cards, test_cards, sides, device))
    path = write_report(results)
    state = {
        "trial": "baudot_margin_gc",
        "default": "src.models.flashcard_baudot_opp_train",
        "opp_mean": _table_mean(GATE_OPP_2026_11_14),
        "mixer_mean": _table_mean(GATE),
        "arms": [
            {
                "name": meta["name"],
                "mean_score": _mean(meta["holdout"]["rows"]),
                "epochs_run": meta["epochs_run"],
                "seconds": meta["seconds"],
                "alpha": meta["alpha"],
                "beta": meta["beta"],
                "rows": meta["holdout"]["rows"],
                "position_20": meta["holdout"]["position_20"],
            }
            for meta in results
        ],
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (ART / "baudot_margin_gc.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
