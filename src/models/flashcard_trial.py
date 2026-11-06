"""Lightweight flash-card trial: causal GRU vs a prefix-frequency baseline.

A card is one player-season of fixture columns. Minutes, home, and opponent
are visible on every column. A prefix of the event bars (goals, assists,
clean sheets, goals conceded) is visible. The remainder is the target.

Train on 2022-23, 2023-24, and 2024-25. Score 2025-26 at a 50% cut.
2026-27 is not read.

This is an exploratory trial, not a gated score comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.ingest.fpl_odds import POS_MAP

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
TEST_SEASON = "2025-26"
MIN_LEN = 12
MIN_PREFIX = 4
MIN_REMAINDER = 4
BLANK = 4
NCLASS = (4, 4, 2, 4)
CHANNELS = ("goals", "assists", "clean_sheets", "goals_conceded")
HIDDEN = 64
TEAM_DIM = 8
POS_DIM = 4
EVENT_DIM = 4
EPOCHS = 10
BATCH = 128
LR = 1e-3
HEAD_LR = 1e-2
N_SAMPLES = 16
TEST_FRAC = 0.5
SEED = 0
PLAYED_MINUTES = 60

SHEET_COLUMNS = (
    "name",
    "position",
    "team",
    "element",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "assists",
    "clean_sheets",
    "kickoff_time",
    "minutes",
    "was_home",
    "GW",
)


@dataclass
class Seq:
    """One player-season, one row per fixture, kickoff order."""

    season: str
    element: int
    name: str
    minutes: np.ndarray
    home: np.ndarray
    team: np.ndarray
    opponent: np.ndarray
    position: np.ndarray
    gw: np.ndarray
    events: np.ndarray  # (T, 4) clipped bar heights


def choose_cut(length: int, frac: float) -> int:
    """Prefix length. Leaves at least MIN_PREFIX and MIN_REMAINDER columns."""
    cut = int(round(float(frac) * int(length)))
    return int(min(max(cut, MIN_PREFIX), int(length) - MIN_REMAINDER))


def shift_events(
    events: np.ndarray,
    cut: np.ndarray,
    length: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Previous-column event ids. Index t never receives the label at t.

    ``teacher`` feeds the true previous bar, including earlier remainder bars.
    ``blank`` feeds the true previous bar only when that bar is still in the
    prefix. Later remainder steps see the blank id.
    """
    if mode not in {"teacher", "blank"}:
        raise ValueError(f"unknown shift mode: {mode}")
    prev = np.full(events.shape, BLANK, dtype=np.int64)
    prev[:, 1:] = events[:, :-1]
    for i, length_i in enumerate(length):
        end = int(length_i)
        if end < prev.shape[1]:
            prev[i, end:] = BLANK
        if mode == "blank":
            start = int(cut[i]) + 1
            prev[i, start:] = BLANK
    return prev


def minute_bin(minutes: np.ndarray) -> np.ndarray:
    """0 = did not play, 1 = under 60, 2 = 60 or more."""
    bins = np.zeros(minutes.shape, dtype=np.int64)
    bins[minutes >= 1] = 1
    bins[minutes >= PLAYED_MINUTES] = 2
    return bins


def _opponent_names(frame: pd.DataFrame) -> pd.Series:
    """The other club on the same fixture id. Unpaired rows become UNK."""
    rows: list[tuple[object, object, str]] = []
    for fixture, names in frame.groupby("fixture")["team"].agg(lambda s: list(pd.unique(s))).items():
        clubs = [str(name) for name in names if str(name)]
        if len(clubs) == 2:
            rows.append((fixture, clubs[0], clubs[1]))
            rows.append((fixture, clubs[1], clubs[0]))
        elif len(clubs) == 1:
            rows.append((fixture, clubs[0], "UNK"))
    if not rows:
        return pd.Series("UNK", index=frame.index)
    paired = pd.DataFrame(rows, columns=["fixture", "team", "opponent"])
    merged = frame[["fixture", "team"]].merge(paired, on=["fixture", "team"], how="left")
    return merged["opponent"].fillna("UNK")


def load_season(season: str) -> pd.DataFrame:
    """Player-fixture rows for one closed season. Drops non-playing positions."""
    path = CACHE / f"merged_gw_{season.replace('-', '_')}.csv"
    frame = pd.read_csv(path, usecols=lambda name: name in SHEET_COLUMNS)
    frame["position"] = frame["position"].astype(str).str.upper().map(POS_MAP)
    frame = frame.dropna(subset=["position", "team", "kickoff_time", "element"]).copy()
    for column in ("minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "GW"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    home = frame["was_home"].astype(str).str.lower().isin({"true", "1"})
    frame["is_home"] = home.astype(np.int64)
    frame["opponent"] = _opponent_names(frame).to_numpy()
    frame["kickoff"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["kickoff"]).copy()
    frame["season"] = season
    frame["element"] = frame["element"].astype(int)
    frame["name"] = frame["name"].astype(str)
    frame["team"] = frame["team"].astype(str)
    return frame


def build_sequences(frame: pd.DataFrame, min_len: int = MIN_LEN) -> list[Seq]:
    """Group a season sheet into kickoff-ordered player sequences."""
    ordered = frame.sort_values(["element", "kickoff", "fixture"], kind="mergesort")
    sequences: list[Seq] = []
    for element, group in ordered.groupby("element", sort=False):
        if len(group) < min_len:
            continue
        events = np.stack(
            [
                np.minimum(group["goals_scored"].to_numpy(dtype=np.int64), 3),
                np.minimum(group["assists"].to_numpy(dtype=np.int64), 3),
                np.minimum(group["clean_sheets"].to_numpy(dtype=np.int64), 1),
                np.minimum(group["goals_conceded"].to_numpy(dtype=np.int64), 3),
            ],
            axis=1,
        )
        sequences.append(
            Seq(
                season=str(group["season"].iloc[0]),
                element=int(element),
                name=str(group["name"].iloc[0]),
                minutes=group["minutes"].to_numpy(dtype=np.float32),
                home=group["is_home"].to_numpy(dtype=np.int64),
                team=group["team"].to_numpy(dtype=object),
                opponent=group["opponent"].to_numpy(dtype=object),
                position=group["position"].to_numpy(dtype=object),
                gw=group["GW"].to_numpy(dtype=np.int64),
                events=events,
            )
        )
    return sequences


def fit_team_index(sequences: list[Seq]) -> dict[str, int]:
    """Map club names seen in training to ids. 0 is unknown."""
    names: set[str] = set()
    for seq in sequences:
        names.update(str(name) for name in seq.team)
        names.update(str(name) for name in seq.opponent)
    names.discard("UNK")
    return {name: i + 1 for i, name in enumerate(sorted(names))}


POS_INDEX = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _encode_club(names: np.ndarray, index: dict[str, int]) -> np.ndarray:
    return np.array([index.get(str(name), 0) for name in names], dtype=np.int64)


def _encode_pos(names: np.ndarray) -> np.ndarray:
    return np.array([POS_INDEX[str(name)] for name in names], dtype=np.int64)


class FlashGRU(nn.Module):
    """One GRU layer. Four softmax heads, one per event bar."""

    def __init__(self, n_teams: int) -> None:
        super().__init__()
        self.hidden = HIDDEN
        self.team = nn.Embedding(n_teams, TEAM_DIM)
        self.opponent = nn.Embedding(n_teams, TEAM_DIM)
        self.position = nn.Embedding(len(POS_INDEX), POS_DIM)
        self.event = nn.ModuleList(
            [nn.Embedding(BLANK + 1, EVENT_DIM) for _ in CHANNELS]
        )
        input_size = 2 + TEAM_DIM + TEAM_DIM + POS_DIM + EVENT_DIM * len(CHANNELS)
        self.gru = nn.GRU(input_size, HIDDEN, num_layers=1, batch_first=True)
        # Minutes and home are concatenated again at the head. The GRU carries
        # history; the current column's minutes must be able to zero a bar
        # without being stored through the update gate.
        # minutes/90, benched indicator, played indicator, home.
        self.heads = nn.ModuleList([nn.Linear(HIDDEN + 4, n) for n in NCLASS])

    def features(
        self,
        minutes: torch.Tensor,
        home: torch.Tensor,
        team: torch.Tensor,
        opponent: torch.Tensor,
        position: torch.Tensor,
        prev: torch.Tensor,
    ) -> torch.Tensor:
        """Build GRU inputs. ``prev`` is (B, T, 4) or (B, 4) for one step."""
        single = minutes.dim() == 1
        if single:
            minutes = minutes.unsqueeze(1)
            home = home.unsqueeze(1)
            team = team.unsqueeze(1)
            opponent = opponent.unsqueeze(1)
            position = position.unsqueeze(1)
            prev = prev.unsqueeze(1)
        pos = self.position(position)
        parts = [
            (minutes / 90.0).unsqueeze(-1),
            home.unsqueeze(-1).float(),
            self.team(team),
            self.opponent(opponent),
            pos,
        ]
        for channel, embedding in enumerate(self.event):
            parts.append(embedding(prev[..., channel]))
        features = torch.cat(parts, dim=-1)
        return features.squeeze(1) if single else features

    def classify(
        self,
        state: torch.Tensor,
        minutes: torch.Tensor,
        home: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Softmax logits. ``state`` is (B, H) or (B, T, H)."""
        single = state.dim() == 2
        if single:
            state = state.unsqueeze(1)
            minutes = minutes.unsqueeze(1)
            home = home.unsqueeze(1)
        context = torch.cat(
            [
                state,
                (minutes / 90.0).unsqueeze(-1),
                (minutes < 1).float().unsqueeze(-1),
                (minutes >= PLAYED_MINUTES).float().unsqueeze(-1),
                home.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        logits = [head(context) for head in self.heads]
        if single:
            return [item.squeeze(1) for item in logits]
        return logits

    def forward(
        self,
        minutes: torch.Tensor,
        home: torch.Tensor,
        team: torch.Tensor,
        opponent: torch.Tensor,
        position: torch.Tensor,
        prev: torch.Tensor,
    ) -> list[torch.Tensor]:
        packed = self.features(minutes, home, team, opponent, position, prev)
        hidden, _ = self.gru(packed)
        return self.classify(hidden, minutes, home)


def _pad_batch(
    seqs: list[Seq],
    team_index: dict[str, int],
    cuts: np.ndarray,
) -> dict[str, np.ndarray]:
    length = np.array([len(seq.minutes) for seq in seqs], dtype=np.int64)
    width = int(length.max())
    batch = len(seqs)
    minutes = np.zeros((batch, width), dtype=np.float32)
    home = np.zeros((batch, width), dtype=np.int64)
    team = np.zeros((batch, width), dtype=np.int64)
    opponent = np.zeros((batch, width), dtype=np.int64)
    position = np.zeros((batch, width), dtype=np.int64)
    events = np.zeros((batch, width, 4), dtype=np.int64)
    gw = np.zeros((batch, width), dtype=np.int64)
    for i, seq in enumerate(seqs):
        end = int(length[i])
        minutes[i, :end] = seq.minutes
        home[i, :end] = seq.home
        team[i, :end] = _encode_club(seq.team, team_index)
        opponent[i, :end] = _encode_club(seq.opponent, team_index)
        position[i, :end] = _encode_pos(seq.position)
        events[i, :end] = seq.events
        gw[i, :end] = seq.gw
    return {
        "minutes": minutes,
        "home": home,
        "team": team,
        "opponent": opponent,
        "position": position,
        "events": events,
        "gw": gw,
        "length": length,
        "cut": cuts.astype(np.int64),
    }


def _tensors(batch: dict[str, np.ndarray], mode: str, device: torch.device) -> dict[str, torch.Tensor]:
    prev = shift_events(batch["events"], batch["cut"], batch["length"], mode)
    out = {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.items()
        if key != "gw"
    }
    out["prev"] = torch.as_tensor(prev, device=device)
    out["gw"] = torch.as_tensor(batch["gw"], device=device)
    return out


def loss_mask(length: torch.Tensor, cut: torch.Tensor, width: int) -> torch.Tensor:
    """1 on remainder columns that exist, else 0."""
    index = torch.arange(width, device=length.device).view(1, -1)
    alive = index < length.view(-1, 1)
    hidden = index >= cut.view(-1, 1)
    return (alive & hidden).float()


def mean_nll(
    logits: list[torch.Tensor],
    events: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy per hidden bar, plus a per-channel breakdown."""
    total = torch.zeros((), device=events.device)
    weight = mask.sum().clamp_min(1.0)
    per_channel: dict[str, float] = {}
    for channel, scores in enumerate(logits):
        target = events[..., channel]
        flat = F.cross_entropy(
            scores.reshape(-1, scores.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)
        total = total + (flat * mask).sum()
        per_channel[CHANNELS[channel]] = float((flat * mask).sum().detach() / weight)
    mean = total / (weight * len(CHANNELS))
    return mean, per_channel


def played_mask(mask: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
    return mask * (minutes >= PLAYED_MINUTES).float()


class FrequencyBaseline:
    """Prefix histogram shrunk toward a position × minutes-bin prior.

    The prior is Laplace-smoothed counts from the training seasons only.
    """

    def __init__(self, sequences: list[Seq]) -> None:
        self.counts = [np.ones((len(POS_INDEX), 3, n), dtype=np.float64) for n in NCLASS]
        for seq in sequences:
            self._add(seq, 0, len(seq.minutes), self.counts, prior=True)

    def _add(
        self,
        seq: Seq,
        start: int,
        end: int,
        counts: list[np.ndarray],
        prior: bool,
    ) -> None:
        bins = minute_bin(seq.minutes[start:end])
        pos = _encode_pos(seq.position[start:end])
        for channel, n_class in enumerate(NCLASS):
            labels = seq.events[start:end, channel]
            for bin_id in range(3):
                for class_id in range(n_class):
                    hit = (bins == bin_id) & (labels == class_id)
                    if prior:
                        for pos_id in range(len(POS_INDEX)):
                            counts[channel][pos_id, bin_id, class_id] += np.count_nonzero(
                                hit & (pos == pos_id)
                            )
                    else:
                        counts[channel][0, bin_id, class_id] += np.count_nonzero(hit)

    def probabilities(self, seq: Seq, cut: int) -> list[np.ndarray]:
        """Class probabilities for every column. Shape (T, n_class) per channel.

        Prefix columns are not scored. Remainder columns use prefix counts.
        """
        local = [np.zeros((1, 3, n), dtype=np.float64) for n in NCLASS]
        self._add(seq, 0, cut, local, prior=False)
        bins = minute_bin(seq.minutes)
        pos = _encode_pos(seq.position)
        out: list[np.ndarray] = []
        for channel, n_class in enumerate(NCLASS):
            probs = np.zeros((len(seq.minutes), n_class), dtype=np.float64)
            for t in range(len(seq.minutes)):
                player = local[channel][0, bins[t]]
                prior = self.counts[channel][pos[t], bins[t]]
                mixed = player + prior
                probs[t] = mixed / mixed.sum()
            out.append(probs)
        return out


def baseline_nll(baseline: FrequencyBaseline, sequences: list[Seq], frac: float) -> dict[str, float]:
    """Mean cross-entropy of the frequency baseline on a fixed cut."""
    totals = {name: 0.0 for name in CHANNELS}
    played_totals = {name: 0.0 for name in CHANNELS}
    n_all = 0
    n_played = 0
    expected = {name: 0.0 for name in CHANNELS}
    actual = {name: 0.0 for name in CHANNELS}
    abs_err = {name: 0.0 for name in CHANNELS}
    for seq in sequences:
        cut = choose_cut(len(seq.minutes), frac)
        probs = baseline.probabilities(seq, cut)
        remainder = slice(cut, len(seq.minutes))
        played = seq.minutes[remainder] >= PLAYED_MINUTES
        n_all += int(len(seq.minutes) - cut)
        n_played += int(played.sum())
        for channel, name in enumerate(CHANNELS):
            labels = seq.events[remainder, channel]
            p = probs[channel][remainder]
            pick = p[np.arange(len(labels)), labels]
            totals[name] += float(-np.log(pick).sum())
            if played.any():
                played_totals[name] += float(-np.log(pick[played]).sum())
            height = np.arange(NCLASS[channel])
            pred_sum = float((p * height).sum())
            true_sum = float(labels.sum())
            expected[name] += pred_sum
            actual[name] += true_sum
            abs_err[name] += abs(pred_sum - true_sum)
    n_cards = max(len(sequences), 1)
    return {
        "nll_all": {name: totals[name] / max(n_all, 1) for name in CHANNELS},
        "nll_played": {name: played_totals[name] / max(n_played, 1) for name in CHANNELS},
        "mean_remainder_sum": {name: actual[name] / n_cards for name in CHANNELS},
        "mean_expected_sum": {name: expected[name] / n_cards for name in CHANNELS},
        "mae_sum": {name: abs_err[name] / n_cards for name in CHANNELS},
        "n_columns": n_all,
        "n_played": n_played,
    }


def _batches(seqs: list[Seq], batch_size: int, rng: np.random.Generator) -> list[list[Seq]]:
    order = rng.permutation(len(seqs))
    groups = []
    for start in range(0, len(seqs), batch_size):
        pick = order[start : start + batch_size]
        groups.append([seqs[int(i)] for i in pick])
    return groups


def train_model(
    model: FlashGRU,
    sequences: list[Seq],
    team_index: dict[str, int],
    rng: np.random.Generator,
    device: torch.device,
    epochs: int = EPOCHS,
) -> list[float]:
    """Teacher-forced cross-entropy on a random 30–70% prefix. Returns epoch NLL."""
    opt = torch.optim.AdamW(
        [
            {"params": model.heads.parameters(), "lr": HEAD_LR},
            {
                "params": [p for name, p in model.named_parameters() if not name.startswith("heads.")],
                "lr": LR,
            },
        ]
    )
    history: list[float] = []
    model.train()
    for _epoch in range(epochs):
        total = 0.0
        seen = 0.0
        for group in _batches(sequences, BATCH, rng):
            fracs = rng.uniform(0.3, 0.7, size=len(group))
            cuts = np.array(
                [choose_cut(len(seq.minutes), float(frac)) for seq, frac in zip(group, fracs, strict=True)]
            )
            batch = _tensors(_pad_batch(group, team_index, cuts), "teacher", device)
            mask = loss_mask(batch["length"], batch["cut"], batch["minutes"].size(1))
            opt.zero_grad()
            logits = model(
                batch["minutes"],
                batch["home"],
                batch["team"],
                batch["opponent"],
                batch["position"],
                batch["prev"],
            )
            loss, _ = mean_nll(logits, batch["events"], mask)
            # Sum the channel scores so one channel is not shrunk by the average.
            (loss * len(CHANNELS)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(mask.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        epoch_nll = total / max(seen, 1.0)
        history.append(epoch_nll)
        if epochs == EPOCHS:
            print(f"epoch {_epoch + 1} train nll {epoch_nll:.3f}", flush=True)
    return history


@torch.no_grad()
def evaluate_gru(
    model: FlashGRU,
    sequences: list[Seq],
    team_index: dict[str, int],
    frac: float,
    device: torch.device,
) -> dict[str, object]:
    """Blank-remainder NLL, teacher-forced NLL, and free-run sample totals."""
    model.eval()
    cuts = [choose_cut(len(seq.minutes), frac) for seq in sequences]
    blank_all = {name: 0.0 for name in CHANNELS}
    blank_played = {name: 0.0 for name in CHANNELS}
    teacher_played = {name: 0.0 for name in CHANNELS}
    n_all = 0
    n_played = 0
    hit_prob = {name: 0.0 for name in CHANNELS}
    hit_actual = {name: 0.0 for name in CHANNELS}
    argmax_sum = {name: 0.0 for name in CHANNELS}
    sample_sum = {name: 0.0 for name in CHANNELS}
    actual_sum = {name: 0.0 for name in CHANNELS}
    sample_mae = {name: 0.0 for name in CHANNELS}
    spike_on_zero = 0.0
    spike_total = 0.0
    sample_by_card = np.zeros((len(sequences), len(CHANNELS)))
    actual_by_card = np.zeros((len(sequences), len(CHANNELS)))

    for start in range(0, len(sequences), BATCH):
        group = sequences[start : start + BATCH]
        cut = np.array(cuts[start : start + BATCH])
        raw = _pad_batch(group, team_index, cut)
        blank = _tensors(raw, "blank", device)
        teacher = _tensors(raw, "teacher", device)
        mask = loss_mask(blank["length"], blank["cut"], blank["minutes"].size(1))
        played = played_mask(mask, blank["minutes"])
        logits = model(
            blank["minutes"],
            blank["home"],
            blank["team"],
            blank["opponent"],
            blank["position"],
            blank["prev"],
        )
        forced = model(
            teacher["minutes"],
            teacher["home"],
            teacher["team"],
            teacher["opponent"],
            teacher["position"],
            teacher["prev"],
        )
        n_all += int(mask.sum())
        n_played += int(played.sum())
        for channel, name in enumerate(CHANNELS):
            target = blank["events"][..., channel]
            flat = F.cross_entropy(
                logits[channel].reshape(-1, NCLASS[channel]),
                target.reshape(-1),
                reduction="none",
            ).view_as(target)
            blank_all[name] += float((flat * mask).sum())
            blank_played[name] += float((flat * played).sum())
            forced_flat = F.cross_entropy(
                forced[channel].reshape(-1, NCLASS[channel]),
                target.reshape(-1),
                reduction="none",
            ).view_as(target)
            teacher_played[name] += float((forced_flat * played).sum())
            probs = torch.softmax(logits[channel], dim=-1)
            positive = probs[..., 1:].sum(-1)
            hit_prob[name] += float((positive * played).sum())
            hit_actual[name] += float(((target > 0).float() * played).sum())
            for row, seq_cut in enumerate(cut):
                end = int(raw["length"][row])
                actual_by_card[start + row, channel] = float(target[row, seq_cut:end].sum())
                actual_sum[name] += float(target[row, seq_cut:end].sum())

        greedy, _, _ = _decode_once(model, blank, device, "greedy")
        for channel, name in enumerate(CHANNELS):
            for row, seq_cut in enumerate(cut):
                end = int(raw["length"][row])
                argmax_sum[name] += float(greedy[row, seq_cut:end, channel].sum())
        draws = _sample_sums(model, blank, raw, device)
        actual_slice = actual_by_card[start : start + len(group)]
        for channel, name in enumerate(CHANNELS):
            sample_sum[name] += float(draws[:, channel].sum())
            sample_mae[name] += float(np.abs(draws[:, channel] - actual_slice[:, channel]).sum())
            sample_by_card[start : start + len(group), channel] = draws[:, channel]
        spike_on_zero += float(draws[:, len(CHANNELS)].sum())
        spike_total += float(draws[:, len(CHANNELS) + 1].sum())

    n_cards = max(len(sequences), 1)
    denom_played = max(n_played, 1)
    return {
        "nll_all": {name: blank_all[name] / max(n_all, 1) for name in CHANNELS},
        "nll_played_blank": {name: blank_played[name] / denom_played for name in CHANNELS},
        "nll_played_teacher": {name: teacher_played[name] / denom_played for name in CHANNELS},
        "p_positive_played": {name: hit_prob[name] / denom_played for name in CHANNELS},
        "rate_positive_played": {name: hit_actual[name] / denom_played for name in CHANNELS},
        "mean_argmax_sum": {name: argmax_sum[name] / n_cards for name in CHANNELS},
        "mean_sample_sum": {name: sample_sum[name] / n_cards for name in CHANNELS},
        "mean_actual_sum": {name: actual_sum[name] / n_cards for name in CHANNELS},
        "mae_sample_sum": {name: sample_mae[name] / n_cards for name in CHANNELS},
        "spike_on_zero_minutes": spike_on_zero / max(spike_total, 1.0),
        "sample_by_card": sample_by_card,
        "actual_by_card": actual_by_card,
        "cuts": cuts,
        "n_columns": n_all,
        "n_played": n_played,
    }


@torch.no_grad()
def _sample_sums(
    model: FlashGRU,
    batch: dict[str, torch.Tensor],
    raw: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Mean remainder sums over N_SAMPLES free runs, plus goal-spike placement.

    Columns 0–3 are mean sampled remainder sums. Column 4 is the count of
    sampled goal bars on zero-minute weeks. Column 5 is the count of sampled
    positive goal bars.
    """
    totals = np.zeros((batch["minutes"].size(0), len(CHANNELS) + 2), dtype=np.float64)
    for _draw in range(N_SAMPLES):
        generated, zero_spikes, spikes = _decode_once(model, batch, device, "sample")
        for row in range(generated.size(0)):
            cut = int(raw["cut"][row])
            end = int(raw["length"][row])
            for channel in range(len(CHANNELS)):
                totals[row, channel] += float(generated[row, cut:end, channel].sum())
            totals[row, len(CHANNELS)] += zero_spikes[row]
            totals[row, len(CHANNELS) + 1] += spikes[row]
    totals[:, : len(CHANNELS)] /= N_SAMPLES
    return totals


@torch.no_grad()
def _decode_once(
    model: FlashGRU,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    how: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Autoregressive fill. Prefix bars stay true. Remainder bars are drawn or argmaxed."""
    if how not in {"sample", "greedy"}:
        raise ValueError(how)
    minutes = batch["minutes"]
    home = batch["home"]
    team = batch["team"]
    opponent = batch["opponent"]
    position = batch["position"]
    events = batch["events"]
    cut = batch["cut"]
    length = batch["length"]
    batch_size, width = minutes.shape
    hidden = torch.zeros(1, batch_size, model.hidden, device=device)
    generated = events.clone()
    prev = torch.full((batch_size, 4), BLANK, dtype=torch.long, device=device)
    zero_spikes = np.zeros(batch_size, dtype=np.float64)
    spikes = np.zeros(batch_size, dtype=np.float64)
    for t in range(width):
        step = model.features(
            minutes[:, t],
            home[:, t],
            team[:, t],
            opponent[:, t],
            position[:, t],
            prev,
        )
        out, hidden = model.gru(step.unsqueeze(1), hidden)
        state = out.squeeze(1)
        drawn = []
        for channel, scores in enumerate(model.classify(state, minutes[:, t], home[:, t])):
            probs = torch.softmax(scores, dim=-1)
            if how == "sample":
                draw = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                draw = probs.argmax(dim=-1)
            use = (t >= cut) & (t < length)
            value = torch.where(use, draw, events[:, t, channel])
            generated[:, t, channel] = value
            drawn.append(value)
        wrote = torch.stack(drawn, dim=1)
        prev = wrote
        goal = generated[:, t, 0]
        active = ((t >= cut) & (t < length)).detach().cpu().numpy()
        goal_np = goal.detach().cpu().numpy()
        minute_np = minutes[:, t].detach().cpu().numpy()
        positive = active & (goal_np > 0)
        spikes += positive.astype(np.float64)
        zero_spikes += (positive & (minute_np <= 0)).astype(np.float64)
    return generated, zero_spikes, spikes


def _example_card(
    model: FlashGRU,
    sequences: list[Seq],
    team_index: dict[str, int],
    cuts: list[int],
    device: torch.device,
) -> dict[str, object]:
    """One 2025-26 card: the remainder with the most actual goal bars."""
    best = max(
        range(len(sequences)),
        key=lambda i: int(sequences[i].events[cuts[i] :, 0].sum()),
    )
    seq = sequences[best]
    cut = cuts[best]
    raw = _pad_batch([seq], team_index, np.array([cut]))
    blank = _tensors(raw, "blank", device)
    torch.manual_seed(SEED + 1)
    generated, _, _ = _decode_once(model, blank, device, "sample")
    greedy, _, _ = _decode_once(model, blank, device, "greedy")
    end = len(seq.minutes)
    argmax_goals = greedy[0, :end, 0].detach().cpu().numpy().astype(int)
    argmax_goals[:cut] = seq.events[:cut, 0]
    return {
        "name": seq.name,
        "team": str(seq.team[0]),
        "position": str(seq.position[0]),
        "cut": cut,
        "gw": seq.gw.tolist(),
        "minutes": seq.minutes.astype(float).tolist(),
        "goals": seq.events[:, 0].astype(int).tolist(),
        "sample_goals": generated[0, :end, 0].detach().cpu().numpy().astype(int).tolist(),
        "argmax_goals": argmax_goals.tolist(),
    }


def _plot_card(example: dict[str, object], path: Path) -> None:
    gw = np.array(example["gw"])
    minutes = np.array(example["minutes"], dtype=float)
    goals = np.array(example["goals"], dtype=float)
    sample = np.array(example["sample_goals"], dtype=float)
    argmax = np.array(example["argmax_goals"], dtype=float)
    cut = int(example["cut"])
    x = np.arange(len(gw))
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    rows = [
        (minutes, "Minutes (always visible)", "#4c78a8"),
        (goals, "Actual goal bars", "#f58518"),
        (sample, "One sampled remainder", "#54a24b"),
        (argmax, "Most likely bar at each column", "#e45756"),
    ]
    for axis, (values, title, color) in zip(axes, rows, strict=True):
        axis.bar(x, values, color=color, width=0.8)
        axis.axvline(cut - 0.5, color="black", linewidth=0.8)
        axis.set_ylabel(title, fontsize=8)
    axes[-1].set_xlabel("Fixture column (line marks the cut)")
    fig.suptitle(
        f"{example['name']} ({example['position']}, {example['team']}) 2025-26",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _deciles(expected: np.ndarray, actual: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(expected)
    folds = np.array_split(order, 5)
    rows = []
    for fold in folds:
        if len(fold) == 0:
            continue
        rows.append(
            {
                "n": int(len(fold)),
                "mean_expected": float(expected[fold].mean()),
                "mean_actual": float(actual[fold].mean()),
            }
        )
    return rows


def _fmt(values: dict[str, float]) -> str:
    return ", ".join(f"{name} {values[name]:.3f}" for name in CHANNELS)


def write_report(payload: dict[str, object], path: Path) -> None:
    """Exploratory note. Not a gated comparison."""
    gru = payload["gru"]
    base = payload["baseline"]
    example = payload["example"]
    lines = [
        "# Flash-card GRU trial",
        "",
        "Exploratory. One held-out season. Not a gated score comparison.",
        "2026-27 was not read.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        f"Cards with at least {MIN_LEN} fixtures: train {payload['n_train']}, test {payload['n_test']}.",
        f"Test cut: first {int(TEST_FRAC * 100)}% of fixture columns visible.",
        "Minutes, home, and opponent stay visible on the remainder.",
        "Event bars on the remainder are hidden. Heights are 0, 1, 2, or 3+ (clean sheet 0 or 1).",
        "",
        "## Fit",
        "",
        "One-layer GRU, hidden 64. The classifier also sees minutes, a benched flag,",
        "a played flag (60 minutes or more), and home.",
        "Teacher forcing. Cross-entropy on the hidden columns.",
        "Prefix fraction during training is uniform between 30% and 70%.",
        f"Epoch mean NLL per hidden bar: {', '.join(f'{v:.3f}' for v in payload['train_nll'])}.",
        "",
        "Baseline: the player's prefix histogram, shrunk toward a position × minutes-bin",
        "prior fit on the training seasons only.",
        "",
        "## Held-out negative log-likelihood",
        "",
        "Lower is better. Played columns are remainder fixtures with at least 60 minutes.",
        "Blank means the remainder event bars are not fed back in. Teacher forcing feeds",
        "true earlier remainder bars, so that number is an easier next-step score.",
        "",
        "| model | goals | assists | clean sheets | goals conceded |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, block, key in (
        ("GRU blank, all columns", gru["nll_all"], None),
        ("GRU blank, played", gru["nll_played_blank"], None),
        ("GRU teacher, played", gru["nll_played_teacher"], None),
        ("Baseline, all columns", base["nll_all"], None),
        ("Baseline, played", base["nll_played"], None),
    ):
        source = block if key is None else block[key]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{source[name]:.3f}" for name in CHANNELS)
            + " |"
        )
    lines += [
        "",
        "## What a played week looks like",
        "",
        "Rate is the share of played remainder columns whose true bar is at least 1.",
        "P is the blank-remainder GRU's mean probability of that event.",
        "Argmax sum is the total height of the single most likely chart.",
        "",
        "| channel | actual rate | GRU P | mean actual sum | mean sample sum | mean argmax sum | baseline expected sum |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CHANNELS:
        lines.append(
            "| {name} | {rate:.3f} | {prob:.3f} | {actual:.2f} | {sample:.2f} | {argmax:.2f} | {base_sum:.2f} |".format(
                name=name,
                rate=gru["rate_positive_played"][name],
                prob=gru["p_positive_played"][name],
                actual=gru["mean_actual_sum"][name],
                sample=gru["mean_sample_sum"][name],
                argmax=gru["mean_argmax_sum"][name],
                base_sum=base["mean_expected_sum"][name],
            )
        )
    lines += [
        "",
        f"Sampled goal bars sitting on a 0-minute column: {gru['spike_on_zero_minutes']:.3f}.",
        "",
        "MAE of the per-card remainder sum (sample mean vs actual, baseline vs actual):",
        "",
        "| channel | GRU sample MAE | baseline MAE |",
        "|---|---:|---:|",
    ]
    for name in CHANNELS:
        lines.append(
            f"| {name} | {gru['mae_sample_sum'][name]:.2f} | {base['mae_sum'][name]:.2f} |"
        )
    lines += [
        "",
        "## Goal-sum quintiles on the test remainder",
        "",
        "Players ordered by mean sampled remainder goal height. Five equal groups.",
        "",
        "| group | GRU sample | GRU actual | baseline expected | baseline actual |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, (left, right) in enumerate(
        zip(payload["gru_goal_deciles"], payload["baseline_goal_deciles"], strict=True),
        start=1,
    ):
        lines.append(
            f"| {i} | {left['mean_expected']:.2f} | {left['mean_actual']:.2f} | "
            f"{right['mean_expected']:.2f} | {right['mean_actual']:.2f} |"
        )
    lines += [
        "",
        "## One card",
        "",
        f"{example['name']}, {example['position']}, {example['team']}. "
        f"Cut after fixture column {example['cut']}. "
        "Chosen as the 2025-26 card with the most goal bars on the remainder, so the spikes are visible.",
        "The sample is the first draw, not a selected draw.",
        "",
        f"Plot: `{payload['plot']}`.",
        "",
        "## Read",
        "",
        "The greedy goal chart is empty: mean argmax sum "
        f"{gru['mean_argmax_sum']['goals']:.2f} against an actual "
        f"{gru['mean_actual_sum']['goals']:.2f}.",
        f"{N_SAMPLES} draws average {gru['mean_sample_sum']['goals']:.2f} goal bars.",
        "Played-week teacher-forced goal NLL is "
        f"{gru['nll_played_teacher']['goals']:.3f} against the baseline "
        f"{base['nll_played']['goals']:.3f}.",
        "The baseline is ahead on clean sheets, goals conceded, and the MAE of every remainder sum.",
        "Blank-token probabilities are not the sampler. On a played week they put probability "
        f"{gru['p_positive_played']['goals']:.2f} on a goal bar whose true rate is "
        f"{gru['rate_positive_played']['goals']:.2f}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, object]:
    """Fit on three seasons and score 2025-26. Writes the report and the plot."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    train = [seq for season in TRAIN_SEASONS for seq in build_sequences(load_season(season))]
    test = build_sequences(load_season(TEST_SEASON))
    team_index = fit_team_index(train)
    model = FlashGRU(n_teams=len(team_index) + 1).to(device)
    history = train_model(model, train, team_index, rng, device)
    gru = evaluate_gru(model, test, team_index, TEST_FRAC, device)
    baseline_model = FrequencyBaseline(train)
    baseline = baseline_nll(baseline_model, test, TEST_FRAC)
    example = _example_card(model, test, team_index, gru["cuts"], device)
    plot_path = PLOTS / "flashcard_gru_trial.png"
    _plot_card(example, plot_path)

    base_expected = []
    for seq in test:
        cut = choose_cut(len(seq.minutes), TEST_FRAC)
        probs = baseline_model.probabilities(seq, cut)
        height = np.arange(NCLASS[0])
        base_expected.append(float((probs[0][cut:] * height).sum()))
    payload: dict[str, object] = {
        "n_train": len(train),
        "n_test": len(test),
        "train_nll": [round(value, 4) for value in history],
        "gru": {
            key: value
            for key, value in gru.items()
            if key not in {"sample_by_card", "actual_by_card", "cuts"}
        },
        "baseline": baseline,
        "gru_goal_deciles": _deciles(gru["sample_by_card"][:, 0], gru["actual_by_card"][:, 0]),
        "baseline_goal_deciles": _deciles(
            np.array(base_expected),
            gru["actual_by_card"][:, 0],
        ),
        "example": {
            "name": example["name"],
            "team": example["team"],
            "position": example["position"],
            "cut": example["cut"],
        },
        "plot": str(plot_path.relative_to(ROOT)),
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(payload))
    (PROCESSED / "flashcard_gru_trial.json").write_text(
        json.dumps(serializable, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(payload, REPORTS / "flashcard_gru_trial.md")
    _plot_card(example, Path("/opt/cursor/artifacts/flashcard_gru_card.png"))
    return payload


def main() -> None:
    payload = run()
    print(
        f"train {payload['n_train']} test {payload['n_test']} "
        f"final train nll {payload['train_nll'][-1]:.3f}"
    )
    print(f"wrote {REPORTS / 'flashcard_gru_trial.md'}")


if __name__ == "__main__":
    main()
