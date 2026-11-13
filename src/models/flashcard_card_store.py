"""Save a reader beside the card it was trained on.

The trial used to write only a markdown table. The weights died with the
process, and a later card change left the old number in the report. A stored
card is the picture, the layout, and the state dict that produced the table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
CURRENT_DIR = ROOT_REPORTS / "cards" / "current"
LEGACY_DIR = ROOT_REPORTS / "cards" / "legacy-odds-venue-opponent"


def save_png(path: Path, image: np.ndarray, scale: int) -> None:
    """Write one card. Nearest-neighbour keeps each square a solid block."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    picture = Image.fromarray(np.ascontiguousarray(image))
    picture.resize((width * scale, height * scale), Image.Resampling.NEAREST).save(path)


def write_layout(directory: Path, layout: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")


def save_reader(directory: Path, name: str, model: nn.Module, meta: dict) -> None:
    """Write `name.pt` and `name.json`. The json records the header the weights fit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "header_h": int(model.header_h),
        "player_h": int(model.player_h),
        **meta,
    }
    torch.save(model.state_dict(), directory / f"{name}.pt")
    (directory / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_reader(directory: Path, name: str) -> nn.Module:
    """Build the reader described by `name.json` and load `name.pt`."""
    from src.models.flashcard_intensity_trial import IntensityReader
    from src.models.flashcard_strip_trial import StripVision

    directory = Path(directory)
    meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    dropout = float(meta.get("dropout", 0.1))
    header_h = int(meta["header_h"])
    player_h = int(meta["player_h"])
    if meta.get("arch") == "fixture_intensity":
        model = IntensityReader(
            header_h,
            mode=str(meta.get("mode", "fixture")),
            dropout=dropout,
            player_h=player_h,
        )
    else:
        model = StripVision(header_h, dropout=dropout, player_h=player_h)
    state = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
