"""numagent_lock freezes a pick without overwrite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.numagent_lock import build_lock_doc, write_lock


def _toy_pick() -> dict:
    return {
        "ok": True,
        "gw": 99,
        "primary": "L3",
        "entry_id": 2632584,
        "team_name": "toy",
        "bank": 10,
        "ft": 1,
        "odds_source": "snapshot",
        "board_path": None,
        "board_t1_path": None,
        "xmi_path": "x.csv",
        "recommendation": {
            "n_transfers": 1,
            "hits": 0,
            "hit_points": 0,
            "sold_names": ["A"],
            "bought_names": ["B"],
            "captain_name": "Haaland",
            "captain_id": 411,
            "formation": "1-3-4-3",
            "objective": 50.0,
            "xi_mu": 40.0,
            "captain_mu": 10.0,
            "bank_after": 5,
            "ft_after": 1,
            "xi": [
                {
                    "player_id": "411",
                    "name": "Haaland",
                    "position": "FWD",
                    "club": "Man City",
                    "mu_post": 10.0,
                    "captain": True,
                }
            ],
            "bench": [
                {
                    "player_id": "1",
                    "name": "Raya",
                    "position": "GKP",
                    "club": "Arsenal",
                    "mu_post": 4.0,
                }
            ],
        },
        "l3": None,
        "l4": None,
    }


def test_write_lock_and_refuse_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.live.numagent_lock as mod

    monkeypatch.setattr(mod, "PRED", tmp_path / "pred")
    monkeypatch.setattr(mod, "REPORTS", tmp_path / "reports")
    doc = _toy_pick()
    # recommendation arm uses recommendation block; set l3 alias
    doc["l3"] = doc["recommendation"]
    lock_doc = build_lock_doc(doc=doc, arm="l3", note="test lock")
    paths = write_lock(gw=99, entry_id=2632584, lock_doc=lock_doc, pick_doc=doc)
    assert paths["lock"].is_file()
    loaded = json.loads(paths["lock"].read_text())
    assert loaded["status"] == "locked"
    assert loaded["decision"]["captain"] == 411
    with pytest.raises(FileExistsError):
        write_lock(gw=99, entry_id=2632584, lock_doc=lock_doc, pick_doc=doc)
