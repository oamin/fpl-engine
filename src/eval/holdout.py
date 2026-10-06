"""Freeze the live 2026/27 snapshots. This module reads them and writes only the manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "live" / "HOLDOUT_FREEZE.json"

TRACKED = (
    "data/cache/player_gw_2026_27.csv",
    "data/cache/E0_2627.csv",
    "data/live/gw_lines.csv",
    "data/live/news_docs_gw15.json",
)
OPTIONAL = (
    "data/live/bootstrap.json",
    "data/live/fixtures.json",
    "data/live/xmi_gw6.csv",
    "data/live/minutes_raw.json",
    "data/live/minutes_prompt.txt",
    "data/live/odds_api_meta.json",
    "data/live/odds_api_trial.json",
    "data/cache/fpl_history_2026_27.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path | None = None) -> dict[str, Any]:
    """Hash snapshots that already exist. Does not rewrite them."""
    base = root or ROOT
    tracked = {rel: sha256_file(base / rel) for rel in TRACKED}
    present = {
        rel: sha256_file(base / rel)
        for rel in OPTIONAL
        if (base / rel).is_file()
    }
    return {
        "season": "2026-27",
        "frozen": "2026-10-06",
        "rule": (
            "Do not modify these snapshots. Do not fit a parameter on 2026-27. "
            "The closed-season comparison does not include this season."
        ),
        "tracked": tracked,
        "present_at_freeze": present,
    }


def verify_freeze(manifest: dict[str, Any], root: Path | None = None) -> list[str]:
    """Tracked paths must match. An optional path is checked only when the file is present."""
    base = root or ROOT
    errors: list[str] = []
    for rel, digest in manifest.get("tracked", {}).items():
        path = base / rel
        if not path.is_file():
            errors.append(f"missing tracked snapshot {rel}")
        elif sha256_file(path) != digest:
            errors.append(f"hash mismatch {rel}")
    for rel, digest in manifest.get("present_at_freeze", {}).items():
        path = base / rel
        if not path.is_file():
            continue
        if sha256_file(path) != digest:
            errors.append(f"hash mismatch {rel}")
    return errors


def write_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest = build_manifest()
    dest = path or MANIFEST_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    write_manifest()
    print(f"Wrote {MANIFEST_PATH}")
