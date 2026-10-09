"""Freeze the live 2026/27 snapshots. This module reads them and writes only the manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "live" / "HOLDOUT_FREEZE.json"

GROWING_CACHE = "data/cache/player_gw_2026_27.csv"
SNAPSHOT_DIR = Path("data/holdout/2026-27/player_gw")
TRACKED = (
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
            "The closed-season comparison does not include this season. "
            "Gameweeks 1-5 were already read. The clean holdout starts at gameweek 6. "
            "Timestamped files under data/predictions/ are the pre-deadline forecasts. "
            "A hash of a file that keeps growing is not a freeze of a future gameweek."
        ),
        "clean_holdout_from_gw": 6,
        "contaminated_through_gw": 5,
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


def write_player_gw_snapshots(root: Path | None = None) -> list[str]:
    """Copy each gameweek of the cache to its own file. An existing file is not rewritten."""
    import pandas as pd

    base = root or ROOT
    frame = pd.read_csv(base / GROWING_CACHE)
    dest = base / SNAPSHOT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for gw, block in frame.groupby(frame["gw"].astype(int), sort=True):
        rel = f"{SNAPSHOT_DIR.as_posix()}/gw{int(gw):02d}.csv"
        path = base / rel
        ordered = block.sort_values(
            ["date", "player_id", "is_home", "team"], kind="mergesort"
        )
        payload = ordered.to_csv(index=False, lineterminator="\n")
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to change snapshot {rel}")
        if not path.exists():
            path.write_text(payload, encoding="utf-8")
        written.append(rel)
    return written


def publish_player_gw_snapshots(root: Path | None = None) -> dict[str, Any]:
    """Add one hash per gameweek and retire the single hash of the growing cache.

    Every hash already in the manifest has to still match. This does not
    rehash those files.
    """
    base = root or ROOT
    rels = write_player_gw_snapshots(base)
    dest = base / "data" / "live" / "HOLDOUT_FREEZE.json"
    manifest = json.loads(dest.read_text(encoding="utf-8"))
    tracked = dict(manifest["tracked"])
    for rel, digest in tracked.items():
        if rel == GROWING_CACHE:
            continue
        if sha256_file(base / rel) != digest:
            raise RuntimeError(f"refusing to publish while {rel} has changed")
    retired = tracked.get(GROWING_CACHE)
    if retired is not None and sha256_file(base / GROWING_CACHE) != retired:
        raise RuntimeError("the growing cache no longer matches the hash being retired")
    for rel in rels:
        digest = sha256_file(base / rel)
        previous = tracked.get(rel)
        if previous is not None and previous != digest:
            raise RuntimeError(f"refusing to change the hash of {rel}")
        tracked[rel] = digest
    if GROWING_CACHE in tracked:
        manifest["retired_growing_cache"] = {
            "path": GROWING_CACHE,
            "sha256": tracked.pop(GROWING_CACHE),
            "retired": "2026-10-07",
            "reason": (
                "One hash cannot freeze a file that grows. "
                "Each finished gameweek has its own file."
            ),
        }
    manifest["tracked"] = tracked
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest = build_manifest()
    dest = path or MANIFEST_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    write_manifest()
    print(f"Wrote {MANIFEST_PATH}")
