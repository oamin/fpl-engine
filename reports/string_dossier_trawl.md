# String dossier and manager notebook

Research only. No squad was chosen and no live feed was stored.

## What Gemini locked

The formula was accepted before the code was written ([dossier trawl](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)). YouTube uses `youtube_published_at`. An RSS HTTP date uses `http_last_modified`. An RSS ISO time uses `feed_updated`. A Reddit listing uses `reddit_created_utc`. A URL already on disk is skipped. The notebook has one row per gameweek. A later prompt sees only earlier rows. A banned score token writes nothing.

## Collector

`python3 -m src.str_agent.trawl --gw 6 --deadline 2026-10-10T10:00:00Z`

The named list is BBC Sport football, the Guardian Premier League RSS, Sky Sports football RSS, the r/FantasyPL and r/PremierLeague RSS feeds, and three YouTube channels: Premier League, Sky Sports Football, and Fantasy Football Scout. The Reddit JSON listing is refused from this network, so the live list uses the RSS feed and `feed_updated`. The JSON clock `reddit_created_utc` remains for a listing that exposes it. A Sky `pubDate` ending in `BST` is read as UTC+1, and the stored raw string still ends in `BST`. The file stored is the feed title and the feed summary, cut at 2,000 characters. The article page is not fetched. Files land in `data/predictions/2026-27/gwNN/string_sources/`. `news_packets/` is not written, so compiled minutes do not move.

## Notebook

A legal plan may include `notes` and `adjustments`. They are appended to `data/predictions/2026-27/string_plans/notebook.jsonl`. The same gameweek replaces its row. The next deadline's dossier includes those notes and still loads only that week's feeds. Missing notes do not fail a plan.

## Checks

An item published at the deadline is dropped. A bare date is dropped. A second run does not rewrite the file. A forum file is invisible to `compile_player_xmi`. A Gameweek 7 note is absent from the Gameweek 6 and Gameweek 7 prompts and present in the Gameweek 8 section. `score_xp` in the notes leaves the directory empty.
