# GW6 audited news packets

Drop one JSON file per note (see `src/live/news_packets.py`).

Required keys: `packet_id`, `source` (whitelist), `url`, `published_at_utc`,
`player_ids`, `headline`, `body`, `sha256` (of headline+body+url+published_at).

`published_at_utc` must be **strictly before** the gameweek deadline.
Forums and unlisted blogs are rejected.

Validate:

```bash
python3 -m src.live.news_packets --gw 6 --validate --with-fpl
```

Empty directory is fine: the loader synthesises FPL bootstrap news as packets.
