# GW6 audited news packets

Drop one JSON file per note (see `src/live/news_packets.py`).

Required keys: `packet_id`, `source` (whitelist), `url`, `published_at_utc`,
`player_ids`, `headline`, `body`, `sha256` (of headline+body+url+published_at).

`published_at_utc` must be **strictly before** the gameweek deadline.
Forums and unlisted blogs are rejected.
The URL must be a real `http(s)` article. A host or path containing
`example` (or `localhost`) is rejected. `fpl:bootstrap` is allowed only
on synthetic FPL packets. Do not hand-write a presser quote.

Every 2026/27 club has a local desk in `CLUB_PRESS` (MEN, Football London,
Birmingham Mail, Bournemouth Echo, The Argus, Liverpool Echo, Hull Daily Mail,
East Anglian Daily Times, Yorkshire Evening Post, Nottingham Post, Sunderland
Echo, Coventry Telegraph, Chronicle, plus Sports Mole and The Standard).
The latest packet's time wins the tag. "Could return this weekend" stays `ask`.

Validate:

```bash
python3 -m src.live.news_packets --gw 6 --validate --with-fpl
```

Empty directory is fine: the loader synthesises FPL bootstrap news as packets.
