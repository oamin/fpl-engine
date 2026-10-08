# String agent — forums, YouTube, and other outlets

Gemini ([open outlets](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)) locked the split. The minutes tag stays on the press whitelist. The string dossier may quote anything filed before the deadline.

## What is allowed

A note in `data/predictions/2026-27/gwNN/string_sources/` has an outlet class: `press`, `forum`, `youtube`, or `other`. The site name is open. Reddit, a YouTube channel, and a podcast all use the same file. The body is an extract of at most 2000 characters, with the URL kept on the quote.

The publish time is re-read from the outlet's own string (`raw_published_at`) through a named clock: YouTube, Reddit unix time, an HTML `time`, JSON-LD, an HTTP date, or a feed. It is stored as `YYYY-MM-DDTHH:MM:SSZ`. A bare date, a naive clock, and "2 hours ago" are refused. `observed_at_utc` is when the note was captured. The order is recorded, then published, then observed, and observed is before the deadline. The hash covers those fields, so moving the capture time by one second breaks it.

The dossier prints the quote with its class, then a sidecar summary line for the compiled minutes. The summary is a footnote.

## What stays shut

`load_gameweek_packets` does not read `string_sources/`. A forum note does not change `xmi`. There is no live crawl in this slice: a page fetched at run time would change the context after the hash was taken. File the note, then the freeze sees it.
