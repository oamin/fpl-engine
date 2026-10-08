# How news packets become numeric `xmi`

Gemini ([compile](bc-90156d5d-ce87-56ee-b772-011f4d43dbd0)): reuse the closed `news_tags` enum and `minutes_for_tag`; do not invent new minute bins. Deterministic phrase rules for tests; LLM classify can sit behind the same cite gate later.

## Pipeline

```
audited packet JSON
  → (optional) synthetic FPL news packet
  → SUPPORT phrase match → tag ∈ {firm_starter, injured, transferred, benched, ask}
  → minutes_for_tag(tag, position, prior, chance, status) → xmi
```

| Tag | xmi rule |
| --- | --- |
| injured + chance in (0,100) | `(chance/100) × prior` |
| injured otherwise / transferred | `0` |
| firm_starter | `clip(prior, 65, 90)` (0 if status `i`) |
| benched GKP | `0` |
| benched outfield | `15` |
| ask / none | leave prior / no write |

Pre-emption: status `s`/`u` or chance `0` → `0`.

## What the context looks like

See the live demo for Saliba / van Ewijk / Haaland:

- `reports/packet_xmi_compile_demo_20261008.md` — markdown the compile sees
- `reports/packet_xmi_compiled_tags_20261008.json` — numeric rows

Note: Saliba’s FPL `news_added` shares a bulk timestamp with 29 other players, so the synthetic FPL loader drops it. Curated `arsenal_fc` packet is what supplies his evidence — that is the point of audited packets.

## Live CSV

This path writes a **sidecar** only. It does not overwrite `data/live/xmi_gw6.csv` until we agree a merge rule (packet xmi vs minutes_llm).
