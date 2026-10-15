# Corrode first-round regression fixture

Source: https://youtu.be/WPm6FMkZ3Qo (NRG vs FNATIC, Champions 2025 grand final).
The captured section starts near 132 seconds and lasts two minutes. Its original
stream-copy boundary is keyframe dependent. The checked-in clip is sampled at
4 FPS, keeps the original 1920x1080 HUD dimensions, and uses H.264 CRF 28.
It is a fixed regression input, not a byte-identical excerpt of every currently
available YouTube rendition.

`vlr.html` contains real match-page markup captured during setup, reduced to the
header, VOD embeds, map navigation, and per-map player/round data. It retains all
five maps so the parser's map indexing is also tested. `series.json` and
`metadata.json` are expected production-parser output and first-map processing
input. `manifest.json` describes bounds, sampling, coverage, thresholds, and hashes.

The initial baseline contains 43 events: 21 ability uses, four ability recharges,
seven deaths, seven kills, one spike plant, match start, round start, and round end.
It reflects the current-main implementation's shared Astra charge validation.
This baseline captures existing behavior; it is not a human-annotated accuracy
benchmark. In particular, the first fixture has no ultimate or full-match end.

Five repeated local Windows runs under the locked dependency environment each
matched at 100%. Processing took approximately 44-57 seconds per run. GitHub's
Linux pipeline check separately verifies that the same baseline is portable.
Review per-event differences before adopting a new baseline. The runner never
updates this file automatically.

SHA-256 values cover exact file bytes. `.gitattributes` disables newline
conversion in this directory. Do not normalize fixtures after updating hashes.
