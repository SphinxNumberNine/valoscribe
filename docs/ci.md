# Pipeline CI

CI runs on GitHub-hosted Ubuntu machines. Once these workflows are merged, the
development computer does not need to be online for tests or merge protection.

## PR checks

`CI Gate` succeeds only when **Unit tests** and **Recorded pipeline** both succeed.
Cancelled, failed, or skipped upstream jobs fail the gate. Main must be up to date
before merging. No separate approving reviewer is required for this solo project.

The recorded job parses a saved real VLR page, checks all five maps' metadata,
processes a fixed two-minute 1080p clip, validates CSV/JSONL output, and compares
events against a reviewed baseline. It has no external VLR/YouTube dependency.
Dependencies come from `uv.lock`; external Actions are pinned to commit hashes.

The initial fixture covers match start, a complete first round, deaths/kills,
abilities, and a spike plant in NRG vs FNATIC, Champions 2025 grand final, Corrode.
It does not yet cover ultimates, overtime, halftime, or full-match finalization.
The clip preserves HUD dimensions but is sampled to 4 FPS and encoded as H.264 to
keep its checked-in size around 16 MB. The hash-verified bytes, not a newly
downloaded/re-encoded video, define the recorded test input.

On PRs, fixture inputs and thresholds come from the PR's base revision. This
prevents an ordinary PR's baseline/threshold edits from changing its own expected
results. The first CI-installation PR bootstraps the initial baseline from its
branch; review that expected output before merging. Workflow/comparator changes
still require code review: a test runner executing PR code is not a security
boundary against a malicious maintainer.

## Compare two logs

```bash
uv run python scripts/compare_event_logs.py expected.jsonl actual.jsonl \
  --sample-fps 4 --frame-tolerance 1 --threshold 98 \
  --type-threshold kill=95 --type-threshold spike_plant=100 \
  --report ci-output/comparison.json
```

Inputs can be JSONL or JSON arrays. `--sample-fps` means the actual processing
cadence. For fractional source FPS on the current VideoReader, also pass the
source FPS, for example `--sample-fps 4 --source-fps 29.97002997002997`. The
comparator accounts for sampling instants rounded up to available source frames.
Use `--frame-interval` instead of `--sample-fps` for legacy integer-stride logs.
`--origin` specifies the common sampling origin; logs must describe the same
video, range, configuration, and sampling policy. The fixture runner verifies
video/metadata hashes and FPS before making a comparison.

An event may match an equivalent event one processed frame earlier, on the same
frame, or one later. Matching is one-to-one and ignores ordering among simultaneous
events. Duplicate and extra events are penalized. Semantic fields must match
exactly, including JSON scalar types. Timestamps/frame bookkeeping are checked
separately; confidence and HUD timer snapshots are diagnostic fields excluded
from semantic equality.

Similarity is event-level F1/Dice:

```text
similarity_percent = 200 * matched_pairs / (reference_count + candidate_count)
```

Current fixture policy is 98% overall, 95% for kills, 100% for spike plants, and
100% for match/round boundaries and scores, all with the same timing tolerance.
Each fixture must pass independently. A critical mismatch cannot be hidden by
many matching ability events. Malformed logs and two empty logs fail; one empty
side against a nonempty side scores zero.

Exit codes: 0 passed; 1 valid comparison below policy; 2 invalid inputs/config.
The JSON report contains precision/recall, per-type results, matched timing deltas,
missing/extra events, and diagnostic confidence/timer differences.

## Reproduce the recorded run

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run python -m valoscribe.validation.pipeline --output ci-output/local-recorded
uv run python scripts/validate_pipeline_outputs.py ci-output/local-recorded \
  --min-frames 300 --require-type kill --require-type round_end
```

Use a fresh output directory. Install Tesseract, FFmpeg and OpenCV's system
libraries first (the workflow contains Ubuntu commands). Frame CSV timestamps
must increase, frame numbers must increase, expected columns and coverage must
exist, and event times must be inside the processed range. Backdated ability
confirmation events may arrive out of order; boundary progression is checked
separately. Use `--full-map` only for complete maps; clips do not need a final
score of 13 or match_end.

The previous shell validator now delegates to this validator, fails for missing
logs, and returns a nonzero status on validation failure.

## Live smoke test

`Live pipeline` runs weekly on Mondays at 09:23 UTC and can be dispatched manually
after merging. It deliberately checks out trusted main code. It scrapes the live
match, discovers its YouTube VOD, downloads only a bounded 120-second section,
checks video dimensions/FPS/duration, processes it, and validates event coverage.
It is advisory, not a required check for ordinary PRs.

```bash
uv run python -m valoscribe.validation.live --output ci-output/local-live
```

Section downloading uses FFmpeg via yt-dlp with exact-cut re-encoding. It never
falls back to downloading the full VOD. The existing normal downloader path and
its provenance-aware full-source cache remain available. Section results have a
separate request-identified cache and measured media metadata.

Network, VLR HTML, YouTube extraction/access, media, processing, and validation
failures are labeled by stage. The live run does not compare a newly encoded clip
to the fixed recorded baseline: source bytes, codecs, and sampling can differ.
Use a successful live check on the exact PR commit if a future risky downloader
change needs to become a merge requirement; a scheduled main run does not test a PR.

Reports and processing logs are retained for seven days. Downloaded live videos
are excluded from artifacts. Superseded PR runs are cancelled; job timeouts bound
compute usage. Unit tests run on one Python version to avoid multiplying costs.

## Baseline changes

Do not regenerate baselines automatically in CI. To deliberately change expected
behavior, run reviewed code on the fixed clip in the CI environment, inspect the
event differences, replace baseline.jsonl, and update its SHA-256 in manifest.json.
Keep the input video/HTML/metadata unchanged unless the fixture itself is changing.
Fixture bytes are exempt from Git line-ending conversion to preserve checksums.

For an intentional behavior change that fails comparison to the base baseline,
first merge a compatible preparation/change in separate PRs, or explicitly adopt
the reviewed new baseline through a documented maintainer exception. Do not lower
the threshold until a failing implementation passes. The first fixture establishes
a regression baseline for current behavior; it is not a claim of perfect detection
accuracy against human annotations.

## Merge rules and $5 monthly budget

`.github/ci-ruleset.json` is the reviewable main-branch ruleset definition. Apply
it only after the initial CI checks exist and pass. It requires a PR, up-to-date
status checks from GitHub Actions, and resolved review threads; it blocks deletion
and force pushes without routine admin bypass.

The Actions budget is an account setting, not YAML. In the repository owner's
[Budgets and alerts](https://github.com/settings/billing/budgets), create/update a
**GitHub Actions product budget** of **$5/month**, scoped to this repository if
available, otherwise the owner account. Enable **Stop usage when budget limit is
reached** and alerts at 75%, 90%, and 100%. Account scope includes other repositories.
Verify the saved amount, scope and stop setting. Do not add a separate $5 budget
for each SKU, because that would allow more than $5 of combined spending.

This repository is public, so standard hosted-runner compute is free under current
GitHub pricing. The budget is still a useful guard against paid Actions usage.
It does not cover Codex, subscriptions, Git LFS, or other services. First-cycle
usage before budget creation is excluded, and retained storage continues to accrue
until removed. The budget cannot guarantee an exact all-inclusive $5 invoice.

The current session has repository-admin API access, but its browser is signed
out of GitHub and the documented budget REST API is for organizations/enterprises.
The personal-account budget must therefore be saved in a signed-in browser;
it has **not** been activated by the code changes.

References: [budget setup](https://docs.github.com/en/billing/how-tos/set-up-budgets),
[Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions),
[repository rulesets](https://docs.github.com/en/rest/repos/rules).
