# CI and event regression plan

Original design plan. See [ci.md](ci.md) for the implemented workflow, usage, coverage, and account-setting status.

## Goals and decisions

- A PR must pass unit tests, a recorded pipeline regression, and event-output validation before merging into main.
- A separate live smoke test exercises VLR scraping, VOD discovery, YouTube download, processing, and validation. Run weekly and on demand initially.
- Use standard Ubuntu CPU runners. The current processing stack does not require a GPU.
- Set an Actions product budget of $5/month with usage stopping enabled. This covers paid Actions usage beyond included allowances, not GitHub subscription fees or Codex usage.
- Interpret the three-frame window as the reference processed frame and its immediate neighbors: index difference at most one. This is configurable, but not a three-frame difference or three-second tolerance.
- Baseline event logs represent reviewed expected behavior. Do not automatically regenerate them from PR output.

## Repository findings

- No .github workflows currently exist.
- The project has pytest tests, uv.lock, OpenCV, Tesseract, Playwright, and yt-dlp dependencies.
- scripts/process_vlr_series.sh already connects the major stages, but is designed for batch operation. CI needs explicit stage failures, expected-map checks, and output validation.
- scripts/validate_event_logs.sh prints issues but does not return a failure status for invalid output. Zero logs can also look successful. Replace its CI role with a Python validator; optionally retain the shell script as a wrapper.
- src/valoscribe/video/youtube.py currently downloads the full video before trimming. A short requested section does not currently limit download size.
- Event logs are JSONL dictionaries with type, timestamp, semantic payload, and sometimes timers/confidence. They do not currently carry frame indices.
- VideoReader computes a source-frame stride using an integer FPS ratio. Actual sampling cadence can differ from the requested rate.
- Existing processed match logs can help choose representative fixtures, but need review before becoming expected output.

## Phase 1: establish reproducible inputs

1. Select two or three short broadcast clips with the supported HUD and native dimensions. Preserve HUD pixel dimensions and avoid resizing or arbitrary re-encoding that alters OCR inputs.
2. Include preround initialization, a complete round transition, killfeed activity, abilities, and a spike or ultimate sequence where available. Include mirrored agents across teams if possible.
3. Save the corresponding VLR HTML, series metadata, per-map processing metadata, and reviewed baseline JSONL logs.
4. Create tests/fixtures/pipeline/manifest.json with fixture ID, asset location, SHA-256, source FPS, sampling stride, timestamp origin, HUD-config hash, metadata hash, baseline version, and expected event coverage.
5. Keep small HTML/metadata/log files in Git. Store video clips in a versioned release asset or other stable artifact store; download by fixed identity and verify their hashes before processing. Select the storage mechanism after measuring total size and checking repository visibility/access.
6. Ensure the PR job needs no production cookies or credentials. Avoid using fork-writable caches as authoritative fixture sources.
7. Freeze Python packages from uv.lock and pin Python, Ubuntu, Tesseract, FFmpeg, and browser versions where practical. Record actual tool versions in run metadata. A container with pinned packages is the next step if host-package variability proves material.
8. Generate reviewed baselines in the same Linux environment used by CI. Compare repeated unchanged runs to measure real variance before enforcing a threshold.

## Phase 2: comparator specification

Implement scripts/compare_event_logs.py with reusable comparison functions and tests under tests/test_validation/.

Proposed CLI:

```bash
python scripts/compare_event_logs.py baseline.jsonl candidate.jsonl \
  --manifest tests/fixtures/pipeline/manifest.json \
  --fixture-id round-one \
  --frame-tolerance 1 \
  --threshold 98 \
  --report output/comparison.json
```

Support existing JSONL logs and JSON arrays of event objects. Print a human-readable percentage and write a structured JSON report. Use exit 0 for passing, 1 for a valid comparison below the gate, and 2 for invalid input/configuration or incompatible run metadata.

### Frame alignment

- Prefer an explicit processed-frame index recorded at the event timestamp, plus source-frame index and sampling metadata in run output. Preserve timestamps that describe when the event occurred; do not overwrite them with the later emission time.
- For existing timestamp-only logs, use the shared fixture sampling lattice: timestamp origin plus source-frame stride divided by source FPS. Map event timestamps onto that lattice, with a documented small numerical epsilon and an audit for timestamps that are off-lattice.
- This is the sampling lattice, not the index of an event in the log. Multiple events may occur on one frame; frames with no events still count.
- Require the same clip hash, origin, sampling stride, FPS, HUD config, and metadata inputs. Reject incompatible runs instead of silently stretching or shifting time.
- At exactly 60 source FPS and stride 15, one processed frame is 0.25 seconds. At fractional FPS, use the measured interval instead of assuming 0.25 seconds.
- Match only when absolute processed-frame index difference is at most one. Reference frame 100 can match candidate 99, 100, or 101; 98 and 102 cannot match.
- Do not apply a global time offset automatically. A systematic offset can indicate a reader or trimming regression.
- During implementation, audit backdated/inferred events and their timestamp origin. If any cannot be mapped to a processed frame, explicitly define their timestamp comparison policy before enabling the gate.

### Semantic equivalence

- Require identical event type and semantic payload after documented canonicalization.
- Normalize JSON object key order; preserve array order, scalar types, identifiers, missing fields versus null, scores, player/team/agent identity, killer/victim, weapon, ability, and charge changes.
- Exclude timestamp and frame bookkeeping from payload equality because timing is checked separately.
- Exclude confidence and HUD timer snapshots from the primary similarity score. Report their differences separately. Timer validity remains part of structural validation where applicable.
- Retain all other fields by default, including unknown fields. Add explicit schema validation for known event types so accidentally omitted semantic fields fail.
- Do not fuzzily match player names or ignore changed numeric payloads. Those are meaningful output changes, even if OCR caused them.
- Derive round context from round_start/round_end boundaries for events that lack round_number. Audit and define preround/postround boundary handling; allow timing tolerance at a boundary without matching a repeated event to an unrelated round.
- Group by semantic fingerprint and compatible round context, then compute maximum-cardinality one-to-one matching within the timing window. Among maximum matches, minimize frame displacement and use deterministic tie-breaking.
- Avoid a naive nearest-event greedy algorithm: overlapping windows and duplicates can produce the wrong score. Same-frame event order does not matter; duplicate event counts do.

### Similarity definition

Let R be the number of baseline events, C the number of candidate events, and M the number of matched event pairs:

```text
similarity_percent = 100 * 2M / (R + C)
precision_percent  = 100 * M / C
recall_percent     = 100 * M / R
```

This is event-level F1/Dice similarity. It is symmetric and penalizes missing events and extra/duplicate events. An event with a changed semantic payload counts as one unmatched baseline event and one unmatched candidate event.

Example: R=100, C=102, M=98 gives similarity 97.03%, precision 96.08%, recall 98.00%.

- One empty log against a nonempty log scores 0%.
- Two empty logs have undefined similarity and fail the CI comparison. Never report an empty pipeline as a successful 100% regression run.
- Compare the unrounded score against the threshold; round only for presentation.
- Report per-type matches, precision/recall/F1, missing and extra events with line numbers, matched time deltas, and excluded-field differences. Provide a bounded console preview and complete machine-readable diagnostics.
- Malformed records, nonfinite timestamps, invalid indices, absent required fields, or incompatible manifests fail independently of the percentage.

### Threshold policy

- Begin with 98% overall similarity per fixture as a provisional target, not a threshold proven appropriate by existing measurements.
- Require exact semantic agreement for critical match/round boundaries and scores, still allowing the one-frame timing window.
- Report and calibrate per-type floors for kills, spike events, and ultimates so frequent ability events cannot hide a complete failure in a rare category. Require expected coverage for selected critical categories.
- Do not let an aggregate score across fixtures hide a failing fixture.
- Run each unchanged fixture at least five times in the pinned CI environment. Investigate differences before attributing them to OCR; fixed-input OCR may be stable, while environment or pipeline changes can cause variance.
- Inject deletions, duplicates, player changes, score changes, and two-frame shifts to confirm meaningful regressions fail the intended policy.
- If intentional detector improvements change output, include reviewed baseline updates in the PR and explain changed events. Pin policy/baselines to the trusted main revision for ordinary gate runs so editing the threshold or expected log in a PR cannot make that PR automatically pass. Baseline/policy changes need a distinct explicit maintainer review and controlled adoption step.

### Comparator tests

Cover self-comparison, plus/minus one frame, plus/minus two frames, fractional FPS, boundary timing, same-frame reordering, duplicate multiplicity, ambiguous overlapping windows, changed semantic fields, excluded timer/confidence changes, missing/extra events, malformed JSON, both empty and one empty inputs, incompatible manifests, and exact threshold boundaries. Verify symmetry and maximum matching with small exhaustive cases.

## Phase 3: structural output validation

Implement scripts/validate_pipeline_outputs.py rather than depending on the current shell validator's exit status.

- Require every expected output file and a nonzero processed-frame count.
- Validate JSONL event schemas and CSV headers, finite timestamps, clip bounds, frame/time consistency, and player slots.
- Validate chronological timestamps according to documented inferred/backdated-event behavior; resolve existing valid exceptions explicitly instead of ignoring all ordering checks.
- Check round/score progression and expected event coverage for the chosen fixture.
- Use clip-aware rules: a mid-match clip need not contain match_start/match_end or a final score of 13. Full-map fixtures do require appropriate complete-match checks.
- Fail when no fixtures/maps/logs are actually checked. Exit unsuccessfully on any required violation.
- Keep correctness validation separate from similarity: a log can resemble a bad baseline and still be invalid.

## Phase 4: PR workflow

Create .github/workflows/ci.yml triggered on pull_request and pushes to main. Add merge_group only if a merge queue is enabled later.

Jobs:

1. unit-tests: checkout, locked dependency setup, necessary system tools, pytest. Establish the current suite's baseline first; fix failures before making the job required.
2. pipeline-regression: verify fixtures, parse saved VLR HTML with the production parser, assert discovered VOD/map metadata against the fixture, process the fixed video using PR code, validate CSV/JSONL, and compare event logs to trusted baselines.
3. ci-gate: always evaluate upstream results and fail unless all required jobs succeeded. Never treat a skipped or cancelled upstream test as a passing gate.

Operational controls:

- Start with one Python version and one Linux runner to keep cost low.
- Cache locked dependencies only; cap cache storage at its included limit. Do not cache full VODs.
- Use read-only contents permissions and pin third-party Actions to reviewed commit SHAs.
- Cancel superseded runs for the same PR with concurrency groups.
- Set measured job timeouts; initial targets are 10 minutes for unit tests and 20 minutes for short regression fixtures. Adjust after benchmarking.
- Avoid workflow-level path filters on the required check so required checks are not indefinitely absent.
- Upload reports and diagnostic logs even on failure, with seven-day retention. Cap bulky CSV/sample-frame artifacts; never upload raw full VODs by default.
- Publish a concise job summary: fixture name, similarity/threshold, per-type differences, structural result, baseline revision, and link to artifacts. This makes the PR check useful on a phone.
- Do not execute untrusted PR code in a privileged pull_request_target job.

Recorded regression exercises the production parser and processing/output path with fixed inputs. Unit tests cover download integration logic, but only the live workflow verifies current network download behavior.

## Phase 5: live end-to-end workflow

Create .github/workflows/live-pipeline.yml triggered by weekly schedule and workflow_dispatch, initially on trusted main code.

1. Scrape a fixed supported VLR match; require expected teams, players, map, and a YouTube link.
2. Download the selected VOD through the production downloader; verify it decodes and has the expected dimensions/FPS/duration.
3. Process a bounded representative section, validate outputs, and compare to a baseline if clip identity and alignment are compatible.
4. Fail clearly at the responsible stage. Permit only bounded retries for transient network failures; never retry away a semantic regression.
5. Upload small reports/logs and a job summary with the same fields as PR CI.

First benchmark download size and time. Implement and test reliable section downloading before relying on short clips to control live-run cost. Until then, a full-VOD download must have explicit disk/time limits. Do not assume section downloading is supported just because trimming parameters exist.

If live video bytes change, report an input-integrity failure rather than silently comparing against a mismatched baseline. VLR/YouTube outages or access restrictions should be distinguishable from processing failures. Do not add personal browser cookies as the default solution to runner access problems.

Keep the live job advisory for ordinary PRs initially. A schedule's success on main does not validate a new PR. If a risky downloader/scraper PR needs a live merge gate, run a check on that PR's exact commit and require that check; test credentials must not be exposed to untrusted code.

## Phase 6: merge rules

After successful trial runs, protect main with:

- PRs required for changes.
- Required ci-gate check from GitHub Actions.
- Branch up to date before merging, so CI checks the proposed merge with current main.
- No routine bypass for administrators, direct pushes, or force pushes.
- A documented emergency procedure rather than silent disabling of checks.

Do not require a separate approving reviewer by default for a solo repository: GitHub does not let an author approve their own PR. Use enforced checks plus the maintainer's merge decision. Baseline/policy adoption remains a deliberate, auditable maintainer action.

Check plan eligibility first. Public repository protection is available on Free; private personal repositories need Pro, and private organization repositories need an eligible paid plan. Subscription cost is separate from the Actions budget.

## Phase 7: $5/month budget

Configure this in the repository owner's GitHub billing settings, not workflow YAML:

1. Open Settings -> Billing & licensing -> Budgets and alerts.
2. Inspect existing overlapping budgets so a stricter existing limit is understood before adding another.
3. Create or update a Product-level budget for GitHub Actions.
4. Use repository scope for valoscribe if available to this account. Otherwise use owner-account scope and document that other repositories consume the same $5.
5. Set monthly budget to $5 USD.
6. Enable Stop usage when budget limit is reached. Alerts alone do not stop spending.
7. Enable threshold alerts at 75%, 90%, and 100%, and included-usage alerts where available.
8. Save and reread the budget to confirm product, scope, amount, and stop setting.

The budget covers paid Actions usage after included allowances. It does not cap subscription fees, external fixture hosting, Git LFS charges, or Codex. GitHub documents that usage before budget creation is excluded from its first-cycle calculation, so existing spend can make that month's total exceed $5. Retained storage also continues accruing charges until deleted; combine the budget with small artifacts, short retention, and cache limits rather than promising an exact invoice ceiling.

Standard Linux x64 usage beyond the allowance is currently $0.006/minute: $5 corresponds to approximately 833 extra compute minutes if there are no other Actions charges. GitHub Free includes 2,000 monthly private-repo minutes and Pro includes 3,000; public standard-runner compute is free. Included private usage is shared across the owner's repositories.

Illustrative monthly workload, not a benchmark: 100 PR runs at 10 total runner-minutes plus four live runs at 30 minutes is 1,120 minutes. That can fit the Free allowance if other repositories leave sufficient capacity. Install time, reruns, and parallel jobs all contribute.

When billing blocks required jobs, merging should remain blocked. Wait for reset or deliberately revise the budget; never bypass the gate automatically.

## Delivery order and acceptance

1. Comparator plus meaningful tests and structural validator.
2. Fixture manifest, reviewed clips, and Linux baseline calibration.
3. Required PR workflow; verify a good PR passes and deliberately broken PRs fail.
4. Benchmark and harden live downloading; enable weekly/manual live workflow.
5. Configure and verify $5 Actions budget.
6. Enable main protection after check names exist and workflows are green.

Done means: main cannot merge a failing regression PR; within-one-frame identical events match; beyond-one-frame events do not; missing/extra/changed events affect the score; empty or malformed outputs fail; reports identify regressions; network failures are identifiable; and the owner's billing UI confirms the $5 stop-usage budget.

## Sources checked

- GitHub budgets: https://docs.github.com/en/billing/how-tos/set-up-budgets
- Budget behavior/limitations: https://docs.github.com/en/billing/concepts/budgets-and-alerts
- Actions billing and included usage: https://docs.github.com/en/billing/concepts/product-billing/github-actions
- Runner rates: https://docs.github.com/en/billing/reference/actions-runner-pricing
- Branch protection: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches
