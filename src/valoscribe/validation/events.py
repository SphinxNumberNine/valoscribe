"""Compare event multisets using one-to-one, frame-tolerant semantic matching.

No OCR, video libraries, or network access are required by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

IGNORED_FIELDS = {"timestamp", "frame_number", "processed_frame_index", "confidence", "timers"}
CRITICAL_TYPES = {"match_start", "match_end", "round_start", "round_end"}
REQUIRED_FIELDS = {
    "match_start": ("team1", "team2"),
    "ability_used": ("player", "team", "agent", "ability", "charges_used", "remaining_charges"),
    "ability_recharged": ("player", "team", "agent", "ability", "charges_gained", "total_charges"),
    "ultimate_used": ("player", "team", "agent", "previous_charges", "current_charges"),
    "death": ("player", "team", "agent"),
    "revival": ("player", "team", "agent"),
    "kill": ("killer_agent", "killer_side", "victim_agent", "victim_side"),
}


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    if isinstance(value, list):
        for item in value:
            _finite_json(item)


def validate_events(events: list[dict]) -> None:
    if not events:
        raise ValueError("Event log is empty")
    for number, event in enumerate(events, 1):
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("type"), str)
            or not event["type"]
        ):
            raise ValueError(f"Event {number}: nonempty type required")
        timestamp = event.get("timestamp")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"Event {number}: finite nonnegative timestamp required")
        _finite_json(event)
        for field in REQUIRED_FIELDS.get(event["type"], ()):
            value = event.get(field)
            if field in {
                "charges_used",
                "remaining_charges",
                "charges_gained",
                "total_charges",
                "previous_charges",
                "current_charges",
            }:
                if type(value) is not int or value < 0:
                    raise ValueError(f"Event {number}: invalid {field}")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"Event {number}: nonempty {field} required")
        for field in ("frame_number", "processed_frame_index"):
            if field in event and (type(event[field]) is not int or event[field] < 0):
                raise ValueError(f"Event {number}: invalid {field}")
        if event["type"] in {"round_start", "round_end"}:
            for field in ("round_number", "score_team1", "score_team2"):
                if type(event.get(field)) is not int or event[field] < (
                    1 if field == "round_number" else 0
                ):
                    raise ValueError(f"Event {number}: invalid {field}")
        if "confidence" in event:
            confidence = event["confidence"]
            if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                raise ValueError(f"Event {number}: invalid confidence")
        if "timers" in event:
            if not isinstance(event["timers"], dict):
                raise ValueError(f"Event {number}: timers must be an object")
            for timer in event["timers"].values():
                if timer is not None and (type(timer) not in (int, float) or timer < 0):
                    raise ValueError(f"Event {number}: invalid timer")


def load_events(path: Path, *, allow_empty=False) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        events = json.loads(text)
        if not isinstance(events, list):
            raise ValueError("JSON log must be an array")
    else:
        events = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if events or not allow_empty:
        validate_events(events)
    return events


def _frame(event: dict, interval: float, origin: float, source_fps=None) -> int:
    position = (event["timestamp"] - origin) / interval
    index = math.floor(position + 0.5)
    if source_fps is not None:
        # VideoReader rounds sampling instants up to the next available source frame.
        source_position = (event["timestamp"] - origin) * source_fps
        source_index = round(source_position)
        index = math.floor(source_index / (interval * source_fps) + 1e-8)
        expected_source = math.ceil(index * interval * source_fps - 1e-8)
        if abs(source_index - source_position) > 1e-3 or source_index != expected_source:
            raise ValueError("Event timestamp is off the source sampling lattice")
        position = index
    # Float serialization is tolerated, but off-grid events cannot silently widen the window.
    if index < 0 or abs(position - index) > 1e-3:
        raise ValueError(f"Event timestamp {event['timestamp']} is off the sampling lattice")
    if "processed_frame_index" in event and event["processed_frame_index"] != index:
        raise ValueError("Processed frame index disagrees with timestamp")
    return index


def _match(left, right, tolerance):
    """Ordered interval matching: maximize count, then minimize total displacement.

    Equal semantic payloads form each group. On a line, uncrossing valid pairs
    preserves cardinality and cannot increase absolute displacement, so this
    dynamic program also finds an optimum among arbitrary bipartite matchings.
    """
    rows, cols = len(left), len(right)
    scores = [[(0, 0)] * (cols + 1) for _ in range(rows + 1)]
    steps = [[None] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            options = [(scores[i - 1][j], "left"), (scores[i][j - 1], "right")]
            distance = abs(left[i - 1][0] - right[j - 1][0])
            if distance <= tolerance:
                count, cost = scores[i - 1][j - 1]
                options.append(((count + 1, cost - distance), "pair"))
            scores[i][j], steps[i][j] = max(options, key=lambda item: item[0])
    pairs = []
    i, j = rows, cols
    while i and j:
        action = steps[i][j]
        if action == "pair":
            pairs.append((left[i - 1][1], right[j - 1][1]))
            i, j = i - 1, j - 1
        elif action == "left":
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def compare_events(
    reference,
    candidate,
    *,
    frame_interval=0.25,
    origin=0.0,
    frame_tolerance=1,
    threshold=98.0,
    critical_types=CRITICAL_TYPES,
    type_thresholds=None,
    source_fps=None,
):
    """Return F1/Dice percentage and diagnostics; invalid/empty logs raise ValueError."""
    if not math.isfinite(frame_interval) or frame_interval <= 0:
        raise ValueError("frame_interval must be positive and finite")
    if not math.isfinite(origin) or origin < 0:
        raise ValueError("origin must be nonnegative and finite")
    if source_fps is not None and (
        not math.isfinite(source_fps) or source_fps <= 0 or frame_interval * source_fps < 1
    ):
        raise ValueError("source-fps must be finite and at least the sampling FPS")
    if type(frame_tolerance) is not int or frame_tolerance < 0:
        raise ValueError("frame_tolerance must be a nonnegative integer")
    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    for floor in (type_thresholds or {}).values():
        if not math.isfinite(floor) or not 0 <= floor <= 100:
            raise ValueError("Per-type thresholds must be between 0 and 100")
    # Empty sides are meaningful only if the other side contains events.
    if not reference and not candidate:
        raise ValueError("Both event logs are empty")
    for events in (reference, candidate):
        if events:
            validate_events(events)
    groups = [defaultdict(list), defaultdict(list)]
    frames = []
    for events, group in zip((reference, candidate), groups):
        indices = [_frame(event, frame_interval, origin, source_fps) for event in events]
        frames.append(indices)
        for number, (event, frame) in enumerate(zip(events, indices)):
            payload = {key: value for key, value in event.items() if key not in IGNORED_FIELDS}
            key = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            group[key].append((frame, number))
    pairs = []
    for key in sorted(groups[0].keys() & groups[1].keys()):
        pairs.extend(_match(sorted(groups[0][key]), sorted(groups[1][key]), frame_tolerance))
    matched_reference = {i for i, _ in pairs}
    matched_candidate = {j for _, j in pairs}

    def metrics(r, c, m):
        return {
            "reference": r,
            "candidate": c,
            "matched": m,
            "similarity_percent": 200 * m / (r + c) if r + c else None,
            "precision_percent": 100 * m / c if c else 0.0,
            "recall_percent": 100 * m / r if r else 0.0,
        }

    result = metrics(len(reference), len(candidate), len(pairs))
    types = sorted({e["type"] for e in reference + candidate})
    result["per_type"] = {
        kind: metrics(
            sum(e["type"] == kind for e in reference),
            sum(e["type"] == kind for e in candidate),
            sum(reference[i]["type"] == kind for i, _ in pairs),
        )
        for kind in types
    }
    failures = []
    if result["similarity_percent"] < threshold:
        failures.append(f"Overall similarity below {threshold}%")
    for kind, values in result["per_type"].items():
        floor = 100 if kind in critical_types else (type_thresholds or {}).get(kind, 0)
        if values["similarity_percent"] < floor:
            failures.append(f"{kind} similarity below {floor}%")
    for kind in type_thresholds or {}:
        if kind not in result["per_type"]:
            failures.append(f"Required event type absent: {kind}")
    result.update(
        {
            "passed": not failures,
            "failures": failures,
            "threshold": threshold,
            "frame_interval": frame_interval,
            "origin": origin,
            "frame_tolerance": frame_tolerance,
            "missing": [
                {"event_number": i + 1, "event": event}
                for i, event in enumerate(reference)
                if i not in matched_reference
            ],
            "extra": [
                {"event_number": j + 1, "event": event}
                for j, event in enumerate(candidate)
                if j not in matched_candidate
            ],
            "matches": [
                {
                    "reference_event": i + 1,
                    "candidate_event": j + 1,
                    "frame_delta": frames[1][j] - frames[0][i],
                    "diagnostic_differences": {
                        field: {
                            "reference": reference[i].get(field),
                            "candidate": candidate[j].get(field),
                        }
                        for field in ("confidence", "timers")
                        if reference[i].get(field) != candidate[j].get(field)
                    },
                }
                for i, j in sorted(pairs)
            ],
        }
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--frame-interval", type=float, help="Actual seconds per processed frame")
    timing.add_argument("--sample-fps", type=float, help="Actual processed FPS, not video FPS")
    parser.add_argument("--origin", type=float, default=0.0)
    parser.add_argument(
        "--source-fps", type=float, help="Source video FPS for rounded sampling grids"
    )
    parser.add_argument("--frame-tolerance", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=98.0)
    parser.add_argument("--type-threshold", action="append", default=[], metavar="TYPE=PERCENT")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        interval = args.frame_interval
        if interval is None:
            if not math.isfinite(args.sample_fps) or args.sample_fps <= 0:
                raise ValueError("sample-fps must be positive and finite")
            interval = 1 / args.sample_fps
        floors = dict(item.split("=", 1) for item in args.type_threshold)
        # Allow one empty file so a missing/extra entire output scores zero.
        logs = [load_events(path, allow_empty=True) for path in (args.reference, args.candidate)]
        result = compare_events(
            *logs,
            frame_interval=interval,
            origin=args.origin,
            frame_tolerance=args.frame_tolerance,
            threshold=args.threshold,
            type_thresholds={key: float(value) for key, value in floors.items()},
            source_fps=args.source_fps,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        print(
            f"Similarity: {result['similarity_percent']:.2f}% "
            f"({result['matched']} matched, {len(result['missing'])} missing, "
            f"{len(result['extra'])} extra); threshold {args.threshold:.2f}%"
        )
        for failure in result["failures"]:
            print(f"FAIL: {failure}")
        return 0 if result["passed"] else 1
    except (ValueError, OSError, TypeError) as error:
        print(f"Invalid comparison: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
