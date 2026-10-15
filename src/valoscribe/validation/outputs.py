"""Fail-closed structural validation for clip and complete-map output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from valoscribe.validation.events import load_events


def validate_outputs(directory: Path, *, full_map=False, required_types=(), min_frames=1):
    events = load_events(directory / "event_log.jsonl")
    csv_path = directory / "frame_states.csv"
    last_timestamp, last_frame, count = -1.0, 0, 0
    columns = {"timestamp", "frame_number", "phase", "round_number", "score_team1", "score_team2"}
    columns |= {
        f"player_{i}_{field}"
        for i in range(10)
        for field in ("name", "team", "agent", "alive", "health", "armor")
    }
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not columns.issubset(reader.fieldnames or ()):
            raise ValueError("Missing frame CSV columns")
        first_timestamp = None
        for row in reader:
            timestamp, frame = float(row["timestamp"]), int(row["frame_number"])
            if (
                not math.isfinite(timestamp)
                or timestamp < 0
                or timestamp <= last_timestamp
                or frame <= last_frame
            ):
                raise ValueError(f"Invalid frame sequence at CSV row {count + 2}")
            if any(int(row[key]) < 0 for key in ("score_team1", "score_team2")):
                raise ValueError("Negative score in CSV")
            if row["phase"] not in {"NON_GAME", "PREROUND", "ACTIVE_ROUND", "POST_ROUND"}:
                raise ValueError("Unknown phase in CSV")
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp, last_frame = timestamp, frame
            count += 1
    if count < min_frames:
        raise ValueError(f"Insufficient frames: {count} < {min_frames}")
    # The writer rounds CSV time to milliseconds. Events can be backdated by confirmation.
    for event in events:
        if (
            event["timestamp"] < first_timestamp - 0.001
            or event["timestamp"] > last_timestamp + 0.001
        ):
            raise ValueError("Event outside processed clip")
    types = {event["type"] for event in events}
    missing = set(required_types) - types
    if missing:
        raise ValueError(f"Missing required event coverage: {sorted(missing)}")
    # Inferred/backdated non-boundary events may arrive out of chronological order.
    for kind in ("round_start", "round_end"):
        boundaries = [e for e in events if e["type"] == kind]
        for previous, current in zip(boundaries, boundaries[1:]):
            if (
                current["timestamp"] <= previous["timestamp"]
                or current["round_number"] <= previous["round_number"]
                or any(current[key] < previous[key] for key in ("score_team1", "score_team2"))
            ):
                raise ValueError(f"Invalid {kind} progression")
    if full_map:
        if not {"match_start", "match_end", "round_start", "round_end"}.issubset(types):
            raise ValueError("Incomplete full-map lifecycle")
        starts = [e for e in events if e["type"] == "round_start"]
        ends = [e for e in events if e["type"] == "round_end"]
        if [e["round_number"] for e in starts] != [e["round_number"] for e in ends]:
            raise ValueError("Unbalanced round lifecycle")
        scores = ends[-1]["score_team1"], ends[-1]["score_team2"]
        if max(scores) < 13 or abs(scores[0] - scores[1]) < 2:
            raise ValueError("Invalid complete-map final score")
    return {"passed": True, "frames": count, "events": len(events), "event_types": sorted(types)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--full-map", action="store_true")
    parser.add_argument("--require-type", action="append", default=[])
    parser.add_argument("--min-frames", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.min_frames < 1:
            raise ValueError("min-frames must be positive")
        result = validate_outputs(
            args.directory,
            full_map=args.full_map,
            required_types=args.require_type,
            min_frames=args.min_frames,
        )
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Invalid pipeline output: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
