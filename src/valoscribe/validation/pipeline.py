"""Run the real recorded video pipeline against hash-verified, trusted fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from valoscribe.validation.events import compare_events, load_events
from valoscribe.validation.outputs import validate_outputs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processing_metadata(selected):
    teams = selected["teams"]
    if len(teams) != 2 or any(len(team["players"]) != 5 for team in teams):
        raise ValueError("Scraper must return two teams of five players")
    return {
        "map": selected["map_name"],
        "teams": [{"name": team["name"], "starting_side": team["starting_side"]} for team in teams],
        "players": [
            {"name": p["name"], "team": team["name"], "agent": p["agent"]}
            for team in teams
            for p in team["players"]
        ],
    }


def run_recorded(fixtures: Path, output: Path):
    import cv2

    from valoscribe.orchestration.game_state_manager import GameStateManager
    from valoscribe.scraper.vlr_scraper import VLRScraper

    policy = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in policy["sha256"].items():
        path = (fixtures / name).resolve()
        if not path.is_relative_to(fixtures.resolve()) or sha256(path) != expected:
            raise ValueError(f"Fixture checksum mismatch or unsafe path: {name}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "event_log.jsonl").exists() or (output / "frame_states.csv").exists():
        raise ValueError("Output directory already contains results; use a fresh directory")
    series = VLRScraper().parse_match_html(
        (fixtures / "vlr.html").read_text(encoding="utf-8"), policy["match_url"]
    )
    expected_series = json.loads((fixtures / "series.json").read_text(encoding="utf-8"))
    if series != expected_series:
        raise ValueError("VLR parser output differs from reviewed series metadata")
    metadata = processing_metadata(series["maps"][policy["map_index"]])
    if metadata != json.loads((fixtures / "metadata.json").read_text(encoding="utf-8")):
        raise ValueError("Processing metadata differs from fixture")
    video = fixtures / policy["video"]
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        dimensions = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        if (
            not capture.isOpened()
            or abs(fps - policy["source_fps"]) > 1e-6
            or dimensions != policy["dimensions"]
        ):
            raise ValueError("Video dimensions/FPS differ from manifest")
    finally:
        capture.release()
    stride = int(fps / policy["processing_fps"])
    if stride < 1 or abs(stride / fps - policy["frame_interval"]) > 1e-9:
        raise ValueError("Sampling interval differs from manifest")
    manager = GameStateManager(
        video_path=video, vlr_metadata=metadata, output_dir=output, fps=policy["processing_fps"]
    )
    manager.process_video()
    structure = validate_outputs(
        output, required_types=policy["required_types"], min_frames=policy["min_frames"]
    )
    result = compare_events(
        load_events(fixtures / "baseline.jsonl"),
        load_events(output / "event_log.jsonl"),
        frame_interval=policy["frame_interval"],
        origin=policy["origin"],
        frame_tolerance=policy["frame_tolerance"],
        threshold=policy["threshold"],
        type_thresholds=policy["type_thresholds"],
    )
    result["structure"] = structure
    result["versions"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python", "pytesseract", "yt-dlp")
        },
    }
    result["fixture_id"] = policy["fixture_id"]
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = (
        f"### Recorded pipeline: {policy['fixture_id']}\n\n"
        f"Similarity **{result['similarity_percent']:.2f}%**; required {policy['threshold']}%. "
        f"Matched {result['matched']}; missing {len(result['missing'])}; "
        f"extra {len(result['extra'])}.\n\n"
        f"Frames: {structure['frames']}; structural validation passed.\n\n"
        "| Event | Reference | Candidate | Similarity |\n|---|---:|---:|---:|\n"
    )
    for kind, values in result["per_type"].items():
        summary += (
            f"| {kind} | {values['reference']} | {values['candidate']} | "
            f"{values['similarity_percent']:.2f}% |\n"
        )
    summary += "\n" + "\n".join(result["failures"]) + "\n"
    (output / "summary.md").write_text(summary, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary)
    print(summary)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/pipeline"))
    parser.add_argument("--output", type=Path, default=Path("ci-output/recorded"))
    args = parser.parse_args(argv)
    try:
        return 0 if run_recorded(args.fixtures, args.output)["passed"] else 1
    except (ValueError, OSError, KeyError) as error:
        print(f"Pipeline regression failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
