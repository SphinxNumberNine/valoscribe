"""Bounded, advisory live VLR -> YouTube -> video -> output smoke test."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from valoscribe.validation.outputs import validate_outputs
from valoscribe.validation.pipeline import processing_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ci-output/live"))
    parser.add_argument("--match-url", default="https://www.vlr.gg/542272?game=all&tab=overview")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args(argv)
    stage = "configuration"
    try:
        if args.duration < 110 or args.duration > 180 or args.map_index < 0:
            raise ValueError(
                "Smoke test requires a 110-180 second section and nonnegative map index"
            )
        parsed = urlparse(args.match_url)
        if parsed.scheme != "https" or parsed.hostname not in {"vlr.gg", "www.vlr.gg"}:
            raise ValueError("Only HTTPS VLR match URLs are supported")
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "processed").exists():
            raise ValueError("Use a fresh live output directory")
        import cv2

        from valoscribe.orchestration.game_state_manager import GameStateManager
        from valoscribe.scraper.vlr_scraper import VLRScraper
        from valoscribe.video.youtube import _parse_timestamp, download_youtube

        stage = "VLR scrape"
        series = VLRScraper(timeout=20).scrape_match(args.match_url)
        if not series["maps"] or args.map_index >= len(series["maps"]):
            raise ValueError("No requested map returned by VLR")
        selected = series["maps"][args.map_index]
        metadata = processing_metadata(selected)
        url = selected.get("vod_url")
        if not url or urlparse(url).hostname not in {"youtube.com", "www.youtube.com", "youtu.be"}:
            raise ValueError("No YouTube VOD found for requested map")
        (args.output / "series.json").write_text(json.dumps(series, indent=2), encoding="utf-8")
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        start = _parse_timestamp(parse_qs(urlparse(url).query).get("t", ["0"])[0])
        if start is None:
            raise ValueError("Unrecognized VOD timestamp")
        stage = "YouTube section download"
        download = download_youtube(
            url, args.output / "video", start_time=start, duration=args.duration, section_only=True
        )
        stage = "video integrity"
        video = cv2.VideoCapture(str(download.out_path))
        try:
            fps = video.get(cv2.CAP_PROP_FPS)
            dimensions = [
                int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ]
            duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
            if (
                not video.isOpened()
                or dimensions != [1920, 1080]
                or not args.duration - 2 <= duration <= args.duration + 2
            ):
                raise ValueError(
                    f"Unexpected video: dimensions={dimensions}, FPS={fps}, duration={duration}"
                )
        finally:
            video.release()
        stage = "processing"
        manager = GameStateManager(download.out_path, metadata, args.output / "processed", fps=4)
        manager.process_video()
        stage = "validation"
        report = validate_outputs(
            args.output / "processed",
            min_frames=300,
            required_types=["match_start", "round_start", "round_end", "kill"],
        )
        report.update({"stage": stage, "vod_url": url, "start": start, "duration": duration})
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = (
            f"### Live pipeline passed\n\n{report['frames']} frames; {report['events']} events.\n"
        )
        print(summary)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
                handle.write(summary)
        return 0
    except Exception as error:
        # Stage labeling is intentional; network failures must not look like OCR regressions.
        message = f"Live pipeline failed at {stage}: {type(error).__name__}: {error}"
        print(message, file=sys.stderr)
        if args.output.is_dir():
            (args.output / "failure.json").write_text(
                json.dumps({"stage": stage, "error": str(error)}), encoding="utf-8"
            )
        return 1


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    raise SystemExit(main())
