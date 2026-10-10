"""Detection commands for analyzing Valorant VODs."""

import typer
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.video.reader import VideoReader
from valoscribe.detectors.cropper import Cropper
from valoscribe.detectors.round_detector import RoundDetector
from valoscribe.detectors.template_score_detector import TemplateScoreDetector
from valoscribe.detectors.template_timer_detector import TemplateTimerDetector
from valoscribe.detectors.template_spike_detector import TemplateSpikeDetector
from valoscribe.detectors.template_credits_detector import TemplateCreditsDetector
from valoscribe.detectors.template_health_detector import TemplateHealthDetector
from valoscribe.detectors.template_armor_detector import TemplateArmorDetector
from valoscribe.detectors.template_agent_detector import TemplateAgentDetector
from valoscribe.detectors.active_round_agent_detector import ActiveRoundAgentDetector
from valoscribe.detectors.preround_credits_detector import PreroundCreditsDetector
from valoscribe.detectors.ability_detector import AbilityDetector
from valoscribe.detectors.preround_ability_detector import PreroundAbilityDetector
from valoscribe.detectors.ultimate_detector import UltimateDetector
from valoscribe.detectors.preround_ultimate_detector import PreroundUltimateDetector
from valoscribe.detectors.killfeed_detector import KillfeedDetector
from valoscribe.orchestration.phase_detector import PhaseDetector, Phase
from valoscribe.utils.ocr import OCREngine
from valoscribe.utils.logger import setup_logging

app = typer.Typer(help="Detection commands for analyzing Valorant VODs")
@app.command(name="round")
def detect_round(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    show_debug: bool = typer.Option(
        False,
        "--show-debug",
        "-d",
        help="Show debug visualization windows",
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
) -> None:
    """Detect round numbers from a Valorant VOD using OCR."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize OCR engine and round detector
        ocr_engine = OCREngine()
        round_detector = RoundDetector(cropper, ocr_engine, min_confidence=min_confidence)

        typer.echo(f"Round detector initialized (min_confidence: {min_confidence})")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")

            if fps:
                typer.echo(f"Processing at: {fps} fps")
            if start_time:
                typer.echo(f"Start time: {start_time}s")
            if end_time:
                typer.echo(f"End time: {end_time}s")
            if max_frames:
                typer.echo(f"Max frames: {max_frames}")

            typer.echo("\n" + "=" * 60)
            typer.echo("ROUND DETECTION RESULTS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detections = []

            for frame_info in reader:
                # Check max_frames limit
                if max_frames and frame_count >= max_frames:
                    typer.echo(f"\nReached max frames limit: {max_frames}")
                    break

                # Detect round number
                if show_debug:
                    result, preprocessed = round_detector.detect_with_debug(frame_info.frame)

                    # Save debug images for first 5 frames to see variations
                    if frame_count < 5:
                        debug_dir = Path("./debug_crops")
                        debug_dir.mkdir(exist_ok=True)
                        round_crop = cropper.crop_simple_region(frame_info.frame, "round_number")
                        result_text = f"round{result.round_number}" if result else "nodetection"
                        cv2.imwrite(str(debug_dir / f"frame_{frame_count}_{result_text}_original.png"), round_crop)
                        if preprocessed is not None and preprocessed.size > 0:
                            cv2.imwrite(str(debug_dir / f"frame_{frame_count}_{result_text}_preprocessed.png"), preprocessed)
                        typer.echo(f"  -> Saved debug images to {debug_dir}/")
                else:
                    result = round_detector.detect(frame_info.frame)
                    preprocessed = None

                # Display result
                if result:
                    detections.append(result)
                    typer.secho(
                        f"[Frame {frame_info.frame_number:6d} @ {frame_info.timestamp_sec:7.2f}s] "
                        f"Round {result.round_number:2d} "
                        f"(confidence: {result.confidence:.2%}) "
                        f"[{result.raw_text}]",
                        fg=typer.colors.GREEN,
                    )
                else:
                    typer.secho(
                        f"[Frame {frame_info.frame_number:6d} @ {frame_info.timestamp_sec:7.2f}s] "
                        f"No detection",
                        fg=typer.colors.YELLOW,
                    )

                # Show debug windows if requested
                if show_debug and preprocessed is not None and preprocessed.size > 0:
                    round_crop = cropper.crop_simple_region(frame_info.frame, "round_number")
                    cv2.imshow("Round Number Crop", cv2.resize(round_crop, None, fx=3, fy=3))
                    cv2.imshow("Preprocessed", cv2.resize(preprocessed, None, fx=3, fy=3))

                    # Wait for key press
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Successful detections: {len(detections)}")

            if detections:
                avg_confidence = sum(d.confidence for d in detections) / len(detections)
                typer.echo(f"Average confidence: {avg_confidence:.2%}")

                # Show unique round numbers detected
                unique_rounds = sorted(set(d.round_number for d in detections))
                typer.echo(f"Rounds detected: {', '.join(map(str, unique_rounds))}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    finally:
        if show_debug:
            cv2.destroyAllWindows()

@app.command(name="timer-template")
def detect_timer_template(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--templates",
        "-t",
        help="Path to template directory (default: src/valoscribe/templates/timer_digits/)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.6,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
) -> None:
    """Detect round timer from a Valorant VOD using template matching."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize template timer detector
        timer_detector = TemplateTimerDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
        )

        # Check that templates are loaded
        if len(timer_detector.templates) == 0:
            typer.secho(
                "Error: No templates loaded. Please ensure digit templates (0.png - 9.png) "
                f"are in {timer_detector.template_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Min confidence: {min_confidence}")

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING TIMER")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detection_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Detect timer
                timer_info = timer_detector.detect(frame_info.frame)

                if timer_info:
                    detection_count += 1
                    typer.secho(
                        f"[{frame_info.timestamp_sec:7.2f}s] "
                        f"Timer: {timer_info.time_seconds:6.2f}s "
                        f"({timer_info.raw_text}) "
                        f"[conf: {timer_info.confidence:.2f}]",
                        fg=typer.colors.GREEN,
                    )
                else:
                    typer.secho(
                        f"[{frame_info.timestamp_sec:7.2f}s] Timer: NOT DETECTED",
                        fg=typer.colors.YELLOW,
                    )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Detections: {detection_count}")
            typer.echo(f"Detection rate: {detection_count / frame_count * 100:.1f}%")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="spike-template")
def detect_spike_template(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_path: Optional[Path] = typer.Option(
        None,
        "--template",
        "-t",
        help="Path to spike template (default: src/valoscribe/templates/spike/spike.png)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
) -> None:
    """Detect spike icon from a Valorant VOD using template matching."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize template spike detector
        spike_detector = TemplateSpikeDetector(
            cropper,
            template_path=template_path,
            min_confidence=min_confidence,
        )

        # Check that template is loaded
        if spike_detector.template is None:
            typer.secho(
                "Error: No template loaded. Please ensure spike template (spike.png) "
                f"is at {spike_detector.template_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Min confidence: {min_confidence}")

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING SPIKE")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detection_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Detect spike
                spike_info = spike_detector.detect(frame_info.frame)

                if spike_info and spike_info.spike_planted:
                    detection_count += 1
                    typer.secho(
                        f"[{frame_info.timestamp_sec:7.2f}s] "
                        f"Spike: PLANTED "
                        f"[conf: {spike_info.confidence:.2f}]",
                        fg=typer.colors.GREEN,
                    )
                else:
                    conf = spike_info.confidence if spike_info else 0.0
                    typer.secho(
                        f"[{frame_info.timestamp_sec:7.2f}s] Spike: NOT DETECTED [conf: {conf:.2f}]",
                        fg=typer.colors.YELLOW,
                    )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Spike detections: {detection_count}")
            typer.echo(f"Detection rate: {detection_count / frame_count * 100:.1f}%")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="credits-template")
def detect_credits_template(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_path: Optional[Path] = typer.Option(
        None,
        "--template",
        "-t",
        help="Path to dead credits template (default: src/valoscribe/templates/credits/credits_icon_dead.png)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to check (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect dead credits icon from a Valorant VOD using template matching (high confidence = dead, low = alive)."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize template credits detector
        credits_detector = TemplateCreditsDetector(
            cropper,
            template_path=template_path,
            min_confidence=min_confidence,
        )

        # Check that template is loaded
        if credits_detector.template is None:
            typer.secho(
                "Error: No template loaded. Please ensure dead credits template (credits_icon_dead.png) "
                f"is at {credits_detector.template_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Determine which players to check
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Checking credits for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Checking credits for player {player_index}")

        if show:
            typer.echo("Display mode: ON")
            if step:
                typer.echo("Step mode: Press any key to advance, 'q' to quit")
            else:
                typer.echo("Press 'q' to quit")

        if step and not show:
            typer.secho(
                "Warning: --step requires --show to be enabled. Ignoring --step.",
                fg=typer.colors.YELLOW,
            )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Template shape: {credits_detector.template.shape}")
            typer.echo(f"Min confidence: {min_confidence}")

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING DEAD CREDITS ICON (HIGH CONF = DEAD, LOW CONF = ALIVE)")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detection_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Collect results for display
                player_results = {}

                # Check credits for each player
                for p_idx in player_indices:
                    # Determine side (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Detect credits
                    credits_info = credits_detector.detect(frame_info.frame, p_idx, side)

                    if credits_info:
                        player_results[p_idx] = (side, credits_info)

                        # credits_visible=True means dead credits icon is visible = player is DEAD
                        # credits_visible=False means dead credits icon not visible = player is ALIVE
                        if credits_info.credits_visible:
                            detection_count += 1
                            typer.secho(
                                f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                                f"DEAD (confidence: {credits_info.confidence:.2f})",
                                fg=typer.colors.RED,
                            )
                        else:
                            typer.secho(
                                f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                                f"ALIVE (confidence: {credits_info.confidence:.2f})",
                                fg=typer.colors.GREEN,
                            )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                    )

                    # Add player status
                    y_offset = 80
                    for p_idx in sorted(player_results.keys()):
                        side, credits_info = player_results[p_idx]

                        # credits_visible=True means dead credits icon visible = player is DEAD
                        # credits_visible=False means dead credits icon not visible = player is ALIVE
                        if credits_info.credits_visible:
                            status_text = f"P{p_idx} ({side}): DEAD (conf: {credits_info.confidence:.2f})"
                            color = (0, 0, 255)  # Red
                        else:
                            status_text = f"P{p_idx} ({side}): ALIVE (conf: {credits_info.confidence:.2f})"
                            color = (0, 255, 0)  # Green

                        cv2.putText(
                            display_frame,
                            status_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
                        y_offset += 35

                    # Show frame
                    cv2.imshow("Credits Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            total_checks = frame_count * len(player_indices)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Players checked per frame: {len(player_indices)}")
            typer.echo(f"Total checks: {total_checks}")
            typer.echo(f"Alive detections: {detection_count}")
            if total_checks > 0:
                typer.echo(f"Alive rate: {detection_count / total_checks * 100:.1f}%")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="health-template")
def detect_health_template(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--template-dir",
        "-t",
        help="Directory containing digit templates (default: src/valoscribe/templates/health_digits/)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to check (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect player health from a Valorant VOD using template matching."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize template health detector
        health_detector = TemplateHealthDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
        )

        # Check that templates are loaded
        if len(health_detector.templates) == 0:
            typer.secho(
                "Error: No templates loaded. Please ensure digit templates (0.png - 9.png) "
                f"are at {health_detector.template_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Determine which players to check
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Checking health for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Checking health for player {player_index}")

        if show:
            typer.echo("Display mode: ON")
            if step:
                typer.echo("Step mode: Press any key to advance, 'q' to quit")
            else:
                typer.echo("Press 'q' to quit")

        if step and not show:
            typer.secho(
                "Warning: --step requires --show to be enabled. Ignoring --step.",
                fg=typer.colors.YELLOW,
            )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Templates loaded: {len(health_detector.templates)}")
            typer.echo(f"Min confidence: {min_confidence}")

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING PLAYER HEALTH")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detection_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Collect results for display
                player_results = {}

                # Check health for each player
                for p_idx in player_indices:
                    # Determine side (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Detect health
                    health_info = health_detector.detect(frame_info.frame, p_idx, side)

                    if health_info:
                        player_results[p_idx] = (side, health_info)
                        detection_count += 1

                        # Color code based on health value
                        if health_info.health >= 75:
                            color = typer.colors.GREEN
                        elif health_info.health >= 40:
                            color = typer.colors.YELLOW
                        else:
                            color = typer.colors.RED

                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                            f"Health={health_info.health} (conf: {health_info.confidence:.2f})",
                            fg=color,
                        )
                    else:
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                            f"NO DETECTION",
                            fg=typer.colors.YELLOW,
                        )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )

                    # Display results
                    y_offset = 80
                    for p_idx, (side, health_info) in player_results.items():
                        # Color based on health
                        if health_info.health >= 75:
                            color = (0, 255, 0)  # Green
                        elif health_info.health >= 40:
                            color = (0, 255, 255)  # Yellow
                        else:
                            color = (0, 0, 255)  # Red

                        text = f"P{p_idx} ({side}): {health_info.health} HP ({health_info.confidence:.2f})"
                        cv2.putText(
                            display_frame,
                            text,
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
                        y_offset += 30

                    # Show frame
                    cv2.imshow("Health Detection", display_frame)

                    # Handle key press
                    if step:
                        key = cv2.waitKey(0)
                    else:
                        key = cv2.waitKey(1)

                    if key & 0xFF == ord("q"):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            total_checks = frame_count * len(player_indices)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Players checked per frame: {len(player_indices)}")
            typer.echo(f"Total checks: {total_checks}")
            typer.echo(f"Successful detections: {detection_count}")
            if total_checks > 0:
                typer.echo(f"Detection rate: {detection_count / total_checks * 100:.1f}%")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="armor-template")
def detect_armor_template(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--template-dir",
        "-t",
        help="Directory containing digit templates (default: src/valoscribe/templates/armor/)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to check (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
    overlap_tolerance: int = typer.Option(
        2,
        "--overlap-tolerance",
        help="Allow N pixels of overlap between digit matches (default: 2)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Show debug information including all template matches",
    ),
    debug_crops: bool = typer.Option(
        False,
        "--debug-crops",
        help="Save debug crops to output directory",
    ),
    debug_output_dir: Path = typer.Option(
        Path("./debug_armor"),
        "--debug-output",
        help="Output directory for debug crops (requires --debug-crops)",
    ),
) -> None:
    """Detect player armor from a Valorant VOD using template matching."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize template armor detector
        armor_detector = TemplateArmorDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
            overlap_tolerance_px=overlap_tolerance,
        )

        if len(armor_detector.templates) == 0:
            typer.secho(
                f"Error: No templates found in {armor_detector.template_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo(f"Loaded {len(armor_detector.templates)} digit templates")
        typer.echo(f"Minimum confidence: {min_confidence}")

        # Setup debug crop output
        if debug_crops:
            debug_output_dir.mkdir(parents=True, exist_ok=True)
            typer.echo(f"Debug crops will be saved to: {debug_output_dir}")
            # Clear directory
            import shutil
            if debug_output_dir.exists():
                for file in debug_output_dir.glob("*.png"):
                    file.unlink()
        typer.echo(f"Overlap tolerance: {overlap_tolerance}px")

        # Determine which players to check
        if player_index == -1:
            player_indices = list(range(10))
            typer.echo("Checking armor for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Checking armor for player {player_index}")

        # Validate step mode
        if step and not show:
            typer.secho(
                "Warning: --step requires --show. Ignoring --step.",
                fg=typer.colors.YELLOW,
            )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")

            typer.echo("\n" + "=" * 60)
            typer.echo("ARMOR DETECTION")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detection_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Format timestamp
                time = f"{int(frame_info.timestamp_sec // 60):02d}:{int(frame_info.timestamp_sec % 60):02d}"

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Detect armor for each player
                for p_idx in player_indices:
                    # Determine side (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Use debug detection if debug mode is enabled
                    if debug or debug_crops:
                        armor_info, armor_preprocessed, debug_info = armor_detector.detect_with_debug(
                            frame_info.frame, p_idx, side
                        )

                        # Save debug crops if requested
                        if debug_crops and armor_preprocessed.size > 0:
                            filename = f"p{p_idx}_{side}_f{frame_count:04d}_armor_preprocessed.png"
                            cv2.imwrite(str(debug_output_dir / filename), armor_preprocessed)

                        # Show debug info
                        if debug:
                            all_matches = debug_info.get("all_matches", [])
                            filtered_matches = debug_info.get("filtered_matches", [])

                            typer.echo(
                                f"[{time}] Player {p_idx} ({side}) Debug: "
                                f"{len(all_matches)} total matches, "
                                f"{len(filtered_matches)} after filtering"
                            )

                            if len(all_matches) > 0:
                                # Show top 5 matches
                                sorted_matches = sorted(all_matches, key=lambda m: m["confidence"], reverse=True)
                                typer.echo(f"  Top matches:")
                                for i, match in enumerate(sorted_matches[:5]):
                                    typer.echo(
                                        f"    {i+1}. Digit '{match['digit']}' at x={match['x']} "
                                        f"conf={match['confidence']:.3f}"
                                    )

                                if len(filtered_matches) > 0:
                                    typer.echo(f"  After overlap filtering:")
                                    for match in filtered_matches:
                                        typer.echo(
                                            f"    Digit '{match['digit']}' at x={match['x']} "
                                            f"conf={match['confidence']:.3f}"
                                        )
                    else:
                        # Normal detection
                        armor_info = armor_detector.detect(frame_info.frame, p_idx, side)

                    if armor_info:
                        detection_count += 1

                        # Determine color based on armor value
                        if armor_info.armor >= 40:
                            color = typer.colors.GREEN  # Heavy armor (50)
                        elif armor_info.armor >= 20:
                            color = typer.colors.CYAN  # Light armor (25)
                        elif armor_info.armor > 0:
                            color = typer.colors.YELLOW  # Damaged
                        else:
                            color = typer.colors.RED  # No armor

                        typer.secho(
                            f"[{time}] Player {p_idx} ({side}): Armor = {armor_info.armor} "
                            f"(conf: {armor_info.confidence:.2f}, raw: '{armor_info.raw_text}')",
                            fg=color,
                        )

                        # Add to display if showing
                        if show:
                            text = f"P{p_idx}: {armor_info.armor}"
                            y_pos = 30 + (p_idx * 25)
                            cv2.putText(
                                display_frame,
                                text,
                                (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0) if armor_info.armor >= 40 else (0, 255, 255),
                                2,
                            )
                    else:
                        if not debug:
                            typer.secho(
                                f"[{time}] Player {p_idx} ({side}): No armor detected",
                                fg=typer.colors.WHITE,
                                dim=True,
                            )

                # Show display frame
                if show:
                    cv2.imshow("Armor Detection", display_frame)

                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            total_checks = frame_count * len(player_indices)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Players checked per frame: {len(player_indices)}")
            typer.echo(f"Total checks: {total_checks}")
            typer.echo(f"Successful detections: {detection_count}")
            if total_checks > 0:
                typer.echo(f"Detection rate: {detection_count / total_checks * 100:.1f}%")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="preround")
def detect_preround(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_path: Optional[Path] = typer.Option(
        None,
        "--template",
        "-t",
        help="Path to pre-round credits template (default: src/valoscribe/templates/credits/credits_icon_preround.png)",
    ),
    player_index: int = typer.Option(
        -1,
        "--player",
        "-p",
        help="Player index to check (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
    preround_threshold: float = typer.Option(
        0.5,
        "--preround-threshold",
        help="Fraction of players with visible credits to consider frame as pre-round (0-1, default 0.5)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect pre-round frames from a Valorant VOD using pre-round credits icon detection."""
    from valoscribe.detectors.preround_credits_detector import PreroundCreditsDetector

    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize pre-round credits detector
        preround_detector = PreroundCreditsDetector(
            cropper,
            template_path=template_path,
            min_confidence=min_confidence,
        )

        # Check that template is loaded
        if preround_detector.template is None:
            typer.secho(
                "Error: No template loaded. Please ensure pre-round credits template (credits_icon_preround.png) "
                f"is at {preround_detector.template_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Determine which players to check
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Checking pre-round credits for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Checking pre-round credits for player {player_index}")

        if show:
            typer.echo("Display mode: ON")
            if step:
                typer.echo("Step mode: Press any key to advance, 'q' to quit")
            else:
                typer.echo("Press 'q' to quit")

        if step and not show:
            typer.secho(
                "Warning: --step requires --show to be enabled. Ignoring --step.",
                fg=typer.colors.YELLOW,
            )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Template shape: {preround_detector.template.shape}")
            typer.echo(f"Min confidence: {min_confidence}")
            typer.echo(f"Pre-round threshold: {preround_threshold} ({preround_threshold * 100:.0f}% of players)")

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING PRE-ROUND FRAMES")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            preround_frame_count = 0
            detection_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Check if frame is pre-round
                is_preround = preround_detector.is_preround_frame(
                    frame_info.frame,
                    threshold=preround_threshold
                )

                if is_preround:
                    preround_frame_count += 1

                # Collect results for display
                player_results = {}

                # Check credits for each player
                for p_idx in player_indices:
                    # Determine side (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Detect credits
                    credits_info = preround_detector.detect(frame_info.frame, p_idx, side)

                    if credits_info:
                        player_results[p_idx] = (side, credits_info)

                        if credits_info.credits_visible:
                            detection_count += 1

                # Print frame status
                frame_status = "PRE-ROUND" if is_preround else "IN-ROUND"
                frame_color = typer.colors.CYAN if is_preround else typer.colors.YELLOW

                typer.secho(
                    f"[{frame_info.timestamp_sec:7.2f}s] Frame status: {frame_status} "
                    f"({sum(1 for _, info in player_results.values() if info.credits_visible)}/{len(player_results)} visible)",
                    fg=frame_color,
                )

                # Print individual player results if checking specific players
                if len(player_indices) <= 10:
                    for p_idx in player_indices:
                        if p_idx in player_results:
                            side, credits_info = player_results[p_idx]
                            if credits_info.credits_visible:
                                typer.echo(
                                    f"  Player {p_idx} ({side}): VISIBLE (conf: {credits_info.confidence:.2f})"
                                )
                            else:
                                typer.echo(
                                    f"  Player {p_idx} ({side}): NOT VISIBLE (conf: {credits_info.confidence:.2f})"
                                )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                    )

                    # Add pre-round status
                    status_color = (0, 255, 255) if is_preround else (0, 165, 255)  # Cyan or Orange
                    cv2.putText(
                        display_frame,
                        f"STATUS: {frame_status}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        status_color,
                        3,
                    )

                    # Add player status
                    y_offset = 140
                    for p_idx in sorted(player_results.keys()):
                        side, credits_info = player_results[p_idx]

                        if credits_info.credits_visible:
                            status_text = f"P{p_idx} ({side}): VISIBLE (conf: {credits_info.confidence:.2f})"
                            color = (0, 255, 0)  # Green
                        else:
                            status_text = f"P{p_idx} ({side}): NOT VISIBLE (conf: {credits_info.confidence:.2f})"
                            color = (128, 128, 128)  # Gray

                        cv2.putText(
                            display_frame,
                            status_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
                        y_offset += 35

                    # Show frame
                    cv2.imshow("Pre-Round Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            total_checks = frame_count * len(player_indices)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Pre-round frames detected: {preround_frame_count} ({preround_frame_count / frame_count * 100:.1f}%)" if frame_count > 0 else "Pre-round frames detected: 0")
            typer.echo(f"Players checked per frame: {len(player_indices)}")
            typer.echo(f"Total checks: {total_checks}")
            typer.echo(f"Visible credits detections: {detection_count}")
            if total_checks > 0:
                typer.echo(f"Visible rate: {detection_count / total_checks * 100:.1f}%")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="agents-preround")
def detect_agents_preround(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--template-dir",
        "-t",
        help="Directory containing agent templates (default: src/valoscribe/templates/preround_agents/)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to check (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Show detailed match scores for all templates",
    ),
) -> None:
    """Detect agent selection from preround phase using template matching."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize template agent detector
        agent_detector = TemplateAgentDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
        )

        # Check that templates are loaded
        if len(agent_detector.templates) == 0:
            typer.secho(
                "Error: No templates loaded. Please ensure agent templates are at "
                f"{agent_detector.template_dir}/attack/ and {agent_detector.template_dir}/defense/",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Determine which players to check
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Checking agents for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Checking agent for player {player_index}")

        if show:
            typer.echo("Display mode: ON")
            if step:
                typer.echo("Step mode: Press any key to advance, 'q' to quit")
            else:
                typer.echo("Press 'q' to quit")

        if step and not show:
            typer.secho(
                "Warning: --step requires --show to be enabled. Ignoring --step.",
                fg=typer.colors.YELLOW,
            )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Templates loaded: {len(agent_detector.templates)}")
            typer.echo(f"Min confidence: {min_confidence}")

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING PREROUND AGENTS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detection_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Collect results for display
                player_results = {}

                # Check agent for each player
                for p_idx in player_indices:
                    # Detect agent (use debug mode if requested)
                    if debug:
                        agent_info, agent_crop, debug_info = agent_detector.detect_with_debug(
                            frame_info.frame, p_idx
                        )
                    else:
                        agent_info = agent_detector.detect(frame_info.frame, p_idx)
                        debug_info = None

                    if agent_info:
                        player_results[p_idx] = agent_info
                        detection_count += 1

                        # Color code based on side
                        if agent_info.side == "attack":
                            color = typer.colors.RED
                        else:  # defense
                            color = typer.colors.GREEN

                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx}: "
                            f"{agent_info.agent_name.upper()} ({agent_info.side}) "
                            f"[conf: {agent_info.confidence:.2f}]",
                            fg=color,
                        )
                    else:
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx}: "
                            f"NO DETECTION",
                            fg=typer.colors.YELLOW,
                        )

                    # Print debug info if requested
                    if debug:
                        if not debug_info or not debug_info.get("match_scores"):
                            typer.secho(
                                f"\n  Debug - Player {p_idx}: No template matching performed",
                                fg=typer.colors.MAGENTA
                            )

                            # Show diagnostic info
                            if debug_info:
                                if debug_info.get("error"):
                                    typer.secho(f"    Error: {debug_info['error']}", fg=typer.colors.RED)
                                if "num_preround_crops" in debug_info:
                                    typer.echo(f"    Preround crops returned: {debug_info['num_preround_crops']}")
                                if "crop_keys" in debug_info:
                                    typer.echo(f"    Crop keys available: {debug_info['crop_keys']}")
                                if "agent_crop_shape" in debug_info:
                                    typer.echo(f"    Agent crop shape: {debug_info['agent_crop_shape']}")
                        else:
                            typer.echo(f"\n  Debug - Template match scores for Player {p_idx}:")

                            # Get match scores and sort by confidence
                            match_scores = debug_info.get("match_scores", {})
                            sorted_matches = sorted(
                                match_scores.items(),
                                key=lambda x: x[1].get("confidence", 0.0),
                                reverse=True
                            )

                            typer.echo(f"  Total templates: {len(match_scores)}")

                            # Show top 10 matches
                            for i, (template_key, score_data) in enumerate(sorted_matches[:10]):
                                agent = score_data.get("agent", "unknown")
                                side = score_data.get("side", "unknown")
                                conf = score_data.get("confidence", 0.0)

                                if "skipped" in score_data:
                                    typer.secho(
                                        f"    {i+1:2d}. {agent:12s} ({side:7s}): SKIPPED - {score_data['skipped']}",
                                        fg=typer.colors.CYAN
                                    )
                                else:
                                    loc = score_data.get("location", (0, 0))
                                    # Color code by confidence
                                    if conf >= 0.7:
                                        color = typer.colors.GREEN
                                    elif conf >= 0.5:
                                        color = typer.colors.YELLOW
                                    else:
                                        color = typer.colors.WHITE

                                    typer.secho(
                                        f"    {i+1:2d}. {agent:12s} ({side:7s}): {conf:.4f} at {loc}",
                                        fg=color
                                    )

                        typer.echo("")  # Blank line after debug output

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )

                    # Display results
                    y_offset = 80
                    for p_idx, agent_info in player_results.items():
                        # Color based on side
                        if agent_info.side == "attack":
                            color = (0, 0, 255)  # Red
                        else:  # defense
                            color = (0, 255, 0)  # Green

                        text = f"P{p_idx}: {agent_info.agent_name} ({agent_info.side}) [{agent_info.confidence:.2f}]"
                        cv2.putText(
                            display_frame,
                            text,
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
                        y_offset += 30

                    # Show frame
                    cv2.imshow("Agent Detection", display_frame)

                    # Handle key press
                    if step:
                        key = cv2.waitKey(0)
                    else:
                        key = cv2.waitKey(1)

                    if key & 0xFF == ord("q"):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            total_checks = frame_count * len(player_indices)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Players checked per frame: {len(player_indices)}")
            typer.echo(f"Total checks: {total_checks}")
            typer.echo(f"Successful detections: {detection_count}")
            if total_checks > 0:
                typer.echo(f"Detection rate: {detection_count / total_checks * 100:.1f}%")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="abilities-preround")
def detect_abilities_preround(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to detect (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    brightness_threshold: int = typer.Option(
        127,
        "--brightness",
        help="Brightness threshold for blob detection (0-255)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect pre-round ability charges from a Valorant VOD using blob detection."""
    from valoscribe.detectors.preround_ability_detector import PreroundAbilityDetector

    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize pre-round ability detector
        ability_detector = PreroundAbilityDetector(
            cropper,
            brightness_threshold=brightness_threshold,
        )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Brightness threshold: {brightness_threshold}")

            if player_index == -1:
                typer.echo("Detecting pre-round abilities for all players")
            else:
                typer.echo(f"Detecting pre-round abilities for player {player_index}")

            if show:
                typer.echo("Display mode: ON")
                if step:
                    typer.echo("Step mode: Press any key to advance, 'q' to quit")
                else:
                    typer.echo("Press 'q' to quit")

            if step and not show:
                typer.secho(
                    "Warning: --step requires --show to be enabled. Ignoring --step.",
                    fg=typer.colors.YELLOW,
                )

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING PRE-ROUND ABILITIES")
            typer.echo("=" * 60 + "\n")

            frame_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Determine which players to detect
                if player_index == -1:
                    player_indices = range(10)  # All players
                else:
                    player_indices = [player_index]

                # Collect all detections for display
                all_detections = {}

                # Detect abilities for each player
                for p_idx in player_indices:
                    # Determine side for display (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Pass full player index (0-9) to detector
                    abilities = ability_detector.detect_player_abilities(
                        frame_info.frame, p_idx, side
                    )

                    # Store detections
                    all_detections[p_idx] = (side, abilities)

                    # Display results
                    if all(info is None for info in abilities.values()):
                        continue

                    ability_strs = []
                    for ability_name, ability_info in abilities.items():
                        if ability_info:
                            ability_strs.append(
                                f"{ability_name}: {ability_info.charges} charges"
                            )

                    if ability_strs:
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                            + " | ".join(ability_strs),
                            fg=typer.colors.GREEN,
                        )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                    )

                    # Add ability detections
                    y_offset = 80
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    for p_idx in sorted(all_detections.keys()):
                        side, abilities = all_detections[p_idx]

                        # Skip if no abilities detected
                        if all(info is None for info in abilities.values()):
                            continue

                        # Build ability data for multi-color rendering
                        ability_data = []
                        for ability_name, ability_info in abilities.items():
                            if ability_info:
                                ability_data.append((ability_name[-1], ability_info.charges))

                        if not ability_data:
                            continue

                        # Draw player prefix in green
                        prefix = f"P{p_idx} ({side}): "
                        x_offset = 10
                        cv2.putText(
                            display_frame,
                            prefix,
                            (x_offset, y_offset),
                            font,
                            font_scale,
                            (0, 255, 0),
                            thickness,
                        )

                        # Calculate width of prefix to offset ability numbers
                        (prefix_width, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)
                        x_offset += prefix_width

                        # Draw each ability number in red
                        for i, (ability_num, charges) in enumerate(ability_data):
                            ability_text = f"{ability_num}:{charges}"
                            cv2.putText(
                                display_frame,
                                ability_text,
                                (x_offset, y_offset),
                                font,
                                font_scale,
                                (0, 0, 255),  # Red for ability counts
                                thickness,
                            )

                            # Add separator if not last ability
                            if i < len(ability_data) - 1:
                                (text_width, _), _ = cv2.getTextSize(
                                    ability_text, font, font_scale, thickness
                                )
                                x_offset += text_width
                                separator = " | "
                                cv2.putText(
                                    display_frame,
                                    separator,
                                    (x_offset, y_offset),
                                    font,
                                    font_scale,
                                    (0, 255, 0),
                                    thickness,
                                )
                                (sep_width, _), _ = cv2.getTextSize(
                                    separator, font, font_scale, thickness
                                )
                                x_offset += sep_width
                            else:
                                (text_width, _), _ = cv2.getTextSize(
                                    ability_text, font, font_scale, thickness
                                )
                                x_offset += text_width

                        y_offset += 35

                    # Show frame
                    cv2.imshow("Pre-Round Ability Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="ultimates-preround")
def detect_ultimates_preround(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to detect (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    brightness_threshold: int = typer.Option(
        127,
        "--brightness",
        help="Brightness threshold for segment detection (0-255)",
    ),
    fullness_threshold: float = typer.Option(
        0.4,
        "--fullness",
        help="White pixel ratio threshold for full ultimate detection (0-1)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect pre-round ultimate charges from a Valorant VOD using blob detection."""
    from valoscribe.detectors.preround_ultimate_detector import PreroundUltimateDetector

    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize pre-round ultimate detector
        ultimate_detector = PreroundUltimateDetector(
            cropper,
            brightness_threshold=brightness_threshold,
            fullness_threshold=fullness_threshold,
        )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Brightness threshold: {brightness_threshold}")
            typer.echo(f"Fullness threshold: {fullness_threshold}")

            if player_index == -1:
                typer.echo("Detecting pre-round ultimates for all players")
            else:
                typer.echo(f"Detecting pre-round ultimate for player {player_index}")

            if show:
                typer.echo("Display mode: ON")
                if step:
                    typer.echo("Step mode: Press any key to advance, 'q' to quit")
                else:
                    typer.echo("Press 'q' to quit")

            if step and not show:
                typer.secho(
                    "Warning: --step requires --show to be enabled. Ignoring --step.",
                    fg=typer.colors.YELLOW,
                )

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING PRE-ROUND ULTIMATES")
            typer.echo("=" * 60 + "\n")

            frame_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Determine which players to detect
                if player_index == -1:
                    player_indices = range(10)  # All players
                else:
                    player_indices = [player_index]

                # Collect all detections for display
                all_detections = {}

                # Detect ultimates for each player
                for p_idx in player_indices:
                    # Determine side (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Detect ultimate
                    result = ultimate_detector.detect_ultimate(
                        frame_info.frame, p_idx, side
                    )

                    if result:
                        ultimate_info, white_pixel_ratio = result

                        # Store detections
                        all_detections[p_idx] = (side, ultimate_info, white_pixel_ratio)

                        # Display results
                        status = "FULL" if ultimate_info.is_full else f"{ultimate_info.charges} charges"
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                            f"{status} (white ratio: {white_pixel_ratio:.2f})",
                            fg=typer.colors.GREEN if ultimate_info.is_full else typer.colors.CYAN,
                        )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                    )

                    # Add ultimate detections
                    y_offset = 80
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    for p_idx in sorted(all_detections.keys()):
                        side, ultimate_info, white_pixel_ratio = all_detections[p_idx]

                        # Draw player prefix in green
                        prefix = f"P{p_idx} ({side}): "
                        x_offset = 10
                        cv2.putText(
                            display_frame,
                            prefix,
                            (x_offset, y_offset),
                            font,
                            font_scale,
                            (0, 255, 0),
                            thickness,
                        )

                        # Calculate width of prefix
                        (prefix_width, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)
                        x_offset += prefix_width

                        # Draw ultimate status in yellow (full) or blue (partial)
                        if ultimate_info.is_full:
                            ult_text = f"FULL ({ultimate_info.charges})"
                            color = (0, 255, 255)  # Yellow
                        else:
                            ult_text = f"{ultimate_info.charges} charges"
                            color = (255, 0, 0)  # Blue

                        cv2.putText(
                            display_frame,
                            ult_text,
                            (x_offset, y_offset),
                            font,
                            font_scale,
                            color,
                            thickness,
                        )

                        y_offset += 35

                    # Show frame
                    cv2.imshow("Pre-Round Ultimate Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="game-state")
def detect_game_state(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    timer_template_dir: Optional[Path] = typer.Option(
        None,
        "--timer-templates",
        help="Path to timer template directory (default: src/valoscribe/templates/timer_digits/)",
    ),
    spike_template_path: Optional[Path] = typer.Option(
        None,
        "--spike-template",
        help="Path to spike template (default: src/valoscribe/templates/spike/spike.png)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    timer_min_confidence: float = typer.Option(
        0.6,
        "--timer-confidence",
        help="Minimum confidence for timer detection (0-1)",
    ),
    spike_min_confidence: float = typer.Option(
        0.7,
        "--spike-confidence",
        help="Minimum confidence for spike detection (0-1)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Show video frames with detection results",
    ),
) -> None:
    """Detect game state by combining timer and spike detection to identify readable game frames."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize timer detector
        timer_detector = TemplateTimerDetector(
            cropper,
            template_dir=timer_template_dir,
            min_confidence=timer_min_confidence,
        )

        # Initialize spike detector
        spike_detector = TemplateSpikeDetector(
            cropper,
            template_path=spike_template_path,
            min_confidence=spike_min_confidence,
        )

        # Check that templates are loaded
        if len(timer_detector.templates) == 0:
            typer.secho(
                "Error: No timer templates loaded. Please ensure digit templates (0.png - 9.png) "
                f"are in {timer_detector.template_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        if spike_detector.template is None:
            typer.secho(
                "Error: No spike template loaded. Please ensure spike template (spike.png) "
                f"is at {spike_detector.template_path}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Timer confidence: {timer_min_confidence}")
            typer.echo(f"Spike confidence: {spike_min_confidence}")

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING GAME STATE")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            timer_count = 0
            spike_count = 0
            non_game_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Try timer detection first
                timer_info = timer_detector.detect(frame_info.frame)

                # Determine state and create display text
                state_text = ""
                state_color = (255, 255, 255)  # White default

                if timer_info:
                    # Readable game frame - timer visible
                    timer_count += 1
                    state_text = f"Timer: {timer_info.time_seconds:.2f}s ({timer_info.raw_text}) [conf: {timer_info.confidence:.2f}]"
                    state_color = (0, 255, 0)  # Green
                    typer.secho(
                        f"[{frame_info.timestamp_sec:7.2f}s] {state_text}",
                        fg=typer.colors.GREEN,
                    )
                else:
                    # Timer not visible - check for spike
                    spike_info = spike_detector.detect(frame_info.frame)

                    if spike_info and spike_info.spike_planted:
                        # Readable game frame - spike planted
                        spike_count += 1
                        state_text = f"Spike: PLANTED [conf: {spike_info.confidence:.2f}]"
                        state_color = (255, 255, 0)  # Cyan
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] {state_text}",
                            fg=typer.colors.CYAN,
                        )
                    else:
                        # Non-game frame
                        non_game_count += 1
                        state_text = "Non-game frame (buy phase / between rounds)"
                        state_color = (0, 255, 255)  # Yellow
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] {state_text}",
                            fg=typer.colors.YELLOW,
                        )

                # Show frame if requested
                if show:
                    display_frame = frame_info.frame.copy()

                    # Add timestamp and state text overlay
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        state_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        state_color,
                        2,
                    )

                    cv2.imshow("Game State Detection", display_frame)

                    # Wait 1ms and check for 'q' key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        typer.echo("\nUser quit (pressed 'q')")
                        break

                frame_count += 1

            # Clean up display window
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Timer frames: {timer_count} ({timer_count / frame_count * 100:.1f}%)")
            typer.echo(f"Spike frames: {spike_count} ({spike_count / frame_count * 100:.1f}%)")
            typer.echo(f"Non-game frames: {non_game_count} ({non_game_count / frame_count * 100:.1f}%)")
            typer.echo(f"Total game frames: {timer_count + spike_count} ({(timer_count + spike_count) / frame_count * 100:.1f}%)")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="score-template")
def detect_score_template(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--templates",
        "-t",
        help="Path to template directory (default: src/valoscribe/templates/score_digits/)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
) -> None:
    """Detect team scores from a Valorant VOD using template matching."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize template score detector
        score_detector = TemplateScoreDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
        )

        typer.echo(f"Template score detector initialized (min_confidence: {min_confidence})")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")

            if fps:
                typer.echo(f"Processing at: {fps} fps")
            if start_time:
                typer.echo(f"Start time: {start_time}s")
            if end_time:
                typer.echo(f"End time: {end_time}s")
            if max_frames:
                typer.echo(f"Max frames: {max_frames}")

            typer.echo("\n" + "=" * 60)
            typer.echo("TEMPLATE SCORE DETECTION RESULTS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            detections = []

            for frame_info in reader:
                # Check max_frames limit
                if max_frames and frame_count >= max_frames:
                    typer.echo(f"\nReached max frames limit: {max_frames}")
                    break

                # Detect scores
                result = score_detector.detect(frame_info.frame)

                # Display result
                if result:
                    detections.append(result)
                    typer.secho(
                        f"[Frame {frame_info.frame_number:6d} @ {frame_info.timestamp_sec:7.2f}s] "
                        f"Score {result.team1_score:2d} - {result.team2_score:2d} "
                        f"(confidence: {result.confidence:.2%}) "
                        f"[{result.team1_raw_text} - {result.team2_raw_text}]",
                        fg=typer.colors.GREEN,
                    )
                else:
                    typer.secho(
                        f"[Frame {frame_info.frame_number:6d} @ {frame_info.timestamp_sec:7.2f}s] "
                        f"No detection",
                        fg=typer.colors.YELLOW,
                    )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Successful detections: {len(detections)}")

            if detections:
                avg_confidence = sum(d.confidence for d in detections) / len(detections)
                typer.echo(f"Average confidence: {avg_confidence:.2%}")

                # Show unique scores detected
                unique_scores = sorted(set((d.team1_score, d.team2_score) for d in detections))
                typer.echo(f"Unique scores detected:")
                for team1, team2 in unique_scores:
                    typer.echo(f"  {team1} - {team2}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="abilities")
def detect_abilities(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to detect (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    brightness_threshold: int = typer.Option(
        127,
        "--brightness",
        help="Brightness threshold for blob detection (0-255)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect ability charges from a Valorant VOD using blob detection."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize ability detector
        ability_detector = AbilityDetector(
            cropper,
            brightness_threshold=brightness_threshold,
        )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Brightness threshold: {brightness_threshold}")

            if player_index == -1:
                typer.echo("Detecting abilities for all players")
            else:
                typer.echo(f"Detecting abilities for player {player_index}")

            if show:
                typer.echo("Display mode: ON")
                if step:
                    typer.echo("Step mode: Press any key to advance, 'q' to quit")
                else:
                    typer.echo("Press 'q' to quit")

            if step and not show:
                typer.secho(
                    "Warning: --step requires --show to be enabled. Ignoring --step.",
                    fg=typer.colors.YELLOW,
                )

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING ABILITIES")
            typer.echo("=" * 60 + "\n")

            frame_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Determine which players to detect
                if player_index == -1:
                    player_indices = range(10)  # All players
                else:
                    player_indices = [player_index]

                # Collect all detections for display
                all_detections = {}

                # Detect abilities for each player
                for p_idx in player_indices:
                    # Determine side for display (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Pass full player index (0-9) to detector
                    abilities = ability_detector.detect_player_abilities(
                        frame_info.frame, p_idx, side
                    )

                    # Store detections
                    all_detections[p_idx] = (side, abilities)

                    # Display results
                    if all(info is None for info in abilities.values()):
                        continue

                    ability_strs = []
                    for ability_name, ability_info in abilities.items():
                        if ability_info:
                            ability_strs.append(
                                f"{ability_name}: {ability_info.charges} charges"
                            )

                    if ability_strs:
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                            + " | ".join(ability_strs),
                            fg=typer.colors.GREEN,
                        )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                    )

                    # Add ability detections
                    y_offset = 80
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    for p_idx in sorted(all_detections.keys()):
                        side, abilities = all_detections[p_idx]

                        # Skip if no abilities detected
                        if all(info is None for info in abilities.values()):
                            continue

                        # Build ability data for multi-color rendering
                        ability_data = []
                        for ability_name, ability_info in abilities.items():
                            if ability_info:
                                ability_data.append((ability_name[-1], ability_info.charges))

                        if not ability_data:
                            continue

                        # Draw player prefix in green
                        prefix = f"P{p_idx} ({side}): "
                        x_offset = 10
                        cv2.putText(
                            display_frame,
                            prefix,
                            (x_offset, y_offset),
                            font,
                            font_scale,
                            (0, 255, 0),  # Green
                            thickness,
                        )

                        # Calculate width of prefix text
                        (prefix_width, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)
                        x_offset += prefix_width

                        # Draw each ability with label in green, count in red
                        for i, (ability_num, charge_count) in enumerate(ability_data):
                            # Add separator if not first
                            if i > 0:
                                separator = " | "
                                cv2.putText(
                                    display_frame,
                                    separator,
                                    (x_offset, y_offset),
                                    font,
                                    font_scale,
                                    (0, 255, 0),  # Green
                                    thickness,
                                )
                                (sep_width, _), _ = cv2.getTextSize(separator, font, font_scale, thickness)
                                x_offset += sep_width

                            # Draw ability label in green
                            label = f"{ability_num}: "
                            cv2.putText(
                                display_frame,
                                label,
                                (x_offset, y_offset),
                                font,
                                font_scale,
                                (0, 255, 0),  # Green
                                thickness,
                            )
                            (label_width, _), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            x_offset += label_width

                            # Draw count in red
                            count = str(charge_count)
                            cv2.putText(
                                display_frame,
                                count,
                                (x_offset, y_offset),
                                font,
                                font_scale,
                                (0, 0, 255),  # Red
                                thickness,
                            )
                            (count_width, _), _ = cv2.getTextSize(count, font, font_scale, thickness)
                            x_offset += count_width

                        y_offset += 35

                    # Show frame
                    cv2.imshow("Ability Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="abilities-live")
def detect_abilities_live(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to detect (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    brightness_threshold: int = typer.Option(
        127,
        "--brightness",
        help="Brightness threshold for blob detection (0-255)",
    ),
    health_template_dir: Optional[Path] = typer.Option(
        None,
        "--health-template-dir",
        help="Directory containing health digit templates (default: templates/health_digits/)",
    ),
    health_min_confidence: float = typer.Option(
        0.7,
        "--health-confidence",
        help="Minimum confidence for health detection (0-1)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect health, abilities, and ultimates for alive players (using health as alive/dead indicator)."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize ability detector
        ability_detector = AbilityDetector(
            cropper,
            brightness_threshold=brightness_threshold,
        )

        # Initialize ultimate detector
        ultimate_detector = UltimateDetector(
            cropper,
            brightness_threshold=brightness_threshold,
        )

        # Initialize health detector
        health_detector = TemplateHealthDetector(
            cropper,
            template_dir=health_template_dir,
            min_confidence=health_min_confidence,
        )

        # Check that health templates are loaded
        if len(health_detector.templates) == 0:
            typer.secho(
                "Error: No health templates loaded. Please ensure digit templates (0.png - 9.png) "
                f"are at {health_detector.template_dir}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Brightness threshold: {brightness_threshold}")
            typer.echo(f"Health confidence: {health_min_confidence}")

            if player_index == -1:
                typer.echo("Detecting health, abilities, and ultimates for all alive players")
            else:
                typer.echo(f"Detecting health, abilities, and ultimates for player {player_index} (if alive)")

            if show:
                typer.echo("Display mode: ON")
                if step:
                    typer.echo("Step mode: Press any key to advance, 'q' to quit")
                else:
                    typer.echo("Press 'q' to quit")

            if step and not show:
                typer.secho(
                    "Warning: --step requires --show to be enabled. Ignoring --step.",
                    fg=typer.colors.YELLOW,
                )

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING HEALTH, ABILITIES & ULTIMATES (ALIVE PLAYERS ONLY)")
            typer.echo("=" * 60 + "\n")

            frame_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Determine which players to detect
                if player_index == -1:
                    player_indices = range(10)  # All players
                else:
                    player_indices = [player_index]

                # Collect all detections for display
                all_detections = {}

                # Detect health, abilities, and ultimates for each player
                for p_idx in player_indices:
                    # Determine side (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # First, check if player has health (alive indicator)
                    health_info = health_detector.detect(frame_info.frame, p_idx, side)

                    if health_info is None:
                        # No health detected = player is dead
                        all_detections[p_idx] = (side, None, None, None, None)
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): DEAD (no health detected)",
                            fg=typer.colors.RED,
                        )
                    else:
                        # Player is alive - detect abilities and ultimate
                        abilities = ability_detector.detect_player_abilities(
                            frame_info.frame, p_idx, side
                        )

                        ultimate_result = ultimate_detector.detect_ultimate(frame_info.frame, p_idx, side)

                        # Unpack ultimate result (returns tuple of (UltimateInfo, white_pixel_ratio))
                        ultimate_info = ultimate_result[0] if ultimate_result else None

                        # Store detections
                        all_detections[p_idx] = (side, health_info, abilities, ultimate_info, health_info)

                        # Build output strings
                        output_parts = [f"HP: {health_info.health}"]

                        # Add abilities
                        ability_strs = []
                        for ability_name, ability_info in abilities.items():
                            if ability_info:
                                ability_strs.append(
                                    f"{ability_name}: {ability_info.charges}"
                                )

                        if ability_strs:
                            output_parts.extend(ability_strs)

                        # Add ultimate
                        if ultimate_info:
                            if ultimate_info.is_full:
                                output_parts.append(f"Ult: READY")
                            else:
                                output_parts.append(f"Ult: {ultimate_info.charges}")

                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): ALIVE | "
                            + " | ".join(output_parts) +
                            f" (health conf: {health_info.confidence:.2f})",
                            fg=typer.colors.GREEN,
                        )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                    )

                    # Add ability/status detections
                    y_offset = 80
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    for p_idx in sorted(all_detections.keys()):
                        side, health_info, abilities, ultimate_info, _ = all_detections[p_idx]

                        if health_info is None:
                            # Dead player - show in gray
                            status_text = f"P{p_idx} ({side}): DEAD"
                            cv2.putText(
                                display_frame,
                                status_text,
                                (10, y_offset),
                                font,
                                font_scale,
                                (128, 128, 128),  # Gray
                                thickness,
                            )
                            y_offset += 35
                        else:
                            # Alive player - show health, abilities, and ultimate
                            # Draw player prefix in green
                            prefix = f"P{p_idx} ({side}): HP {health_info.health} | "
                            x_offset = 10
                            cv2.putText(
                                display_frame,
                                prefix,
                                (x_offset, y_offset),
                                font,
                                font_scale,
                                (0, 255, 0),  # Green
                                thickness,
                            )

                            # Calculate width of prefix text
                            (prefix_width, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)
                            x_offset += prefix_width

                            # Draw each ability with label in green, count in red
                            ability_data = []
                            for ability_name, ability_info in abilities.items():
                                if ability_info:
                                    ability_data.append((ability_name[-1], ability_info.charges))

                            for i, (ability_num, charge_count) in enumerate(ability_data):
                                # Add separator if not first
                                if i > 0:
                                    separator = " | "
                                    cv2.putText(
                                        display_frame,
                                        separator,
                                        (x_offset, y_offset),
                                        font,
                                        font_scale,
                                        (0, 255, 0),  # Green
                                        thickness,
                                    )
                                    (sep_width, _), _ = cv2.getTextSize(separator, font, font_scale, thickness)
                                    x_offset += sep_width

                                # Draw ability label in green
                                label = f"{ability_num}: "
                                cv2.putText(
                                    display_frame,
                                    label,
                                    (x_offset, y_offset),
                                    font,
                                    font_scale,
                                    (0, 255, 0),  # Green
                                    thickness,
                                )
                                (label_width, _), _ = cv2.getTextSize(label, font, font_scale, thickness)
                                x_offset += label_width

                                # Draw count in red
                                count = str(charge_count)
                                cv2.putText(
                                    display_frame,
                                    count,
                                    (x_offset, y_offset),
                                    font,
                                    font_scale,
                                    (0, 0, 255),  # Red
                                    thickness,
                                )
                                (count_width, _), _ = cv2.getTextSize(count, font, font_scale, thickness)
                                x_offset += count_width

                            # Add ultimate status
                            if ultimate_info:
                                separator = " | Ult: "
                                cv2.putText(
                                    display_frame,
                                    separator,
                                    (x_offset, y_offset),
                                    font,
                                    font_scale,
                                    (0, 255, 0),  # Green
                                    thickness,
                                )
                                (sep_width, _), _ = cv2.getTextSize(separator, font, font_scale, thickness)
                                x_offset += sep_width

                                if ultimate_info.is_full:
                                    ult_text = "READY"
                                    ult_color = (0, 255, 255)  # Yellow
                                else:
                                    ult_text = f"{ultimate_info.charges}"
                                    ult_color = (0, 0, 255)  # Red

                                cv2.putText(
                                    display_frame,
                                    ult_text,
                                    (x_offset, y_offset),
                                    font,
                                    font_scale,
                                    ult_color,
                                    thickness,
                                )

                            y_offset += 35

                    # Show frame
                    cv2.imshow("Live Ability Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="ultimates")
def detect_ultimates(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to detect (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    brightness_threshold: int = typer.Option(
        127,
        "--brightness",
        help="Brightness threshold for segment detection (0-255)",
    ),
    fullness_threshold: float = typer.Option(
        0.4,
        "--fullness",
        help="White pixel ratio threshold for full ultimate detection (0-1)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
    debug_crops: bool = typer.Option(
        False,
        "--debug-crops",
        help="Save preprocessed ultimate crops with mask overlays for debugging",
    ),
    debug_output_dir: Path = typer.Option(
        Path("./ultimate_debug"),
        "--debug-output",
        help="Output directory for debug crops (default: ./ultimate_debug)",
    ),
) -> None:
    """Detect ultimate charges from a Valorant VOD using blob detection."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize ultimate detector
        ultimate_detector = UltimateDetector(
            cropper,
            brightness_threshold=brightness_threshold,
            fullness_threshold=fullness_threshold,
        )

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Brightness threshold: {brightness_threshold}")
            typer.echo(f"Fullness threshold: {fullness_threshold}")

            if player_index == -1:
                typer.echo("Detecting ultimates for all players")
            else:
                typer.echo(f"Detecting ultimate for player {player_index}")

            if show:
                typer.echo("Display mode: ON")
                if step:
                    typer.echo("Step mode: Press any key to advance, 'q' to quit")
                else:
                    typer.echo("Press 'q' to quit")

            if step and not show:
                typer.secho(
                    "Warning: --step requires --show to be enabled. Ignoring --step.",
                    fg=typer.colors.YELLOW,
                )

            # Create debug output directory if needed
            if debug_crops:
                debug_output_dir.mkdir(parents=True, exist_ok=True)
                typer.echo(f"Debug crops will be saved to: {debug_output_dir}")
                import shutil
                # Clear directory
                if debug_output_dir.exists():
                    for file in debug_output_dir.glob("*.png"):
                        file.unlink()

            typer.echo("\n" + "=" * 60)
            typer.echo("DETECTING ULTIMATES")
            typer.echo("=" * 60 + "\n")

            frame_count = 0

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames and frame_count >= max_frames:
                    break

                # Create display frame if showing
                if show:
                    display_frame = frame_info.frame.copy()

                # Determine which players to detect
                if player_index == -1:
                    player_indices = range(10)  # All players
                else:
                    player_indices = [player_index]

                # Collect all detections for display
                all_detections = {}

                # Detect ultimates for each player
                for p_idx in player_indices:
                    # Determine side (0-4 = left, 5-9 = right)
                    side = "left" if p_idx < 5 else "right"

                    # Use debug detection if debug_crops is enabled
                    if debug_crops:
                        ultimate_info, preprocessed, debug_info = ultimate_detector.detect_with_debug(
                            frame_info.frame, p_idx, side
                        )

                        # Get white pixel ratio from debug info
                        white_pixel_ratio = debug_info.get("white_pixel_ratio", 0.0)

                        # Save preprocessed image exactly as the detector sees it
                        if preprocessed.size > 0 and ultimate_info is not None:
                            # Save the raw preprocessed image (no overlays)
                            filename = (
                                f"p{p_idx}_{side}_f{frame_count:04d}_"
                                f"fullness{white_pixel_ratio:.3f}_"
                                f"charges{ultimate_info.charges}.png"
                            )
                            output_path = debug_output_dir / filename
                            cv2.imwrite(str(output_path), preprocessed)

                        # Result is already unpacked from detect_with_debug
                        result = (ultimate_info, white_pixel_ratio) if ultimate_info else None
                    else:
                        # Normal detection
                        result = ultimate_detector.detect_ultimate(
                            frame_info.frame, p_idx, side
                        )

                    # Unpack result
                    if result is None:
                        all_detections[p_idx] = (side, None, 0.0)
                        continue

                    ultimate_info, white_pixel_ratio = result

                    # Store detections
                    all_detections[p_idx] = (side, ultimate_info, white_pixel_ratio)

                    # Display results
                    if ultimate_info.is_full:
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                            f"FULL ({ultimate_info.charges} charges) [fullness: {white_pixel_ratio:.3f}]",
                            fg=typer.colors.CYAN,
                        )
                    else:
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): "
                            f"{ultimate_info.charges} charges "
                            f"[fullness: {white_pixel_ratio:.3f}, blobs: {ultimate_info.total_blobs_detected}]",
                            fg=typer.colors.GREEN,
                        )

                # Draw detections on frame
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {frame_info.timestamp_sec:.2f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                    )

                    # Add ultimate detections
                    y_offset = 80
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    for p_idx in sorted(all_detections.keys()):
                        side, ultimate_info, white_pixel_ratio = all_detections[p_idx]

                        if ultimate_info is None:
                            continue

                        if ultimate_info.is_full:
                            # Full ultimate - cyan
                            text = f"P{p_idx} ({side}): FULL ({ultimate_info.charges} charges) [{white_pixel_ratio:.3f}]"
                            color = (255, 255, 0)  # Cyan
                        else:
                            # Partial ultimate - green
                            text = f"P{p_idx} ({side}): {ultimate_info.charges} charges [{white_pixel_ratio:.3f}]"
                            color = (0, 255, 0)  # Green

                        cv2.putText(
                            display_frame,
                            text,
                            (10, y_offset),
                            font,
                            font_scale,
                            color,
                            thickness,
                        )
                        y_offset += 35

                    # Show frame
                    cv2.imshow("Ultimate Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Clean up
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="killfeed")
def detect_killfeed(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--template-dir",
        "-t",
        help="Path to killfeed agent templates directory (default: templates/killfeed_agents/)",
    ),
    min_confidence: float = typer.Option(
        0.85,
        "--confidence",
        help="Minimum template match confidence (0-1)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        1.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 1s)",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display visual output of detections",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Step through frames one at a time (requires --show)",
    ),
    agents: Optional[str] = typer.Option(
        None,
        "--agents",
        "-a",
        help="Comma-separated list of agent names to match (e.g., 'jett,raze,cypher'). If not provided, matches all agents.",
    ),
    full_frame: bool = typer.Option(
        False,
        "--full-frame",
        "-f",
        help="Show full frame instead of just killfeed crops (requires --show)",
    ),
) -> None:
    """Detect kills from killfeed using agent icon template matching."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Parse agent list if provided
        agent_list = None
        if agents:
            agent_list = [a.strip() for a in agents.split(",")]
            typer.echo(f"Limiting matching to {len(agent_list)} agents: {', '.join(agent_list)}")

        # Initialize killfeed detector
        killfeed_detector = KillfeedDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
            agents=agent_list,
        )

        # Check if templates loaded
        if len(killfeed_detector.templates) == 0:
            typer.secho(
                f"Error: No templates loaded. Please ensure agent templates are at "
                f"{killfeed_detector.template_dir}/attack/ and "
                f"{killfeed_detector.template_dir}/defense/",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo(f"Loaded {len(killfeed_detector.templates)} agent templates")
        typer.echo(f"Minimum confidence: {min_confidence}")

        # Validate step and full_frame require show
        if step and not show:
            typer.secho(
                "Warning: --step requires --show to be enabled. Ignoring --step.",
                fg=typer.colors.YELLOW,
            )
        if full_frame and not show:
            typer.secho(
                "Warning: --full-frame requires --show to be enabled. Ignoring --full-frame.",
                fg=typer.colors.YELLOW,
            )

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")

            if show:
                typer.echo("Display mode: ON")
                if step:
                    typer.echo("Step mode: Press any key to advance, 'q' to quit")
                else:
                    typer.echo("Press 'q' to quit")

            typer.echo("\n" + "=" * 60)
            typer.echo("KILLFEED DETECTION")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            total_kills = 0

            for frame_info in reader:
                # Detect kills
                detections = killfeed_detector.detect(frame_info.frame)

                # Format timestamp
                time = f"{int(frame_info.timestamp_sec // 60):02d}:{int(frame_info.timestamp_sec % 60):02d}"

                if len(detections) == 0:
                    typer.secho(f"[{time}] No kills detected", fg=typer.colors.WHITE, dim=True)
                else:
                    for entry_idx, detection in enumerate(detections):
                        if detection:
                            # Format: "atk agent_name killed def agent_name"
                            kill_msg = (
                                f"{detection.killer_side[:3]} {detection.killer_agent} killed "
                                f"{detection.victim_side[:3]} {detection.victim_agent}"
                            )

                            # Color code based on confidence
                            if detection.confidence >= 0.95:
                                color = typer.colors.GREEN
                            elif detection.confidence >= 0.90:
                                color = typer.colors.YELLOW
                            else:
                                color = typer.colors.WHITE

                            typer.secho(
                                f"[{time}] Entry {entry_idx}: {kill_msg} (conf: {detection.confidence:.2f})",
                                fg=color,
                            )
                            total_kills += 1

                # Display mode
                if show:
                    if full_frame:
                        # Show full frame with detection text overlay (always, even if no detections)
                        display_frame = frame_info.frame.copy()

                        # Add detection text to frame
                        y_offset = 50
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2

                        if len(detections) > 0:
                            for entry_idx, detection in enumerate(detections):
                                if detection:
                                    kill_text = (
                                        f"Entry {entry_idx}: {detection.killer_side[:3]} {detection.killer_agent} -> "
                                        f"{detection.victim_side[:3]} {detection.victim_agent} "
                                        f"(conf: {detection.confidence:.2f})"
                                    )

                                    # Color based on confidence
                                    if detection.confidence >= 0.95:
                                        text_color = (0, 255, 0)  # Green
                                    elif detection.confidence >= 0.90:
                                        text_color = (0, 255, 255)  # Yellow
                                    else:
                                        text_color = (255, 255, 255)  # White

                                    cv2.putText(
                                        display_frame,
                                        kill_text,
                                        (10, y_offset),
                                        font,
                                        font_scale,
                                        text_color,
                                        thickness,
                                    )
                                    y_offset += 35
                        else:
                            # Show "No detections" message
                            cv2.putText(
                                display_frame,
                                "No kills detected",
                                (10, y_offset),
                                font,
                                font_scale,
                                (128, 128, 128),  # Gray
                                thickness,
                            )

                        cv2.imshow("Killfeed Detection", display_frame)

                        # Wait for key press
                        if step:
                            key = cv2.waitKey(0) & 0xFF
                        else:
                            key = cv2.waitKey(1) & 0xFF

                        # Check for quit
                        if key == ord('q'):
                            typer.echo("\nQuitting...")
                            break

                    elif len(detections) > 0:
                        # Show individual killfeed crops (only when there are detections)
                        killfeed_entries = cropper.crop_killfeed(frame_info.frame)

                        for entry_idx, detection in enumerate(detections):
                            if detection and entry_idx < len(killfeed_entries):
                                entry_crop = killfeed_entries[entry_idx]

                                # Display entry crop
                                display_crop = cv2.resize(entry_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                                # Add label
                                kill_text = (
                                    f"{detection.killer_side[:3]} {detection.killer_agent} -> "
                                    f"{detection.victim_side[:3]} {detection.victim_agent}"
                                )
                                label_img = 255 * np.ones((40, display_crop.shape[1], 3), dtype=np.uint8)
                                cv2.putText(
                                    label_img,
                                    kill_text,
                                    (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    1,
                                )

                                # Stack label and crop
                                combined = np.vstack([label_img, display_crop])

                                cv2.imshow(f"Killfeed Entry {entry_idx}", combined)

                        # Wait for key press
                        if step:
                            key = cv2.waitKey(0) & 0xFF
                        else:
                            key = cv2.waitKey(1) & 0xFF

                        # Check for quit
                        if key == ord('q'):
                            typer.echo("\nQuitting...")
                            break

                frame_count += 1

            # Cleanup display mode
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Total kills detected: {total_kills}")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="phase")
def detect_phase(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        4.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 4 frames/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display video frames with phase detection overlays",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Wait for key press before proceeding to next frame (requires --show)",
    ),
) -> None:
    """Detect game phase from a Valorant VOD."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if step and not show:
        typer.secho(
            "Warning: --step requires --show to be enabled. Ignoring --step.",
            fg=typer.colors.YELLOW,
        )

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize detectors
        timer_detector = TemplateTimerDetector(cropper)
        spike_detector = TemplateSpikeDetector(cropper)
        score_detector = TemplateScoreDetector(cropper)
        credits_detector = PreroundCreditsDetector(cropper)

        # Initialize phase detector
        phase_detector = PhaseDetector(
            timer_detector=timer_detector,
            spike_detector=spike_detector,
            score_detector=score_detector,
            credits_detector=credits_detector,
        )

        typer.echo(f"\nProcessing video: {video_path}")
        typer.echo(f"FPS: {fps}")
        if start_time is not None:
            typer.echo(f"Start time: {start_time}s")
        if end_time is not None:
            typer.echo(f"End time: {end_time}s")
        if max_frames is not None:
            typer.echo(f"Max frames: {max_frames}")

        if show:
            typer.echo("Display mode: ON")
            if step:
                typer.echo("Step mode: Press any key to advance, 'q' to quit")
            else:
                typer.echo("Press 'q' to quit")

        typer.echo()

        # Phase tracking
        current_phase = Phase.NON_GAME
        phase_changes = []
        phase_counts = {phase: 0 for phase in Phase}
        frame_count = 0

        # Process frames
        with VideoReader(
            str(video_path),
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo("=" * 80)
            typer.echo(f"{'Frame':<8} {'Time':<10} {'Phase':<15} {'Timer':<10} {'Spike':<8} {'Score':<10} {'Credits':<10}")
            typer.echo("=" * 80)

            for frame_info in reader:
                # Stop if max_frames reached
                if max_frames is not None and frame_count >= max_frames:
                    break

                timestamp = frame_info.timestamp_sec
                frame = frame_info.frame

                # Create display frame if showing
                if show:
                    display_frame = frame.copy()

                # Detect phase
                detected_phase, detections = phase_detector.detect_phase(frame, current_phase)

                # Track phase changes
                if detected_phase != current_phase:
                    phase_changes.append({
                        "frame": frame_count,
                        "timestamp": timestamp,
                        "old_phase": current_phase,
                        "new_phase": detected_phase,
                    })
                    current_phase = detected_phase

                # Track phase counts
                phase_counts[detected_phase] += 1

                # Format detection results
                timer_str = (
                    f"{detections['timer'].time_seconds:.1f}s"
                    if detections['timer']
                    else "N/A"
                )
                spike_str = (
                    "YES" if detections['spike'] and detections['spike'].spike_planted else "NO"
                )
                score_str = (
                    f"{detections['score'].team1_score}-{detections['score'].team2_score}"
                    if detections['score']
                    else "N/A"
                )
                credits_str = (
                    "YES" if detections['preround_credits'] else "NO"
                )

                # Print frame result
                typer.echo(
                    f"{frame_count:<8} {timestamp:<10.2f} {detected_phase.name:<15} "
                    f"{timer_str:<10} {spike_str:<8} {score_str:<10} {credits_str:<10}"
                )

                # Display frame with annotations
                if show:
                    # Add timestamp
                    cv2.putText(
                        display_frame,
                        f"Time: {timestamp:.2f}s | Frame: {frame_count}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )

                    # Add phase with background box for visibility
                    phase_color = {
                        Phase.NON_GAME: (128, 128, 128),  # Gray
                        Phase.PREROUND: (0, 255, 255),    # Yellow
                        Phase.ACTIVE_ROUND: (0, 255, 0),  # Green
                        Phase.POST_ROUND: (0, 165, 255),  # Orange
                    }
                    color = phase_color.get(detected_phase, (255, 255, 255))

                    # Draw background rectangle for phase
                    phase_text = f"PHASE: {detected_phase.name}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        phase_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                    )
                    cv2.rectangle(
                        display_frame,
                        (10, 70),
                        (20 + text_width, 105 + text_height),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        display_frame,
                        phase_text,
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        3,
                    )

                    # Add detection details
                    y_offset = 160
                    details = [
                        f"Timer: {timer_str}",
                        f"Spike: {spike_str}",
                        f"Score: {score_str}",
                        f"Credits: {credits_str}",
                    ]
                    for detail in details:
                        cv2.putText(
                            display_frame,
                            detail,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        y_offset += 35

                    # Show frame
                    cv2.imshow("Phase Detection", display_frame)

                    # Wait for key press
                    if step:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    # Check for quit
                    if key == ord('q'):
                        typer.echo("\nQuitting...")
                        break

                frame_count += 1

            # Cleanup display mode
            if show:
                cv2.destroyAllWindows()

            # Summary
            typer.echo("\n" + "=" * 80)
            typer.echo("SUMMARY")
            typer.echo("=" * 80)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"\nPhase distribution:")
            for phase, count in phase_counts.items():
                percentage = (count / frame_count * 100) if frame_count > 0 else 0
                typer.echo(f"  {phase.name:<15}: {count:>4} frames ({percentage:>5.1f}%)")

            if phase_changes:
                typer.echo(f"\nPhase transitions ({len(phase_changes)}):")
                for change in phase_changes:
                    typer.echo(
                        f"  Frame {change['frame']:>4} @ {change['timestamp']:>7.2f}s: "
                        f"{change['old_phase'].name:>15} → {change['new_phase'].name}"
                    )

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@app.command(name="agents-active")
def detect_agents_active(
    image_path: Path = typer.Argument(..., help="Path to test image"),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to test (0-9, or -1 for all players)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--template-dir",
        "-t",
        help="Directory containing agent templates (default: src/valoscribe/templates/active_round_agents/)",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Show detailed match scores for all templates",
    ),
) -> None:
    """Detect agent icons from active round using template matching.

    Note: This only detects agent names. Side (attack/defense) is inferred from
    screen position (left = 0-4, right = 5-9) and round state.
    """
    setup_logging()

    if not image_path.exists():
        typer.secho(f"Error: Image file not found: {image_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Load image
        typer.echo(f"Loading image: {image_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            typer.secho(f"Error: Could not load image: {image_path}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Image loaded: {frame.shape[1]}x{frame.shape[0]}")

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize detector
        detector = ActiveRoundAgentDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
        )

        # Check that templates are loaded
        if len(detector.templates) == 0:
            typer.secho(
                "Error: No templates loaded. Please ensure agent templates are at "
                f"{detector.template_dir}/",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo(f"Loaded {len(detector.templates)} templates")
        typer.echo("")

        # Determine which players to check
        if player_index == -1:
            players_to_check = list(range(10))
        else:
            if not 0 <= player_index <= 9:
                typer.secho(
                    f"Error: Player index must be between 0 and 9, got {player_index}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)
            players_to_check = [player_index]

        # Process each player
        for p_idx in players_to_check:
            if debug:
                agent_name, _, debug_info = detector.detect_with_debug(frame, p_idx)

                typer.echo(f"=== PLAYER {p_idx} ===")
                if debug_info.get("error"):
                    typer.secho(f"  Error: {debug_info['error']}", fg=typer.colors.RED)
                else:
                    typer.echo(f"  Crop shape: {debug_info.get('agent_crop_shape', 'N/A')}")

                    if "match_scores" in debug_info:
                        # Sort by confidence
                        scores = sorted(
                            debug_info["match_scores"].items(),
                            key=lambda x: x[1].get("confidence", 0),
                            reverse=True,
                        )

                        typer.echo("  Top 5 matches:")
                        for agent, score_data in scores[:5]:
                            confidence = score_data.get("confidence", 0)

                            if "skipped" in score_data:
                                typer.echo(f"    {agent:30s} - SKIPPED")
                            else:
                                color_code = typer.colors.GREEN if confidence >= min_confidence else typer.colors.WHITE
                                typer.secho(
                                    f"    {agent:30s} - {confidence:.3f}",
                                    fg=color_code,
                                )

                if agent_name:
                    side_label = "LEFT" if p_idx < 5 else "RIGHT"
                    typer.secho(f"  ✓ Detected: {agent_name} (screen: {side_label})", fg=typer.colors.GREEN)
                else:
                    typer.secho(f"  ✗ No agent detected", fg=typer.colors.YELLOW)

                typer.echo("")
            else:
                agent_name = detector.detect(frame, p_idx)

                if agent_name:
                    side_label = "LEFT" if p_idx < 5 else "RIGHT"
                    typer.secho(
                        f"Player {p_idx}: {agent_name} (screen: {side_label})",
                        fg=typer.colors.GREEN,
                    )
                else:
                    typer.secho(f"Player {p_idx}: No agent detected", fg=typer.colors.YELLOW)

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@app.command(name="agents-active-vod")
def detect_agents_active_vod(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    template_dir: Optional[Path] = typer.Option(
        None,
        "--template-dir",
        "-t",
        help="Directory containing agent templates (default: src/valoscribe/templates/active_round_agents/)",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        help="Minimum confidence threshold (0-1)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    fps: Optional[float] = typer.Option(
        1.0,
        "--fps",
        "-f",
        help="Process frames at this FPS (default: 1 frame/sec)",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
    ),
    show_all: bool = typer.Option(
        False,
        "--show-all",
        help="Show detection results for every frame (default: only show when all 10 agents detected)",
    ),
) -> None:
    """Detect agents from active round VOD frames.
    
    Processes video frames and detects agent icons. By default, only prints results
    when all 10 agents are successfully detected. Use --show-all to see every frame.
    
    Example:
        valoscribe detect agents-active-vod video.mp4 --fps 1 --start 60
    """
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Initialize detector
        detector = ActiveRoundAgentDetector(
            cropper,
            template_dir=template_dir,
            min_confidence=min_confidence,
        )

        # Check that templates are loaded
        if len(detector.templates) == 0:
            typer.secho(
                "Error: No templates loaded. Please ensure agent templates are at "
                f"{detector.template_dir}/",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo(f"Loaded {len(detector.templates)} templates")
        typer.echo(f"Processing video: {video_path}")
        typer.echo(f"FPS: {fps}, Start: {start_time}s, End: {end_time}s")
        typer.echo("")

        # Create video reader
        with VideoReader(
            str(video_path),
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as video_reader:

            frame_count = 0
            agents_detected_frame = None

            # Process frames
            for frame_info in video_reader:
                frame_count += 1
                timestamp = frame_info.timestamp_sec
                frame = frame_info.frame

                # Detect agents for all 10 players
                detected_agents = []
                all_detected = True

                for player_idx in range(10):
                    agent_name = detector.detect(frame, player_idx)
                    if agent_name:
                        detected_agents.append((player_idx, agent_name))
                    else:
                        all_detected = False
                        detected_agents.append((player_idx, None))

                # Show results
                if show_all or all_detected:
                    typer.echo(f"[{timestamp:7.2f}s] Frame {frame_count}:")
                
                    # Left side (0-4)
                    typer.echo("  LEFT side (0-4):")
                    for player_idx in range(5):
                        agent_name = detected_agents[player_idx][1]
                        if agent_name:
                            typer.secho(f"    Player {player_idx}: {agent_name}", fg=typer.colors.GREEN)
                        else:
                            typer.secho(f"    Player {player_idx}: Not detected", fg=typer.colors.YELLOW)
                
                    # Right side (5-9)
                    typer.echo("  RIGHT side (5-9):")
                    for player_idx in range(5, 10):
                        agent_name = detected_agents[player_idx][1]
                        if agent_name:
                            typer.secho(f"    Player {player_idx}: {agent_name}", fg=typer.colors.GREEN)
                        else:
                            typer.secho(f"    Player {player_idx}: Not detected", fg=typer.colors.YELLOW)
                
                    typer.echo("")

                # Track first successful detection of all 10
                if all_detected and agents_detected_frame is None:
                    agents_detected_frame = frame_count
                    typer.secho(
                        f"✓ All 10 agents detected at frame {frame_count} ({timestamp:.2f}s)!",
                        fg=typer.colors.GREEN,
                        bold=True,
                    )
                    typer.echo("")
                
                    # Print summary
                    typer.echo("Agent Summary:")
                    typer.echo("  LEFT side (0-4):")
                    for player_idx in range(5):
                        typer.echo(f"    Player {player_idx}: {detected_agents[player_idx][1]}")
                    typer.echo("  RIGHT side (5-9):")
                    for player_idx in range(5, 10):
                        typer.echo(f"    Player {player_idx}: {detected_agents[player_idx][1]}")
                    typer.echo("")
                
                    # Exit unless show_all is enabled
                    if not show_all:
                        break

                # Check max frames
                if max_frames and frame_count >= max_frames:
                    typer.echo(f"Reached max frames limit ({max_frames})")
                    break

        # Final summary
        typer.echo("=" * 60)
        typer.echo(f"Processed {frame_count} frames")
        if agents_detected_frame:
            typer.secho(
                f"All 10 agents first detected at frame {agents_detected_frame}",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho("Could not detect all 10 agents in the processed frames", fg=typer.colors.YELLOW)

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
