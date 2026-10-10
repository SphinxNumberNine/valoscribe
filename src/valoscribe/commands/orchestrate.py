"""Orchestration commands for full VOD processing."""

import typer
import json
import logging
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.orchestration.game_state_manager import GameStateManager
from valoscribe.orchestration.phase_detector import Phase
from valoscribe.video.reader import VideoReader
from valoscribe.utils.logger import setup_logging

app = typer.Typer(help="Orchestration commands for full VOD processing")


@app.command(name="process-vod")
def process_vod(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    metadata_path: Path = typer.Argument(..., help="Path to VLR metadata JSON file"),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output",
        "-o",
        help="Output directory for CSV and JSONL files",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    fps: float = typer.Option(
        4.0,
        "--fps",
        "-f",
        help="Frames per second to process",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display frames during processing",
    ),
    step: bool = typer.Option(
        False,
        "--step",
        help="Step through frames (press any key to advance, 'q' to quit)",
    ),
    print_frame_data: bool = typer.Option(
        False,
        "--print-frame-data",
        help="Print full frame state data (default: only events)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging",
    ),
    player_filter: Optional[int] = typer.Option(
        None,
        "--player",
        "-p",
        help="Filter debug output to specific player index (0-9)",
    ),
    mute_agent_detector: bool = typer.Option(
        False,
        "--mute-agent-detector",
        help="Mute logs from the agent detector",
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
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode: only show events, suppress progress messages",
    ),
    show_prev_frame: bool = typer.Option(
        False,
        "--show-prev-frame",
        help="Show previous frame alongside current frame when events are fired (requires --show)",
    ),
    debug_phase: bool = typer.Option(
        False,
        "--debug-phase",
        help="Show phase detection details (timer, score, spike, credits)",
    ),
):
    """
    Process a Valorant VOD using the GameStateManager orchestrator.

    This command processes the entire video frame-by-frame, detecting game state
    and events, and outputs structured CSV and JSONL files.

    Example:
        valoscribe orchestrate process-vod video.mp4 metadata.json --show --step
        valoscribe orchestrate process-vod video.mp4 metadata.json --quiet  # Only show events
        valoscribe orchestrate process-vod video.mp4 metadata.json --show --show-prev-frame --step  # Debug with previous frame
        valoscribe orchestrate process-vod video.mp4 metadata.json --debug-phase  # Show phase detection details
    """
    setup_logging(level=logging.DEBUG if debug else logging.INFO)

    # Mute agent detector logs if requested
    if mute_agent_detector:
        logging.getLogger('valoscribe.detectors.template_agent_detector').setLevel(logging.WARNING)

    # Enable phase detection debug logs if requested
    if debug_phase:
        logging.getLogger('valoscribe.orchestration.phase_detector').setLevel(logging.DEBUG)
        logging.getLogger('valoscribe.detectors.template_score_detector').setLevel(logging.DEBUG)
        logging.getLogger('valoscribe.detectors.template_timer_detector').setLevel(logging.DEBUG)
        logging.getLogger('valoscribe.detectors.template_spike_detector').setLevel(logging.DEBUG)
        logging.getLogger('valoscribe.detectors.preround_credits_detector').setLevel(logging.DEBUG)

    # Validate player_filter
    if player_filter is not None and not (0 <= player_filter <= 9):
        typer.echo(f"Error: Player filter must be between 0 and 9, got {player_filter}", err=True)
        raise typer.Exit(1)

    # Validate show_prev_frame requires show
    if show_prev_frame and not show:
        typer.echo("Error: --show-prev-frame requires --show to be enabled", err=True)
        raise typer.Exit(1)

    # Load VLR metadata
    if not quiet:
        typer.echo(f"Loading VLR metadata from {metadata_path}...")
    try:
        with open(metadata_path) as f:
            vlr_metadata = json.load(f)
    except Exception as e:
        typer.echo(f"Error loading metadata: {e}", err=True)
        raise typer.Exit(1)

    # Validate metadata
    if "teams" not in vlr_metadata or "players" not in vlr_metadata:
        typer.echo("Error: Invalid metadata format (missing 'teams' or 'players')", err=True)
        raise typer.Exit(1)

    if not quiet:
        typer.echo(f"Teams: {vlr_metadata['teams'][0]['name']} vs {vlr_metadata['teams'][1]['name']}")
        typer.echo(f"Players: {len(vlr_metadata['players'])}")
        typer.echo(f"Map: {vlr_metadata.get('map', 'Unknown')}")
        typer.echo("")

    # Initialize GameStateManager
    if not quiet:
        typer.echo(f"Initializing GameStateManager...")
        typer.echo(f"  Video: {video_path}")
        typer.echo(f"  Output: {output_dir}")
        typer.echo(f"  FPS: {fps}")
        typer.echo("")

    manager = GameStateManager(
        video_path=video_path,
        vlr_metadata=vlr_metadata,
        output_dir=output_dir,
        config_path=config_path,
        fps=fps,
        debug_player_filter=player_filter,
    )

    # Create display window if needed
    if show:
        cv2.namedWindow("GameStateManager - VOD Processing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GameStateManager - VOD Processing", 1280, 720)

        # Create previous frame window if requested
        if show_prev_frame:
            cv2.namedWindow("Previous Frame (Before Events)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Previous Frame (Before Events)", 1280, 720)

    # Track events for display
    last_event_count = 0
    previous_frame = None  # Store previous frame for debugging

    if not quiet:
        typer.echo("Starting video processing...")
        typer.echo("=" * 80)
        typer.echo("")

    try:
        # Open output files
        with manager.output_writer, VideoReader(
                str(video_path),
                fps_filter=fps,
                start_time_sec=start_time,
                end_time_sec=end_time,
        ) as video_reader:
            try:
                for frame_info in video_reader:
                    timestamp = frame_info.timestamp_sec
                    frame = frame_info.frame

                    # Process frame with manager (handles all core logic)
                    manager.process_frame(timestamp, frame)

                    # UI/Display logic below (CLI-specific)

                    # 1. Print events (new events since last frame)
                    current_event_count = len(manager.event_collector.get_all_events())
                    if current_event_count > last_event_count:
                        new_events = manager.event_collector.get_all_events()[last_event_count:]
                        for event in new_events:
                            _print_event(event)

                        # Show previous frame if requested and available
                        if show and show_prev_frame and previous_frame is not None:
                            prev_display = _create_display_frame(
                                previous_frame,
                                timestamp - (1.0 / fps),  # Approximate previous timestamp
                                manager.frame_count - 1,
                                manager.current_phase,
                                manager.round_manager,
                                manager.timer_manager,
                            )
                            cv2.imshow("Previous Frame (Before Events)", prev_display)

                        last_event_count = current_event_count

                    # 2. Print frame data if requested
                    if print_frame_data and manager.current_phase:
                        _print_frame_data(
                            timestamp,
                            manager.frame_count,
                            manager.current_phase,
                            manager.round_manager,
                            manager.timer_manager,
                        )

                    # 3. Display frame if requested
                    if show and manager.current_phase:
                        display_frame = _create_display_frame(
                            frame,
                            timestamp,
                            manager.frame_count,
                            manager.current_phase,
                            manager.round_manager,
                            manager.timer_manager,
                        )
                        cv2.imshow("GameStateManager - VOD Processing", display_frame)

                        # Handle keyboard input
                        if step:
                            key = cv2.waitKey(0) & 0xFF
                        else:
                            key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            typer.echo("\nProcessing stopped by user")
                            break

                    # 4. Progress feedback (every 100 frames)
                    if not quiet and manager.frame_count % 100 == 0 and manager.current_phase:
                        typer.echo(
                            f"[{timestamp:7.2f}s] Frame {manager.frame_count:5d} | "
                            f"Phase: {manager.current_phase.name:12s} | "
                            f"Round {manager.round_manager.current_round:2d} | "
                            f"Score: {manager.round_manager.current_score['team1']}-{manager.round_manager.current_score['team2']}"
                        )

                    # Store current frame as previous for next iteration
                    if show_prev_frame:
                        previous_frame = frame.copy()
            except StopIteration:
                # Match ended normally - this is expected when match completes
                if not quiet:
                    typer.echo("\nMatch ended, processing complete")

            # Finalize: check for missing round_end and match_end events
            manager._finalize_events()

    finally:
        if show:
            cv2.destroyAllWindows()

    # Summary
    if not quiet:
        typer.echo("")
        typer.echo("=" * 80)
        typer.echo("PROCESSING COMPLETE")
        typer.echo("=" * 80)
        typer.echo(f"Total frames processed: {manager.frame_count}")
        typer.echo(f"Total events detected: {len(manager.event_collector.get_all_events())}")
        typer.echo("")
        typer.echo("Output files:")
        typer.echo(f"  - {manager.output_writer.frame_states_path}")
        typer.echo(f"  - {manager.output_writer.event_log_path}")


def _print_event(event: dict) -> None:
    """Print an event to console with formatting."""
    event_type = event.get("type", "unknown")
    timestamp = event.get("timestamp", 0.0)

    # Color codes (if terminal supports it)
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Format based on event type
    if event_type == "round_start":
        typer.echo(
            f"{GREEN}[{timestamp:7.2f}s] ROUND START{RESET} | "
            f"Round {event.get('round_number', '?')} | "
            f"Score: {event.get('score_team1', 0)}-{event.get('score_team2', 0)}"
        )
    elif event_type == "round_end":
        typer.echo(
            f"{RED}[{timestamp:7.2f}s] ROUND END{RESET} | "
            f"Round {event.get('round_number', '?')} | "
            f"Winner: {event.get('winner', '?')} | "
            f"Score: {event.get('score_team1', 0)}-{event.get('score_team2', 0)}"
        )
    elif event_type == "spike_plant":
        typer.echo(
            f"{YELLOW}[{timestamp:7.2f}s] SPIKE PLANTED{RESET}"
        )
    elif event_type == "kill":
        # Build killer string with name if available
        killer_agent = event.get('killer_agent', '?')
        killer_name = event.get('killer_name')
        killer_side = event.get('killer_side', '?')
        killer_team = event.get('killer_team', '?')

        if killer_name:
            killer_str = f"{killer_name} ({killer_agent}, {killer_team})"
        else:
            killer_str = f"{killer_agent} ({killer_side})"

        # Build victim string with name if available
        victim_agent = event.get('victim_agent', '?')
        victim_name = event.get('victim_name')
        victim_side = event.get('victim_side', '?')
        victim_team = event.get('victim_team', '?')

        if victim_name:
            victim_str = f"{victim_name} ({victim_agent}, {victim_team})"
        else:
            victim_str = f"{victim_agent} ({victim_side})"

        typer.echo(
            f"{CYAN}[{timestamp:7.2f}s] KILL{RESET} | "
            f"{killer_str} -> {victim_str}"
        )
    elif event_type == "death":
        typer.echo(
            f"[{timestamp:7.2f}s] DEATH | "
            f"Player: {event.get('player', event.get('player_name', '?'))}"
        )
    elif event_type == "revival":
        typer.echo(
            f"[{timestamp:7.2f}s] REVIVAL | "
            f"Player: {event.get('player', event.get('player_name', '?'))}"
        )
    elif event_type == "ability_used":
        typer.echo(
            f"[{timestamp:7.2f}s] ABILITY USED | "
            f"{event.get('player', event.get('player_name', '?'))} - {event.get('ability', '?')}"
        )
    elif event_type == "ultimate_used":
        typer.echo(
            f"[{timestamp:7.2f}s] ULTIMATE USED | "
            f"{event.get('player', event.get('player_name', '?'))}"
        )
    else:
        # Generic event format
        typer.echo(f"[{timestamp:7.2f}s] {event_type.upper()} | {event}")


def _print_frame_data(
    timestamp: float,
    frame_number: int,
    phase: Phase,
    round_manager,
    timer_manager,
) -> None:
    """Print full frame data to console."""
    typer.echo(f"\n--- Frame {frame_number} @ {timestamp:.3f}s ---")
    typer.echo(f"Phase: {phase.name}")
    typer.echo(f"Round: {round_manager.current_round}")
    typer.echo(f"Score: {round_manager.current_score['team1']}-{round_manager.current_score['team2']}")

    # Get timers
    timers = timer_manager.get_timers(timestamp, phase, None)
    if timers["game_timer"] is not None:
        typer.echo(f"Game Timer: {timers['game_timer']:.1f}s")
    if timers["spike_timer"] is not None:
        typer.echo(f"Spike Timer: {timers['spike_timer']:.1f}s")
    if timers["post_round_timer"] is not None:
        typer.echo(f"Post-Round Timer: {timers['post_round_timer']:.1f}s")
    typer.echo("")


def _create_display_frame(
    frame: np.ndarray,
    timestamp: float,
    frame_number: int,
    phase: Phase,
    round_manager,
    timer_manager,
) -> np.ndarray:
    """Create display frame with overlay information."""
    display = frame.copy()

    # Add overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background for text
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (600, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Text
    y_offset = 40
    line_height = 30

    cv2.putText(
        display,
        f"Frame: {frame_number} | Time: {timestamp:.2f}s",
        (20, y_offset),
        font,
        0.6,
        (255, 255, 255),
        2,
    )
    y_offset += line_height

    cv2.putText(
        display,
        f"Phase: {phase.name}",
        (20, y_offset),
        font,
        0.6,
        (255, 255, 255),
        2,
    )
    y_offset += line_height

    cv2.putText(
        display,
        f"Round: {round_manager.current_round} | Score: {round_manager.current_score['team1']}-{round_manager.current_score['team2']}",
        (20, y_offset),
        font,
        0.6,
        (255, 255, 255),
        2,
    )
    y_offset += line_height

    # Timers
    timers = timer_manager.get_timers(timestamp, phase, None)
    timer_text = []
    if timers["game_timer"] is not None:
        timer_text.append(f"Game: {timers['game_timer']:.1f}s")
    if timers["spike_timer"] is not None:
        timer_text.append(f"Spike: {timers['spike_timer']:.1f}s")
    if timers["post_round_timer"] is not None:
        timer_text.append(f"Post: {timers['post_round_timer']:.1f}s")

    if timer_text:
        cv2.putText(
            display,
            " | ".join(timer_text),
            (20, y_offset),
            font,
            0.6,
            (100, 255, 100),
            2,
        )

    return display
