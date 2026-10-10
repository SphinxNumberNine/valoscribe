"""Main orchestrator for processing Valorant VOD frames."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np

from valoscribe.orchestration.phase_detector import PhaseDetector, Phase
from valoscribe.orchestration.round_manager import RoundManager
from valoscribe.orchestration.player_state_tracker import PlayerStateTracker
from valoscribe.orchestration.detector_registry import DetectorRegistry
from valoscribe.orchestration.state_validator import StateValidator
from valoscribe.orchestration.event_collector import EventCollector
from valoscribe.orchestration.output_writer import OutputWriter
from valoscribe.orchestration.timer_manager import TimerManager
from valoscribe.video.reader import VideoReader
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class GameStateManager:
    """Main orchestrator for processing Valorant VOD frames."""

    def __init__(
        self,
        video_path: Path,
        vlr_metadata: dict,
        output_dir: Path,
        config_path: Optional[Path] = None,
        fps: float = 4.0,
        debug_player_filter: Optional[int] = None,
    ):
        """
        Initialize GameStateManager.

        Args:
            video_path: Path to VOD file
            vlr_metadata: VLR scraper output with team/player info
            output_dir: Directory for output files
            config_path: Optional path to HUD config.yaml
            fps: Target FPS for processing (default: 4.0)
            debug_player_filter: Optional player index to filter debug logs (0-9)
        """
        self.video_path = Path(video_path)
        self.vlr_metadata = vlr_metadata
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.debug_player_filter = debug_player_filter

        # Initialize components
        self._init_components(config_path)

        # Internal state
        self.current_phase = None  # Will be set to first detected game phase
        self.frame_count = 0

        # Match tracking
        self.match_started = False  # Track if match_start event has been emitted
        self.match_ended = False  # Track if match_end event has been emitted
        self.frames_after_match_end = 0  # Counter for frames processed after match end

        # Player trackers - initialized lazily on first preround
        self.player_trackers: Optional[list[PlayerStateTracker]] = None

        # Team composition cache (for revival validation)
        self.team_has_sage_cache: dict[str, bool] = {}  # team_name -> has_sage

        log.info(f"GameStateManager initialized: {self.video_path.name} @ {fps}fps")
        if debug_player_filter is not None:
            log.info(f"Debug output filtered to player {debug_player_filter}")

    def _init_components(self, config_path: Optional[Path]) -> None:
        """Initialize all orchestration components."""
        # Detector registry (initializes all detectors without OCR usage in GameStateManager)
        self.detector_registry = DetectorRegistry(config_path=config_path)

        # Phase detector (not in registry, needs specific detectors)
        self.phase_detector = PhaseDetector(
            timer_detector=self.detector_registry.timer_detector,
            spike_detector=self.detector_registry.spike_detector,
            score_detector=self.detector_registry.score_detector,
            credits_detector=self.detector_registry.preround_credits_detector,
        )

        # Round manager
        self.round_manager = RoundManager(vlr_metadata=self.vlr_metadata)

        # Player state trackers will be initialized on first preround

        # State validator
        self.state_validator = StateValidator()

        # Event collector
        self.event_collector = EventCollector(dedup_window_sec=5.0)

        # Timer manager
        self.timer_manager = TimerManager()

        # Output writer
        self.output_writer = OutputWriter(output_dir=self.output_dir)

        log.info("All components initialized successfully")

    def _initialize_player_trackers_from_agents(
        self, detected_agents: list[str], detected_sides: list[str]
    ) -> None:
        """
        Initialize player state trackers from detected agents in scoreboard order.

        Matches teams by using side detection (attack/defense) to determine which
        team is on which side of the screen, then matches to VLR starting_side metadata.

        Args:
            detected_agents: List of 10 agent names in scoreboard order (positions 0-9)
            detected_sides: List of 10 sides ("attack"/"defense") in scoreboard order
        """
        if len(detected_agents) != 10:
            log.warning(f"Expected 10 agents, got {len(detected_agents)}. Cannot initialize player trackers.")
            return

        if len(detected_sides) != 10:
            log.warning(f"Expected 10 sides, got {len(detected_sides)}. Cannot initialize player trackers.")
            return

        # Split detected agents and sides by scoreboard side
        left_agents = detected_agents[0:5]   # Positions 0-4
        right_agents = detected_agents[5:10]  # Positions 5-9
        left_sides = detected_sides[0:5]
        right_sides = detected_sides[5:10]

        # Count attack/defense detections on each screen side
        left_attack_count = left_sides.count("attack")
        left_defense_count = left_sides.count("defense")
        right_attack_count = right_sides.count("attack")
        right_defense_count = right_sides.count("defense")

        log.info(
            f"Side detection counts - Left: {left_attack_count} atk / {left_defense_count} def, "
            f"Right: {right_attack_count} atk / {right_defense_count} def"
        )

        # Determine which screen side is attacking/defending
        # Majority vote on each side
        left_side_is_attack = left_attack_count > left_defense_count
        right_side_is_attack = right_attack_count > right_defense_count

        log.info(
            f"Screen side determination: Left={'ATTACK' if left_side_is_attack else 'DEFENSE'}, "
            f"Right={'ATTACK' if right_side_is_attack else 'DEFENSE'}"
        )

        # Group VLR players by team and get team metadata
        vlr_teams = {}  # team_name -> list of players
        team_metadata = {}  # team_name -> team info (including starting_side)

        for player in self.vlr_metadata["players"]:
            team_name = player["team"]
            if team_name not in vlr_teams:
                vlr_teams[team_name] = []
            vlr_teams[team_name].append(player)

        for team_info in self.vlr_metadata["teams"]:
            team_metadata[team_info["name"]] = team_info

        # Match teams to screen sides using starting_side metadata
        # First, try using side detection
        left_team = None
        right_team = None

        # Find which VLR team starts on attack and which starts on defense
        attacking_team = None
        defending_team = None

        for team_name, team_info in team_metadata.items():
            if team_info["starting_side"] == "attack":
                attacking_team = team_name
            elif team_info["starting_side"] == "defense":
                defending_team = team_name

        if attacking_team and defending_team:
            # Match screen sides to teams based on side detection
            if left_side_is_attack:
                left_team = attacking_team
                right_team = defending_team
                log.info(f"Matched by side detection: Left (attack) = {left_team}, Right (defense) = {right_team}")
            else:
                left_team = defending_team
                right_team = attacking_team
                log.info(f"Matched by side detection: Left (defense) = {left_team}, Right (attack) = {right_team}")
        else:
            # Fallback: match by agent composition (old logic)
            log.warning("Could not determine teams from starting_side, falling back to agent matching")

            def find_matching_team(side_agents: list[str], vlr_teams: dict) -> tuple[str, list]:
                """Find which team has the most agent matches for this side."""
                best_team = None
                best_matches = 0
                best_players = []

                for team_name, team_players in vlr_teams.items():
                    team_agents = {p["agent"] for p in team_players}
                    matches = len(set(side_agents) & team_agents)

                    if matches > best_matches:
                        best_matches = matches
                        best_team = team_name
                        best_players = team_players

                return best_team, best_players

            left_team, _ = find_matching_team(left_agents, vlr_teams)
            right_team, _ = find_matching_team(right_agents, vlr_teams)
            log.info(f"Matched by agent composition: Left = {left_team}, Right = {right_team}")

        left_vlr_players = vlr_teams[left_team]
        right_vlr_players = vlr_teams[right_team]

        log.info(f"Team matching: Left side (0-4) = {left_team}, Right side (5-9) = {right_team}")

        # CRITICAL: Reorder round_manager.team_names to match detected screen positions
        # team_names[0] should be left team, team_names[1] should be right team
        # This ensures round_win events are attributed to the correct team
        if self.round_manager.team_names[0] != left_team:
            log.warning(
                f"VLR metadata team order does not match screen positions! "
                f"Reordering: {self.round_manager.team_names} -> [{left_team}, {right_team}]"
            )
            self.round_manager.team_names = [left_team, right_team]

            # Also need to swap starting sides to match the new team order
            old_starting_sides = self.round_manager.starting_sides.copy()
            # Find which original team corresponds to the new team1/team2
            for team_data in self.vlr_metadata["teams"]:
                if team_data["name"] == left_team:
                    self.round_manager.starting_sides["team1"] = team_data["starting_side"]
                elif team_data["name"] == right_team:
                    self.round_manager.starting_sides["team2"] = team_data["starting_side"]

            log.info(
                f"Updated starting sides: {old_starting_sides} -> {self.round_manager.starting_sides}"
            )

        # Create trackers with team-aware matching
        trackers = []

        for scoreboard_idx, agent_name in enumerate(detected_agents):
            # Determine which side this position is on
            if scoreboard_idx < 5:
                team_name = left_team
                vlr_players = left_vlr_players
            else:
                team_name = right_team
                vlr_players = right_vlr_players

            # Find matching player by agent within this team
            matched_player = None
            for vlr_player in vlr_players:
                if vlr_player["agent"] == agent_name:
                    matched_player = vlr_player
                    break

            if matched_player:
                metadata = {
                    "name": matched_player["name"],
                    "team": matched_player["team"],
                    "agent": agent_name,
                }
                log.debug(f"Matched position {scoreboard_idx} ({agent_name}) → {matched_player['name']} ({team_name})")
            else:
                # Agent not found in expected team - create placeholder
                metadata = {
                    "name": f"Unknown_{agent_name}",
                    "team": team_name or ("team1" if scoreboard_idx < 5 else "team2"),
                    "agent": agent_name,
                }
                log.warning(f"Could not match {agent_name} to {team_name} in VLR metadata, using placeholder")

            trackers.append(PlayerStateTracker(
                player_index=scoreboard_idx,
                metadata=metadata,
                agent_config=self.state_validator.agent_config,
            ))

        self.player_trackers = trackers
        log.info(f"Initialized {len(trackers)} player trackers from detected agents")

        # Populate Sage cache for revival validation
        self._populate_sage_cache()

        # Log the final mapping
        for idx, tracker in enumerate(trackers):
            side_label = "LEFT" if idx < 5 else "RIGHT"
            log.info(f"  Position {idx} ({side_label}): {tracker.metadata['name']} - {tracker.metadata['agent']} ({tracker.metadata['team']})")

    def process_video(self) -> None:
        """Main processing loop - process entire video."""
        log.info(f"Starting video processing: {self.video_path.name}")

        # Create video reader with fps filter
        video_reader = VideoReader(str(self.video_path), fps_filter=self.fps)

        # Context manager ensures files are closed
        with self.output_writer:
            try:
                # Read frames at target FPS using iterator
                for frame_info in video_reader:
                    self.process_frame(frame_info.timestamp_sec, frame_info.frame)
            except StopIteration:
                # Match ended, exit early
                log.info("Early exit: Match ended")

            # Finalize: check for missing round_end and match_end events
            self._finalize_events()

        log.info(f"Video processing complete: {self.frame_count} frames processed")

    def process_frame(self, timestamp: float, frame: np.ndarray) -> None:
        """Process a single frame."""
        with self.detector_registry.cropper.frame_scope(frame):
            self._process_frame(timestamp, frame)

    def _process_frame(self, timestamp: float, frame: np.ndarray) -> None:
        """Process a frame while its detector crops are shared."""
        self.frame_count += 1

        # Check if match has ended - process 5 more frames then stop
        if self.match_ended:
            self.frames_after_match_end += 1
            if self.frames_after_match_end > 5:
                log.info(f"Match ended 5 frames ago, stopping processing @ {timestamp:.2f}s")
                # Signal to stop processing by raising StopIteration
                raise StopIteration

        # 1. Detect phase
        phase, phase_detections = self.phase_detector.detect_phase(
            frame, self.current_phase
        )

        # Skip NON_GAME frames entirely - don't process, don't update current_phase
        if phase == Phase.NON_GAME:
            log.debug(f"Skipping NON_GAME frame @ {timestamp:.2f}s")
            return

        # Handle phase transitions (only between valid game phases)
        if phase != self.current_phase:
            self._handle_phase_transition(self.current_phase, phase, timestamp)
            self.current_phase = phase

        # 2. Route detection based on phase
        if phase == Phase.PREROUND:
            self._process_preround(timestamp, frame)
        elif phase == Phase.ACTIVE_ROUND:
            self._process_active_round(timestamp, frame, phase_detections)
        elif phase == Phase.POST_ROUND:
            self._process_post_round(timestamp, frame, phase_detections)

        # 3. Write frame state
        self._write_frame_state(timestamp, frame)

        # 4. Progress feedback (every 100 frames)
        if self.frame_count % 100 == 0:
            self._print_progress(timestamp)

    def _add_timers_to_event(self, event: dict, timestamp: float, frame: np.ndarray = None) -> dict:
        """
        Add all timer values to an event.

        Args:
            event: Event dictionary to enrich
            timestamp: Current video timestamp
            frame: Optional frame for detecting game timer (if None, game_timer will be None)

        Returns:
            Event dictionary with timers added
        """
        # Get game timer from frame if provided
        game_timer = None
        if frame is not None:
            timer_info = self.detector_registry.timer_detector.detect(frame)
            game_timer = timer_info.time_seconds if timer_info else None

        # Get all timers from timer manager
        timers = self.timer_manager.get_timers(timestamp, self.current_phase, game_timer)

        # Add timers to event
        event["timers"] = {
            "game_timer": timers["game_timer"],
            "spike_timer": timers["spike_timer"],
            "post_round_timer": timers["post_round_timer"],
        }

        return event

    def _populate_sage_cache(self) -> None:
        """
        Populate the cache that tracks which teams have Sage.

        Called once after player trackers are initialized.
        """
        if self.player_trackers is None:
            return

        # Get unique team names
        teams = set(tracker.metadata.get("team") for tracker in self.player_trackers)

        # Check each team for Sage
        for team_name in teams:
            has_sage = any(
                tracker.metadata.get("team") == team_name
                and tracker.metadata.get("agent", "").lower() == "sage"
                for tracker in self.player_trackers
            )
            self.team_has_sage_cache[team_name] = has_sage

            log.debug(f"Team {team_name} has Sage: {has_sage}")

    def _team_has_sage(self, team_name: str) -> bool:
        """
        Check if a team has a Sage agent (for revival validation).

        Uses cached result populated during player tracker initialization.

        Args:
            team_name: Team name to check

        Returns:
            True if the team has a Sage, False otherwise
        """
        return self.team_has_sage_cache.get(team_name, False)

    def _handle_phase_transition(
        self, old_phase: Phase, new_phase: Phase, timestamp: float
    ) -> None:
        """Handle phase transition events and state updates."""
        old_phase_name = old_phase.name if old_phase else "None"
        log.debug(f"Phase transition: {old_phase_name} -> {new_phase.name} @ {timestamp:.2f}s")

        # Entering PREROUND: Increment round (new round starting)
        # This happens either:
        # 1. After POST_ROUND (normal case - score change detected)
        # 2. After ACTIVE_ROUND (score detection failed, but round still ended)
        # 3. At the start of the game (old_phase is None)
        if new_phase == Phase.PREROUND and (old_phase == Phase.POST_ROUND or old_phase == Phase.ACTIVE_ROUND or old_phase is None):
            # Increment round number for new round
            round_number = self.round_manager.start_new_round(timestamp)

            # Reset player states, timers, and state validator for new round
            # (not needed for first round, but safe to do)
            if self.player_trackers is not None:
                for tracker in self.player_trackers:
                    tracker.reset_for_new_round()
            self.timer_manager.reset_for_new_round()
            self.state_validator.reset_for_new_round()

            if old_phase is None:
                log.info(f"Entering preround for round {round_number} (first round)")
            elif old_phase == Phase.ACTIVE_ROUND:
                log.info(
                    f"Entering preround for round {round_number} from ACTIVE_ROUND "
                    f"(score will be detected during preround processing)"
                )
            else:
                log.info(f"Entering preround for round {round_number}")

        # PREROUND -> ACTIVE_ROUND: Round start event
        elif old_phase == Phase.PREROUND and new_phase == Phase.ACTIVE_ROUND:
            # Reset state validator to clear any pending changes from preround
            self.state_validator.reset_for_new_round()

            # Set grace period timestamp - UI fade-in happens NOW
            # Also force all players to alive=True and sync state to prevent preround events
            if self.player_trackers is not None:
                for tracker in self.player_trackers:
                    tracker.round_start_timestamp = timestamp
                    # Force alive status (players start round alive, any prior "dead" status is stale)
                    tracker.current_state["alive"] = True
                    tracker.previous_state["alive"] = True
                    # Sync previous_state to current_state to prevent preround changes from firing events
                    # (StateValidator is not called in preround, so transitions there shouldn't generate events)
                    tracker.previous_state = tracker.current_state.copy()
                    log.debug(
                        f"[STATE SYNC] Player {tracker.player_index} ({tracker.metadata.get('name')}): "
                        f"Synced state at round start"
                    )

            event = self._add_timers_to_event({
                "type": "round_start",
                "timestamp": timestamp,
                "round_number": self.round_manager.current_round,
                "score_team1": self.round_manager.current_score["team1"],
                "score_team2": self.round_manager.current_score["team2"],
            }, timestamp)
            self.event_collector.add_event(event)
            log.info(f"Round {self.round_manager.current_round} started @ {timestamp:.2f}s")

            # Emit match_start event on round 1 (only once)
            if self.round_manager.current_round == 1 and not self.match_started:
                team1_name = self.round_manager.team_names[0]
                team2_name = self.round_manager.team_names[1]
                match_start_event = self._add_timers_to_event({
                    "type": "match_start",
                    "timestamp": timestamp,
                    "team1": team1_name,
                    "team2": team2_name,
                }, timestamp)
                self.event_collector.add_event(match_start_event)
                self.match_started = True
                log.info(f"Match started: {team1_name} vs {team2_name} @ {timestamp:.2f}s")

        # Entering ACTIVE_ROUND directly (preround skipped): Increment round and fire round_start
        # This happens either:
        # 1. After POST_ROUND (preround skipped)
        # 2. At the start of the game (old_phase is None, first round skipped preround)
        elif new_phase == Phase.ACTIVE_ROUND and (old_phase == Phase.POST_ROUND or old_phase is None):
            # Increment round number (preround was skipped)
            round_number = self.round_manager.start_new_round(timestamp)

            if old_phase == Phase.POST_ROUND:
                # Reset player states
                if self.player_trackers is not None:
                    for tracker in self.player_trackers:
                        tracker.reset_for_new_round()
                # Reset timers
                self.timer_manager.reset_for_new_round()
                # Reset state validator (clear pending ability changes)
                self.state_validator.reset_for_new_round()

            # Set grace period timestamp - UI fade-in happens NOW (for all cases)
            # Also force all players to alive=True to prevent revival events during fade-in
            if self.player_trackers is not None:
                for tracker in self.player_trackers:
                    tracker.round_start_timestamp = timestamp
                    # Force alive status (players start round alive, any prior "dead" status is stale)
                    tracker.current_state["alive"] = True
                    tracker.previous_state["alive"] = True
                    log.debug(f"Player {tracker.player_index}: Grace period started, forced alive @ {timestamp:.2f}s")

            # Round start event
            event = self._add_timers_to_event({
                "type": "round_start",
                "timestamp": timestamp,
                "round_number": round_number,
                "score_team1": self.round_manager.current_score["team1"],
                "score_team2": self.round_manager.current_score["team2"],
            }, timestamp)
            self.event_collector.add_event(event)

            if old_phase is None:
                log.info(f"Round {round_number} started (first round, skipped preround) @ {timestamp:.2f}s")
            else:
                log.info(f"Round {round_number} started (skipped preround) @ {timestamp:.2f}s")

            # Emit match_start event on round 1 (only once)
            if round_number == 1 and not self.match_started:
                team1_name = self.round_manager.team_names[0]
                team2_name = self.round_manager.team_names[1]
                match_start_event = self._add_timers_to_event({
                    "type": "match_start",
                    "timestamp": timestamp,
                    "team1": team1_name,
                    "team2": team2_name,
                }, timestamp)
                self.event_collector.add_event(match_start_event)
                self.match_started = True
                log.info(f"Match started: {team1_name} vs {team2_name} @ {timestamp:.2f}s")

    def _process_preround(self, timestamp: float, frame: np.ndarray) -> None:
        """Process PREROUND phase - detect agents, abilities, ultimates."""
        # Only detect agents if player trackers not yet initialized
        if self.player_trackers is None:
            agents_detected = []
            sides_detected = []

            for player_idx in range(10):
                # Detect agent
                agent_info = self.detector_registry.preround_agent_detector.detect(frame, player_idx)

                if agent_info:
                    agents_detected.append(agent_info.agent_name)
                    sides_detected.append(agent_info.side)  # "attack" or "defense"
                else:
                    agents_detected.append(None)  # Placeholder for missing detection
                    sides_detected.append(None)

            # Initialize player trackers on first successful detection of all 10 agents
            if all(a is not None for a in agents_detected):
                log.info(f"Detected all 10 agents in scoreboard order: {agents_detected}")
                log.info(f"Detected sides: {sides_detected}")
                self._initialize_player_trackers_from_agents(agents_detected, sides_detected)
                # Reinitialize killfeed with detected agents
                self.detector_registry.reinitialize_killfeed(agents_detected)
                # Optimize agent detectors for future use (filter to only these 10 agents)
                self.detector_registry.preround_agent_detector.set_agent_filter(agents_detected)
                self.detector_registry.inround_agent_detector.set_agent_filter(agents_detected)

        # If player trackers are initialized, update abilities and ultimates
        # Map UI slot detections to actual player trackers based on agent detection
        if self.player_trackers is not None:
            for slot_idx in range(10):
                # Determine side (left = 0-4, right = 5-9)
                side = "left" if slot_idx < 5 else "right"

                # Step 1: Detect which agent is in this slot
                agent_info = self.detector_registry.preround_agent_detector.detect(frame, slot_idx)

                if not agent_info:
                    # Could not identify agent in this slot
                    log.debug(f"Preround slot {slot_idx}: Agent not identified")
                    continue

                detected_agent = agent_info.agent_name
                detected_side = agent_info.side

                # Step 2: Find the matching player tracker with this agent and side
                # PREROUND slots are ordered by scoreboard (changes each round)
                # We need to find which tracker has this agent AND is on the correct team

                # Determine which side (left/right) this slot is on
                # Left = team1, Right = team2 (this is fixed based on initial agent detection)
                slot_side = "left" if slot_idx < 5 else "right"

                # Find which team occupies this side by checking any tracker on that side
                # (We know from initialization that left=one team, right=other team)
                slot_team_name = None
                for tracker in self.player_trackers:
                    tracker_slot = tracker.player_index
                    if (tracker_slot < 5 and slot_side == "left") or (tracker_slot >= 5 and slot_side == "right"):
                        slot_team_name = tracker.metadata.get("team")
                        break

                if not slot_team_name:
                    log.warning(f"Preround slot {slot_idx}: Could not determine team for {slot_side} side")
                    continue

                # Convert team name to team_key (team1/team2) for round manager lookup
                if slot_team_name == self.round_manager.team_names[0]:
                    slot_team_key = "team1"
                elif slot_team_name == self.round_manager.team_names[1]:
                    slot_team_key = "team2"
                else:
                    log.warning(f"Preround slot {slot_idx}: Unknown team {slot_team_name}")
                    continue

                # Get the expected side for this team
                current_sides = self.round_manager.get_current_sides()
                expected_side_for_slot = current_sides[slot_team_key]

                # Verify detected visual styling matches expected role for this slot's team
                if detected_side != expected_side_for_slot:
                    log.warning(
                        f"Preround slot {slot_idx}: Detected {detected_side} styling for agent {detected_agent}, "
                        f"but expected {expected_side_for_slot} for {slot_team_name} ({slot_team_key})"
                    )
                    continue

                # Now find the tracker with this agent on this team
                # (PREROUND slots are NOT fixed - ordered by scoreboard rank)
                matching_tracker = None
                for tracker in self.player_trackers:
                    if tracker.metadata.get("agent") == detected_agent:
                        # Verify this player is on the correct team (based on side)
                        tracker_team = tracker.metadata.get("team")
                        if tracker_team == slot_team_name:
                            matching_tracker = tracker
                            break

                if not matching_tracker:
                    log.warning(
                        f"Preround slot {slot_idx}: Detected agent {detected_agent} ({detected_side}) "
                        f"but could not find matching player tracker from {slot_team_name}"
                    )
                    continue

                # Step 3: Detect abilities and ultimate for this slot
                # Note: In preround, ability charge increases are allowed (buying utility)
                # so we don't validate transitions via StateValidator
                detections = {}

                # Detect abilities - returns dict of ability_name -> AbilityInfo
                ability_dict = self.detector_registry.preround_ability_detector.detect_player_abilities(
                    frame, slot_idx, side
                )
                if ability_dict:
                    for ability_name, ability_info in ability_dict.items():
                        if ability_info is not None:
                            detections[ability_name] = ability_info

                # Detect ultimate - returns tuple of (UltimateInfo, white_pixel_ratio)
                ultimate_result = self.detector_registry.preround_ultimate_detector.detect_ultimate(
                    frame, slot_idx, side
                )
                if ultimate_result is not None:
                    ultimate_info, _ = ultimate_result
                    detections["ultimate"] = ultimate_info
                    if self.debug_player_filter is None or self.debug_player_filter == matching_tracker.player_index:
                        log.debug(
                            f"Preround slot {slot_idx} -> Player {matching_tracker.player_index} "
                            f"ultimate = {ultimate_info.charges} charges (full: {ultimate_info.is_full})"
                        )

                # Step 4: Update the matched player tracker (no event generation in preround)
                if detections:
                    matching_tracker.update(detections, timestamp, allow_charge_increase=True)

        # Check for score changes during preround (indicates previous round ended)
        # This handles cases where score detection failed during ACTIVE_ROUND→PREROUND transition
        score_info = self.detector_registry.score_detector.detect(frame)
        if score_info:
            new_score = {
                "team1": score_info.team1_score,
                "team2": score_info.team2_score,
            }

            # Check if score changed from last known score
            old_score = self.round_manager.current_score
            if new_score != old_score:
                # Determine winner by comparing scores
                winner = None
                if new_score["team1"] > old_score["team1"]:
                    winner = "team1"
                elif new_score["team2"] > old_score["team2"]:
                    winner = "team2"

                if winner:
                    # Fire retroactive round_end event with timestamp of most recent event
                    all_events = self.event_collector.get_all_events()
                    if all_events:
                        last_event_timestamp = all_events[-1]["timestamp"]
                    else:
                        last_event_timestamp = timestamp

                    # Convert winner from "team1"/"team2" to actual team name
                    winner_name = self.round_manager.team_names[0 if winner == "team1" else 1]

                    event = self._add_timers_to_event({
                        "type": "round_end",
                        "timestamp": last_event_timestamp,
                        "round_number": self.round_manager.current_round - 1,  # Previous round
                        "winner": winner_name,
                        "score_team1": new_score["team1"],
                        "score_team2": new_score["team2"],
                    }, timestamp, frame)
                    self.event_collector.add_event(event)

                    # Update current score in round manager
                    self.round_manager.current_score = new_score.copy()

                    log.info(
                        f"PREROUND score change detected @ {timestamp:.2f}s: "
                        f"{old_score['team1']}-{old_score['team2']} → {new_score['team1']}-{new_score['team2']} "
                        f"(round {self.round_manager.current_round - 1} ended, {winner} wins, "
                        f"event timestamp: {last_event_timestamp:.2f}s)"
                    )

    def _process_active_round(
        self, timestamp: float, frame: np.ndarray, phase_detections: dict
    ) -> None:
        """Process ACTIVE_ROUND phase - detect player states and killfeed."""
        # Skip if player trackers not initialized yet
        if self.player_trackers is None:
            log.warning("Skipping active round processing - player trackers not initialized")
            return

        # Check for spike plant
        spike_info = phase_detections.get("spike")
        if spike_info and spike_info.spike_planted and self.timer_manager.spike_planted_at is None:
            # Spike just planted
            self.timer_manager.on_spike_planted(timestamp)
            event = self._add_timers_to_event({
                "type": "spike_plant",
                "timestamp": timestamp,
            }, timestamp, frame)
            self.event_collector.add_event(event)
            log.info(f"Spike planted @ {timestamp:.2f}s")

        # Detect states by searching for each player's agent icon
        # Loop through all player trackers and find them in the UI
        for player_tracker in self.player_trackers:
            # Skip if already dead (optimization - dead players stay dead until revival)
            # But still process during grace period in case UI is fading in
            in_grace_period = False
            if player_tracker.round_start_timestamp is not None:
                time_since_round_start = timestamp - player_tracker.round_start_timestamp
                in_grace_period = time_since_round_start < player_tracker.ROUND_START_GRACE_PERIOD

            if not player_tracker.current_state["alive"] and not in_grace_period:
                continue

            # Search for this player's agent icon in UI slots
            found_slot = self._find_player_agent_slot(frame, player_tracker)

            if found_slot is not None:
                # Agent icon found - reset missing counter
                player_tracker.frames_icon_missing = 0
                player_tracker.last_detected_slot = found_slot

                # Determine side for detection
                side = "left" if found_slot < 5 else "right"

                # Gather all detections from this slot
                detections = {}

                # Detect health
                health = self.detector_registry.health_detector.detect(frame, found_slot, side)
                detections["health"] = health

                # Detect armor
                armor = self.detector_registry.armor_detector.detect(frame, found_slot, side)
                if armor is not None:
                    detections["armor"] = armor

                # Detect abilities
                ability_dict = self.detector_registry.inround_ability_detector.detect_player_abilities(
                    frame, found_slot, side
                )
                if ability_dict:
                    for ability_name, ability_info in ability_dict.items():
                        if ability_info is not None:
                            detections[ability_name] = ability_info

                # Detect ultimate
                ultimate_result = self.detector_registry.inround_ultimate_detector.detect_ultimate(
                    frame, found_slot, side
                )
                if ultimate_result is not None:
                    ultimate_info, _ = ultimate_result
                    detections["ultimate"] = ultimate_info

                # Update player tracker
                player_tracker.update(detections, timestamp)

                # Debug logging
                if self.debug_player_filter is None or self.debug_player_filter == player_tracker.player_index:
                    log.debug(
                        f"Player {player_tracker.player_index} ({player_tracker.metadata.get('name')}) "
                        f"found in slot {found_slot}, detections: {detections}"
                    )
            else:
                # Agent icon not found anywhere
                player_tracker.frames_icon_missing += 1

                log.debug(
                    f"Player {player_tracker.player_index} ({player_tracker.metadata.get('name')}) "
                    f"agent icon not found (missing {player_tracker.frames_icon_missing} frames)"
                )

                # Mark as dead if icon missing for threshold frames (but NOT during grace period)
                if player_tracker.frames_icon_missing >= 2 and not in_grace_period:
                    log.info(
                        f"Player {player_tracker.player_index} ({player_tracker.metadata.get('name')}) "
                        f"agent icon disappeared for {player_tracker.frames_icon_missing} frames → marking as dead"
                    )
                    # Update with None health to trigger death detection
                    player_tracker.update({"health": None}, timestamp)

            # Check for state changes and generate events
            if player_tracker.is_death_transition():
                event = self._add_timers_to_event({
                    "type": "death",
                    "timestamp": timestamp,
                    "player": player_tracker.metadata.get("name"),
                    "team": player_tracker.metadata.get("team"),
                    "agent": player_tracker.metadata.get("agent"),
                    "player_index": player_tracker.player_index,
                }, timestamp, frame)
                self.event_collector.add_event(event)

            # Validate state changes for ability/ultimate events
            current_state = player_tracker.current_state
            previous_state = player_tracker.previous_state
            team_name = player_tracker.metadata.get("team")
            validation_events = self.state_validator.validate_player_state(
                current=current_state,
                previous=previous_state,
                metadata=player_tracker.metadata,
                timestamp=timestamp,
                team_has_sage=self._team_has_sage(team_name),
            )

            for event in validation_events:
                event_with_timers = self._add_timers_to_event(event, timestamp, frame)
                self.event_collector.add_event(event_with_timers)

        # Detect killfeed - returns ALL candidate combinations tagged with entry index
        # Format: list of (entry_idx, KillfeedAgentDetection)
        killfeed_candidates = self.detector_registry.killfeed_detector.detect(frame)

        # Log all candidates before validation
        if killfeed_candidates:
            log.debug(f"Killfeed candidates @ {timestamp:.2f}s:")
            victim_counts = {}
            for entry_idx, candidate in killfeed_candidates:
                victim_key = f"{candidate.victim_agent} ({candidate.victim_side})"
                victim_counts[victim_key] = victim_counts.get(victim_key, 0) + 1
                log.debug(
                    f"  Entry {entry_idx}: {candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) [conf: {candidate.confidence:.3f}]"
                )

            # Warn if same victim appears in multiple candidates
            for victim_key, count in victim_counts.items():
                if count > 1:
                    log.debug(f"Same victim appears in {count} candidates: {victim_key}")

        # Validate candidates in order, accept first valid one PER ENTRY
        valid_kills = []
        rejected_count = 0
        accepted_entries = set()  # Track which entries already have accepted kills

        for entry_idx, candidate in killfeed_candidates:
            # Skip if we already accepted a kill for this entry
            if entry_idx in accepted_entries:
                log.debug(
                    f"Skipping entry {entry_idx} candidate (already accepted): "
                    f"{candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) "
                    f"[conf: {candidate.confidence:.3f}]"
                )
                continue

            if self._validate_kill_event(candidate):
                # Mark victim as killed (prevents duplicate kills)
                self._mark_victim_as_killed(candidate)

                # Verify the killer was set correctly
                victim_player = self._find_victim_player(candidate)
                if victim_player:
                    actual_killer = victim_player.current_state.get("killer")
                    if actual_killer != candidate.killer_agent:
                        log.error(
                            f"BUG: Killer not set correctly! Expected {candidate.killer_agent}, "
                            f"got {actual_killer} for victim {candidate.victim_agent}"
                        )

                valid_kills.append(candidate)
                accepted_entries.add(entry_idx)
                log.debug(
                    f"Valid kill (entry {entry_idx}): {candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) "
                    f"[conf: {candidate.confidence:.3f}]"
                )
            else:
                rejected_count += 1
                log.debug(
                    f"Rejected kill (entry {entry_idx}): {candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) "
                    f"[conf: {candidate.confidence:.3f}]"
                )

        if rejected_count > 0:
            log.debug(f"Rejected {rejected_count} invalid kill candidates")

        # Enrich valid kills with player information (name, team) and timers
        enriched_kills = []

        # Get timers for this frame (calculate once for all kills)
        timer_info = self.detector_registry.timer_detector.detect(frame)
        game_timer = timer_info.time_seconds if timer_info else None
        timers = self.timer_manager.get_timers(timestamp, self.current_phase, game_timer)

        for kill_detection in valid_kills:
            # Find killer and victim players
            killer_player = self._find_killer_player(kill_detection)
            victim_player = self._find_victim_player(kill_detection)

            # Build player info dict
            player_info = {}
            if killer_player:
                player_info["killer_name"] = killer_player.metadata.get("name")
                player_info["killer_team"] = killer_player.metadata.get("team")
            if victim_player:
                player_info["victim_name"] = victim_player.metadata.get("name")
                player_info["victim_team"] = victim_player.metadata.get("team")

            # Add timers to player_info
            player_info["timers"] = {
                "game_timer": timers["game_timer"],
                "spike_timer": timers["spike_timer"],
                "post_round_timer": timers["post_round_timer"],
            }

            enriched_kills.append((kill_detection, player_info))

        num_kills = self.event_collector.add_killfeed_events(timestamp, enriched_kills)
        if num_kills > 0:
            log.debug(f"Accepted {num_kills} valid kills after deduplication @ {timestamp:.2f}s")

        # Debug: Print summary of all player states every 25 frames (every ~6 seconds at 4fps)
        if self.frame_count % 25 == 0:
            log.debug("=" * 80)
            if self.debug_player_filter is None:
                log.debug(f"ACTIVE ROUND PLAYER STATE SUMMARY @ {timestamp:.2f}s")
                log.debug("=" * 80)
                for idx, tracker in enumerate(self.player_trackers):
                    state = tracker.current_state
                    log.debug(
                        f"Player {idx} ({tracker.metadata.get('name', '?'):12s} - {tracker.metadata.get('agent', '?'):10s}): "
                        f"alive={state['alive']:5} | hp={str(state['health']):>3} | armor={str(state['armor']):>3} | "
                        f"ab1={str(state['ability_1']):>4} ab2={str(state['ability_2']):>4} ab3={str(state['ability_3']):>4} | "
                        f"ult={state['ultimate']}"
                    )
            else:
                log.debug(f"PLAYER {self.debug_player_filter} STATE @ {timestamp:.2f}s")
                log.debug("=" * 80)
                tracker = self.player_trackers[self.debug_player_filter]
                state = tracker.current_state
                log.debug(
                    f"Player {self.debug_player_filter} ({tracker.metadata.get('name', '?'):12s} - {tracker.metadata.get('agent', '?'):10s}): "
                    f"alive={state['alive']:5} | hp={str(state['health']):>3} | armor={str(state['armor']):>3} | "
                    f"ab1={str(state['ability_1']):>4} ab2={str(state['ability_2']):>4} ab3={str(state['ability_3']):>4} | "
                    f"ult={state['ultimate']}"
                )
            log.debug("=" * 80)

    def _process_post_round(
        self, timestamp: float, frame: np.ndarray, phase_detections: dict
    ) -> None:
        """Process POST_ROUND phase - detect score changes, round winner, and player states."""
        # Skip if player trackers not initialized yet
        if self.player_trackers is None:
            log.warning("Skipping post round processing - player trackers not initialized")
            return

        # Detect score from scoreboard and infer winner
        # Note: We add the round_end event at the END to ensure it fires after all other events
        score_info = phase_detections.get("score")
        round_end_data = None  # Store round end info to add later

        if score_info:
            new_score = {
                "team1": score_info.team1_score,
                "team2": score_info.team2_score,
            }

            # Infer round end from score change
            winner = self.round_manager.infer_round_end(new_score, timestamp)

            if winner:
                # Mark round end in timer manager
                if self.timer_manager.round_ended_at is None:
                    self.timer_manager.on_round_ended(timestamp)

                # Store round end data to add event later (at the end of processing)
                round_end_data = {
                    "winner": winner,
                    "score": new_score,
                    "round_number": self.round_manager.current_round,
                }
                log.info(
                    f"Round {self.round_manager.current_round} ended @ {timestamp:.2f}s: "
                    f"{winner} wins ({new_score['team1']}-{new_score['team2']})"
                )

        # Detect states by searching for each player's agent icon
        # Loop through all player trackers and find them in the UI
        for player_tracker in self.player_trackers:
            # Check if in grace period (for clutch mode death detection)
            in_grace_period = False
            if player_tracker.round_start_timestamp is not None:
                time_since_round_start = timestamp - player_tracker.round_start_timestamp
                in_grace_period = time_since_round_start < player_tracker.ROUND_START_GRACE_PERIOD

            # Skip if already dead (optimization)
            if not player_tracker.current_state["alive"]:
                continue

            # Search for this player's agent icon in UI slots
            found_slot = self._find_player_agent_slot(frame, player_tracker)

            if found_slot is not None:
                # Agent icon found - reset missing counter
                player_tracker.frames_icon_missing = 0
                player_tracker.last_detected_slot = found_slot

                # Determine side for detection
                side = "left" if found_slot < 5 else "right"

                # Gather all detections from this slot
                detections = {}

                # Detect health
                health = self.detector_registry.health_detector.detect(frame, found_slot, side)
                detections["health"] = health

                # Detect armor
                armor = self.detector_registry.armor_detector.detect(frame, found_slot, side)
                if armor is not None:
                    detections["armor"] = armor

                # Detect abilities
                ability_dict = self.detector_registry.inround_ability_detector.detect_player_abilities(
                    frame, found_slot, side
                )
                if ability_dict:
                    for ability_name, ability_info in ability_dict.items():
                        if ability_info is not None:
                            detections[ability_name] = ability_info

                # Detect ultimate
                ultimate_result = self.detector_registry.inround_ultimate_detector.detect_ultimate(
                    frame, found_slot, side
                )
                if ultimate_result is not None:
                    ultimate_info, _ = ultimate_result
                    detections["ultimate"] = ultimate_info

                # Update player tracker
                player_tracker.update(detections, timestamp)

                # Debug logging
                if self.debug_player_filter is None or self.debug_player_filter == player_tracker.player_index:
                    log.debug(
                        f"Player {player_tracker.player_index} ({player_tracker.metadata.get('name')}) "
                        f"found in slot {found_slot}, detections: {detections}"
                    )
            else:
                # Agent icon not found anywhere
                player_tracker.frames_icon_missing += 1

                log.debug(
                    f"Player {player_tracker.player_index} ({player_tracker.metadata.get('name')}) "
                    f"agent icon not found (missing {player_tracker.frames_icon_missing} frames)"
                )

                # Mark as dead if icon missing for threshold frames (but NOT during grace period)
                if player_tracker.frames_icon_missing >= 2 and not in_grace_period:
                    log.info(
                        f"Player {player_tracker.player_index} ({player_tracker.metadata.get('name')}) "
                        f"agent icon disappeared for {player_tracker.frames_icon_missing} frames → marking as dead"
                    )
                    # Update with None health to trigger death detection
                    player_tracker.update({"health": None}, timestamp)

            # Check for death transition and fire event
            if player_tracker.is_death_transition():
                event = self._add_timers_to_event({
                    "type": "death",
                    "timestamp": timestamp,
                    "player": player_tracker.metadata.get("name"),
                    "team": player_tracker.metadata.get("team"),
                    "agent": player_tracker.metadata.get("agent"),
                    "player_index": player_tracker.player_index,
                }, timestamp, frame)
                self.event_collector.add_event(event)

            # Validate state changes for ability/ultimate events
            current_state = player_tracker.current_state
            previous_state = player_tracker.previous_state
            team_name = player_tracker.metadata.get("team")
            validation_events = self.state_validator.validate_player_state(
                current=current_state,
                previous=previous_state,
                metadata=player_tracker.metadata,
                timestamp=timestamp,
                team_has_sage=self._team_has_sage(team_name),
            )

            for event in validation_events:
                event_with_timers = self._add_timers_to_event(event, timestamp, frame)
                self.event_collector.add_event(event_with_timers)

        # Detect killfeed - returns ALL candidate combinations tagged with entry index
        # Format: list of (entry_idx, KillfeedAgentDetection)
        killfeed_candidates = self.detector_registry.killfeed_detector.detect(frame)

        # Log all candidates before validation
        if killfeed_candidates:
            log.debug(f"Killfeed candidates @ {timestamp:.2f}s:")
            victim_counts = {}
            for entry_idx, candidate in killfeed_candidates:
                victim_key = f"{candidate.victim_agent} ({candidate.victim_side})"
                victim_counts[victim_key] = victim_counts.get(victim_key, 0) + 1
                log.debug(
                    f"  Entry {entry_idx}: {candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) [conf: {candidate.confidence:.3f}]"
                )

            # Warn if same victim appears in multiple candidates
            for victim_key, count in victim_counts.items():
                if count > 1:
                    log.debug(f"Same victim appears in {count} candidates: {victim_key}")

        # Validate candidates in order, accept first valid one PER ENTRY
        valid_kills = []
        rejected_count = 0
        accepted_entries = set()  # Track which entries already have accepted kills

        for entry_idx, candidate in killfeed_candidates:
            # Skip if we already accepted a kill for this entry
            if entry_idx in accepted_entries:
                log.debug(
                    f"Skipping entry {entry_idx} candidate (already accepted): "
                    f"{candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) "
                    f"[conf: {candidate.confidence:.3f}]"
                )
                continue

            if self._validate_kill_event(candidate):
                # Mark victim as killed (prevents duplicate kills)
                self._mark_victim_as_killed(candidate)

                # Verify the killer was set correctly
                victim_player = self._find_victim_player(candidate)
                if victim_player:
                    actual_killer = victim_player.current_state.get("killer")
                    if actual_killer != candidate.killer_agent:
                        log.error(
                            f"BUG: Killer not set correctly! Expected {candidate.killer_agent}, "
                            f"got {actual_killer} for victim {candidate.victim_agent}"
                        )

                valid_kills.append(candidate)
                accepted_entries.add(entry_idx)
                log.debug(
                    f"Valid kill (entry {entry_idx}): {candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) "
                    f"[conf: {candidate.confidence:.3f}]"
                )
            else:
                rejected_count += 1
                log.debug(
                    f"Rejected kill (entry {entry_idx}): {candidate.killer_agent} ({candidate.killer_side}) -> "
                    f"{candidate.victim_agent} ({candidate.victim_side}) "
                    f"[conf: {candidate.confidence:.3f}]"
                )

        if rejected_count > 0:
            log.debug(f"Rejected {rejected_count} invalid kill candidates")

        # Enrich valid kills with player information (name, team) and timers
        enriched_kills = []

        # Get timers for this frame (calculate once for all kills)
        timer_info = self.detector_registry.timer_detector.detect(frame)
        game_timer = timer_info.time_seconds if timer_info else None
        timers = self.timer_manager.get_timers(timestamp, self.current_phase, game_timer)

        for kill_detection in valid_kills:
            # Find killer and victim players
            killer_player = self._find_killer_player(kill_detection)
            victim_player = self._find_victim_player(kill_detection)

            # Build player info dict
            player_info = {}
            if killer_player:
                player_info["killer_name"] = killer_player.metadata.get("name")
                player_info["killer_team"] = killer_player.metadata.get("team")
            if victim_player:
                player_info["victim_name"] = victim_player.metadata.get("name")
                player_info["victim_team"] = victim_player.metadata.get("team")

            # Add timers to player_info
            player_info["timers"] = {
                "game_timer": timers["game_timer"],
                "spike_timer": timers["spike_timer"],
                "post_round_timer": timers["post_round_timer"],
            }

            enriched_kills.append((kill_detection, player_info))

        num_kills = self.event_collector.add_killfeed_events(timestamp, enriched_kills)
        if num_kills > 0:
            log.debug(f"Accepted {num_kills} valid kills after deduplication @ {timestamp:.2f}s")

        # Debug: Print summary of all player states every 25 frames (every ~6 seconds at 4fps)
        if self.frame_count % 25 == 0:
            log.debug("=" * 80)
            if self.debug_player_filter is None:
                log.debug(f"POST ROUND PLAYER STATE SUMMARY @ {timestamp:.2f}s")
                log.debug("=" * 80)
                for idx, tracker in enumerate(self.player_trackers):
                    state = tracker.current_state
                    log.debug(
                        f"Player {idx} ({tracker.metadata.get('name', '?'):12s} - {tracker.metadata.get('agent', '?'):10s}): "
                        f"alive={state['alive']:5} | hp={str(state['health']):>3} | armor={str(state['armor']):>3} | "
                        f"ab1={str(state['ability_1']):>4} ab2={str(state['ability_2']):>4} ab3={str(state['ability_3']):>4} | "
                        f"ult={state['ultimate']}"
                    )
            else:
                log.debug(f"PLAYER {self.debug_player_filter} STATE @ {timestamp:.2f}s")
                log.debug("=" * 80)
                tracker = self.player_trackers[self.debug_player_filter]
                state = tracker.current_state
                log.debug(
                    f"Player {self.debug_player_filter} ({tracker.metadata.get('name', '?'):12s} - {tracker.metadata.get('agent', '?'):10s}): "
                    f"alive={state['alive']:5} | hp={str(state['health']):>3} | armor={str(state['armor']):>3} | "
                    f"ab1={str(state['ability_1']):>4} ab2={str(state['ability_2']):>4} ab3={str(state['ability_3']):>4} | "
                    f"ult={state['ultimate']}"
                )
            log.debug("=" * 80)

        # Add round_end event LAST to ensure it fires after all other events on this frame
        if round_end_data:
            # Convert winner from "team1"/"team2" to actual team name
            winner_key = round_end_data["winner"]  # "team1" or "team2"
            winner_name = self.round_manager.team_names[0 if winner_key == "team1" else 1]

            event = self._add_timers_to_event({
                "type": "round_end",
                "timestamp": timestamp,
                "round_number": round_end_data["round_number"],
                "winner": winner_name,
                "score_team1": round_end_data["score"]["team1"],
                "score_team2": round_end_data["score"]["team2"],
            }, timestamp, frame)
            self.event_collector.add_event(event)

            # Check if match has ended (one team won)
            # Winning condition: First to 13 rounds with at least 2 round lead
            score_team1 = round_end_data["score"]["team1"]
            score_team2 = round_end_data["score"]["team2"]
            max_score = max(score_team1, score_team2)
            score_diff = abs(score_team1 - score_team2)

            if max_score >= 13 and score_diff >= 2 and not self.match_ended:
                # Match is over - determine winner
                if score_team1 > score_team2:
                    match_winner = self.round_manager.team_names[0]
                else:
                    match_winner = self.round_manager.team_names[1]

                match_end_event = self._add_timers_to_event({
                    "type": "match_end",
                    "timestamp": timestamp,
                    "winner": match_winner,
                    "final_score_team1": score_team1,
                    "final_score_team2": score_team2,
                    "team1": self.round_manager.team_names[0],
                    "team2": self.round_manager.team_names[1],
                }, timestamp, frame)
                self.event_collector.add_event(match_end_event)
                self.match_ended = True
                log.info(
                    f"Match ended @ {timestamp:.2f}s: {match_winner} wins "
                    f"({score_team1}-{score_team2})"
                )

    def _find_player_agent_slot(self, frame: np.ndarray, player_tracker) -> Optional[int]:
        """
        Search UI slots on the player's side to find which slot contains their agent icon.

        Args:
            frame: Current frame
            player_tracker: PlayerStateTracker to search for

        Returns:
            Slot index (0-9) if found, None if agent icon not visible
        """
        player_agent = player_tracker.metadata.get("agent")
        if not player_agent:
            return None

        # Determine which slots to search based on player's original screen position
        # Players were assigned to indices 0-4 (left) or 5-9 (right) during initialization
        # This mapping stays consistent throughout the match
        if player_tracker.player_index < 5:
            slots_to_search = range(0, 5)  # Left side
        else:
            slots_to_search = range(5, 10)  # Right side

        # Search slots on this player's side for their agent
        detected_agents = []
        for slot_idx in slots_to_search:
            # Detect agent in this slot (color first)
            detected_agent = self.detector_registry.inround_agent_detector.detect(frame, slot_idx)
            detected_agents.append((slot_idx, detected_agent))

            if detected_agent == player_agent:
                return slot_idx

        # Try greyscale detection (for dead players)
        for slot_idx in slots_to_search:
            detected_agent = self.detector_registry.inround_agent_detector.detect(
                frame, slot_idx, greyscale=True
            )

            if detected_agent == player_agent:
                log.debug(f"Player {player_tracker.player_index} found in slot {slot_idx} via greyscale")
                return slot_idx

        # Not found - log what was detected for debugging
        log.warning(
            f"Could not find {player_agent} for player {player_tracker.player_index} ({player_tracker.metadata.get('name')}). "
            f"Detected in slots {slots_to_search.start}-{slots_to_search.stop-1}: {detected_agents}"
        )

        return None

    def _find_victim_player(self, kill_detection):
        """
        Find the player tracker for a kill victim.

        Args:
            kill_detection: KillfeedAgentDetection object

        Returns:
            PlayerStateTracker for victim, or None if not found
        """
        current_sides = self.round_manager.get_current_sides()

        for tracker in self.player_trackers:
            agent = tracker.metadata.get("agent")
            team_name = tracker.metadata.get("team")

            # Check if agent matches
            if agent != kill_detection.victim_agent:
                continue

            # Find team key for this player
            team_key = None
            if team_name == self.round_manager.team_names[0]:
                team_key = "team1"
            elif team_name == self.round_manager.team_names[1]:
                team_key = "team2"

            if team_key:
                # Get current side for this player's team
                player_side = current_sides[team_key]

                # Check if this player is on the victim's side
                if player_side == kill_detection.victim_side:
                    return tracker

        return None

    def _find_killer_player(self, kill_detection):
        """
        Find the player tracker for a killer.

        Args:
            kill_detection: KillfeedAgentDetection object

        Returns:
            PlayerStateTracker for killer, or None if not found
        """
        current_sides = self.round_manager.get_current_sides()

        for tracker in self.player_trackers:
            agent = tracker.metadata.get("agent")
            team_name = tracker.metadata.get("team")

            # Check if agent matches
            if agent != kill_detection.killer_agent:
                continue

            # Find team key for this player
            team_key = None
            if team_name == self.round_manager.team_names[0]:
                team_key = "team1"
            elif team_name == self.round_manager.team_names[1]:
                team_key = "team2"

            if team_key:
                # Get current side for this player's team
                player_side = current_sides[team_key]

                # Check if this player is on the killer's side
                if player_side == kill_detection.killer_side:
                    return tracker

        return None

    def _mark_victim_as_killed(self, kill_detection) -> None:
        """
        Mark the victim as killed by setting the killer agent.

        This is called after a kill has been validated to prevent duplicate kills
        of the same player in a round.

        Args:
            kill_detection: KillfeedAgentDetection object
        """
        victim_player = self._find_victim_player(kill_detection)

        if victim_player:
            victim_player.current_state["killer"] = kill_detection.killer_agent
            log.info(
                f"Marked {kill_detection.victim_agent} ({kill_detection.victim_side}) as "
                f"killed by {kill_detection.killer_agent} (player_index={victim_player.player_index})"
            )
        else:
            log.warning(
                f"Could not find victim player to mark as killed: "
                f"{kill_detection.victim_agent} ({kill_detection.victim_side})"
            )

    def _validate_kill_event(self, kill_detection) -> bool:
        """
        Validate a killfeed detection before accepting it as an event.

        A valid kill must satisfy:
        1. Killer's team (atk/def) has the killer agent
        2. Victim's team (atk/def) has the victim agent
        3. Killer and victim are on opposite sides (no team kills)
        4. Victim is dead in the UI (no health detected)
        5. Victim does not already have a killer (can only be killed once per round)

        Args:
            kill_detection: KillfeedAgentDetection object

        Returns:
            True if kill is valid, False otherwise
        """
        # Build team compositions from player trackers
        attack_agents = set()
        defense_agents = set()

        # Get current sides for both teams
        current_sides = self.round_manager.get_current_sides()

        for tracker in self.player_trackers:
            agent = tracker.metadata.get("agent")
            if not agent:
                continue

            # Determine side from team name and current round
            team_name = tracker.metadata.get("team")

            # Find which team this is (team1 or team2)
            team_key = None
            if team_name == self.round_manager.team_names[0]:
                team_key = "team1"
            elif team_name == self.round_manager.team_names[1]:
                team_key = "team2"

            if team_key:
                # Get current side based on round number
                current_side = current_sides[team_key]

                if current_side == "attack":
                    attack_agents.add(agent)
                elif current_side == "defense":
                    defense_agents.add(agent)

        # Validate killer's team has killer agent
        if kill_detection.killer_side == "attack":
            if kill_detection.killer_agent not in attack_agents:
                log.debug(
                    f"Killer validation failed: {kill_detection.killer_agent} not in attack team "
                    f"(attack agents: {attack_agents})"
                )
                return False
        elif kill_detection.killer_side == "defense":
            if kill_detection.killer_agent not in defense_agents:
                log.debug(
                    f"Killer validation failed: {kill_detection.killer_agent} not in defense team "
                    f"(defense agents: {defense_agents})"
                )
                return False

        # Validate victim's team has victim agent
        if kill_detection.victim_side == "attack":
            if kill_detection.victim_agent not in attack_agents:
                log.debug(
                    f"Victim validation failed: {kill_detection.victim_agent} not in attack team "
                    f"(attack agents: {attack_agents})"
                )
                return False
        elif kill_detection.victim_side == "defense":
            if kill_detection.victim_agent not in defense_agents:
                log.debug(
                    f"Victim validation failed: {kill_detection.victim_agent} not in defense team "
                    f"(defense agents: {defense_agents})"
                )
                return False

        # Validate killer and victim are on opposite sides (no team kills allowed)
        if kill_detection.killer_side == kill_detection.victim_side:
            log.debug(
                f"Kill validation failed: Killer and victim on same side "
                f"({kill_detection.killer_agent} -> {kill_detection.victim_agent}, both {kill_detection.killer_side})"
            )
            return False

        # Validate victim is dead (find player with victim agent AND on victim's side)
        victim_player = self._find_victim_player(kill_detection)

        if victim_player:
            victim_alive = victim_player.current_state["alive"]
            victim_health = victim_player.current_state.get("health")
            victim_killer = victim_player.current_state.get("killer")

            log.debug(
                f"Victim check: {kill_detection.victim_agent} ({kill_detection.victim_side}) "
                f"player_index={victim_player.player_index} - "
                f"alive={victim_alive}, health={victim_health}, killer={victim_killer}"
            )

            # Check if victim is dead (not alive or no health detected)
            if victim_alive:
                log.debug(
                    f"Victim validation failed: {kill_detection.victim_agent} ({kill_detection.victim_side}) "
                    f"player_index={victim_player.player_index} is marked alive (health: {victim_health})"
                )
                return False

            # Check if victim already has a killer (can only be killed once per round)
            if victim_killer is not None:
                log.debug(
                    f"Victim validation failed: {kill_detection.victim_agent} ({kill_detection.victim_side}) "
                    f"player_index={victim_player.player_index} already killed by {victim_killer}"
                )
                return False
        else:
            log.debug(
                f"Victim validation warning: Could not find player with agent {kill_detection.victim_agent} "
                f"on {kill_detection.victim_side} side"
            )

        # All validations passed
        return True

    def _detect_alive_players(self, frame: np.ndarray) -> list[bool]:
        """
        Detect which players are alive (from health detection heuristic).

        Args:
            frame: Current frame

        Returns:
            List of 10 booleans indicating if each player is alive
        """
        alive_list = []
        for i in range(10):
            # Determine side (left = 0-4, right = 5-9)
            side = "left" if i < 5 else "right"
            # Heuristic: If health detected, player is likely alive
            health = self.detector_registry.health_detector.detect(frame, i, side)
            alive_list.append(health is not None)

        return alive_list

    def _write_frame_state(self, timestamp: float, frame: np.ndarray) -> None:
        """Write current frame state to CSV."""
        # Detect visible game timer
        timer_info = self.detector_registry.timer_detector.detect(frame)
        game_timer = timer_info.time_seconds if timer_info else None

        # Get all timer values
        timers = self.timer_manager.get_timers(timestamp, self.current_phase, game_timer)

        # Collect player states (or empty list if not initialized)
        player_states = []
        if self.player_trackers is not None:
            for tracker in self.player_trackers:
                player_states.append({
                    "metadata": tracker.metadata,
                    "current_state": tracker.current_state,
                })

        # Write to CSV
        self.output_writer.write_frame_state(
            timestamp=timestamp,
            frame_number=self.frame_count,
            phase=self.current_phase,
            round_number=self.round_manager.current_round,
            scores=(self.round_manager.current_score["team1"], self.round_manager.current_score["team2"]),
            player_states=player_states,
            timers=timers,
        )

        # Write events (if any accumulated)
        events = self.event_collector.get_events_since_last_write()
        if events:
            self.output_writer.write_events(events)

    def _finalize_events(self) -> None:
        """
        Finalize events when video processing completes.

        If the last round started but didn't end, infer the winner and fire
        round_end and match_end events.
        """
        # Check if we have any events at all
        all_events = self.event_collector.get_all_events()
        if not all_events:
            log.warning("No events collected during processing")
            return

        # Get last event timestamp for retroactive events
        last_event_timestamp = all_events[-1]["timestamp"]

        # Count round_start and round_end events
        round_starts = sum(1 for e in all_events if e["type"] == "round_start")
        round_ends = sum(1 for e in all_events if e["type"] == "round_end")

        # Check if we're missing a round_end event
        if round_starts > round_ends:
            log.warning(
                f"Missing round_end event: {round_starts} round_start events "
                f"but only {round_ends} round_end events"
            )

            # Infer winner from current score (team with higher score won the last round)
            # and increment their score
            score_team1 = self.round_manager.current_score["team1"]
            score_team2 = self.round_manager.current_score["team2"]

            if score_team1 > score_team2:
                winner_name = self.round_manager.team_names[0]
                winner_key = "team1"
                score_team1 += 1  # Increment winner's score
            elif score_team2 > score_team1:
                winner_name = self.round_manager.team_names[1]
                winner_key = "team2"
                score_team2 += 1  # Increment winner's score
            else:
                log.error(
                    f"Cannot infer round winner: scores are tied at "
                    f"{score_team1}-{score_team2}"
                )
                return

            # Update round manager with new score
            self.round_manager.current_score[winner_key] += 1

            # Fire retroactive round_end event with incremented score
            round_end_event = {
                "type": "round_end",
                "timestamp": last_event_timestamp,
                "round_number": self.round_manager.current_round,
                "winner": winner_name,
                "score_team1": score_team1,
                "score_team2": score_team2,
                "timers": {
                    "game_timer": None,
                    "spike_timer": None,
                    "post_round_timer": None,
                }
            }
            self.event_collector.add_event(round_end_event)
            log.info(
                f"Fired retroactive round_end @ {last_event_timestamp:.2f}s: "
                f"Round {self.round_manager.current_round}, Winner: {winner_name}, "
                f"Score: {score_team1}-{score_team2} (incremented from previous)"
            )

            # Write the round_end event immediately
            self.output_writer.write_events([round_end_event])

        # Check if we need to fire match_end event
        if not self.match_ended:
            # Check if the match should have ended (score >= 13 with 2 round lead)
            score_team1 = self.round_manager.current_score["team1"]
            score_team2 = self.round_manager.current_score["team2"]
            max_score = max(score_team1, score_team2)
            score_diff = abs(score_team1 - score_team2)

            if max_score >= 13 and score_diff >= 2:
                # Match should have ended
                if score_team1 > score_team2:
                    match_winner = self.round_manager.team_names[0]
                else:
                    match_winner = self.round_manager.team_names[1]

                match_end_event = {
                    "type": "match_end",
                    "timestamp": last_event_timestamp,
                    "winner": match_winner,
                    "final_score_team1": score_team1,
                    "final_score_team2": score_team2,
                    "team1": self.round_manager.team_names[0],
                    "team2": self.round_manager.team_names[1],
                    "timers": {
                        "game_timer": None,
                        "spike_timer": None,
                        "post_round_timer": None,
                    }
                }
                self.event_collector.add_event(match_end_event)
                self.match_ended = True
                log.info(
                    f"Fired retroactive match_end @ {last_event_timestamp:.2f}s: "
                    f"{match_winner} wins ({score_team1}-{score_team2})"
                )

                # Write the match_end event immediately
                self.output_writer.write_events([match_end_event])

    def _print_progress(self, timestamp: float) -> None:
        """Print progress information."""
        phase_name = self.current_phase.name if self.current_phase else "Unknown"
        print(
            f"[{timestamp:7.2f}s] Frame {self.frame_count:5d} | "
            f"Phase: {phase_name:12s} | "
            f"Round {self.round_manager.current_round:2d} | "
            f"Score: {self.round_manager.current_score['team1']}-{self.round_manager.current_score['team2']}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"GameStateManager(video={self.video_path.name}, fps={self.fps})"
