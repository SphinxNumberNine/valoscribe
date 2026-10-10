"""Player state tracking for game state orchestration."""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import json

from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class PlayerStateTracker:
    """
    Tracks state for a single player.

    Responsibilities:
    - Maintain current and previous state for one player
    - Track: alive/dead, health, armor, abilities, ultimate
    - Handle revival detection (requires 3 consecutive alive detections)
    - Provide state history for event detection
    - Validate ability updates against agent config (max_charges, rechargeable)
    """

    # Revival detection threshold
    REVIVAL_THRESHOLD = 3
    # Death detection threshold (consecutive None health detections)
    DEATH_THRESHOLD = 2
    # Grace period after round start (seconds) - no death detection during UI fade-in
    ROUND_START_GRACE_PERIOD = 2.0

    def __init__(
        self,
        player_index: int,
        metadata: Optional[dict] = None,
        config_path: Optional[Path] = None,
        *,
        agent_config: Optional[dict] = None,
    ):
        """
        Initialize player state tracker.

        Args:
            player_index: Player index (0-9, where 0-4 is team1, 5-9 is team2)
            metadata: Optional player metadata dict with keys:
                - name: Player name
                - team: Team name
                - agent: Agent name
            config_path: Optional path to agent rules (default: agents_champs2025.json)
            agent_config: Already loaded rules shared by the processing session
        """
        if not 0 <= player_index <= 9:
            raise ValueError(f"Invalid player index: {player_index}, must be 0-9")

        self.player_index = player_index

        # Metadata (populated during initialization or later)
        self.metadata = metadata or {
            "name": f"Player{player_index}",
            "team": "Unknown",
            "agent": "unknown",
        }

        self.agent_config = (
            agent_config if agent_config is not None else self._load_agent_config(config_path)
        )

        # Current state
        self.current_state = {
            "alive": True,  # Assume alive at start
            "health": None,  # None = not detected yet
            "armor": None,
            "ability_1": None,  # Ability charges
            "ability_2": None,
            "ability_3": None,
            "ultimate": None,  # Ultimate charges/full status
            "killer": None,  # Agent name of killer (set when killed, cleared on revival/round reset)
        }

        # Previous state (for detecting transitions)
        self.previous_state = self.current_state.copy()

        # Revival detection tracking
        self.revival_candidate_count = 0
        self.revival_first_detection_timestamp: Optional[float] = None

        # Death detection tracking (consecutive None health detections)
        self.death_candidate_count = 0

        # Round start tracking (for grace period during UI fade-in)
        self.round_start_timestamp: Optional[float] = None

        # Agent icon tracking (for clutch mode death detection)
        self.last_detected_slot: Optional[int] = None  # Last UI slot where agent was found
        self.frames_icon_missing: int = 0  # Consecutive frames where agent icon not found

        log.debug(f"PlayerStateTracker initialized for player {player_index}")

    def _load_agent_config(self, config_path: Optional[Path] = None) -> dict:
        """
        Load agent configuration from JSON file.

        Args:
            config_path: Optional path to config file

        Returns:
            Dictionary of agent configurations
        """
        if config_path is None:
            package_dir = Path(__file__).parent.parent
            config_path = package_dir / "config" / "agents_champs2025.json"

        with open(config_path, "r") as f:
            agents_config = json.load(f)
        log.debug(f"Loaded agent config from {config_path} ({len(agents_config)} agents)")
        return agents_config

    def update(
        self, detections: dict, timestamp: float, *, allow_charge_increase: bool = False
    ) -> None:
        """
        Update player state from detections.

        Args:
            detections: Dictionary of detection results:
                - health: HealthInfo or None
                - armor: ArmorInfo or None
                - ability_1: AbilityInfo or None
                - ability_2: AbilityInfo or None
                - ability_3: AbilityInfo or None
                - ultimate: UltimateInfo or None
            timestamp: Current timestamp (seconds)
            allow_charge_increase: Allow buying/refunding utility during preround;
                maximum charge limits still apply
        """
        # Copy current → previous
        self.previous_state = self.current_state.copy()

        # Update health and alive status (with revival detection)
        # Only update if health is in detections (not present during PREROUND)
        if "health" in detections:
            self._update_health(detections["health"], timestamp)

        # Update armor
        if detections.get("armor"):
            self.current_state["armor"] = detections["armor"].armor

        # Update abilities (only if alive) with validation
        if self.current_state["alive"]:
            if detections.get("ability_1"):
                self._validate_and_update_ability(
                    "ability_1", detections["ability_1"].charges, allow_charge_increase
                )
            if detections.get("ability_2"):
                self._validate_and_update_ability(
                    "ability_2", detections["ability_2"].charges, allow_charge_increase
                )
            if detections.get("ability_3"):
                self._validate_and_update_ability(
                    "ability_3", detections["ability_3"].charges, allow_charge_increase
                )

        # Update ultimate (for all players, alive or dead)
        if detections.get("ultimate"):
            self.current_state["ultimate"] = {
                "charges": detections["ultimate"].charges,
                "is_full": detections["ultimate"].is_full,
            }

    def _update_health(self, health_info, timestamp: float) -> None:
        """
        Update health and alive status with revival detection.

        Revival Detection Logic:
        - Dead → Alive requires 3 consecutive successful health detections
        - Event timestamp = first detection time
        - If detections fail, reset revival candidate count

        Args:
            health_info: HealthInfo or None
            timestamp: Current timestamp (seconds)
        """
        was_alive = self.previous_state["alive"]

        if health_info:
            # Health detected - reset death tracking
            self.death_candidate_count = 0

            health_value = health_info.health
            self.current_state["health"] = health_value

            # Check if player is alive based on health
            is_alive = health_value > 0

            if not was_alive and is_alive:
                # Potential revival detected
                if self.revival_candidate_count == 0:
                    # First alive detection after being dead
                    self.revival_first_detection_timestamp = timestamp
                    log.debug(
                        f"Player {self.player_index}: Revival candidate detected "
                        f"(1/{self.REVIVAL_THRESHOLD})"
                    )

                self.revival_candidate_count += 1

                if self.revival_candidate_count >= self.REVIVAL_THRESHOLD:
                    # Confirmed revival!
                    self.current_state["alive"] = True
                    # Clear killer on revival (can be killed again)
                    self.current_state["killer"] = None
                    # Reset icon tracking (can be found in UI again)
                    self.frames_icon_missing = 0
                    log.info(
                        f"Player {self.player_index}: Revival confirmed "
                        f"(timestamp: {self.revival_first_detection_timestamp:.2f}s)"
                    )
                    # Reset revival tracking
                    self.revival_candidate_count = 0
                    self.revival_first_detection_timestamp = None
                else:
                    # Keep as dead until threshold reached
                    self.current_state["alive"] = False
            else:
                # No revival in progress, normal state
                self.current_state["alive"] = is_alive
                # Reset revival tracking if alive or stayed dead
                if self.revival_candidate_count > 0 and (is_alive or not is_alive):
                    self.revival_candidate_count = 0
                    self.revival_first_detection_timestamp = None
        else:
            # No health detected (None)
            # If we were tracking a revival, reset it (detection failed)
            if self.revival_candidate_count > 0:
                log.debug(
                    f"Player {self.player_index}: Revival tracking reset (detection failed)"
                )
                self.revival_candidate_count = 0
                self.revival_first_detection_timestamp = None

            # Death detection: None health = player died (health removed from UI)
            # Skip death detection during grace period (UI fade-in at round start)
            in_grace_period = False
            if self.round_start_timestamp is not None:
                time_since_round_start = timestamp - self.round_start_timestamp
                in_grace_period = time_since_round_start < self.ROUND_START_GRACE_PERIOD

            if in_grace_period:
                # During grace period, don't detect deaths (UI may be fading in)
                log.debug(
                    f"Player {self.player_index}: In grace period "
                    f"({timestamp - self.round_start_timestamp:.1f}s < {self.ROUND_START_GRACE_PERIOD}s), "
                    f"skipping death detection"
                )
                # Reset death candidate count during grace period
                self.death_candidate_count = 0
            elif was_alive:
                # Player was alive but now health is not detected
                self.death_candidate_count += 1

                if self.death_candidate_count >= self.DEATH_THRESHOLD:
                    # Confirmed death - mark as dead
                    self.current_state["alive"] = False
                    self.current_state["health"] = None
                    log.debug(
                        f"Player {self.player_index}: Death confirmed after "
                        f"{self.death_candidate_count} consecutive None health detections"
                    )
                    # Reset death tracking
                    self.death_candidate_count = 0
                else:
                    # Keep as alive until threshold reached
                    log.debug(
                        f"Player {self.player_index}: Death candidate "
                        f"({self.death_candidate_count}/{self.DEATH_THRESHOLD})"
                    )
            else:
                # Player was already dead, keep dead status
                # Reset death tracking since we're already dead
                self.death_candidate_count = 0

    def _validate_and_update_ability(
        self, ability_key: str, detected_charges: int, allow_charge_increase: bool = False
    ) -> None:
        """
        Validate ability charges and update state if valid.

        Validation rules:
        1. Detected charges must not exceed max_charges from config
        2. Non-rechargeable abilities cannot gain charges

        Invalid detections are rejected and logged as warnings.

        Args:
            ability_key: Ability key (ability_1, ability_2, ability_3)
            detected_charges: Detected charge count
        """
        agent_name = self.metadata.get("agent", "unknown")
        player_name = self.metadata.get("name", f"Player{self.player_index}")

        # Get agent config
        if agent_name not in self.agent_config:
            # No config available - accept the detection (can't validate)
            self.current_state[ability_key] = detected_charges
            return

        agent_config = self.agent_config[agent_name]

        # Get ability config
        if ability_key not in agent_config:
            # Ability not in config - accept detection
            self.current_state[ability_key] = detected_charges
            return

        ability_config = agent_config[ability_key]
        max_charges = ability_config.get("max_charges", 10)
        rechargeable = ability_config.get("rechargeable", False)
        ability_name = ability_config.get("name", ability_key)

        # Get previous charges for comparison
        prev_charges = self.previous_state.get(ability_key)

        # Validation 1: Check max_charges
        if detected_charges > max_charges:
            log.warning(
                f"[ABILITY VALIDATION] {player_name} ({agent_name}): {ability_name} detected with "
                f"{detected_charges} charges but max is {max_charges}. Rejecting misdetection."
            )
            # Don't update state - keep previous value
            return

        # Validation 2: Check invalid recharge (non-rechargeable ability gained charges)
        if prev_charges is not None and detected_charges > prev_charges:
            if not rechargeable and not allow_charge_increase:
                log.warning(
                    f"[ABILITY VALIDATION] {player_name} ({agent_name}): {ability_name} is not rechargeable "
                    f"but charges increased {prev_charges}→{detected_charges}. Rejecting misdetection."
                )
                # Don't update state - keep previous value
                return

        # Valid detection - update state
        self.current_state[ability_key] = detected_charges

    def set_metadata(self, metadata: dict) -> None:
        """
        Update player metadata.

        Args:
            metadata: Dict with keys: name, team, agent
        """
        self.metadata.update(metadata)
        log.debug(
            f"Player {self.player_index} metadata updated: "
            f"{self.metadata['name']} ({self.metadata['team']}) - {self.metadata['agent']}"
        )

    def get_state_changes(self) -> dict:
        """
        Get changes between previous and current state.

        Returns:
            Dictionary with changed fields and their (old, new) values
        """
        changes = {}

        for key in self.current_state:
            old_value = self.previous_state.get(key)
            new_value = self.current_state.get(key)

            if old_value != new_value:
                changes[key] = {"old": old_value, "new": new_value}

        return changes

    def is_alive_transition(self) -> bool:
        """Check if player transitioned from dead to alive."""
        return not self.previous_state["alive"] and self.current_state["alive"]

    def is_death_transition(self) -> bool:
        """Check if player transitioned from alive to dead."""
        return self.previous_state["alive"] and not self.current_state["alive"]

    def reset_for_new_round(self) -> None:
        """
        Reset state for a new round.

        All players start alive with unknown stats.
        Note: round_start_timestamp is set separately when ACTIVE_ROUND begins.
        """
        self.current_state = {
            "alive": True,
            "health": None,
            "armor": None,
            "ability_1": None,
            "ability_2": None,
            "ability_3": None,
            "ultimate": None,
            "killer": None,  # Clear killer for new round
        }
        self.previous_state = self.current_state.copy()
        self.revival_candidate_count = 0
        self.revival_first_detection_timestamp = None
        self.death_candidate_count = 0

        # Reset icon tracking for new round
        self.last_detected_slot = None
        self.frames_icon_missing = 0

        # Note: round_start_timestamp NOT reset here - it's set when ACTIVE_ROUND begins
        # to align with UI fade-in timing

        log.debug(f"Player {self.player_index}: State reset for new round")

    def __repr__(self) -> str:
        """String representation."""
        name = self.metadata.get("name", f"Player{self.player_index}")
        alive_status = "alive" if self.current_state["alive"] else "dead"
        health = self.current_state.get("health", "?")

        return f"PlayerStateTracker({name}, {alive_status}, health={health})"
