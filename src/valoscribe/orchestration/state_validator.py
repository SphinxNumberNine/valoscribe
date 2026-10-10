"""State validator for detecting and validating state transitions."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class StateValidator:
    """
    Validates player state transitions and generates events.

    Checks for valid transitions based on game rules and agent abilities:
    - Revival (dead → alive, already validated by PlayerStateTracker)
    - Ability usage (charges decreased)
    - Ability recharge (charges increased, only if rechargeable)
    - Ultimate usage (full → not full)

    Note: Death events are handled in GameStateManager to include player_index field.
    """

    def __init__(self, agent_config_path: Optional[Path] = None):
        """
        Initialize state validator.

        Args:
            agent_config_path: Path to agent config JSON file
                             (default: src/valoscribe/config/agents_champs2025.json)
        """
        if agent_config_path is None:
            agent_config_path = Path(__file__).parent.parent / "config" / "agents_champs2025.json"

        self.agent_config = self._load_agent_config(agent_config_path)

        # Ability change confirmation tracking
        # Format: {player_name: {ability_key: {"first_timestamp": float, "old": int, "new": int, "count": int}}}
        self.pending_ability_changes: dict[str, dict[str, dict]] = {}

        log.info(f"StateValidator initialized with {len(self.agent_config)} agents")

    def _load_agent_config(self, config_path: Path) -> dict:
        """
        Load agent configuration from JSON file.

        Args:
            config_path: Path to agent config file

        Returns:
            Dictionary mapping agent names to ability configs
        """
        with open(config_path) as f:
            config = json.load(f)

        log.debug(f"Loaded agent config from {config_path}")
        return config

    def validate_player_state(
        self,
        current: dict,
        previous: dict,
        metadata: dict,
        timestamp: float,
        team_has_sage: bool = False,
    ) -> list[dict]:
        """
        Validate player state transition and generate events.

        Args:
            current: Current player state dict
            previous: Previous player state dict
            metadata: Player metadata (name, team, agent)
            timestamp: Current timestamp in seconds
            team_has_sage: Whether the player's team has a Sage (for revival validation)

        Returns:
            List of event dictionaries
        """
        events = []

        # Note: Death events are handled in GameStateManager (with player_index field)
        # Not detecting deaths here to prevent duplicates

        # Check for revival (already validated by PlayerStateTracker with 3 detections)
        # Revivals are only possible if the team has a Sage (only agent with revival ultimate)
        if not previous["alive"] and current["alive"]:
            if team_has_sage:
                events.append({
                    "type": "revival",
                    "timestamp": timestamp,
                    "player": metadata["name"],
                    "team": metadata["team"],
                    "agent": metadata["agent"],
                })
            else:
                log.warning(
                    f"Revival detected for {metadata['name']} ({metadata['agent']}) but team "
                    f"{metadata['team']} has no Sage - ignoring false positive"
                )

        # Check abilities (only if agent is known and exists in config)
        agent_name = metadata.get("agent")
        if agent_name and agent_name in self.agent_config:
            ability_events = self._validate_abilities(
                current, previous, metadata, timestamp, agent_name
            )
            events.extend(ability_events)

        # Check ultimate
        ultimate_events = self._validate_ultimate(current, previous, metadata, timestamp)
        events.extend(ultimate_events)

        return events

    def _validate_abilities(
        self,
        current: dict,
        previous: dict,
        metadata: dict,
        timestamp: float,
        agent_name: str,
    ) -> list[dict]:
        """
        Validate ability transitions and generate usage/recharge events.

        Uses 2-frame confirmation to prevent false events during UI transitions (e.g., death effects).
        Events are fired with the timestamp of the FIRST detection, not the confirmation frame.

        Args:
            current: Current player state
            previous: Previous player state
            metadata: Player metadata
            timestamp: Current timestamp
            agent_name: Agent name

        Returns:
            List of ability event dictionaries
        """
        events = []
        agent_config = self.agent_config[agent_name]
        player_name = metadata["name"]

        # Skip ability validation if player is dead
        if not current.get("alive", True):
            # Clear any pending changes for dead players
            if player_name in self.pending_ability_changes:
                self.pending_ability_changes[player_name].clear()
            return events

        # Initialize pending changes for this player if needed
        if player_name not in self.pending_ability_changes:
            self.pending_ability_changes[player_name] = {}

        for ability_key in ["ability_1", "ability_2", "ability_3"]:
            prev_charges = previous.get(ability_key)
            curr_charges = current.get(ability_key)

            # Skip if either is None (not detected)
            if prev_charges is None or curr_charges is None:
                # Clear pending if we lost detection
                if ability_key in self.pending_ability_changes[player_name]:
                    del self.pending_ability_changes[player_name][ability_key]
                continue

            ability_config = agent_config[ability_key]
            pending = self.pending_ability_changes[player_name].get(ability_key)

            # Check if we have a pending change
            if pending:
                # Check if current value matches the pending "new" value (stable confirmation)
                if curr_charges == pending["new"]:
                    # Value stayed stable - increment confirmation count
                    pending["count"] += 1
                    log.debug(
                        f"[ABILITY] {player_name} {ability_key}: Stable confirmation {pending['count']}/2 "
                        f"(value={curr_charges})"
                    )

                    # Fire event if threshold reached
                    if pending["count"] >= 2:
                        log.debug(
                            f"[ABILITY] {player_name} {ability_key}: Confirmed after 2 frames, "
                            f"firing event with timestamp {pending['first_timestamp']:.2f}s"
                        )

                        # Generate event with FIRST timestamp
                        event = self._create_ability_event(
                            pending["old"],
                            pending["new"],
                            ability_key,
                            ability_config,
                            metadata,
                            pending["first_timestamp"],  # Use first detection timestamp
                        )
                        if event:
                            events.append(event)

                        # Clear pending
                        del self.pending_ability_changes[player_name][ability_key]
                elif curr_charges == pending["old"]:
                    # Value reverted back to old - cancel pending
                    log.debug(
                        f"[ABILITY] {player_name} {ability_key}: Reverted to old value, clearing pending"
                    )
                    del self.pending_ability_changes[player_name][ability_key]
                else:
                    # Value changed to something different - start new pending
                    self.pending_ability_changes[player_name][ability_key] = {
                        "first_timestamp": timestamp,
                        "old": prev_charges,
                        "new": curr_charges,
                        "count": 1,
                    }
                    log.debug(
                        f"[ABILITY] {player_name} {ability_key}: New change detected (1/2): "
                        f"{prev_charges} → {curr_charges} @ {timestamp:.2f}s"
                    )
            elif prev_charges != curr_charges:
                # No pending and value changed - start new pending
                self.pending_ability_changes[player_name][ability_key] = {
                    "first_timestamp": timestamp,
                    "old": prev_charges,
                    "new": curr_charges,
                    "count": 1,
                }
                log.debug(
                    f"[ABILITY] {player_name} {ability_key}: New change detected (1/2): "
                    f"{prev_charges} → {curr_charges} @ {timestamp:.2f}s"
                )

        return events

    def _create_ability_event(
        self,
        old_charges: int,
        new_charges: int,
        ability_key: str,
        ability_config: dict,
        metadata: dict,
        timestamp: float,
    ) -> Optional[dict]:
        """
        Create an ability event (used or recharged).

        Args:
            old_charges: Previous charge count
            new_charges: New charge count
            ability_key: Ability key (ability_1, ability_2, ability_3)
            ability_config: Ability config from agents_champs2025.json
            metadata: Player metadata
            timestamp: Event timestamp

        Returns:
            Event dict or None
        """
        # Get ability name from config (fallback to ability_key if not found)
        ability_name = ability_config.get("name", ability_key)

        # Ability usage (charges decreased)
        if new_charges < old_charges:
            charges_used = old_charges - new_charges
            return {
                "type": "ability_used",
                "timestamp": timestamp,
                "player": metadata["name"],
                "team": metadata["team"],
                "agent": metadata["agent"],
                "ability": ability_name,
                "charges_used": charges_used,
                "remaining_charges": new_charges,
            }

        # Ability recharge (charges increased)
        elif new_charges > old_charges:
            # Only valid if ability is rechargeable
            if ability_config["rechargeable"]:
                charges_gained = new_charges - old_charges
                return {
                    "type": "ability_recharged",
                    "timestamp": timestamp,
                    "player": metadata["name"],
                    "team": metadata["team"],
                    "agent": metadata["agent"],
                    "ability": ability_name,
                    "charges_gained": charges_gained,
                    "total_charges": new_charges,
                }
            else:
                # Invalid transition: non-rechargeable ability gained charges
                log.warning(
                    f"Invalid ability recharge: {metadata['name']} {ability_name} "
                    f"is not rechargeable but charges increased {old_charges}→{new_charges}"
                )
                return None

        return None

    def _validate_ultimate(
        self,
        current: dict,
        previous: dict,
        metadata: dict,
        timestamp: float,
    ) -> list[dict]:
        """
        Validate ultimate transitions and generate usage events.

        Args:
            current: Current player state
            previous: Previous player state
            metadata: Player metadata
            timestamp: Current timestamp

        Returns:
            List of ultimate event dictionaries
        """
        events = []

        prev_ult = previous.get("ultimate")
        curr_ult = current.get("ultimate")

        # Skip if either is None
        if prev_ult is None or curr_ult is None:
            return events

        # Check for ultimate usage (full → not full)
        if prev_ult.get("is_full") and not curr_ult.get("is_full"):
            # Get ultimate name from config
            agent_name = metadata.get("agent")
            ultimate_name = None
            if agent_name and agent_name in self.agent_config:
                ultimate_config = self.agent_config[agent_name].get("ultimate", {})
                ultimate_name = ultimate_config.get("name")

            event = {
                "type": "ultimate_used",
                "timestamp": timestamp,
                "player": metadata["name"],
                "team": metadata["team"],
                "agent": metadata["agent"],
                "previous_charges": prev_ult["charges"],
                "current_charges": curr_ult["charges"],
            }

            # Add ultimate name if found
            if ultimate_name:
                event["ultimate"] = ultimate_name

            events.append(event)
            log.debug(
                f"{metadata['name']} used ultimate: "
                f"charges {prev_ult['charges']}→{curr_ult['charges']}"
            )

        return events

    def reset_for_new_round(self) -> None:
        """Clear all pending ability changes for a new round."""
        self.pending_ability_changes.clear()
        log.debug("StateValidator: Cleared pending ability changes for new round")

    def __repr__(self) -> str:
        """String representation."""
        return f"StateValidator(agents={len(self.agent_config)})"
