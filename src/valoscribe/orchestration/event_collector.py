"""Event collector for aggregating discrete game events."""

from __future__ import annotations
from typing import Optional

from valoscribe.orchestration.killfeed_deduplicator import KillfeedDeduplicator
from valoscribe.types.detections import KillfeedAgentDetection
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class EventCollector:
    """
    Collects discrete game events from various sources.

    Events include:
    - Kills (from killfeed, with deduplication)
    - Deaths (from player state transitions)
    - Revivals (Sage ultimate)
    - Ability usage
    - Ultimate usage
    - Round start/end
    - Spike plant/defuse

    The collector maintains a chronological list of all events
    and handles deduplication for killfeed events.
    """

    def __init__(self, dedup_window_sec: float = 5.0):
        """
        Initialize event collector.

        Args:
            dedup_window_sec: Time window for killfeed deduplication (default: 5.0s)
        """
        self.events: list[dict] = []
        self.killfeed_dedup = KillfeedDeduplicator(dedup_window_sec=dedup_window_sec)
        self._last_write_index = 0  # Track last written event index

        log.debug(f"EventCollector initialized with {dedup_window_sec}s dedup window")

    def add_event(self, event: dict | str, timestamp: Optional[float] = None, **data) -> None:
        """
        Add a generic event to the collection.

        Can be called two ways:
        1. add_event({"type": "...", "timestamp": ..., ...})  # Dict
        2. add_event("death", 10.0, player="name")           # Separate params

        Args:
            event: Event dict or event type string
            timestamp: Event timestamp in seconds (required if event is string)
            **data: Additional event data as keyword arguments
        """
        if isinstance(event, dict):
            # Dict format
            self.events.append(event)
            log.debug(f"Event added: {event.get('type', 'unknown')} at {event.get('timestamp', 0):.2f}s")
        else:
            # Separate params format
            if timestamp is None:
                raise ValueError("timestamp is required when event is a string")
            event_dict = {
                "type": event,
                "timestamp": timestamp,
                **data,
            }
            self.events.append(event_dict)
            log.debug(f"Event added: {event} at {timestamp:.2f}s")

    def add_player_events(self, player_events: list[dict]) -> None:
        """
        Add multiple player events from StateValidator.

        Args:
            player_events: List of event dictionaries from StateValidator
        """
        for event in player_events:
            # Events from StateValidator already have type, timestamp, and data
            self.events.append(event)
            log.debug(f"Player event added: {event['type']} at {event['timestamp']:.2f}s")

    def add_killfeed_events(
        self,
        timestamp: float,
        killfeed_detections: list[tuple[KillfeedAgentDetection, dict]],
    ) -> int:
        """
        Add killfeed events with deduplication.

        Args:
            timestamp: Current frame timestamp in seconds
            killfeed_detections: List of tuples (detection, player_info) where player_info contains:
                - killer_name: str (optional)
                - killer_team: str (optional)
                - victim_name: str (optional)
                - victim_team: str (optional)

        Returns:
            Number of new (non-duplicate) kills added
        """
        added_count = 0

        for detection, player_info in killfeed_detections:
            # Check if duplicate
            if self.killfeed_dedup.is_duplicate(timestamp, detection):
                continue

            # Build event with player information
            event_data = {
                "killer_agent": detection.killer_agent,
                "killer_side": detection.killer_side,
                "victim_agent": detection.victim_agent,
                "victim_side": detection.victim_side,
                "confidence": detection.confidence,
                "weapon": detection.weapon,
            }

            # Add player names and teams if available
            if player_info.get("killer_name"):
                event_data["killer_name"] = player_info["killer_name"]
            if player_info.get("killer_team"):
                event_data["killer_team"] = player_info["killer_team"]
            if player_info.get("victim_name"):
                event_data["victim_name"] = player_info["victim_name"]
            if player_info.get("victim_team"):
                event_data["victim_team"] = player_info["victim_team"]

            # Add timers if available
            if player_info.get("timers"):
                event_data["timers"] = player_info["timers"]

            # Not a duplicate - add to events
            self.add_event("kill", timestamp, **event_data)
            added_count += 1

        if added_count > 0:
            log.debug(f"Added {added_count} kills at {timestamp:.2f}s")

        return added_count

    def get_events(self, event_type: Optional[str] = None) -> list[dict]:
        """
        Get all events, optionally filtered by type.

        Args:
            event_type: Optional event type to filter by

        Returns:
            List of event dictionaries
        """
        if event_type is None:
            return self.events.copy()

        return [e for e in self.events if e["type"] == event_type]

    def get_all_events(self) -> list[dict]:
        """
        Get all events (alias for get_events()).

        Returns:
            List of all event dictionaries
        """
        return self.events.copy()

    def get_events_since_last_write(self) -> list[dict]:
        """
        Get events added since last write.

        Returns:
            List of new event dictionaries
        """
        new_events = self.events[self._last_write_index:]
        self._last_write_index = len(self.events)
        return new_events

    def get_events_in_range(
        self,
        start_time: float,
        end_time: float,
        event_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Get events within a time range.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            event_type: Optional event type to filter by

        Returns:
            List of event dictionaries in the time range
        """
        events = self.events if event_type is None else self.get_events(event_type)

        return [
            e for e in events
            if start_time <= e["timestamp"] <= end_time
        ]

    def get_event_count(self, event_type: Optional[str] = None) -> int:
        """
        Get count of events, optionally filtered by type.

        Args:
            event_type: Optional event type to filter by

        Returns:
            Number of events
        """
        if event_type is None:
            return len(self.events)

        return len(self.get_events(event_type))

    def reset(self) -> None:
        """Reset the event collector, clearing all events and dedup cache."""
        self.events.clear()
        self.killfeed_dedup.reset()
        self._last_write_index = 0
        log.debug("EventCollector reset")

    def get_event_summary(self) -> dict[str, int]:
        """
        Get summary of events by type.

        Returns:
            Dictionary mapping event types to counts
        """
        summary: dict[str, int] = {}

        for event in self.events:
            event_type = event["type"]
            summary[event_type] = summary.get(event_type, 0) + 1

        return summary

    def __repr__(self) -> str:
        """String representation."""
        event_types = set(e["type"] for e in self.events)
        return (
            f"EventCollector("
            f"total_events={len(self.events)}, "
            f"event_types={len(event_types)})"
        )
