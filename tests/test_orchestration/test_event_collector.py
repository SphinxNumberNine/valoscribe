"""Unit tests for EventCollector."""

from __future__ import annotations
import pytest

from valoscribe.orchestration.event_collector import EventCollector
from valoscribe.types.detections import KillfeedAgentDetection


class TestEventCollector:
    """Tests for EventCollector class."""

    @pytest.fixture
    def collector(self):
        """Create EventCollector instance."""
        return EventCollector()

    def test_init(self, collector):
        """Test EventCollector initialization."""
        assert collector.events == []
        assert collector.killfeed_dedup is not None

    def test_init_custom_dedup_window(self):
        """Test initialization with custom dedup window."""
        collector = EventCollector(dedup_window_sec=10.0)
        assert collector.killfeed_dedup.dedup_window_sec == 10.0

    def test_add_event_basic(self, collector):
        """Test adding a basic event."""
        collector.add_event("round_start", timestamp=0.0, round_number=1)

        assert len(collector.events) == 1
        assert collector.events[0]["type"] == "round_start"
        assert collector.events[0]["timestamp"] == 0.0
        assert collector.events[0]["round_number"] == 1

    def test_add_event_with_data(self, collector):
        """Test adding event with multiple data fields."""
        collector.add_event(
            "death",
            timestamp=10.5,
            player="brawk",
            team="NRG",
            agent="sova",
        )

        assert len(collector.events) == 1
        event = collector.events[0]
        assert event["type"] == "death"
        assert event["timestamp"] == 10.5
        assert event["player"] == "brawk"
        assert event["team"] == "NRG"
        assert event["agent"] == "sova"

    def test_add_multiple_events(self, collector):
        """Test adding multiple events."""
        collector.add_event("round_start", timestamp=0.0, round_number=1)
        collector.add_event("death", timestamp=5.0, player="player1")
        collector.add_event("kill", timestamp=5.0, killer="player2")

        assert len(collector.events) == 3
        assert collector.events[0]["type"] == "round_start"
        assert collector.events[1]["type"] == "death"
        assert collector.events[2]["type"] == "kill"

    def test_add_player_events(self, collector):
        """Test adding player events from StateValidator."""
        player_events = [
            {
                "type": "death",
                "timestamp": 10.0,
                "player": "brawk",
                "team": "NRG",
                "agent": "sova",
            },
            {
                "type": "ability_used",
                "timestamp": 12.0,
                "player": "marved",
                "ability": "ability_1",
            },
        ]

        collector.add_player_events(player_events)

        assert len(collector.events) == 2
        assert collector.events[0]["type"] == "death"
        assert collector.events[1]["type"] == "ability_used"

    def test_add_killfeed_events_single(self, collector):
        """Test adding single killfeed event."""
        detection = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )

        added = collector.add_killfeed_events(timestamp=5.0, killfeed_detections=[(detection, {})])

        assert added == 1
        assert len(collector.events) == 1
        assert collector.events[0]["type"] == "kill"
        assert collector.events[0]["killer_agent"] == "jett"
        assert collector.events[0]["victim_agent"] == "sova"

    def test_add_killfeed_events_multiple(self, collector):
        """Test adding multiple killfeed events."""
        detections = [
            KillfeedAgentDetection(
                killer_agent="jett",
                killer_side="attack",
                victim_agent="sova",
                victim_side="defense",
                confidence=0.95,
            ),
            KillfeedAgentDetection(
                killer_agent="raze",
                killer_side="attack",
                victim_agent="cypher",
                victim_side="defense",
                confidence=0.92,
            ),
        ]

        added = collector.add_killfeed_events(timestamp=5.0, killfeed_detections=[(detection, {}) for detection in detections])

        assert added == 2
        assert len(collector.events) == 2

    def test_killfeed_deduplication(self, collector):
        """Test that duplicate killfeed events are filtered."""
        detection = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )

        # Add same kill multiple times (simulating consecutive frames)
        added1 = collector.add_killfeed_events(timestamp=5.0, killfeed_detections=[(detection, {})])
        added2 = collector.add_killfeed_events(timestamp=5.25, killfeed_detections=[(detection, {})])
        added3 = collector.add_killfeed_events(timestamp=5.5, killfeed_detections=[(detection, {})])

        assert added1 == 1  # First is new
        assert added2 == 0  # Duplicate
        assert added3 == 0  # Duplicate
        assert len(collector.events) == 1  # Only one kill logged

    def test_killfeed_expiration(self, collector):
        """Test that kills expire after dedup window."""
        detection = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )

        # Add kill at t=0
        added1 = collector.add_killfeed_events(timestamp=0.0, killfeed_detections=[(detection, {})])

        # Same kill at t=6 (beyond 5s window) - should be new
        added2 = collector.add_killfeed_events(timestamp=6.0, killfeed_detections=[(detection, {})])

        assert added1 == 1
        assert added2 == 1
        assert len(collector.events) == 2  # Two separate kill instances

    def test_get_events_all(self, collector):
        """Test getting all events."""
        collector.add_event("round_start", timestamp=0.0)
        collector.add_event("death", timestamp=5.0)
        collector.add_event("kill", timestamp=5.0)

        events = collector.get_events()

        assert len(events) == 3
        # Should be a copy
        events.append({"type": "test"})
        assert len(collector.events) == 3

    def test_get_events_filtered(self, collector):
        """Test getting events filtered by type."""
        collector.add_event("death", timestamp=5.0, player="player1")
        collector.add_event("kill", timestamp=5.0, killer="player2")
        collector.add_event("death", timestamp=10.0, player="player3")

        deaths = collector.get_events(event_type="death")
        kills = collector.get_events(event_type="kill")

        assert len(deaths) == 2
        assert len(kills) == 1
        assert all(e["type"] == "death" for e in deaths)
        assert all(e["type"] == "kill" for e in kills)

    def test_get_events_in_range(self, collector):
        """Test getting events within time range."""
        collector.add_event("event1", timestamp=0.0)
        collector.add_event("event2", timestamp=5.0)
        collector.add_event("event3", timestamp=10.0)
        collector.add_event("event4", timestamp=15.0)

        events = collector.get_events_in_range(start_time=5.0, end_time=10.0)

        assert len(events) == 2
        assert events[0]["timestamp"] == 5.0
        assert events[1]["timestamp"] == 10.0

    def test_get_events_in_range_filtered(self, collector):
        """Test getting events in range filtered by type."""
        collector.add_event("death", timestamp=5.0)
        collector.add_event("kill", timestamp=7.0)
        collector.add_event("death", timestamp=10.0)
        collector.add_event("kill", timestamp=15.0)

        deaths = collector.get_events_in_range(
            start_time=0.0,
            end_time=12.0,
            event_type="death",
        )

        assert len(deaths) == 2
        assert all(e["type"] == "death" for e in deaths)
        assert all(0.0 <= e["timestamp"] <= 12.0 for e in deaths)

    def test_get_event_count_all(self, collector):
        """Test getting total event count."""
        collector.add_event("event1", timestamp=0.0)
        collector.add_event("event2", timestamp=5.0)
        collector.add_event("event3", timestamp=10.0)

        assert collector.get_event_count() == 3

    def test_get_event_count_filtered(self, collector):
        """Test getting event count filtered by type."""
        collector.add_event("death", timestamp=5.0)
        collector.add_event("kill", timestamp=5.0)
        collector.add_event("death", timestamp=10.0)

        assert collector.get_event_count(event_type="death") == 2
        assert collector.get_event_count(event_type="kill") == 1
        assert collector.get_event_count(event_type="revival") == 0

    def test_reset(self, collector):
        """Test resetting the collector."""
        collector.add_event("event1", timestamp=0.0)
        collector.add_event("event2", timestamp=5.0)

        detection = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )
        collector.add_killfeed_events(timestamp=10.0, killfeed_detections=[(detection, {})])

        assert len(collector.events) == 3
        assert collector.killfeed_dedup.get_recent_kill_count() == 1

        # Reset
        collector.reset()

        assert len(collector.events) == 0
        assert collector.killfeed_dedup.get_recent_kill_count() == 0

    def test_get_event_summary(self, collector):
        """Test getting event summary."""
        collector.add_event("death", timestamp=5.0)
        collector.add_event("kill", timestamp=5.0)
        collector.add_event("death", timestamp=10.0)
        collector.add_event("death", timestamp=15.0)
        collector.add_event("ability_used", timestamp=20.0)

        summary = collector.get_event_summary()

        assert summary["death"] == 3
        assert summary["kill"] == 1
        assert summary["ability_used"] == 1
        assert len(summary) == 3

    def test_event_summary_empty(self, collector):
        """Test event summary when empty."""
        summary = collector.get_event_summary()
        assert summary == {}

    def test_repr(self, collector):
        """Test string representation."""
        collector.add_event("death", timestamp=5.0)
        collector.add_event("kill", timestamp=5.0)

        repr_str = repr(collector)
        assert "EventCollector" in repr_str
        assert "total_events=2" in repr_str
        assert "event_types=2" in repr_str

    def test_realistic_scenario(self, collector):
        """Test realistic game scenario with mixed events."""
        # Round start
        collector.add_event("round_start", timestamp=0.0, round_number=1)

        # Player events
        player_events = [
            {"type": "death", "timestamp": 10.0, "player": "brawk"},
            {"type": "ability_used", "timestamp": 8.0, "player": "marved"},
        ]
        collector.add_player_events(player_events)

        # Killfeed events (with duplicates across frames)
        kill1 = KillfeedAgentDetection(
            killer_agent="jett",
            killer_side="attack",
            victim_agent="sova",
            victim_side="defense",
            confidence=0.95,
        )

        # Same kill detected over 5 frames
        for i in range(5):
            collector.add_killfeed_events(
                timestamp=10.0 + i * 0.25,
                killfeed_detections=[(kill1, {})],
            )

        # Round end
        collector.add_event("round_end", timestamp=100.0, round_number=1)

        # Should have: 1 round_start + 2 player_events + 1 kill + 1 round_end = 5 events
        assert len(collector.events) == 5

        summary = collector.get_event_summary()
        assert summary["round_start"] == 1
        assert summary["death"] == 1
        assert summary["ability_used"] == 1
        assert summary["kill"] == 1
        assert summary["round_end"] == 1
