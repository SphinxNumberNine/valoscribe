"""Unit tests for PlayerStateTracker."""

from __future__ import annotations
from unittest.mock import Mock
import pytest

from valoscribe.orchestration.player_state_tracker import PlayerStateTracker
from valoscribe.types.detections import HealthInfo, ArmorInfo, AbilityInfo, UltimateInfo


class TestPlayerStateTracker:
    """Tests for PlayerStateTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create PlayerStateTracker instance."""
        metadata = {
            "name": "brawk",
            "team": "NRG",
            "agent": "sova",
        }
        return PlayerStateTracker(player_index=0, metadata=metadata)

    @pytest.fixture
    def mock_detections(self):
        """Create mock detection results."""
        return {
            "health": HealthInfo(health=100, confidence=0.9, raw_text="100"),
            "armor": ArmorInfo(armor=50, confidence=0.9, raw_text="50"),
            "ability_1": AbilityInfo(charges=2, total_blobs_detected=2),
            "ability_2": AbilityInfo(charges=1, total_blobs_detected=1),
            "ability_3": AbilityInfo(charges=1, total_blobs_detected=1),
            "ultimate": UltimateInfo(charges=6, is_full=False, total_blobs_detected=6),
        }

    def test_init(self, tracker):
        """Test PlayerStateTracker initialization."""
        assert tracker.player_index == 0
        assert tracker.metadata["name"] == "brawk"
        assert tracker.metadata["team"] == "NRG"
        assert tracker.metadata["agent"] == "sova"
        assert tracker.current_state["alive"] is True
        assert tracker.current_state["health"] is None
        assert tracker.revival_candidate_count == 0

    def test_init_invalid_index(self):
        """Test initialization with invalid player index."""
        with pytest.raises(ValueError, match="Invalid player index"):
            PlayerStateTracker(player_index=-1)

        with pytest.raises(ValueError, match="Invalid player index"):
            PlayerStateTracker(player_index=10)

    def test_init_without_metadata(self):
        """Test initialization without metadata."""
        tracker = PlayerStateTracker(player_index=5)
        assert tracker.metadata["name"] == "Player5"
        assert tracker.metadata["team"] == "Unknown"
        assert tracker.metadata["agent"] == "unknown"

    def test_update_basic_detections(self, tracker, mock_detections):
        """Test updating state with basic detections."""
        tracker.update(mock_detections, timestamp=10.0)

        assert tracker.current_state["alive"] is True
        assert tracker.current_state["health"] == 100
        assert tracker.current_state["armor"] == 50
        assert tracker.current_state["ability_1"] == 2
        assert tracker.current_state["ability_2"] == 1
        assert tracker.current_state["ability_3"] == 1
        assert tracker.current_state["ultimate"]["charges"] == 6
        assert tracker.current_state["ultimate"]["is_full"] is False

    def test_update_preserves_previous_state(self, tracker, mock_detections):
        """Test that update preserves previous state."""
        tracker.update(mock_detections, timestamp=10.0)

        # Update with different values
        new_detections = {
            "health": HealthInfo(health=75, confidence=0.9, raw_text="75"),
            "armor": ArmorInfo(armor=25, confidence=0.9, raw_text="25"),
        }
        tracker.update(new_detections, timestamp=11.0)

        assert tracker.previous_state["health"] == 100
        assert tracker.previous_state["armor"] == 50
        assert tracker.current_state["health"] == 75
        assert tracker.current_state["armor"] == 25

    def test_death_detection(self, tracker):
        """Test detecting player death."""
        # Start alive with health
        detections = {
            "health": HealthInfo(health=100, confidence=0.9, raw_text="100"),
        }
        tracker.update(detections, timestamp=10.0)
        assert tracker.current_state["alive"] is True

        # Health drops to 0 = death
        detections = {
            "health": HealthInfo(health=0, confidence=0.9, raw_text="0"),
        }
        tracker.update(detections, timestamp=11.0)
        assert tracker.current_state["alive"] is False
        assert tracker.current_state["health"] == 0

    def test_revival_requires_three_detections(self, tracker):
        """Test that revival requires 3 consecutive alive detections."""
        # Start dead
        tracker.current_state["alive"] = False
        tracker.previous_state["alive"] = False

        # First alive detection - not enough
        detections = {"health": HealthInfo(health=100, confidence=0.9, raw_text="100")}
        tracker.update(detections, timestamp=10.0)
        assert tracker.current_state["alive"] is False
        assert tracker.revival_candidate_count == 1

        # Second alive detection - not enough
        tracker.update(detections, timestamp=11.0)
        assert tracker.current_state["alive"] is False
        assert tracker.revival_candidate_count == 2

        # Third alive detection - revival confirmed!
        tracker.update(detections, timestamp=12.0)
        assert tracker.current_state["alive"] is True
        assert tracker.revival_candidate_count == 0

    def test_revival_reset_on_detection_failure(self, tracker):
        """Test that revival tracking resets when detection fails."""
        # Start dead
        tracker.current_state["alive"] = False
        tracker.previous_state["alive"] = False

        # First alive detection
        detections = {"health": HealthInfo(health=100, confidence=0.9, raw_text="100")}
        tracker.update(detections, timestamp=10.0)
        assert tracker.revival_candidate_count == 1

        # Detection fails (no health)
        tracker.update({"health": None}, timestamp=11.0)
        assert tracker.revival_candidate_count == 0
        assert tracker.revival_first_detection_timestamp is None

    def test_revival_timestamp_tracking(self, tracker):
        """Test that revival timestamp is tracked correctly."""
        # Start dead
        tracker.current_state["alive"] = False
        tracker.previous_state["alive"] = False

        # Track revival timestamp
        detections = {"health": HealthInfo(health=100, confidence=0.9, raw_text="100")}
        tracker.update(detections, timestamp=10.5)
        assert tracker.revival_first_detection_timestamp == 10.5

        # Continue detections
        tracker.update(detections, timestamp=11.0)
        tracker.update(detections, timestamp=12.0)

        # After revival, timestamp should be reset
        assert tracker.revival_first_detection_timestamp is None

    def test_abilities_only_update_when_alive(self, tracker):
        """Test that abilities only update when player is alive."""
        # Alive: abilities update
        detections = {
            "health": HealthInfo(health=100, confidence=0.9, raw_text="100"),
            "ability_1": AbilityInfo(charges=2, total_blobs_detected=2),
        }
        tracker.update(detections, timestamp=10.0)
        assert tracker.current_state["ability_1"] == 2

        # Dead: abilities don't update
        tracker.current_state["alive"] = False
        detections = {
            "health": HealthInfo(health=0, confidence=0.9, raw_text="0"),
            "ability_1": AbilityInfo(charges=1, total_blobs_detected=1),
        }
        tracker.update(detections, timestamp=11.0)
        # ability_1 should stay at 2 (not update to 1)
        assert tracker.current_state["ability_1"] == 2

    def test_ultimate_updates_regardless_of_alive(self, tracker):
        """Test that ultimate updates even when dead."""
        # Dead player
        tracker.current_state["alive"] = False

        detections = {
            "health": HealthInfo(health=0, confidence=0.9, raw_text="0"),
            "ultimate": UltimateInfo(charges=7, is_full=True, total_blobs_detected=7),
        }
        tracker.update(detections, timestamp=10.0)

        # Ultimate should update even when dead
        assert tracker.current_state["ultimate"]["charges"] == 7
        assert tracker.current_state["ultimate"]["is_full"] is True

    def test_set_metadata(self, tracker):
        """Test updating player metadata."""
        new_metadata = {
            "name": "s0m",
            "team": "NRG",
            "agent": "omen",
        }
        tracker.set_metadata(new_metadata)

        assert tracker.metadata["name"] == "s0m"
        assert tracker.metadata["agent"] == "omen"
        assert tracker.metadata["team"] == "NRG"

    def test_get_state_changes(self, tracker):
        """Test detecting state changes."""
        # Initial state
        detections = {
            "health": HealthInfo(health=100, confidence=0.9, raw_text="100"),
            "armor": ArmorInfo(armor=50, confidence=0.9, raw_text="50"),
        }
        tracker.update(detections, timestamp=10.0)

        # Change health and armor
        detections = {
            "health": HealthInfo(health=75, confidence=0.9, raw_text="75"),
            "armor": ArmorInfo(armor=25, confidence=0.9, raw_text="25"),
        }
        tracker.update(detections, timestamp=11.0)

        changes = tracker.get_state_changes()
        assert "health" in changes
        assert changes["health"]["old"] == 100
        assert changes["health"]["new"] == 75
        assert "armor" in changes
        assert changes["armor"]["old"] == 50
        assert changes["armor"]["new"] == 25

    def test_is_alive_transition(self, tracker):
        """Test detecting alive transitions."""
        # Start dead
        tracker.current_state["alive"] = False
        tracker.previous_state["alive"] = False

        # Simulate revival (instant for this test)
        tracker.current_state["alive"] = True

        assert tracker.is_alive_transition() is True

    def test_is_death_transition(self, tracker):
        """Test detecting death transitions."""
        # Start alive
        tracker.current_state["alive"] = True
        tracker.previous_state["alive"] = True

        # Die
        detections = {
            "health": HealthInfo(health=0, confidence=0.9, raw_text="0"),
        }
        tracker.update(detections, timestamp=10.0)

        assert tracker.is_death_transition() is True

    def test_reset_for_new_round(self, tracker, mock_detections):
        """Test resetting state for a new round."""
        # Set some state
        tracker.update(mock_detections, timestamp=10.0)
        tracker.current_state["alive"] = False
        tracker.revival_candidate_count = 2

        # Reset for new round
        tracker.reset_for_new_round()

        assert tracker.current_state["alive"] is True
        assert tracker.current_state["health"] is None
        assert tracker.current_state["armor"] is None
        assert tracker.current_state["ability_1"] is None
        assert tracker.revival_candidate_count == 0
        assert tracker.revival_first_detection_timestamp is None

    def test_repr(self, tracker):
        """Test string representation."""
        detections = {
            "health": HealthInfo(health=100, confidence=0.9, raw_text="100"),
        }
        tracker.update(detections, timestamp=10.0)

        repr_str = repr(tracker)
        assert "brawk" in repr_str
        assert "alive" in repr_str
        assert "100" in repr_str

    def test_no_detections_preserves_state(self, tracker):
        """Test that lack of detections preserves existing state."""
        # Set initial state
        detections = {
            "health": HealthInfo(health=100, confidence=0.9, raw_text="100"),
            "armor": ArmorInfo(armor=50, confidence=0.9, raw_text="50"),
        }
        tracker.update(detections, timestamp=10.0)

        # Update with no detections
        tracker.update({"health": None}, timestamp=11.0)

        # State should be preserved (except moved to previous)
        assert tracker.current_state["health"] == 100
        assert tracker.current_state["armor"] == 50


def test_preround_update_without_health_preserves_revival_candidate():
    tracker = PlayerStateTracker(0)
    tracker.current_state["alive"] = False
    tracker.update({"health": HealthInfo(health=100, confidence=0.9)}, 10.0)
    tracker.update({}, 10.25)
    assert tracker.revival_candidate_count == 1
    assert tracker.revival_first_detection_timestamp == 10.0


def test_missing_health_confirms_death_after_multiple_gameplay_observations():
    tracker = PlayerStateTracker(0)
    tracker.update({"health": HealthInfo(health=100, confidence=0.9)}, 10.0)
    tracker.update({"health": None}, 10.25)
    assert tracker.current_state["alive"] is True
    tracker.update({"health": None}, 10.5)
    assert tracker.current_state["alive"] is False
