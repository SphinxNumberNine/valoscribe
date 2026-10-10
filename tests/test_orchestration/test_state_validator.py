"""Unit tests for StateValidator."""

from __future__ import annotations
import pytest
from pathlib import Path

from valoscribe.orchestration.state_validator import StateValidator


class TestStateValidator:
    """Tests for StateValidator class."""

    @pytest.fixture
    def validator(self):
        """Create StateValidator instance."""
        return StateValidator()

    @pytest.fixture
    def metadata(self):
        """Sample player metadata."""
        return {"name": "brawk", "team": "NRG", "agent": "sova"}

    def test_init(self, validator):
        """Test StateValidator initialization."""
        assert validator.agent_config is not None
        assert len(validator.agent_config) > 0
        assert "sova" in validator.agent_config

    def test_load_agent_config(self, validator):
        """Test loading agent config."""
        assert "jett" in validator.agent_config
        assert "ability_1" in validator.agent_config["jett"]
        assert "ability_2" in validator.agent_config["jett"]
        assert "ability_3" in validator.agent_config["jett"]
        assert "ultimate" in validator.agent_config["jett"]

    def test_ability_config_structure(self, validator):
        """Test that ability config has correct structure."""
        jett_config = validator.agent_config["jett"]

        # Check ability_1 structure
        assert "max_charges" in jett_config["ability_1"]
        assert "rechargeable" in jett_config["ability_1"]
        assert isinstance(jett_config["ability_1"]["max_charges"], int)
        assert isinstance(jett_config["ability_1"]["rechargeable"], bool)

    def test_death_handled_by_manager(self, validator, metadata):
        """Validator must not duplicate deaths emitted by GameStateManager."""
        previous = {"alive": True, "health": 100}
        current = {"alive": False, "health": 0}
        assert validator.validate_player_state(current, previous, metadata, 10.0) == []

    def test_revival_detection(self, validator, metadata):
        """Test detection of revival event."""
        previous = {"alive": False, "health": 0}
        current = {"alive": True, "health": 100}

        events = validator.validate_player_state(current, previous, metadata, 15.0, team_has_sage=True)

        assert len(events) == 1
        assert events[0]["type"] == "revival"
        assert events[0]["player"] == "brawk"
        assert events[0]["timestamp"] == 15.0

    def test_no_alive_change_no_event(self, validator, metadata):
        """Test that no event is generated when alive status doesn't change."""
        previous = {"alive": True, "health": 100}
        current = {"alive": True, "health": 80}

        events = validator.validate_player_state(current, previous, metadata, 10.0)

        # No death or revival events
        assert not any(e["type"] in ["death", "revival"] for e in events)

    def test_ability_usage(self, validator, metadata):
        previous = {"alive": True, "ability_2": 2}
        current = {"alive": True, "ability_2": 1}
        assert validator.validate_player_state(current, previous, metadata, 10.0) == []
        events = validator.validate_player_state(current, current, metadata, 10.25)
        assert len(events) == 1
        assert events[0]["type"] == "ability_used"
        assert events[0]["ability"] == "shock dart"
        assert events[0]["charges_used"] == 1
        assert events[0]["remaining_charges"] == 1
        assert events[0]["player"] == "brawk"
        assert events[0]["timestamp"] == 10.0
        assert validator.validate_player_state(current, current, metadata, 10.5) == []

    def test_ability_recharge_valid(self, validator, metadata):
        """Test detection of valid ability recharge."""
        # Omen's ability_3 is rechargeable
        metadata_omen = {"name": "marved", "team": "NRG", "agent": "omen"}
        previous = {"alive": True, "ability_3": 1}
        current = {"alive": True, "ability_3": 2}

        assert validator.validate_player_state(current, previous, metadata_omen, 10.0) == []
        events = validator.validate_player_state(current, current, metadata_omen, 10.25)

        assert len(events) == 1
        assert events[0]["type"] == "ability_recharged"
        assert events[0]["ability"] == "dark cover"
        assert events[0]["charges_gained"] == 1
        assert events[0]["total_charges"] == 2

    def test_ability_recharge_invalid(self, validator, metadata):
        """Confirmed owl drone charge gains are rejected during gameplay."""
        previous = {"alive": True, "ability_1": 0}
        current = {"alive": True, "ability_1": 1}
        assert validator.validate_player_state(current, previous, metadata, 10.0) == []
        assert validator.validate_player_state(current, current, metadata, 10.25) == []

    def test_ability_no_change(self, validator, metadata):
        """Test that no event is generated when ability charges don't change."""
        previous = {"alive": True, "ability_1": 0}
        current = {"alive": True, "ability_1": 0}

        events = validator.validate_player_state(current, previous, metadata, 10.0)

        assert not any(e["type"] in ["ability_used", "ability_recharged"] for e in events)

    def test_ability_none_skipped(self, validator, metadata):
        """Test that abilities with None values are skipped."""
        previous = {"alive": True, "ability_1": None}
        current = {"alive": True, "ability_1": 0}

        events = validator.validate_player_state(current, previous, metadata, 10.0)

        # No event because previous was None
        assert not any(e["type"] in ["ability_used", "ability_recharged"] for e in events)

    def test_ultimate_usage(self, validator, metadata):
        """Test detection of ultimate usage."""
        previous = {"alive": True, "ultimate": {"charges": 8, "is_full": True}}
        current = {"alive": True, "ultimate": {"charges": 0, "is_full": False}}

        events = validator.validate_player_state(current, previous, metadata, 10.0)

        assert len(events) == 1
        assert events[0]["type"] == "ultimate_used"
        assert events[0]["player"] == "brawk"
        assert events[0]["previous_charges"] == 8
        assert events[0]["current_charges"] == 0

    def test_ultimate_charging_no_event(self, validator, metadata):
        """Test that ultimate charging doesn't generate event."""
        previous = {"alive": True, "ultimate": {"charges": 5, "is_full": False}}
        current = {"alive": True, "ultimate": {"charges": 6, "is_full": False}}

        events = validator.validate_player_state(current, previous, metadata, 10.0)

        # No ultimate_used event when just charging
        assert not any(e["type"] == "ultimate_used" for e in events)

    def test_ultimate_becomes_full_no_event(self, validator, metadata):
        """Test that ultimate becoming full doesn't generate event."""
        previous = {"alive": True, "ultimate": {"charges": 7, "is_full": False}}
        current = {"alive": True, "ultimate": {"charges": 8, "is_full": True}}

        events = validator.validate_player_state(current, previous, metadata, 10.0)

        # No event when ultimate becomes full (only when used)
        assert not any(e["type"] == "ultimate_used" for e in events)

    def test_multiple_events_same_frame(self, validator, metadata):
        """Test multiple events in single validation."""
        previous = {
            "alive": True,
            "ability_1": 1,
            "ability_2": 1,
            "ultimate": {"charges": 8, "is_full": True},
        }
        current = {
            "alive": True,
            "ability_1": 0,
            "ability_2": 0,
            "ultimate": {"charges": 0, "is_full": False},
        }

        first = validator.validate_player_state(current, previous, metadata, 10.0)
        assert [event["type"] for event in first] == ["ultimate_used"]
        confirmed = validator.validate_player_state(current, current, metadata, 10.25)
        assert len(confirmed) == 2
        assert all(event["timestamp"] == 10.0 for event in confirmed)
        events = first + confirmed

        # Should have 3 events: 2 abilities + 1 ultimate
        assert len(events) == 3
        event_types = [e["type"] for e in events]
        assert event_types.count("ability_used") == 2
        assert event_types.count("ultimate_used") == 1

    def test_unknown_agent_skips_ability_validation(self, validator):
        """Test that unknown agent skips ability validation."""
        metadata = {"name": "test", "team": "TEST", "agent": "unknown_agent"}
        previous = {"alive": True, "ability_1": 1}
        current = {"alive": True, "ability_1": 0}

        events = validator.validate_player_state(current, previous, metadata, 10.0)

        # No ability events for unknown agent
        assert not any(e["type"] in ["ability_used", "ability_recharged"] for e in events)

    def test_all_three_abilities(self, validator, metadata):
        """Test validation of all three abilities."""
        previous = {
            "alive": True,
            "ability_1": 1,
            "ability_2": 1,
            "ability_3": 1,
        }
        current = {
            "alive": True,
            "ability_1": 0,
            "ability_2": 0,
            "ability_3": 0,
        }

        assert validator.validate_player_state(current, previous, metadata, 10.0) == []
        events = validator.validate_player_state(current, current, metadata, 10.25)

        # Should detect usage of all 3 abilities
        assert len(events) == 3
        abilities_used = [e["ability"] for e in events]
        assert "owl drone" in abilities_used
        assert "shock dart" in abilities_used
        assert "recon bolt" in abilities_used

    def test_repr(self, validator):
        """Test string representation."""
        repr_str = repr(validator)
        assert "StateValidator" in repr_str
        assert "agents=" in repr_str

    def test_real_agent_configs(self, validator):
        """Test that common agents have correct config."""
        # Test a few key agents
        for agent in ["jett", "sova", "omen", "sage", "raze"]:
            assert agent in validator.agent_config
            config = validator.agent_config[agent]

            # All should have 3 abilities + ultimate
            assert "ability_1" in config
            assert "ability_2" in config
            assert "ability_3" in config
            assert "ultimate" in config

            # All abilities should have max_charges and rechargeable
            for ability in ["ability_1", "ability_2", "ability_3"]:
                assert "max_charges" in config[ability]
                assert "rechargeable" in config[ability]
                assert config[ability]["max_charges"] > 0


@pytest.mark.parametrize("interruption", [None, 2])
def test_ability_confirmation_resets_on_missing_or_reverted_observation(interruption):
    validator = StateValidator()
    metadata = {"name": "test", "team": "NRG", "agent": "sova"}
    old = {"alive": True, "ability_2": 2}
    used = {"alive": True, "ability_2": 1}
    interrupted = {"alive": True, "ability_2": interruption}
    assert validator.validate_player_state(used, old, metadata, 10.0) == []
    assert validator.validate_player_state(interrupted, used, metadata, 10.25) == []
    assert validator.validate_player_state(used, old, metadata, 10.5) == []
    events = validator.validate_player_state(used, used, metadata, 10.75)
    assert len(events) == 1
    assert events[0]["timestamp"] == 10.5


def test_revival_requires_sage_on_team():
    validator = StateValidator()
    metadata = {"name": "test", "team": "NRG", "agent": "sova"}
    assert validator.validate_player_state(
        {"alive": True}, {"alive": False}, metadata, 10.0, team_has_sage=False
    ) == []
