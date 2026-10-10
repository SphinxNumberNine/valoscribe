"""Unit tests for TimerManager."""

from __future__ import annotations
import pytest

from valoscribe.orchestration.timer_manager import TimerManager
from valoscribe.orchestration.phase_detector import Phase


class TestTimerManager:
    """Tests for TimerManager class."""

    @pytest.fixture
    def manager(self):
        """Create TimerManager instance."""
        return TimerManager()

    def test_init(self, manager):
        """Test TimerManager initialization."""
        assert manager.spike_planted_at is None
        assert manager.round_ended_at is None

    def test_on_spike_planted(self, manager):
        """Test spike plant tracking."""
        manager.on_spike_planted(45.0)

        assert manager.spike_planted_at == 45.0

    def test_on_round_ended(self, manager):
        """Test round end tracking."""
        manager.on_round_ended(100.0)

        assert manager.round_ended_at == 100.0

    def test_reset_for_new_round(self, manager):
        """Test reset for new round."""
        manager.on_spike_planted(45.0)
        manager.on_round_ended(100.0)

        manager.reset_for_new_round()

        assert manager.spike_planted_at is None
        assert manager.round_ended_at is None

    def test_get_timers_preround(self, manager):
        """Test get_timers during PREROUND."""
        timers = manager.get_timers(
            timestamp=5.0,
            phase=Phase.PREROUND,
            game_timer=None,
        )

        assert timers["game_timer"] is None
        assert timers["spike_timer"] is None
        assert timers["post_round_timer"] is None

    def test_get_timers_active_round_pre_plant(self, manager):
        """Test get_timers during ACTIVE_ROUND before spike plant."""
        timers = manager.get_timers(
            timestamp=10.0,
            phase=Phase.ACTIVE_ROUND,
            game_timer=95.5,
        )

        assert timers["game_timer"] == 95.5
        assert timers["spike_timer"] is None
        assert timers["post_round_timer"] is None

    def test_get_timers_active_round_post_plant(self, manager):
        """Test get_timers during ACTIVE_ROUND after spike plant."""
        manager.on_spike_planted(45.0)

        timers = manager.get_timers(
            timestamp=46.0,
            phase=Phase.ACTIVE_ROUND,
            game_timer=None,  # Timer not visible post-plant
        )

        assert timers["game_timer"] is None
        assert timers["spike_timer"] == 1.0
        assert timers["post_round_timer"] is None

    def test_get_timers_post_round(self, manager):
        """Test get_timers during POST_ROUND."""
        manager.on_spike_planted(45.0)
        manager.on_round_ended(100.0)

        timers = manager.get_timers(
            timestamp=105.0,
            phase=Phase.POST_ROUND,
            game_timer=None,
        )

        assert timers["game_timer"] is None
        assert timers["spike_timer"] is None  # Hidden after round end
        assert timers["post_round_timer"] == 5.0  # 105 - 100

    def test_spike_timer_increments(self, manager):
        """Test that spike timer increments correctly."""
        manager.on_spike_planted(45.0)

        timers1 = manager.get_timers(45.0, Phase.ACTIVE_ROUND, None)
        assert timers1["spike_timer"] == 0.0

        timers2 = manager.get_timers(45.25, Phase.ACTIVE_ROUND, None)
        assert timers2["spike_timer"] == 0.25

        timers3 = manager.get_timers(50.0, Phase.ACTIVE_ROUND, None)
        assert timers3["spike_timer"] == 5.0

    def test_post_round_timer_increments(self, manager):
        """Test that post_round_timer increments correctly."""
        manager.on_round_ended(100.0)

        timers1 = manager.get_timers(100.0, Phase.POST_ROUND, None)
        assert timers1["post_round_timer"] == 0.0

        timers2 = manager.get_timers(100.5, Phase.POST_ROUND, None)
        assert timers2["post_round_timer"] == 0.5

        timers3 = manager.get_timers(107.0, Phase.POST_ROUND, None)
        assert timers3["post_round_timer"] == 7.0

    def test_game_timer_not_shown_post_plant(self, manager):
        """Test that game_timer is None after spike plant even if provided."""
        manager.on_spike_planted(45.0)

        timers = manager.get_timers(
            timestamp=46.0,
            phase=Phase.ACTIVE_ROUND,
            game_timer=50.0,  # Shouldn't be visible, but provide anyway
        )

        # game_timer should be None because spike is planted
        assert timers["game_timer"] is None
        assert timers["spike_timer"] == 1.0

    def test_repr(self, manager):
        """Test string representation."""
        repr_str = repr(manager)

        assert "TimerManager" in repr_str
        assert "spike_planted_at" in repr_str
        assert "round_ended_at" in repr_str

    def test_multiple_rounds(self, manager):
        """Test timer manager across multiple rounds."""
        # Round 1
        manager.on_spike_planted(45.0)
        manager.on_round_ended(100.0)

        timers1 = manager.get_timers(105.0, Phase.POST_ROUND, None)
        assert timers1["spike_timer"] is None
        assert timers1["post_round_timer"] == 5.0

        # Reset for round 2
        manager.reset_for_new_round()

        timers2 = manager.get_timers(110.0, Phase.ACTIVE_ROUND, 95.0)
        assert timers2["game_timer"] == 95.0
        assert timers2["spike_timer"] is None
        assert timers2["post_round_timer"] is None

        # Round 2 spike plant
        manager.on_spike_planted(150.0)

        timers3 = manager.get_timers(155.0, Phase.ACTIVE_ROUND, None)
        assert timers3["game_timer"] is None
        assert timers3["spike_timer"] == 5.0
        assert timers3["post_round_timer"] is None
