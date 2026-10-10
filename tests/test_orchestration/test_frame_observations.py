from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from valoscribe.orchestration.game_state_manager import GameStateManager
from valoscribe.orchestration.phase_detector import Phase


@pytest.fixture
def manager(tmp_path):
    instance = GameStateManager(
        Path("test.mp4"),
        {
            "teams": [
                {"name": "A", "starting_side": "attack"},
                {"name": "B", "starting_side": "defense"},
            ],
            "players": [],
        },
        tmp_path,
    )
    instance.player_trackers = []
    instance.output_writer = Mock()
    instance._handle_phase_transition = Mock()
    instance.detector_registry.killfeed_detector.detect = Mock(return_value=[])
    return instance


@pytest.mark.parametrize(
    "timer,spike,score,credits,initial,expected",
    [
        (90, False, True, False, Phase.ACTIVE_ROUND, Phase.ACTIVE_ROUND),
        (None, True, True, False, Phase.ACTIVE_ROUND, Phase.ACTIVE_ROUND),
        (5, False, True, False, Phase.POST_ROUND, Phase.POST_ROUND),
        (None, False, True, False, Phase.ACTIVE_ROUND, Phase.NON_GAME),
        (90, False, False, False, Phase.ACTIVE_ROUND, Phase.NON_GAME),
        (30, False, False, True, Phase.PREROUND, Phase.PREROUND),
    ],
)
def test_reuse_preserves_game_frame_gates(manager, timer, spike, score, credits, initial, expected):
    registry = manager.detector_registry
    registry.timer_detector.detect = Mock(
        return_value=(SimpleNamespace(time_seconds=timer) if timer is not None else None)
    )
    registry.spike_detector.detect = Mock(return_value=SimpleNamespace(spike_planted=spike))
    registry.score_detector.detect = Mock(
        return_value=(SimpleNamespace(team1_score=0, team2_score=0) if score else None)
    )
    registry.preround_credits_detector.detect = Mock(
        return_value=SimpleNamespace(credits_visible=credits)
    )
    manager.current_phase = initial
    if expected == Phase.PREROUND:
        # Isolate player observations; the existing preround score check remains necessary.
        manager._process_preround = Mock()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    manager.process_frame(10.0, frame)

    registry.timer_detector.detect.assert_called_once_with(frame)
    if expected == Phase.NON_GAME:
        assert manager.current_phase == initial
        manager.output_writer.write_frame_state.assert_not_called()
        manager._handle_phase_transition.assert_not_called()
    else:
        assert manager.current_phase == expected
        manager.output_writer.write_frame_state.assert_called_once()
        if expected == Phase.ACTIVE_ROUND and spike:
            assert manager.timer_manager.spike_planted_at == 10.0
    if expected != Phase.PREROUND and (timer is not None or spike):
        registry.score_detector.detect.assert_called_once_with(frame)


def test_failed_timer_observation_is_reused_and_next_frame_is_fresh(manager):
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    detector = manager.detector_registry.timer_detector
    detector.detect = Mock(side_effect=AssertionError("must not redetect"))
    manager.current_phase = Phase.ACTIVE_ROUND
    assert (
        manager._add_timers_to_event({}, 1, frame, {"timer": None})["timers"]["game_timer"] is None
    )
    assert (
        manager._add_timers_to_event({}, 2, frame, {"timer": SimpleNamespace(time_seconds=88)})[
            "timers"
        ]["game_timer"]
        == 88
    )
    detector.detect.assert_not_called()
