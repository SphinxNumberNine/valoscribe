from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from valoscribe.detectors.active_round_agent_detector import ActiveRoundAgentDetector
from valoscribe.orchestration.game_state_manager import GameStateManager
from valoscribe.orchestration.player_state_tracker import PlayerStateTracker


@pytest.fixture
def detector(tmp_path):
    rng = np.random.default_rng(42)
    for name in ["sova", "jett"]:
        cv2.imwrite(str(tmp_path / f"{name}.png"), rng.integers(
            0, 256, (12, 12, 3), dtype=np.uint8
        ))
    return ActiveRoundAgentDetector(Mock(), template_dir=tmp_path)


@pytest.mark.parametrize("greyscale", [False, True])
def test_prepared_templates_match_original_conversion(detector, greyscale):
    template = detector.templates["sova"]["image"]
    crop = np.pad(template, ((3, 3), (3, 3), (0, 0)))
    prepared = detector._preprocess_crop(crop)
    if greyscale:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    scores = {}
    for name, data in detector.templates.items():
        original = data["image"]
        if greyscale:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        scores[name] = cv2.minMaxLoc(cv2.matchTemplate(
            prepared, original, detector.match_method
        ))[1]
    result = detector._match_all_templates(crop, greyscale)
    assert result[0] == max(scores, key=scores.get)
    assert result[1] == pytest.approx(scores[result[0]])


def test_repeated_greyscale_lookup_converts_only_current_crop(detector):
    crop = detector.templates["sova"]["image"].copy()
    with patch.object(cv2, "cvtColor", wraps=cv2.cvtColor) as convert:
        for _ in range(3):
            assert detector._match_all_templates(crop, True)[0] == "sova"
    assert convert.call_count == 3
    detector.set_agent_filter(["sova"])
    assert set(detector.templates) == {"sova"}
    assert detector._match_all_templates(crop, True)[0] == "sova"


def test_slot_search_tracks_clutch_movement_and_greyscale_fallback(detector):
    manager = GameStateManager.__new__(GameStateManager)
    manager.detector_registry = Mock(inround_agent_detector=detector)
    tracker = PlayerStateTracker(0, {"agent": "sova", "name": "test", "team": "A"})
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    icon = detector.templates["sova"]["image"]
    blank = np.zeros_like(icon)
    for slot in [4, 0, 2, None]:
        detector.cropper.crop_player_info.return_value = [
            {"agent_icon": icon if index == slot else blank} for index in range(10)
        ]
        assert manager._find_player_agent_slot(frame, tracker) == slot
    grey = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    grey = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    detector.cropper.crop_player_info.return_value = [
        {"agent_icon": grey if index == 3 else blank} for index in range(10)
    ]
    assert detector.detect(frame, 3) is None
    assert detector.detect(frame, 3, greyscale=True) == "sova"
    assert manager._find_player_agent_slot(frame, tracker) == 3


def test_slot_search_does_not_match_opposing_team_icon(detector):
    manager = GameStateManager.__new__(GameStateManager)
    manager.detector_registry = Mock(inround_agent_detector=detector)
    icon = detector.templates["sova"]["image"]
    detector.cropper.crop_player_info.return_value = [
        {"agent_icon": icon if index == 7 else np.zeros_like(icon)} for index in range(10)
    ]
    tracker = PlayerStateTracker(0, {"agent": "sova", "name": "test", "team": "A"})
    assert manager._find_player_agent_slot(np.zeros((1, 1, 3)), tracker) is None
