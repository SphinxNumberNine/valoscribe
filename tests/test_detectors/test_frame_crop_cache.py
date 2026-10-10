"""Sharing crops must preserve pixel values and never cross frame boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from valoscribe.detectors.cropper import Cropper
from valoscribe.orchestration.game_state_manager import GameStateManager


@pytest.fixture
def frame():
    return np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)


def assert_same_crops(actual, expected):
    assert len(actual) == len(expected)
    for actual_player, expected_player in zip(actual, expected):
        assert actual_player.keys() == expected_player.keys()
        for key in actual_player:
            if key == "side":
                assert actual_player[key] == expected_player[key]
            else:
                np.testing.assert_array_equal(actual_player[key], expected_player[key])


@pytest.mark.parametrize("method", ["crop_player_info", "crop_player_info_preround"])
def test_scoped_crops_are_identical_and_computed_once(frame, method):
    cropper = Cropper()
    crop = getattr(cropper, method)
    expected = crop(frame)
    with patch.object(cropper, "_crop_side_players", wraps=cropper._crop_side_players) as sides:
        with cropper.frame_scope(frame):
            first = crop(frame)
            for _ in range(60):
                assert crop(frame) is first
            assert_same_crops(first, expected)
        assert sides.call_count == 2
        assert crop(frame) is not first
        assert sides.call_count == 4


def test_preround_and_active_layouts_have_separate_caches(frame):
    cropper = Cropper()
    expected_active = cropper.crop_player_info(frame)
    expected_preround = cropper.crop_player_info_preround(frame)
    with cropper.frame_scope(frame):
        active = cropper.crop_player_info(frame)
        preround = cropper.crop_player_info_preround(frame)
        assert active is not preround
        assert_same_crops(active, expected_active)
        assert_same_crops(preround, expected_preround)


def test_reused_frame_buffer_gets_fresh_crops_in_next_scope(frame):
    cropper = Cropper()
    with cropper.frame_scope(frame):
        first = cropper.crop_player_info(frame)
    frame.fill(0)
    with cropper.frame_scope(frame):
        second = cropper.crop_player_info(frame)
    assert second is not first
    assert not np.any(second[0]["health"])
    assert np.any(first[0]["health"])


def test_other_frame_is_not_cached_and_nested_scope_restores_outer(frame):
    cropper = Cropper()
    other = np.zeros_like(frame)
    with cropper.frame_scope(frame):
        first = cropper.crop_player_info(frame)
        assert cropper.crop_player_info(other) is not cropper.crop_player_info(other)
        with cropper.frame_scope(other):
            second = cropper.crop_player_info(other)
            assert cropper.crop_player_info(other) is second
        assert cropper.crop_player_info(frame) is first


def test_exception_releases_cached_frame(frame):
    cropper = Cropper()
    with pytest.raises(RuntimeError):
        with cropper.frame_scope(frame):
            cropper.crop_player_info(frame)
            raise RuntimeError("detector failure")
    assert cropper._scoped_frame is None
    assert cropper._player_crop_cache == {}


def test_manager_shares_crops_and_cleans_up_after_early_exit(frame):
    # Exercise the public boundary without initializing every detector.
    manager = GameStateManager.__new__(GameStateManager)
    cropper = Cropper()
    manager.detector_registry = SimpleNamespace(cropper=cropper)

    def process(timestamp, current_frame):
        assert timestamp == 1.0
        assert cropper.crop_player_info(current_frame) is cropper.crop_player_info(current_frame)
        raise StopIteration

    manager._process_frame = process
    with pytest.raises(StopIteration):
        manager.process_frame(1.0, frame)
    assert cropper._scoped_frame is None
    assert cropper._player_crop_cache == {}
