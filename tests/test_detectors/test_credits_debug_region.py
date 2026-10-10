from unittest.mock import Mock

import numpy as np
import pytest

from valoscribe.detectors.template_credits_detector import TemplateCreditsDetector


@pytest.mark.parametrize("side", ["left", "right"])
def test_debug_uses_same_dead_credits_pixels_as_detection(side):
    crop = np.random.default_rng(42).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    cropper = Mock()
    cropper.crop_player_info.return_value = [{
        "side": side,
        "dead_credits": crop,
        "credits": np.zeros_like(crop),
    }]
    detector = TemplateCreditsDetector(cropper)
    detector.template = detector._preprocess_crop(crop)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    regular = detector.detect(frame, 0, side)
    debug, pixels, details = detector.detect_with_debug(frame, 0, side)

    assert debug == regular
    assert debug.credits_visible
    assert np.array_equal(pixels, detector.template)
    assert details["max_confidence"] == pytest.approx(regular.confidence)


def test_debug_supports_crop_with_only_dead_credits():
    cropper = Mock()
    cropper.crop_player_info.return_value = [{
        "side": "left", "dead_credits": np.zeros((20, 20, 3), dtype=np.uint8)
    }]
    detector = TemplateCreditsDetector(cropper)
    detector.template = np.ones((10, 10), dtype=np.uint8)
    result, pixels, details = detector.detect_with_debug(np.zeros((1, 1, 3)), 0)
    assert result is not None
    assert pixels.size > 0
    assert details


@pytest.mark.parametrize("dead_icon_present", [False, True])
def test_normal_credits_location_does_not_determine_dead_credits_signal(dead_icon_present):
    icon = np.random.default_rng(42).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    blank = np.zeros_like(icon)
    cropper = Mock()
    cropper.crop_player_info.return_value = [{
        "side": "left",
        "credits": blank if dead_icon_present else icon,
        "dead_credits": icon if dead_icon_present else blank,
    }]
    detector = TemplateCreditsDetector(cropper)
    detector.template = detector._preprocess_crop(icon)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    regular = detector.detect(frame, 0)
    debug, pixels, details = detector.detect_with_debug(frame, 0)
    assert regular.credits_visible is dead_icon_present
    assert debug == regular
    assert np.array_equal(pixels, detector._preprocess_crop(
        cropper.crop_player_info.return_value[0]["dead_credits"]
    ))
    assert details["max_confidence"] == pytest.approx(regular.confidence)
