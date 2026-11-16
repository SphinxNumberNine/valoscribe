from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from valoscribe.detectors.killfeed_detector import KillfeedDetector


@pytest.fixture
def detector(tmp_path):
    rng = np.random.default_rng(7)
    for side in ("attack", "defense"):
        side_dir = tmp_path / side
        side_dir.mkdir()
        for agent in ("sova", "jett"):
            cv2.imwrite(
                str(side_dir / f"{agent}.png"),
                rng.integers(0, 256, (14, 18, 3), dtype=np.uint8),
            )
    return KillfeedDetector(Mock(), template_dir=tmp_path)


def entries_with_hits(*hit_slots):
    entries = [np.zeros((35, 120, 3), dtype=np.uint8) for _ in range(10)]
    for slot in hit_slots:
        entries[slot] = np.full((35, 120, 3), 40, dtype=np.uint8)
    return entries


def test_prepared_templates_match_their_own_image(detector):
    crops = [
        np.pad(data["image"], ((4, 4), (4, 4), (0, 0)))
        for data in detector.templates.values()
    ]
    for data, crop in zip(detector.templates.values(), crops):
        preprocessed = detector._preprocess_crop(crop)
        result = detector._match_all_templates(crop, flipped=False)
        assert result is not None
        assert result[0] == data["agent"]
        score = cv2.minMaxLoc(
            cv2.matchTemplate(preprocessed, data["image_match"], detector.match_method)
        )[1]
        assert score == pytest.approx(result[2])


def test_downscaled_templates_are_cached_at_match_scale(tmp_path):
    side_dir = tmp_path / "attack"
    side_dir.mkdir()
    cv2.imwrite(str(side_dir / "sova.png"), np.full((20, 30, 3), 128, dtype=np.uint8))
    scaled = KillfeedDetector(Mock(), template_dir=tmp_path, match_scale=0.5)
    template = scaled.templates["sova_attack"]
    assert template["image_match"].shape[:2] == (10, 15)
    assert template["image_match_flipped"].shape == template["image_match"].shape
    assert template["image"].shape[:2] == (20, 30)


@pytest.mark.parametrize("scale", [0.0, 0.05, 1.5])
def test_match_scale_is_validated(tmp_path, scale):
    with pytest.raises(ValueError):
        KillfeedDetector(Mock(), template_dir=tmp_path, match_scale=scale)


def _entry_candidates(hit_slots):
    def fake_entry(preprocessed, entry_idx):
        return [(entry_idx, Mock())] if entry_idx in hit_slots else []

    return fake_entry


def test_scan_all_slots_reaches_entries_below_gaps(detector):
    detector.cropper.crop_killfeed.return_value = entries_with_hits(0, 4)
    detector._detect_entry = _entry_candidates((0, 4))
    candidates = detector.detect(np.zeros((1, 1, 3), dtype=np.uint8))
    assert [idx for idx, _ in candidates] == [0, 4]


def test_stopping_after_consecutive_empty_entries_can_hide_later_hits(detector):
    detector.max_consecutive_empty_entries = 2
    detector.cropper.crop_killfeed.return_value = entries_with_hits(2)
    detector._detect_entry = _entry_candidates((2,))
    assert detector.detect(np.zeros((1, 1, 3), dtype=np.uint8)) == []

    detector.max_consecutive_empty_entries = 3
    assert len(detector.detect(np.zeros((1, 1, 3), dtype=np.uint8))) == 1


def test_preprocess_does_not_copy_bgr_frames(detector):
    crop = np.zeros((35, 120, 3), dtype=np.uint8)
    assert detector._preprocess_crop(crop) is crop
