"""Unit tests for pre-round credits detector."""

from __future__ import annotations
from unittest.mock import Mock
from pathlib import Path
import pytest
import numpy as np
import cv2

from valoscribe.detectors.preround_credits_detector import PreroundCreditsDetector
from valoscribe.types.detections import CreditsInfo


class TestPreroundCreditsDetector:
    """Tests for PreroundCreditsDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        return cropper

    @pytest.fixture
    def mock_template(self):
        """Create a simple mock template (white circle on black background)."""
        # Create a raw template that will be preprocessed
        template = np.zeros((20, 20), dtype=np.uint8)
        cv2.circle(template, (10, 10), 8, 255, -1)
        return template

    def preprocess_template_for_storage(self, template: np.ndarray) -> np.ndarray:
        """
        Preprocess template before saving to disk (simulates extract-preround-credits-crops).

        This matches the preprocessing done in _preprocess_crop:
        - Already grayscale
        - Upscale 3x
        - Otsu's thresholding
        """
        # Upscale 3x
        h, w = template.shape
        upscaled = cv2.resize(template, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        # Otsu's threshold
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    @pytest.fixture
    def detector_with_template(self, mock_cropper, mock_template, tmp_path):
        """Create detector with a valid preprocessed template."""
        template_path = tmp_path / "credits_icon_preround.png"

        # Preprocess template before saving (simulates extract-preround-credits-crops output)
        preprocessed_template = self.preprocess_template_for_storage(mock_template)
        cv2.imwrite(str(template_path), preprocessed_template)

        return PreroundCreditsDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.7,
        )

    def create_matching_crop(self, template: np.ndarray, crop_size: tuple = (30, 80)) -> np.ndarray:
        """
        Create a crop that will match the template after preprocessing.

        This simulates how the template would appear in a preround game frame.
        """
        h, w = crop_size
        crop = np.zeros((h, w, 3), dtype=np.uint8)

        # Place bright pattern that will survive preprocessing
        th, tw = template.shape
        y_offset = (h - th) // 2
        x_offset = (w - tw) // 2

        # Convert template to color and make it bright (white)
        for i in range(th):
            for j in range(tw):
                if template[i, j] > 0:
                    crop[y_offset + i, x_offset + j] = 255

        return crop

    def test_init_with_custom_template(self, mock_cropper, tmp_path):
        """Test detector initialization with custom template path."""
        template_path = tmp_path / "custom_preround_credits.png"
        template = np.zeros((20, 20), dtype=np.uint8)
        cv2.imwrite(str(template_path), template)

        detector = PreroundCreditsDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.8,
        )

        assert detector.cropper == mock_cropper
        assert detector.min_confidence == 0.8
        assert detector.template_path == template_path
        assert detector.template is not None

    def test_init_with_default_template_path(self, mock_cropper):
        """Test detector initialization with default template path."""
        # Don't provide template_path, let it use default
        detector = PreroundCreditsDetector(mock_cropper)

        # Default template path should be set correctly
        assert "templates/credits/credits_icon_preround.png" in detector.template_path.as_posix()

    def test_load_template_success(self, mock_cropper, mock_template, tmp_path):
        """Test successful template loading."""
        template_path = tmp_path / "credits_icon_preround.png"
        cv2.imwrite(str(template_path), mock_template)

        detector = PreroundCreditsDetector(mock_cropper, template_path=template_path)

        assert detector.template is not None
        assert detector.template.shape == mock_template.shape
        assert np.array_equal(detector.template, mock_template)

    def test_load_template_file_not_exists(self, mock_cropper, tmp_path):
        """Test loading template when file doesn't exist."""
        template_path = tmp_path / "nonexistent.png"

        detector = PreroundCreditsDetector(mock_cropper, template_path=template_path)

        assert detector.template is None

    def test_detect_no_template_loaded(self, mock_cropper):
        """Test detection fails gracefully when no template is loaded."""
        detector = PreroundCreditsDetector(mock_cropper, template_path=Path("/nonexistent/path.png"))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame, player_index=0)

        assert result is None

    def test_detect_player_out_of_range(self, detector_with_template, mock_cropper):
        """Test detection with invalid player index."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": np.zeros((30, 80, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=10)

        assert result is None

    def test_detect_wrong_side(self, detector_with_template, mock_cropper):
        """Test detection with wrong side specified."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": np.zeros((30, 80, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0, side="right")

        assert result is None

    def test_detect_missing_credits_region(self, detector_with_template, mock_cropper):
        """Test detection when credits region is not in crop data."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left"}  # No credits region
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0)

        assert result is None

    def test_detect_empty_crop(self, detector_with_template, mock_cropper):
        """Test detection with empty credits crop."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0)

        assert result is None

    def test_detect_credits_visible_high_confidence(self, detector_with_template, mock_cropper, mock_template):
        """Test detection when pre-round credits icon is clearly visible."""
        # Create credits crop that will match after preprocessing
        credits_crop = self.create_matching_crop(mock_template)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": credits_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0)

        assert result is not None
        assert isinstance(result, CreditsInfo)
        assert result.credits_visible is True
        assert result.confidence >= 0.7
        assert result.confidence <= 1.0

    def test_detect_credits_not_visible_low_confidence(self, detector_with_template, mock_cropper):
        """Test detection when credits icon is not visible."""
        # Create credits crop with no matching pattern
        credits_crop = np.random.randint(0, 50, (30, 80, 3), dtype=np.uint8)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": credits_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0)

        assert result is not None
        assert isinstance(result, CreditsInfo)
        assert result.credits_visible is False
        assert result.confidence < 0.7

    def test_detect_right_side_player(self, detector_with_template, mock_cropper, mock_template):
        """Test detection for right-side player."""
        credits_crop = self.create_matching_crop(mock_template)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": np.zeros((30, 80, 3), dtype=np.uint8)},
            {"side": "right", "credits": credits_crop},
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=1, side="right")

        assert result is not None
        assert result.credits_visible is True

    def test_detect_with_debug_success(self, detector_with_template, mock_cropper, mock_template):
        """Test debug detection with successful detection."""
        credits_crop = self.create_matching_crop(mock_template)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": credits_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        credits_info, preprocessed, debug_info = detector_with_template.detect_with_debug(
            frame, player_index=0
        )

        assert credits_info is not None
        assert credits_info.credits_visible is True
        assert isinstance(preprocessed, np.ndarray)
        assert preprocessed.size > 0
        assert "max_confidence" in debug_info
        assert "max_location" in debug_info
        assert "match_result_shape" in debug_info
        assert debug_info["max_confidence"] >= 0.7

    def test_detect_with_debug_no_match(self, detector_with_template, mock_cropper):
        """Test debug detection when no match found."""
        credits_crop = np.random.randint(0, 50, (30, 80, 3), dtype=np.uint8)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": credits_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        credits_info, preprocessed, debug_info = detector_with_template.detect_with_debug(
            frame, player_index=0
        )

        assert credits_info is not None
        assert credits_info.credits_visible is False
        assert preprocessed.size > 0
        assert debug_info["max_confidence"] < 0.7

    def test_is_preround_frame_all_visible(self, detector_with_template, mock_cropper, mock_template):
        """Test is_preround_frame when all players have visible credits (clear pre-round)."""
        # Create matching crops for all 10 players
        credits_crop = self.create_matching_crop(mock_template)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": credits_crop},  # Player 0
            {"side": "left", "credits": credits_crop},  # Player 1
            {"side": "left", "credits": credits_crop},  # Player 2
            {"side": "left", "credits": credits_crop},  # Player 3
            {"side": "left", "credits": credits_crop},  # Player 4
            {"side": "right", "credits": credits_crop},  # Player 5
            {"side": "right", "credits": credits_crop},  # Player 6
            {"side": "right", "credits": credits_crop},  # Player 7
            {"side": "right", "credits": credits_crop},  # Player 8
            {"side": "right", "credits": credits_crop},  # Player 9
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        is_preround = detector_with_template.is_preround_frame(frame, threshold=0.5)

        assert is_preround is True

    def test_is_preround_frame_none_visible(self, detector_with_template, mock_cropper):
        """Test is_preround_frame when no players have visible credits (clear in-round)."""
        # Create non-matching crops for all 10 players
        credits_crop = np.random.randint(0, 50, (30, 80, 3), dtype=np.uint8)

        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": credits_crop},  # Player 0
            {"side": "left", "credits": credits_crop},  # Player 1
            {"side": "left", "credits": credits_crop},  # Player 2
            {"side": "left", "credits": credits_crop},  # Player 3
            {"side": "left", "credits": credits_crop},  # Player 4
            {"side": "right", "credits": credits_crop},  # Player 5
            {"side": "right", "credits": credits_crop},  # Player 6
            {"side": "right", "credits": credits_crop},  # Player 7
            {"side": "right", "credits": credits_crop},  # Player 8
            {"side": "right", "credits": credits_crop},  # Player 9
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        is_preround = detector_with_template.is_preround_frame(frame, threshold=0.5)

        assert is_preround is False

    def test_is_preround_frame_exactly_at_threshold(self, detector_with_template, mock_cropper, mock_template):
        """Test is_preround_frame when exactly at threshold (50%)."""
        matching_crop = self.create_matching_crop(mock_template)
        non_matching_crop = np.random.randint(0, 50, (30, 80, 3), dtype=np.uint8)

        # 5 players with visible credits, 5 without (exactly 50%)
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": matching_crop},  # Player 0 - visible
            {"side": "left", "credits": matching_crop},  # Player 1 - visible
            {"side": "left", "credits": matching_crop},  # Player 2 - visible
            {"side": "left", "credits": matching_crop},  # Player 3 - visible
            {"side": "left", "credits": matching_crop},  # Player 4 - visible
            {"side": "right", "credits": non_matching_crop},  # Player 5 - not visible
            {"side": "right", "credits": non_matching_crop},  # Player 6 - not visible
            {"side": "right", "credits": non_matching_crop},  # Player 7 - not visible
            {"side": "right", "credits": non_matching_crop},  # Player 8 - not visible
            {"side": "right", "credits": non_matching_crop},  # Player 9 - not visible
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # At threshold (0.5), should be considered pre-round
        is_preround_at = detector_with_template.is_preround_frame(frame, threshold=0.5)
        assert is_preround_at is True

        # Just above threshold, should still be pre-round
        is_preround_above = detector_with_template.is_preround_frame(frame, threshold=0.49)
        assert is_preround_above is True

        # Just below threshold, should not be pre-round
        is_preround_below = detector_with_template.is_preround_frame(frame, threshold=0.51)
        assert is_preround_below is False

    def test_is_preround_frame_custom_threshold(self, detector_with_template, mock_cropper, mock_template):
        """Test is_preround_frame with custom thresholds."""
        matching_crop = self.create_matching_crop(mock_template)
        non_matching_crop = np.random.randint(0, 50, (30, 80, 3), dtype=np.uint8)

        # 7 out of 10 players have visible credits (70%)
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": matching_crop},  # Player 0
            {"side": "left", "credits": matching_crop},  # Player 1
            {"side": "left", "credits": matching_crop},  # Player 2
            {"side": "left", "credits": matching_crop},  # Player 3
            {"side": "left", "credits": matching_crop},  # Player 4
            {"side": "right", "credits": matching_crop},  # Player 5
            {"side": "right", "credits": matching_crop},  # Player 6
            {"side": "right", "credits": non_matching_crop},  # Player 7
            {"side": "right", "credits": non_matching_crop},  # Player 8
            {"side": "right", "credits": non_matching_crop},  # Player 9
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # With low threshold (0.3 = 30%), should detect as pre-round
        is_preround_low = detector_with_template.is_preround_frame(frame, threshold=0.3)
        assert is_preround_low is True

        # With medium threshold (0.7 = 70%), should detect as pre-round (exactly at threshold)
        is_preround_medium = detector_with_template.is_preround_frame(frame, threshold=0.7)
        assert is_preround_medium is True

        # With high threshold (0.8 = 80%), should not detect as pre-round
        is_preround_high = detector_with_template.is_preround_frame(frame, threshold=0.8)
        assert is_preround_high is False

    def test_is_preround_frame_empty_crops(self, detector_with_template, mock_cropper):
        """Test is_preround_frame when some crops are empty."""
        matching_crop = np.zeros((30, 80, 3), dtype=np.uint8)

        # Some players have empty crops (simulates crop failure)
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": matching_crop},  # Player 0
            {"side": "left", "credits": np.array([])},  # Player 1 - empty
            {"side": "left", "credits": matching_crop},  # Player 2
            {"side": "left", "credits": np.array([])},  # Player 3 - empty
            {"side": "left", "credits": matching_crop},  # Player 4
            {"side": "right", "credits": matching_crop},  # Player 5
            {"side": "right", "credits": np.array([])},  # Player 6 - empty
            {"side": "right", "credits": matching_crop},  # Player 7
            {"side": "right", "credits": matching_crop},  # Player 8
            {"side": "right", "credits": matching_crop},  # Player 9
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Should not crash, should handle gracefully
        is_preround = detector_with_template.is_preround_frame(frame, threshold=0.5)

        # Result depends on whether empty crops are skipped or counted as not visible
        assert isinstance(is_preround, bool)

    def test_is_preround_frame_all_crops_empty(self, detector_with_template, mock_cropper):
        """Test is_preround_frame when all crops are empty."""
        mock_cropper.crop_player_info_preround.return_value = [
            {"side": "left", "credits": np.array([])},
            {"side": "left", "credits": np.array([])},
            {"side": "left", "credits": np.array([])},
            {"side": "left", "credits": np.array([])},
            {"side": "left", "credits": np.array([])},
            {"side": "right", "credits": np.array([])},
            {"side": "right", "credits": np.array([])},
            {"side": "right", "credits": np.array([])},
            {"side": "right", "credits": np.array([])},
            {"side": "right", "credits": np.array([])},
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        is_preround = detector_with_template.is_preround_frame(frame, threshold=0.5)

        # Should return False when no players can be checked
        assert is_preround is False

    def test_inherits_from_template_credits_detector(self, detector_with_template):
        """Test that PreroundCreditsDetector properly inherits from TemplateCreditsDetector."""
        from valoscribe.detectors.template_credits_detector import TemplateCreditsDetector

        assert isinstance(detector_with_template, TemplateCreditsDetector)
        assert isinstance(detector_with_template, PreroundCreditsDetector)

        # Should have inherited methods
        assert hasattr(detector_with_template, '_preprocess_crop')
        assert hasattr(detector_with_template, 'detect')
        assert hasattr(detector_with_template, 'detect_with_debug')

        # Should have unique method
        assert hasattr(detector_with_template, 'is_preround_frame')
