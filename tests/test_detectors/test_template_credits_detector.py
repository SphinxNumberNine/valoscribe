"""Unit tests for template-based credits detector."""

from __future__ import annotations
from unittest.mock import Mock, patch
from pathlib import Path
import pytest
import numpy as np
import cv2

from valoscribe.detectors.template_credits_detector import TemplateCreditsDetector
from valoscribe.types.detections import CreditsInfo


class TestTemplateCreditsDetector:
    """Tests for TemplateCreditsDetector class."""

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
        Preprocess template before saving to disk (simulates extract-credits-crops).

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
        template_path = tmp_path / "credits_icon.png"

        # Preprocess template before saving (simulates extract-credits-crops output)
        preprocessed_template = self.preprocess_template_for_storage(mock_template)
        cv2.imwrite(str(template_path), preprocessed_template)

        return TemplateCreditsDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.7,
        )

    def create_matching_crop(self, template: np.ndarray, crop_size: tuple = (40, 60)) -> np.ndarray:
        """
        Create a crop that will match the template after preprocessing.

        This simulates how the template would appear in an actual game frame.
        We need to ensure the crop, when preprocessed, will contain the template pattern.
        """
        h, w = crop_size
        crop = np.zeros((h, w, 3), dtype=np.uint8)

        # Place bright pattern that will survive preprocessing
        # Template is 20x20, place it in center with some padding
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
        template_path = tmp_path / "custom_credits.png"
        template = np.zeros((20, 20), dtype=np.uint8)
        cv2.imwrite(str(template_path), template)

        detector = TemplateCreditsDetector(
            mock_cropper,
            template_path=template_path,
            min_confidence=0.8,
            match_method=cv2.TM_CCORR_NORMED,
        )

        assert detector.cropper == mock_cropper
        assert detector.min_confidence == 0.8
        assert detector.match_method == cv2.TM_CCORR_NORMED
        assert detector.template_path == template_path
        assert detector.template is not None

    def test_init_with_default_template_path(self, mock_cropper):
        """Test detector initialization with default template path."""
        # Don't provide template_path, let it use default
        detector = TemplateCreditsDetector(mock_cropper)

        # Default template path should be set correctly
        assert "templates/credits/credits_icon_dead.png" in detector.template_path.as_posix()
        # Template may or may not exist depending on setup

    def test_load_template_success(self, mock_cropper, mock_template, tmp_path):
        """Test successful template loading."""
        template_path = tmp_path / "credits_icon.png"
        cv2.imwrite(str(template_path), mock_template)

        detector = TemplateCreditsDetector(mock_cropper, template_path=template_path)

        assert detector.template is not None
        assert detector.template.shape == mock_template.shape
        assert np.array_equal(detector.template, mock_template)

    def test_load_template_file_not_exists(self, mock_cropper, tmp_path):
        """Test loading template when file doesn't exist."""
        template_path = tmp_path / "nonexistent.png"

        detector = TemplateCreditsDetector(mock_cropper, template_path=template_path)

        assert detector.template is None

    def test_load_template_invalid_file(self, mock_cropper, tmp_path):
        """Test loading template with invalid image file."""
        template_path = tmp_path / "invalid.png"
        template_path.write_text("not an image")

        detector = TemplateCreditsDetector(mock_cropper, template_path=template_path)

        assert detector.template is None

    def test_detect_no_template_loaded(self, mock_cropper):
        """Test detection fails gracefully when no template is loaded."""
        detector = TemplateCreditsDetector(mock_cropper, template_path=Path("/nonexistent/path.png"))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame, player_index=0)

        assert result is None

    def test_detect_player_out_of_range(self, detector_with_template, mock_cropper):
        """Test detection with invalid player index."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": np.zeros((20, 40, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=10)

        assert result is None

    def test_detect_wrong_side(self, detector_with_template, mock_cropper):
        """Test detection with wrong side specified."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": np.zeros((20, 40, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0, side="right")

        assert result is None

    def test_detect_missing_dead_credits_region(self, detector_with_template, mock_cropper):
        """Test detection when the dead credits region is missing from crop data."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left"}  # No dead credits region
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0)

        assert result is None

    def test_detect_empty_crop(self, detector_with_template, mock_cropper):
        """Test detection with empty credits crop."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0)

        assert result is None

    def test_detect_dead_credits_visible_high_confidence(self, detector_with_template, mock_cropper, mock_template):
        """Test detection when the icon in the dead credits location is visible."""
        # Create credits crop that will match after preprocessing
        credits_crop = self.create_matching_crop(mock_template)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": credits_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=0)

        assert result is not None
        assert isinstance(result, CreditsInfo)
        assert result.credits_visible is True
        assert result.confidence >= 0.7
        assert result.confidence <= 1.0

    def test_detect_dead_credits_not_visible_low_confidence(self, detector_with_template, mock_cropper):
        """An unmatched dead credits crop supplies no positive death signal."""
        # Create credits crop with no matching pattern
        credits_crop = np.random.randint(0, 50, (40, 60, 3), dtype=np.uint8)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": credits_crop}
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

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": np.zeros((40, 60, 3), dtype=np.uint8)},
            {"side": "right", "dead_credits": credits_crop},
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector_with_template.detect(frame, player_index=1, side="right")

        assert result is not None
        assert result.credits_visible is True

    def test_preprocess_crop_color_input(self, detector_with_template):
        """Test preprocessing with color input."""
        crop = np.zeros((30, 40, 3), dtype=np.uint8)
        # Add bright region
        crop[10:20, 15:25] = 200

        preprocessed = detector_with_template._preprocess_crop(crop)

        # Should be grayscale
        assert len(preprocessed.shape) == 2
        # Should be upscaled 3x
        assert preprocessed.shape[0] == 30 * 3
        assert preprocessed.shape[1] == 40 * 3
        # Should be binary
        assert np.all((preprocessed == 0) | (preprocessed == 255))

    def test_preprocess_crop_grayscale_input(self, detector_with_template):
        """Test preprocessing with grayscale input."""
        crop = np.zeros((30, 40), dtype=np.uint8)
        crop[10:20, 15:25] = 200

        preprocessed = detector_with_template._preprocess_crop(crop)

        assert len(preprocessed.shape) == 2
        assert preprocessed.shape[0] == 30 * 3
        assert preprocessed.shape[1] == 40 * 3
        assert np.all((preprocessed == 0) | (preprocessed == 255))

    def test_preprocess_crop_upscaling(self, detector_with_template):
        """Test that preprocessing upscales by 3x."""
        crop = np.ones((20, 30, 3), dtype=np.uint8) * 128

        preprocessed = detector_with_template._preprocess_crop(crop)

        assert preprocessed.shape == (20 * 3, 30 * 3)

    def test_preprocess_crop_otsu_thresholding(self, detector_with_template):
        """Test that preprocessing uses Otsu's thresholding."""
        # Create image with two distinct intensity levels
        crop = np.zeros((40, 60, 3), dtype=np.uint8)
        crop[:20, :] = 100  # Dark region
        crop[20:, :] = 200  # Bright region

        preprocessed = detector_with_template._preprocess_crop(crop)

        # After Otsu's threshold, should have only 0 and 255
        unique_values = np.unique(preprocessed)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)

    def test_detect_with_debug_success(self, detector_with_template, mock_cropper, mock_template):
        """Test debug detection with successful detection."""
        credits_crop = self.create_matching_crop(mock_template)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": credits_crop}
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
        credits_crop = np.random.randint(0, 50, (40, 60, 3), dtype=np.uint8)

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": credits_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        credits_info, preprocessed, debug_info = detector_with_template.detect_with_debug(
            frame, player_index=0
        )

        assert credits_info is not None
        assert credits_info.credits_visible is False
        assert preprocessed.size > 0
        assert debug_info["max_confidence"] < 0.7

    def test_detect_with_debug_empty_crop(self, detector_with_template, mock_cropper):
        """Test debug detection with empty crop."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": np.array([])}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        credits_info, preprocessed, debug_info = detector_with_template.detect_with_debug(
            frame, player_index=0
        )

        assert credits_info is None
        assert preprocessed.size == 0
        assert debug_info == {}

    def test_detect_with_debug_player_out_of_range(self, detector_with_template, mock_cropper):
        """Test debug detection with invalid player index."""
        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": np.zeros((20, 40, 3), dtype=np.uint8)}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        credits_info, preprocessed, debug_info = detector_with_template.detect_with_debug(
            frame, player_index=10
        )

        assert credits_info is None
        assert preprocessed.size == 0
        assert debug_info == {}

    def test_different_confidence_thresholds(self, mock_cropper, mock_template, tmp_path):
        """Test detector with different confidence thresholds."""
        template_path = tmp_path / "credits_icon.png"
        cv2.imwrite(str(template_path), mock_template)

        # Create crop with partial match (medium confidence)
        credits_crop = np.zeros((40, 60, 3), dtype=np.uint8)
        # Only partially match the template
        partial_template = mock_template[0:15, 0:15]
        credits_crop[10:25, 20:35] = cv2.cvtColor(partial_template, cv2.COLOR_GRAY2BGR)

        # Low threshold detector
        detector_low = TemplateCreditsDetector(mock_cropper, template_path=template_path, min_confidence=0.3)
        mock_cropper.crop_player_info.return_value = [{"side": "left", "dead_credits": credits_crop}]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result_low = detector_low.detect(frame, player_index=0)

        # High threshold detector
        detector_high = TemplateCreditsDetector(mock_cropper, template_path=template_path, min_confidence=0.9)
        result_high = detector_high.detect(frame, player_index=0)

        # Low threshold might detect, high threshold should not
        # (depending on actual confidence, but they should differ in credits_visible if confidence is between thresholds)
        assert result_low is not None
        assert result_high is not None

    def test_different_match_methods(self, mock_cropper, mock_template, tmp_path):
        """Test detector with different OpenCV template matching methods."""
        template_path = tmp_path / "credits_icon.png"

        # Preprocess template before saving
        preprocessed_template = self.preprocess_template_for_storage(mock_template)
        cv2.imwrite(str(template_path), preprocessed_template)

        credits_crop = self.create_matching_crop(mock_template)

        # Test with TM_CCOEFF_NORMED (default)
        detector1 = TemplateCreditsDetector(
            mock_cropper, template_path=template_path, match_method=cv2.TM_CCOEFF_NORMED
        )
        mock_cropper.crop_player_info.return_value = [{"side": "left", "dead_credits": credits_crop}]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result1 = detector1.detect(frame, player_index=0)

        # Test with TM_CCORR_NORMED
        detector2 = TemplateCreditsDetector(
            mock_cropper, template_path=template_path, match_method=cv2.TM_CCORR_NORMED
        )
        result2 = detector2.detect(frame, player_index=0)

        # Both should detect successfully
        assert result1 is not None
        assert result2 is not None
        # Confidence values might differ between methods
        assert result1.credits_visible is True
        assert result2.credits_visible is True

    def test_negative_confidence_clamped_to_zero(self, mock_cropper, tmp_path):
        """Test that negative confidence values from template matching are clamped to 0.0."""
        # Create a checkerboard pattern template (half black, half white after preprocessing)
        template = np.zeros((20, 40), dtype=np.uint8)
        template[:, :20] = 255  # Left half white
        # Right half stays black

        template_path = tmp_path / "credits_icon.png"
        # Save without preprocessing (will be loaded as grayscale)
        # Upscale and threshold it
        h, w = template.shape
        template_upscaled = cv2.resize(template, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        _, template_binary = cv2.threshold(template_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(str(template_path), template_binary)

        detector = TemplateCreditsDetector(
            mock_cropper, template_path=template_path, min_confidence=0.7
        )

        # Create inverse pattern crop (right half bright, left half dark) in the preprocessed space
        # This will create an anti-correlation with the template
        credits_crop = np.zeros((60, 120, 3), dtype=np.uint8)
        credits_crop[:, 60:] = 255  # Right half white (opposite of template)
        # After preprocessing, this should give negative correlation

        mock_cropper.crop_player_info.return_value = [
            {"side": "left", "dead_credits": credits_crop}
        ]

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame, player_index=0)

        # Should not raise validation error (this is the key test)
        assert result is not None
        assert isinstance(result, CreditsInfo)
        # Confidence should be clamped to >= 0.0
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        # Should be marked as not visible (low confidence, pattern doesn't match)
        assert result.credits_visible is False


class TestCreditsInfoValidation:
    """Tests for CreditsInfo Pydantic model validation."""

    def test_valid_credits_info_alive(self):
        """Test creating valid CreditsInfo for alive player."""
        info = CreditsInfo(credits_visible=True, confidence=0.85)

        assert info.credits_visible is True
        assert info.confidence == 0.85

    def test_valid_credits_info_dead(self):
        """Test creating valid CreditsInfo for dead player."""
        info = CreditsInfo(credits_visible=False, confidence=0.32)

        assert info.credits_visible is False
        assert info.confidence == 0.32

    def test_confidence_range_validation(self):
        """Test confidence must be in valid range (0-1)."""
        # Valid cases
        CreditsInfo(credits_visible=True, confidence=0.0)
        CreditsInfo(credits_visible=True, confidence=1.0)
        CreditsInfo(credits_visible=False, confidence=0.5)

        # Invalid cases
        with pytest.raises(Exception):  # Pydantic ValidationError
            CreditsInfo(credits_visible=True, confidence=-0.1)

        with pytest.raises(Exception):
            CreditsInfo(credits_visible=True, confidence=1.5)

    def test_credits_visible_boolean(self):
        """Test credits_visible accepts boolean values."""
        # Valid boolean values
        info1 = CreditsInfo(credits_visible=True, confidence=0.8)
        assert info1.credits_visible is True

        info2 = CreditsInfo(credits_visible=False, confidence=0.3)
        assert info2.credits_visible is False

        # Pydantic coerces some values to boolean (1, 0, "true", "false", etc.)
        # This is expected behavior for Pydantic

    def test_missing_required_fields(self):
        """Test that required fields cannot be omitted."""
        # Missing credits_visible
        with pytest.raises(Exception):
            CreditsInfo(confidence=0.8)

        # Missing confidence
        with pytest.raises(Exception):
            CreditsInfo(credits_visible=True)

        # Missing both
        with pytest.raises(Exception):
            CreditsInfo()

    def test_edge_case_confidences(self):
        """Test edge cases for confidence values."""
        # Exactly at boundaries
        info1 = CreditsInfo(credits_visible=True, confidence=0.0)
        assert info1.confidence == 0.0

        info2 = CreditsInfo(credits_visible=True, confidence=1.0)
        assert info2.confidence == 1.0

        # Very close to threshold
        info3 = CreditsInfo(credits_visible=False, confidence=0.6999)
        assert info3.credits_visible is False
        assert info3.confidence < 0.7

        info4 = CreditsInfo(credits_visible=True, confidence=0.7001)
        assert info4.credits_visible is True
        assert info4.confidence > 0.7
