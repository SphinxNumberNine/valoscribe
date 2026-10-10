"""Unit tests for template-based score detector."""

from __future__ import annotations
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pytest
import numpy as np
import cv2

from valoscribe.detectors.template_score_detector import TemplateScoreDetector
from valoscribe.types.detections import ScoreInfo


class TestTemplateScoreDetector:
    """Tests for TemplateScoreDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        # Default: return non-empty crops
        cropper.crop_simple_region.return_value = np.zeros((36, 70, 3), dtype=np.uint8)
        return cropper

    @pytest.fixture
    def mock_templates(self):
        """Create mock templates for digits 0-9."""
        templates = {}
        for digit in range(10):
            # Create simple template (20x30 white digit on black background)
            template = np.zeros((30, 20), dtype=np.uint8)
            templates[str(digit)] = template
        return templates

    @pytest.fixture
    def detector(self, mock_cropper, mock_templates, tmp_path):
        """Create template score detector with mocked templates."""
        # Create temporary template directory
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Save mock templates
        for digit, template in mock_templates.items():
            cv2.imwrite(str(template_dir / f"{digit}.png"), template)

        detector = TemplateScoreDetector(
            mock_cropper,
            template_dir=template_dir,
            min_confidence=0.7
        )

        return detector

    def test_init(self, mock_cropper, tmp_path):
        """Test template score detector initialization."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        detector = TemplateScoreDetector(
            mock_cropper,
            template_dir=template_dir,
            min_confidence=0.8
        )

        assert detector.cropper == mock_cropper
        assert detector.min_confidence == 0.8
        assert detector.match_method == cv2.TM_CCOEFF_NORMED

    def test_load_templates_success(self, mock_cropper, tmp_path):
        """Test successful template loading."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Create templates for 0-9
        for digit in range(10):
            template = np.zeros((30, 20), dtype=np.uint8)
            cv2.imwrite(str(template_dir / f"{digit}.png"), template)

        detector = TemplateScoreDetector(mock_cropper, template_dir=template_dir)

        assert len(detector.templates) == 10
        assert all(str(i) in detector.templates for i in range(10))

    def test_load_templates_missing_directory(self, mock_cropper, tmp_path):
        """Test loading templates from non-existent directory."""
        template_dir = tmp_path / "nonexistent"

        detector = TemplateScoreDetector(mock_cropper, template_dir=template_dir)

        assert len(detector.templates) == 0

    def test_load_templates_partial(self, mock_cropper, tmp_path):
        """Test loading templates when only some exist."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Only create templates for 0, 1, 2
        for digit in [0, 1, 2]:
            template = np.zeros((30, 20), dtype=np.uint8)
            cv2.imwrite(str(template_dir / f"{digit}.png"), template)

        detector = TemplateScoreDetector(mock_cropper, template_dir=template_dir)

        assert len(detector.templates) == 3
        assert "0" in detector.templates
        assert "1" in detector.templates
        assert "2" in detector.templates
        assert "3" not in detector.templates

    def test_detect_no_templates(self, mock_cropper, tmp_path):
        """Test detection fails when no templates are loaded."""
        template_dir = tmp_path / "empty"
        template_dir.mkdir()

        detector = TemplateScoreDetector(mock_cropper, template_dir=template_dir)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_empty_crop(self, detector, mock_cropper):
        """Test detection with empty crop."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_single_digit(self, detector, mock_cropper):
        """Test detection of single-digit score."""
        # Mock _match_score_region to return single digit
        detector._match_score_region = Mock(side_effect=[
            (5, 0.95, "5"),  # team1
            (3, 0.92, "3"),  # team2
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert result.team1_score == 5
        assert result.team2_score == 3
        assert result.confidence == 0.92  # min of the two
        assert result.team1_raw_text == "5"
        assert result.team2_raw_text == "3"

    def test_detect_double_digit(self, detector, mock_cropper):
        """Test detection of double-digit score."""
        # Mock _match_score_region to return double digit
        detector._match_score_region = Mock(side_effect=[
            (10, 0.88, "10"),  # team1
            (13, 0.90, "13"),  # team2
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert result.team1_score == 10
        assert result.team2_score == 13
        assert result.confidence == 0.88  # min of the two

    def test_detect_mixed_digits(self, detector, mock_cropper):
        """Test detection with mixed single and double digits."""
        detector._match_score_region = Mock(side_effect=[
            (0, 0.95, "0"),   # team1
            (12, 0.93, "12"), # team2
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert result.team1_score == 0
        assert result.team2_score == 12

    def test_detect_out_of_range_team1(self, detector, mock_cropper):
        """Test detection rejects team1 score out of range."""
        detector._match_score_region = Mock(side_effect=[
            (100, 0.95, "100"),  # team1 - out of range
            (5, 0.92, "5"),    # team2
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_out_of_range_team2(self, detector, mock_cropper):
        """Test detection rejects team2 score out of range."""
        detector._match_score_region = Mock(side_effect=[
            (7, 0.95, "7"),    # team1
            (100, 0.92, "100"),  # team2 - out of range
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_match_fails(self, detector, mock_cropper):
        """Test detection when template matching fails."""
        detector._match_score_region = Mock(side_effect=[
            None,  # team1 fails
            (5, 0.92, "5"),
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_find_all_digit_matches(self, detector):
        """Test finding all digit matches in an image."""
        # Create test image with a digit pattern
        image = np.zeros((100, 100), dtype=np.uint8)

        # Mock matchTemplate to return controlled results
        with patch('cv2.matchTemplate') as mock_match:
            # Simulate finding "5" at x=10, y=10 with high confidence
            result = np.zeros((80, 70))  # Result is smaller than input
            result[10, 10] = 0.95
            mock_match.return_value = result

            matches = detector._find_all_digit_matches(image)

            # Should find one match per template (10 templates)
            assert len(matches) > 0
            # Each match should have required fields
            for match in matches:
                assert "digit" in match
                assert "confidence" in match
                assert "x" in match
                assert "y" in match
                assert "w" in match
                assert "h" in match

    def test_filter_overlapping_matches_no_overlap(self, detector):
        """Test filtering when matches don't overlap."""
        matches = [
            {"digit": "1", "confidence": 0.95, "x": 0, "y": 0, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.90, "x": 25, "y": 0, "w": 20, "h": 30},
        ]

        filtered = detector._filter_overlapping_matches(matches)

        # Both should remain since they don't overlap
        assert len(filtered) == 2

    def test_filter_overlapping_matches_with_overlap(self, detector):
        """Test filtering when matches overlap significantly."""
        matches = [
            {"digit": "1", "confidence": 0.95, "x": 0, "y": 0, "w": 20, "h": 30},
            {"digit": "7", "confidence": 0.85, "x": 5, "y": 0, "w": 20, "h": 30},  # Overlaps with "1"
        ]

        filtered = detector._filter_overlapping_matches(matches)

        # Only highest confidence should remain
        assert len(filtered) == 1
        assert filtered[0]["digit"] == "1"
        assert filtered[0]["confidence"] == 0.95

    def test_filter_overlapping_matches_preserves_order(self, detector):
        """Test that filtering and re-sorting preserves left-to-right order."""
        matches = [
            {"digit": "0", "confidence": 0.95, "x": 25, "y": 0, "w": 20, "h": 30},  # Right, higher conf
            {"digit": "1", "confidence": 0.90, "x": 0, "y": 0, "w": 20, "h": 30},   # Left, lower conf
        ]

        filtered = detector._filter_overlapping_matches(matches)

        # Both should remain
        assert len(filtered) == 2

        # After sorting by x (done in _match_score_region), should get "10" not "01"
        filtered.sort(key=lambda m: m["x"])
        digits = [m["digit"] for m in filtered]
        assert digits == ["1", "0"]

    def test_match_score_region_success(self, detector):
        """Test successful score region matching."""
        # Create mock crop
        crop = np.zeros((36, 70, 3), dtype=np.uint8)

        # Mock _find_all_digit_matches to return a single digit
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "7", "confidence": 0.95, "x": 10, "y": 5, "w": 20, "h": 30},
        ])

        result = detector._match_score_region(crop)

        assert result is not None
        score, confidence, raw_text = result
        assert score == 7
        assert confidence == 0.95
        assert raw_text == "7"

    def test_match_score_region_double_digit(self, detector):
        """Test matching double-digit score."""
        crop = np.zeros((36, 70, 3), dtype=np.uint8)

        # Mock _find_all_digit_matches to return two digits
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "1", "confidence": 0.90, "x": 5, "y": 5, "w": 20, "h": 30},
            {"digit": "0", "confidence": 0.88, "x": 30, "y": 5, "w": 20, "h": 30},
        ])

        result = detector._match_score_region(crop)

        assert result is not None
        score, confidence, raw_text = result
        assert score == 10
        assert confidence == 0.88  # minimum
        assert raw_text == "10"

    def test_match_score_region_low_confidence(self, detector):
        """Test matching rejects low confidence."""
        crop = np.zeros((36, 70, 3), dtype=np.uint8)

        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "5", "confidence": 0.50, "x": 10, "y": 5, "w": 20, "h": 30},
        ])

        result = detector._match_score_region(crop)

        # Should return None because 0.50 < 0.7 (min_confidence)
        assert result is None

    def test_match_score_region_no_matches(self, detector):
        """Test matching when no digits are found."""
        crop = np.zeros((36, 70, 3), dtype=np.uint8)

        detector._find_all_digit_matches = Mock(return_value=[])

        result = detector._match_score_region(crop)

        assert result is None

    def test_detect_with_debug(self, detector, mock_cropper):
        """Test debug detection returns additional info."""
        # Mock successful detection
        detector._match_score_region = Mock(side_effect=[
            (8, 0.95, "8"),
            (4, 0.92, "4"),
        ])
        detector._find_all_digit_matches = Mock(return_value=[
            {"digit": "8", "confidence": 0.95, "x": 10, "y": 5, "w": 20, "h": 30},
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result, team1_preprocessed, team2_preprocessed, debug_info = detector.detect_with_debug(frame)

        assert result is not None
        assert result.team1_score == 8
        assert result.team2_score == 4
        assert isinstance(team1_preprocessed, np.ndarray)
        assert isinstance(team2_preprocessed, np.ndarray)
        assert "team1_matches" in debug_info
        assert "team2_matches" in debug_info

    def test_detect_with_debug_empty_crops(self, detector, mock_cropper):
        """Test debug detection with empty crops."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result, team1_preprocessed, team2_preprocessed, debug_info = detector.detect_with_debug(frame)

        assert result is None
        assert team1_preprocessed.size == 0
        assert team2_preprocessed.size == 0
        assert debug_info == {}

    def test_preprocess_crop(self, detector):
        """Test crop preprocessing."""
        # Create color crop
        crop = np.ones((12, 23, 3), dtype=np.uint8) * 200

        preprocessed = detector._preprocess_crop(crop)

        # Should be grayscale
        assert len(preprocessed.shape) == 2

        # Should be upscaled 3x
        assert preprocessed.shape[0] == 12 * 3
        assert preprocessed.shape[1] == 23 * 3

        # Should be binary
        assert np.all((preprocessed == 0) | (preprocessed == 255))


class TestScoreInfoValidation:
    """Tests for ScoreInfo Pydantic model validation."""

    def test_valid_score_info(self):
        """Test creating valid ScoreInfo."""
        info = ScoreInfo(
            team1_score=10,
            team2_score=12,
            confidence=0.88,
            team1_raw_text="10",
            team2_raw_text="12",
        )

        assert info.team1_score == 10
        assert info.team2_score == 12
        assert info.confidence == 0.88
        assert info.team1_raw_text == "10"
        assert info.team2_raw_text == "12"

    def test_score_range_validation(self):
        """Test score must be in valid range including overtime (0-99)."""
        # Valid cases
        ScoreInfo(team1_score=0, team2_score=0, confidence=0.9)
        ScoreInfo(team1_score=13, team2_score=13, confidence=0.9)
        ScoreInfo(team1_score=20, team2_score=18, confidence=0.9)
        ScoreInfo(team1_score=99, team2_score=99, confidence=0.9)

        # Invalid cases - team1
        with pytest.raises(Exception):  # Pydantic ValidationError
            ScoreInfo(team1_score=-1, team2_score=5, confidence=0.9)

        with pytest.raises(Exception):
            ScoreInfo(team1_score=100, team2_score=5, confidence=0.9)

        # Invalid cases - team2
        with pytest.raises(Exception):
            ScoreInfo(team1_score=5, team2_score=-1, confidence=0.9)

        with pytest.raises(Exception):
            ScoreInfo(team1_score=5, team2_score=100, confidence=0.9)

    def test_confidence_range_validation(self):
        """Test confidence must be in valid range (0-1)."""
        # Valid cases
        ScoreInfo(team1_score=5, team2_score=3, confidence=0.0)
        ScoreInfo(team1_score=5, team2_score=3, confidence=1.0)

        # Invalid cases
        with pytest.raises(Exception):
            ScoreInfo(team1_score=5, team2_score=3, confidence=-0.1)

        with pytest.raises(Exception):
            ScoreInfo(team1_score=5, team2_score=3, confidence=1.5)
