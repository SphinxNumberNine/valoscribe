"""Unit tests for round detector."""

from __future__ import annotations
from unittest.mock import Mock, MagicMock
import pytest
import numpy as np

from valoscribe.detectors.round_detector import RoundDetector
from valoscribe.types.detections import RoundInfo


class TestRoundDetector:
    """Tests for RoundDetector class."""

    @pytest.fixture
    def mock_cropper(self):
        """Create mock cropper."""
        cropper = Mock()
        # Default: return a non-empty crop
        cropper.crop_simple_region.return_value = np.zeros((20, 100, 3), dtype=np.uint8)
        return cropper

    @pytest.fixture
    def mock_ocr_engine(self):
        """Create mock OCR engine."""
        ocr = Mock()
        # Default: return valid round text
        ocr.read_single_line.return_value = ("ROUND 5/24", 0.95)
        ocr.preprocess_for_text = Mock(side_effect=lambda x: x)  # Pass through
        return ocr

    @pytest.fixture
    def detector(self, mock_cropper, mock_ocr_engine):
        """Create round detector with mocked dependencies."""
        return RoundDetector(mock_cropper, mock_ocr_engine, min_confidence=0.5)

    def test_init(self, mock_cropper, mock_ocr_engine):
        """Test round detector initialization."""
        detector = RoundDetector(mock_cropper, mock_ocr_engine, min_confidence=0.7)

        assert detector.cropper == mock_cropper
        assert detector.ocr_engine == mock_ocr_engine
        assert detector.min_confidence == 0.7

    def test_init_creates_default_ocr_engine(self, mock_cropper):
        """Test that detector creates default OCR engine if not provided."""
        detector = RoundDetector(mock_cropper, ocr_engine=None)

        assert detector.ocr_engine is not None

    def test_detect_success_full_format(self, detector, mock_cropper, mock_ocr_engine):
        """Test successful detection with 'ROUND X/24' format."""
        mock_ocr_engine.read_single_line.return_value = ("ROUND 12/24", 0.95)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert isinstance(result, RoundInfo)
        assert result.round_number == 12
        assert result.confidence == 0.95
        assert result.raw_text == "ROUND 12/24"

        # Verify cropper was called
        mock_cropper.crop_simple_region.assert_called_once_with(frame, "round_number")

    def test_detect_success_short_format(self, detector, mock_ocr_engine):
        """Test successful detection with 'X/24' format."""
        mock_ocr_engine.read_single_line.return_value = ("5/24", 0.92)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert result.round_number == 5
        assert result.confidence == 0.92

    def test_detect_success_number_only(self, detector, mock_ocr_engine):
        """Test successful detection with just a number."""
        mock_ocr_engine.read_single_line.return_value = ("15", 0.88)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is not None
        assert result.round_number == 15

    def test_detect_empty_crop(self, detector, mock_cropper):
        """Test detection with empty crop."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_low_confidence(self, detector, mock_ocr_engine):
        """Test detection with confidence below threshold."""
        mock_ocr_engine.read_single_line.return_value = ("ROUND 8/24", 0.3)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        # Should return None due to low confidence
        assert result is None

    def test_detect_unparseable_text(self, detector, mock_ocr_engine):
        """Test detection with unparseable text."""
        mock_ocr_engine.read_single_line.return_value = ("INVALID TEXT", 0.95)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert result is None

    def test_detect_out_of_range_number(self, detector, mock_ocr_engine):
        """Test detection with round number out of valid range."""
        mock_ocr_engine.read_single_line.return_value = ("ROUND 30/24", 0.95)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect(frame)

        # 30 is out of range (max 24)
        assert result is None

    def test_parse_round_number_various_formats(self, detector):
        """Test parsing round numbers from various formats."""
        test_cases = [
            ("ROUND 5/24", 5),
            ("5/24", 5),
            ("ROUND 12", 12),
            ("12", 12),
            ("  ROUND  5 / 24  ", 5),  # Extra whitespace
            ("1/24", 1),
            ("24/24", 24),
            ("ROUND 0/24", None),  # Invalid: 0
            ("ROUND 25/24", None),  # Invalid: > 24
            ("ROUND ABC/24", None),  # Invalid: non-numeric
            ("", None),  # Empty
            ("INVALID", None),  # No numbers
        ]

        for text, expected in test_cases:
            result = detector._parse_round_number(text)
            assert result == expected, f"Failed for input: '{text}'"

    def test_detect_with_debug(self, detector, mock_cropper, mock_ocr_engine):
        """Test detection with debug visualization."""
        mock_ocr_engine.read_single_line.return_value = ("ROUND 7/24", 0.90)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result, preprocessed = detector.detect_with_debug(frame)

        assert result is not None
        assert result.round_number == 7
        assert isinstance(preprocessed, np.ndarray)
        # Preprocessed image should be returned
        assert preprocessed.size > 0

    def test_detect_with_debug_empty_crop(self, detector, mock_cropper):
        """Test debug detection with empty crop."""
        mock_cropper.crop_simple_region.return_value = np.array([])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result, preprocessed = detector.detect_with_debug(frame)

        assert result is None
        assert preprocessed.size == 0

    def test_whitelist_passed_to_ocr(self, detector, mock_ocr_engine):
        """Test that whitelist is passed to OCR engine."""
        mock_ocr_engine.read_single_line.return_value = ("5/24", 0.95)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detector.detect(frame)

        # Verify OCR was called with correct whitelist
        call_kwargs = mock_ocr_engine.read_single_line.call_args[1]
        assert "whitelist" in call_kwargs
        assert "0123456789" in call_kwargs["whitelist"]
        assert "ROUND" in call_kwargs["whitelist"]
        assert "/" in call_kwargs["whitelist"]


class TestRoundInfoValidation:
    """Tests for RoundInfo Pydantic model validation."""

    def test_valid_round_info(self):
        """Test creating valid RoundInfo."""
        info = RoundInfo(round_number=5, confidence=0.95, raw_text="ROUND 5/24")

        assert info.round_number == 5
        assert info.confidence == 0.95
        assert info.raw_text == "ROUND 5/24"

    def test_round_number_range_validation(self):
        """Test round number must be in valid range including overtime (1-100)."""
        # Valid cases
        RoundInfo(round_number=1, confidence=0.9)
        RoundInfo(round_number=24, confidence=0.9)
        RoundInfo(round_number=25, confidence=0.9)
        RoundInfo(round_number=100, confidence=0.9)

        # Invalid cases
        with pytest.raises(Exception):  # Pydantic ValidationError
            RoundInfo(round_number=0, confidence=0.9)

        with pytest.raises(Exception):
            RoundInfo(round_number=101, confidence=0.9)

        with pytest.raises(Exception):
            RoundInfo(round_number=-5, confidence=0.9)

    def test_confidence_range_validation(self):
        """Test confidence must be in valid range (0-1)."""
        # Valid cases
        RoundInfo(round_number=5, confidence=0.0)
        RoundInfo(round_number=5, confidence=1.0)
        RoundInfo(round_number=5, confidence=0.5)

        # Invalid cases
        with pytest.raises(Exception):
            RoundInfo(round_number=5, confidence=-0.1)

        with pytest.raises(Exception):
            RoundInfo(round_number=5, confidence=1.5)

    def test_raw_text_optional(self):
        """Test raw_text is optional."""
        info = RoundInfo(round_number=5, confidence=0.9)

        assert info.raw_text is None
