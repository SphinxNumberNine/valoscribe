"""Unit tests for DetectorRegistry."""

from __future__ import annotations
import pytest

from valoscribe.orchestration.detector_registry import DetectorRegistry
from valoscribe.detectors.template_timer_detector import TemplateTimerDetector
from valoscribe.detectors.template_score_detector import TemplateScoreDetector
from valoscribe.detectors.template_spike_detector import TemplateSpikeDetector
from valoscribe.detectors.template_health_detector import TemplateHealthDetector
from valoscribe.detectors.template_armor_detector import TemplateArmorDetector
from valoscribe.detectors.round_detector import RoundDetector
from valoscribe.detectors.preround_credits_detector import PreroundCreditsDetector
from valoscribe.detectors.template_agent_detector import TemplateAgentDetector
from valoscribe.detectors.preround_ability_detector import PreroundAbilityDetector
from valoscribe.detectors.preround_ultimate_detector import PreroundUltimateDetector
from valoscribe.detectors.ability_detector import AbilityDetector
from valoscribe.detectors.ultimate_detector import UltimateDetector
from valoscribe.detectors.killfeed_detector import KillfeedDetector


class TestDetectorRegistry:
    """Tests for DetectorRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create DetectorRegistry instance."""
        return DetectorRegistry()

    def test_init(self, registry):
        """Test DetectorRegistry initialization."""
        assert registry.cropper is not None
        assert registry.ocr_engine is not None
        assert registry.timer_detector is not None
        assert registry.killfeed_detector is not None

    def test_template_detectors_initialized(self, registry):
        """Test that all template detectors are initialized."""
        assert isinstance(registry.timer_detector, TemplateTimerDetector)
        assert isinstance(registry.score_detector, TemplateScoreDetector)
        assert isinstance(registry.spike_detector, TemplateSpikeDetector)
        assert isinstance(registry.health_detector, TemplateHealthDetector)
        assert isinstance(registry.armor_detector, TemplateArmorDetector)

    def test_ocr_detectors_initialized(self, registry):
        """Test that OCR detectors are initialized."""
        assert isinstance(registry.round_detector, RoundDetector)

    def test_preround_detectors_initialized(self, registry):
        """Test that preround detectors are initialized."""
        assert isinstance(registry.preround_credits_detector, PreroundCreditsDetector)
        assert isinstance(registry.preround_agent_detector, TemplateAgentDetector)
        assert isinstance(registry.preround_ability_detector, PreroundAbilityDetector)
        assert isinstance(registry.preround_ultimate_detector, PreroundUltimateDetector)

    def test_inround_detectors_initialized(self, registry):
        """Test that inround detectors are initialized."""
        assert isinstance(registry.inround_ability_detector, AbilityDetector)
        assert isinstance(registry.inround_ultimate_detector, UltimateDetector)

    def test_killfeed_detector_initialized(self, registry):
        """Test that killfeed detector is initialized."""
        assert isinstance(registry.killfeed_detector, KillfeedDetector)

    def test_get_all_detectors(self, registry):
        """Test getting all detector instances."""
        detectors = registry.get_all_detectors()

        # Should have all detector types
        expected_keys = [
            "timer", "score", "spike", "health", "armor",
            "round",
            "preround_credits", "preround_agent", "preround_ability", "preround_ultimate",
            "inround_agent", "inround_ability", "inround_ultimate",
            "killfeed",
        ]

        for key in expected_keys:
            assert key in detectors, f"Missing detector: {key}"

        assert len(detectors) == len(expected_keys)

    def test_reinitialize_killfeed(self, registry):
        """Test reinitializing killfeed with agent list."""
        # Get original killfeed detector
        original_detector = registry.killfeed_detector

        # Reinitialize with specific agents
        agent_list = ["jett", "sova", "omen", "viper", "kayo"]
        registry.reinitialize_killfeed(agent_list)

        # Should have new detector instance
        assert registry.killfeed_detector is not original_detector
        assert isinstance(registry.killfeed_detector, KillfeedDetector)

    def test_shared_cropper(self, registry):
        """Test that all detectors share the same Cropper instance."""
        # All detectors should use the same cropper
        assert registry.timer_detector.cropper is registry.cropper
        assert registry.score_detector.cropper is registry.cropper
        assert registry.spike_detector.cropper is registry.cropper
        assert registry.health_detector.cropper is registry.cropper
        assert registry.armor_detector.cropper is registry.cropper

    def test_shared_ocr_engine(self, registry):
        """Test that OCR-based detectors share the same OCR engine."""
        assert registry.round_detector.ocr_engine is registry.ocr_engine

    def test_repr(self, registry):
        """Test string representation."""
        repr_str = repr(registry)
        assert "DetectorRegistry" in repr_str
        assert "detectors=" in repr_str

    def test_all_detectors_have_cropper(self, registry):
        """Test that all detectors have access to cropper."""
        detectors = registry.get_all_detectors()

        # These detectors should have cropper
        detectors_with_cropper = [
            "timer", "score", "spike", "health", "armor",
            "preround_credits", "preround_agent", "preround_ability", "preround_ultimate",
            "inround_agent", "inround_ability", "inround_ultimate", "killfeed",
        ]

        for name in detectors_with_cropper:
            detector = detectors[name]
            assert hasattr(detector, "cropper"), f"{name} detector missing cropper"
            assert detector.cropper is registry.cropper

    def test_ocr_detectors_have_ocr_engine(self, registry):
        """Test that OCR-based detectors have OCR engine."""
        assert hasattr(registry.round_detector, "ocr_engine")
        assert registry.round_detector.ocr_engine is registry.ocr_engine
