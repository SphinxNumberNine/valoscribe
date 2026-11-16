"""Detector registry for managing all detector instances."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from valoscribe.detectors.cropper import Cropper
from valoscribe.detectors.template_timer_detector import TemplateTimerDetector
from valoscribe.detectors.template_score_detector import TemplateScoreDetector
from valoscribe.detectors.template_spike_detector import TemplateSpikeDetector
from valoscribe.detectors.template_health_detector import TemplateHealthDetector
from valoscribe.detectors.template_armor_detector import TemplateArmorDetector
from valoscribe.detectors.round_detector import RoundDetector
from valoscribe.detectors.preround_credits_detector import PreroundCreditsDetector
from valoscribe.detectors.template_agent_detector import TemplateAgentDetector
from valoscribe.detectors.active_round_agent_detector import ActiveRoundAgentDetector
from valoscribe.detectors.preround_ability_detector import PreroundAbilityDetector
from valoscribe.detectors.preround_ultimate_detector import PreroundUltimateDetector
from valoscribe.detectors.ability_detector import AbilityDetector
from valoscribe.detectors.ultimate_detector import UltimateDetector
from valoscribe.detectors.killfeed_detector import KillfeedDetector
from valoscribe.utils.ocr import OCREngine
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class DetectorRegistry:
    """
    Manages all detector instances.

    Responsibilities:
    - Initialize all detectors once with shared dependencies (Cropper, OCR)
    - Provide access to all detectors
    - Support reinitialization of specific detectors (e.g., Killfeed with agent list)
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        ocr_engine: Optional[OCREngine] = None,
    ):
        """
        Initialize detector registry.

        Args:
            config_path: Path to HUD config file (default: champs2025.json)
            ocr_engine: Optional OCR engine instance (creates new if None)
        """
        # Initialize shared dependencies
        if config_path:
            self.cropper = Cropper(config_path=config_path)
        else:
            self.cropper = Cropper()  # Use default config
        self.ocr_engine = ocr_engine or OCREngine()

        log.info(f"DetectorRegistry initializing with HUD config: {self.cropper.config['name']}")

        # Initialize all detectors
        self._init_template_detectors()
        self._init_ocr_detectors()
        self._init_preround_detectors()
        self._init_inround_detectors()
        self._init_killfeed_detector()

        log.info("DetectorRegistry initialized successfully")

    def _init_template_detectors(self) -> None:
        """Initialize template-based detectors."""
        self.timer_detector = TemplateTimerDetector(self.cropper)
        self.score_detector = TemplateScoreDetector(self.cropper)
        self.spike_detector = TemplateSpikeDetector(self.cropper)
        self.health_detector = TemplateHealthDetector(self.cropper)
        self.armor_detector = TemplateArmorDetector(self.cropper)

        log.debug("Template detectors initialized")

    def _init_ocr_detectors(self) -> None:
        """Initialize OCR-based detectors."""
        self.round_detector = RoundDetector(self.cropper, self.ocr_engine)

        log.debug("OCR detectors initialized")

    def _init_preround_detectors(self) -> None:
        """Initialize preround-specific detectors."""
        self.preround_credits_detector = PreroundCreditsDetector(self.cropper)
        self.preround_agent_detector = TemplateAgentDetector(self.cropper)
        self.preround_ability_detector = PreroundAbilityDetector(self.cropper)
        self.preround_ultimate_detector = PreroundUltimateDetector(self.cropper)

        log.debug("Preround detectors initialized")

    def _init_inround_detectors(self) -> None:
        """Initialize inround-specific detectors."""
        self.inround_agent_detector = ActiveRoundAgentDetector(self.cropper)
        self.inround_ability_detector = AbilityDetector(self.cropper)
        self.inround_ultimate_detector = UltimateDetector(self.cropper)

        log.debug("Inround detectors initialized")

    def _init_killfeed_detector(self, agent_list: Optional[list[str]] = None) -> None:
        """
        Initialize killfeed detector.

        Args:
            agent_list: Optional list of agent names in the game.
                       If None, uses all agents (slower but works initially).
        """
        if agent_list:
            log.info(f"Initializing killfeed detector with {len(agent_list)} agents: {agent_list}")
        else:
            log.info("Initializing killfeed detector with all agents (will reinitialize after first preround)")

        self.killfeed_detector = KillfeedDetector(
            self.cropper,
            agents=agent_list,
            # Killfeed slots are scanned top-down and most frames have at most one entry.
            # Stopping after two consecutive empty entries was 4.9x faster on the recorded
            # fixture with identical detections there (see HANDOFF.md). Set to 0 to scan all
            # ten slots unconditionally.
            max_consecutive_empty_entries=2,
        )

        log.debug("Killfeed detector initialized")

    def reinitialize_killfeed(self, agent_list: list[str]) -> None:
        """
        Reinitialize killfeed detector with specific agent list.

        This should be called after the first preround when we know which
        agents are in the game for better performance.

        Args:
            agent_list: List of agent names in the game (10 agents)
        """
        log.info(f"Reinitializing killfeed detector with agents: {agent_list}")
        self._init_killfeed_detector(agent_list)

    def get_all_detectors(self) -> dict:
        """
        Get all detector instances.

        Returns:
            Dictionary mapping detector names to instances
        """
        return {
            # Template detectors
            "timer": self.timer_detector,
            "score": self.score_detector,
            "spike": self.spike_detector,
            "health": self.health_detector,
            "armor": self.armor_detector,
            # OCR detectors
            "round": self.round_detector,
            # Preround detectors
            "preround_credits": self.preround_credits_detector,
            "preround_agent": self.preround_agent_detector,
            "preround_ability": self.preround_ability_detector,
            "preround_ultimate": self.preround_ultimate_detector,
            # Inround detectors
            "inround_agent": self.inround_agent_detector,
            "inround_ability": self.inround_ability_detector,
            "inround_ultimate": self.inround_ultimate_detector,
            # Killfeed
            "killfeed": self.killfeed_detector,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DetectorRegistry("
            f"config={self.cropper.config['name']}, "
            f"detectors={len(self.get_all_detectors())})"
        )
