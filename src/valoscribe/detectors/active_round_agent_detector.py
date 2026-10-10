"""
Template-based agent detector for Valorant active round HUD.

Uses color template matching to detect agent icons during active round phase.
Matches against attack and defense templates to identify both agent and side.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import AgentInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class ActiveRoundAgentDetector:
    """
    Template matching-based detector for active round agent icons.

    Uses color template matching with minimal preprocessing to detect
    agents and their side (attack/defense) from active round agent crops.
    """

    def __init__(
        self,
        cropper: Cropper,
        template_dir: Optional[Path] = None,
        min_confidence: float = 0.7,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        """
        Initialize active round agent detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_dir: Directory containing agent templates with subdirs 'attack' and 'defense'
                         If None, uses default: src/valoscribe/templates/active_round_agents/
            min_confidence: Minimum template match confidence (0-1)
            match_method: OpenCV template matching method
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method

        # Set default template directory if not provided
        if template_dir is None:
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "active_round_agents"

        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()

        log.info(
            f"Active round agent detector initialized (min_confidence: {min_confidence}, "
            f"templates: {len(self.templates)}, method: {match_method})"
        )

    def _load_templates(self) -> dict[str, dict]:
        """
        Load agent templates from template directory.

        Expects structure:
        - templates/active_round_agents/<agent_name>.png (or .jpg)

        Note: Active round agent icons are the same regardless of side (attack/defense),
        so we only need one template per agent. Side is inferred from screen position.

        Returns:
            Dictionary mapping agent names to template data:
            {
                "agent_name": {"image": ndarray, "agent": "agent_name"},
            }
        """
        templates = {}

        if not self.template_dir.exists():
            log.warning(f"Template directory does not exist: {self.template_dir}")
            return templates

        # Load templates from root directory (no attack/defense subdirs)
        for pattern in ["*.png", "*.jpg"]:
            for template_path in self.template_dir.glob(pattern):
                agent_name = template_path.stem
                template = cv2.imread(str(template_path))

                if template is None:
                    log.warning(f"Failed to load template: {template_path}")
                    continue

                templates[agent_name] = {
                    "image": template,
                    "greyscale_image": cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                    "agent": agent_name,
                }
                log.debug(f"Loaded template for {agent_name}: {template.shape}")

        if len(templates) == 0:
            log.error(f"No templates loaded from {self.template_dir}")
        else:
            log.info(f"Loaded {len(templates)} active round agent templates")

        return templates

    def set_agent_filter(self, agent_names: list[str]) -> None:
        """
        Filter templates to only the specified agents for faster matching.

        Call this after first successful detection of all 10 agents in a match
        to skip templates for agents not in the game.

        Args:
            agent_names: List of agent names in the game (e.g., ["Jett", "Sova", ...])
        """
        if not agent_names:
            log.warning("Empty agent list provided to set_agent_filter, ignoring")
            return

        # Filter templates to only those for agents in the game
        filtered_templates = {}
        agent_set = set(agent_names)

        for key, template_data in self.templates.items():
            if template_data["agent"] in agent_set:
                filtered_templates[key] = template_data

        original_count = len(self.templates)
        self.templates = filtered_templates

        log.info(
            f"Filtered active round agent templates: {original_count} → {len(filtered_templates)} "
            f"(agents: {sorted(agent_set)})"
        )

    def detect(
        self, frame: np.ndarray, player_index: int, greyscale: bool = False
    ) -> Optional[str]:
        """
        Detect agent for a specific player using template matching.

        Note: This only returns the agent name. Side (attack/defense) should be
        inferred from player_index (0-4 = left side, 5-9 = right side) and
        the current round state.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            greyscale: If True, convert both template and crop to greyscale before matching
                      (useful for detecting dead players whose UI turns greyscale)

        Returns:
            Agent name string if successfully detected, None otherwise
        """
        # Check if templates are loaded
        if len(self.templates) == 0:
            log.error("No templates loaded, cannot detect agent")
            return None

        # Get active round crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            log.warning(f"Player index {player_index} out of range (max: {len(player_crops) - 1})")
            return None

        player_crop_data = player_crops[player_index]

        # Get agent crop (check for "agent_icon" key - same as preround)
        if "agent_icon" not in player_crop_data:
            log.warning(f"Agent icon region not found in player crop data")
            return None

        agent_crop = player_crop_data["agent_icon"]

        if agent_crop.size == 0:
            log.warning(f"Agent crop is empty for player {player_index}")
            return None

        # Match against all templates
        best_match = self._match_all_templates(agent_crop, greyscale=greyscale)

        if best_match is None:
            log.debug(f"No agent detected for player {player_index} (greyscale={greyscale})")
            return None

        agent_name, confidence = best_match

        log.debug(
            f"Player {player_index}: {agent_name} with confidence {confidence:.2f} (greyscale={greyscale})"
        )

        return agent_name

    def _match_all_templates(
        self, crop: np.ndarray, greyscale: bool = False
    ) -> Optional[tuple[str, float]]:
        """
        Match all agent templates against a crop and return the best match.

        Args:
            crop: Cropped agent region
            greyscale: If True, convert both crop and templates to greyscale before matching

        Returns:
            Tuple of (agent_name, confidence) for best match, or None if no match
        """
        # Preprocess crop (minimal - just ensure color format)
        preprocessed = self._preprocess_crop(crop)

        # Convert to greyscale if requested (for dead player detection)
        if greyscale:
            if len(preprocessed.shape) == 3:
                preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)

        best_confidence = 0.0
        best_agent = None

        # Try matching each template
        for agent_name, template_data in self.templates.items():
            # Only static templates are reused; each slot is still observed afresh.
            template_img = template_data["greyscale_image" if greyscale else "image"]

            # Skip if template is larger than crop (can't match)
            if template_img.shape[0] > preprocessed.shape[0] or template_img.shape[1] > preprocessed.shape[1]:
                log.debug(
                    f"Template {agent_name} ({template_img.shape[:2]}) larger than crop "
                    f"({preprocessed.shape[:2]}), skipping"
                )
                continue

            # Template matching (color or greyscale)
            result = cv2.matchTemplate(preprocessed, template_img, self.match_method)

            # Get the best match location
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            confidence = float(max_val)

            # Clamp to [0, 1] range
            confidence = max(0.0, min(1.0, confidence))

            log.debug(
                f"Template {agent_name}: confidence = {confidence:.3f} at position {max_loc} (greyscale={greyscale})"
            )

            # Track best match
            if confidence > best_confidence:
                best_confidence = confidence
                best_agent = agent_name

        # Check if best match meets threshold
        if best_confidence < self.min_confidence:
            log.debug(
                f"Best match confidence {best_confidence:.2f} below threshold {self.min_confidence} (greyscale={greyscale})"
            )
            return None

        return best_agent, best_confidence

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess agent crop for template matching.

        Uses minimal preprocessing to preserve color detail.

        Args:
            crop: Cropped agent region

        Returns:
            Preprocessed color image
        """
        # Ensure BGR color format (OpenCV default)
        if len(crop.shape) == 2:
            # Grayscale - convert to BGR
            preprocessed = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            # RGBA - convert to BGR
            preprocessed = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        else:
            # Already BGR
            preprocessed = crop.copy()

        return preprocessed

    def detect_with_debug(
        self, frame: np.ndarray, player_index: int
    ) -> tuple[Optional[str], np.ndarray, dict]:
        """
        Detect agent and return debug information.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)

        Returns:
            Tuple of (agent_name or None, agent_crop, debug_info)
            debug_info contains match scores for all templates
        """
        debug_info = {"error": None}

        # Get active round crops
        player_crops = self.cropper.crop_player_info(frame)

        debug_info["num_crops"] = len(player_crops)

        if player_index >= len(player_crops):
            debug_info["error"] = f"player_index {player_index} >= num_crops {len(player_crops)}"
            return None, np.array([]), debug_info

        player_crop_data = player_crops[player_index]
        debug_info["crop_keys"] = list(player_crop_data.keys())

        if "agent_icon" not in player_crop_data:
            debug_info["error"] = f"'agent_icon' key not in crop_data. Available keys: {list(player_crop_data.keys())}"
            return None, np.array([]), debug_info

        agent_crop = player_crop_data["agent_icon"]
        debug_info["agent_crop_shape"] = agent_crop.shape if agent_crop.size > 0 else "empty"

        if agent_crop.size == 0:
            debug_info["error"] = "agent_crop.size == 0 (empty crop)"
            return None, np.array([]), debug_info

        # Preprocess crop
        preprocessed = self._preprocess_crop(agent_crop)

        # Get match scores for all templates
        match_scores = {}
        for agent_name, template_data in self.templates.items():
            template_img = template_data["image"]

            # Skip if template is larger than crop
            if template_img.shape[0] > preprocessed.shape[0] or template_img.shape[1] > preprocessed.shape[1]:
                match_scores[agent_name] = {
                    "agent": agent_name,
                    "confidence": 0.0,
                    "skipped": "template_too_large",
                }
                continue

            # Color template matching
            result = cv2.matchTemplate(preprocessed, template_img, self.match_method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            confidence = max(0.0, min(1.0, float(max_val)))

            match_scores[agent_name] = {
                "agent": agent_name,
                "confidence": confidence,
                "location": max_loc,
            }

        debug_info["match_scores"] = match_scores

        # Run detection
        agent_name = self.detect(frame, player_index)

        return agent_name, agent_crop, debug_info
