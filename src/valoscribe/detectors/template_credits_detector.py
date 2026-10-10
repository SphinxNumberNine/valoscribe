"""
Template-based credits detector for Valorant HUD.

Uses template matching to detect the dead credits icon in player info regions.
The dead credits icon appears when a player dies, so high confidence indicates
a dead player while low confidence indicates an alive player.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import CreditsInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class TemplateCreditsDetector:
    """
    Template matching-based detector for dead credits icon.

    Detects the dead credits icon in player info regions to determine
    if a player is dead (icon visible = high confidence) or alive (icon not visible = low confidence).
    """

    def __init__(
        self,
        cropper: Cropper,
        template_path: Optional[Path] = None,
        min_confidence: float = 0.7,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        """
        Initialize template credits detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            template_path: Path to dead credits icon template (credits_icon_dead.png)
                          If None, uses default: src/valoscribe/templates/credits/credits_icon_dead.png
            min_confidence: Minimum template match confidence (0-1) for detecting dead players
            match_method: OpenCV template matching method
        """
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.match_method = match_method

        # Set default template path if not provided
        if template_path is None:
            package_dir = Path(__file__).parent.parent
            template_path = package_dir / "templates" / "credits" / "credits_icon_dead.png"

        self.template_path = Path(template_path)
        self.template = self._load_template()

        log.info(
            f"Template credits detector initialized (min_confidence: {min_confidence}, "
            f"template_loaded: {self.template is not None}, method: {match_method})"
        )

    def _load_template(self) -> Optional[np.ndarray]:
        """
        Load credits icon template from template path.

        Returns:
            Template image if successful, None otherwise
        """
        if not self.template_path.exists():
            log.warning(f"Template file does not exist: {self.template_path}")
            return None

        # Load template in grayscale
        template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)

        if template is None:
            log.warning(f"Failed to load template: {self.template_path}")
            return None

        log.info(f"Loaded credits template: {template.shape}")
        return template

    def detect(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> Optional[CreditsInfo]:
        """
        Detect credits icon for a specific player using template matching.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard ("left" or "right")

        Returns:
            CreditsInfo if successfully checked, None if error
        """
        # Check if template is loaded
        if self.template is None:
            log.error("No template loaded, cannot detect credits")
            return None

        # Get player info crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            log.warning(f"Player index {player_index} out of range (max: {len(player_crops) - 1})")
            return None

        player_crop_data = player_crops[player_index]

        # Check side
        if player_crop_data["side"] != side:
            log.warning(
                f"Player {player_index} is on {player_crop_data['side']} side, "
                f"but {side} was requested"
            )
            return None

        # Get dead_credits crop
        if "dead_credits" not in player_crop_data:
            log.warning(f"Dead credits region not found in player crop data")
            return None

        dead_credits_crop = player_crop_data["dead_credits"]

        if dead_credits_crop.size == 0:
            log.warning(f"Dead credits crop is empty for player {player_index}")
            return None

        # Preprocess crop
        preprocessed = self._preprocess_crop(dead_credits_crop)

        # Match template
        result = cv2.matchTemplate(preprocessed, self.template, self.match_method)

        # Get best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # For TM_CCOEFF_NORMED, higher is better
        # Clamp confidence to [0.0, 1.0] range (template matching can return negative values)
        confidence = max(0.0, min(1.0, float(max_val)))

        # Check if confidence meets threshold
        # High confidence = dead credits icon visible = player is DEAD
        # Low confidence = dead credits icon not visible = player is ALIVE
        if confidence >= self.min_confidence:
            log.debug(f"Player {player_index} DEAD - dead credits visible with confidence: {confidence:.2f}")
            return CreditsInfo(
                credits_visible=True,  # True = dead credits visible = player is dead
                confidence=confidence,
            )
        else:
            log.debug(f"Player {player_index} ALIVE - dead credits not visible (confidence: {confidence:.2f})")
            return CreditsInfo(
                credits_visible=False,  # False = dead credits not visible = player is alive
                confidence=confidence,
            )

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess credits crop for template matching.

        Uses same preprocessing as other template detectors for consistency.

        Args:
            crop: Cropped credits region

        Returns:
            Preprocessed grayscale binary image
        """
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Upscale for better matching (3x)
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        # Use Otsu's thresholding - handles varying brightness
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def detect_with_debug(
        self, frame: np.ndarray, player_index: int, side: str = "left"
    ) -> tuple[Optional[CreditsInfo], np.ndarray, dict]:
        """
        Detect credits and return debug visualizations.

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-9)
            side: Which side of scoreboard

        Returns:
            Tuple of (CreditsInfo or None, credits_preprocessed, debug_info)
            debug_info contains match details for debugging
        """
        # Get player info crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            return None, np.array([]), {}

        player_crop_data = player_crops[player_index]

        if "dead_credits" not in player_crop_data:
            return None, np.array([]), {}

        credits_crop = player_crop_data["dead_credits"]

        if credits_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess using the same method as detection
        credits_preprocessed = self._preprocess_crop(credits_crop)

        # Get match result
        debug_info = {}
        if self.template is not None:
            result = cv2.matchTemplate(credits_preprocessed, self.template, self.match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Clamp confidence to [0.0, 1.0] range
            clamped_confidence = max(0.0, min(1.0, float(max_val)))

            debug_info = {
                "max_confidence": clamped_confidence,
                "max_location": max_loc,
                "match_result_shape": result.shape,
            }

        # Run detection
        credits_info = self.detect(frame, player_index, side)

        return credits_info, credits_preprocessed, debug_info
