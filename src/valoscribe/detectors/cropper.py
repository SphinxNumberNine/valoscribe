"""
Cropper for Valorant HUD elements.

This module provides cropping functionality for extracting specific UI regions
from 1080p Valorant esports broadcast frames based on JSON coordinate configuration.
"""

from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Dict, List, Any, Iterator

import cv2
import numpy as np

from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class Cropper:
    """Crops HUD elements from Valorant frames based on JSON configuration."""

    def __init__(self, config_path: str | Path = "src/valoscribe/config/champs2025.json"):
        """
        Initialize cropper with HUD coordinate configuration.

        Args:
            config_path: Path to JSON configuration file with HUD coordinates
        """
        self.config = self._load_config(config_path)
        self.regions = self.config["regions"]
        self._scoped_frame = None
        self._player_crop_cache = {}
        log.info(f"Loaded HUD config: {self.config['name']}")

    @contextmanager
    def frame_scope(self, frame: np.ndarray) -> Iterator[Cropper]:
        """Reuse player crops during one immutable frame's processing.

        Detectors must treat returned crops as read-only. Caches are released
        on scope exit, including early returns and exceptions. Calls outside
        a scope retain the original behavior and return freshly copied crops.
        """
        previous_frame = self._scoped_frame
        previous_cache = self._player_crop_cache
        if frame is not previous_frame:
            self._scoped_frame = frame
            self._player_crop_cache = {}
        try:
            yield self
        finally:
            self._scoped_frame = previous_frame
            self._player_crop_cache = previous_cache

    def _get_player_crops(
        self, frame: np.ndarray, preround: bool
    ) -> List[Dict[str, np.ndarray]]:
        crop = self._crop_player_info_preround if preround else self._crop_player_info
        if frame is not self._scoped_frame:
            return crop(frame)
        if preround not in self._player_crop_cache:
            self._player_crop_cache[preround] = crop(frame)
        return self._player_crop_cache[preround]

    def _load_config(self, config_path: str | Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            return json.load(f)

    def crop_simple_region(self, frame: np.ndarray, region_name: str) -> np.ndarray:
        """
        Crop a simple region using x, y, width, height coordinates.

        Applicable to: round_number, team1_score, team2_score, round_timer, minimap

        Args:
            frame: Input frame (1080p)
            region_name: Name of region to crop

        Returns:
            Cropped region as numpy array
        """
        region = self.regions[region_name]
        x, y = region["x"], region["y"]
        width, height = region["width"], region["height"]

        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + width)
        y2 = min(frame.shape[0], y + height)

        return frame[y:y2, x:x2].copy()

    def crop_killfeed(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Crop killfeed region and subdivide into individual kill entries.

        Process:
        1. Crop main killfeed region using x, y, width, height
        2. Subdivide vertically into 10 individual entries
        3. Each entry has height = individual_height
        4. Entries are separated by offset pixels

        Args:
            frame: Input frame (1080p)

        Returns:
            List of 10 cropped kill entry regions
        """
        killfeed = self.regions["killfeed"]

        # Step 1: Crop main killfeed region
        x, y = killfeed["x"], killfeed["y"]
        width, height = killfeed["width"], killfeed["height"]

        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + width)
        y2 = min(frame.shape[0], y + height)

        killfeed_crop = frame[y:y2, x:x2].copy()

        # Step 2: Subdivide into individual entries
        individual_height = killfeed["individual_height"]
        offset = killfeed["offset"]

        entries = []
        current_y = 0

        for i in range(10):
            # Check if we have enough space for another entry
            if current_y + individual_height > killfeed_crop.shape[0]:
                break

            # Crop individual entry
            entry = killfeed_crop[current_y:current_y + individual_height, :].copy()
            entries.append(entry)

            # Move to next entry position
            current_y += individual_height + offset

        return entries

    def crop_player_info(self, frame: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Crop all ten players, reusing results inside a frame scope."""
        return self._get_player_crops(frame, preround=False)

    def _crop_player_info(self, frame: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Crop player info for all 10 players (5 per side).

        Process:
        LEFT side (players 0-4):
        1. Crop main player_info region
        2. Subdivide into 5 player boxes
        3. For each box, crop all individual_player_info sub-regions

        RIGHT side (players 5-9):
        1. Mirror frame horizontally
        2. Repeat cropping process
        3. Flip agent_icon and player_name back to match left side orientation

        Args:
            frame: Input frame (1080p)

        Returns:
            List of 10 player info dictionaries, each containing cropped sub-regions
        """
        player_info = self.regions["player_info"]
        individual_info = self.regions["individual_player_info"]

        # LEFT side: players 0-4
        left_players = self._crop_side_players(frame, player_info, individual_info)
        for player in left_players:
            player["side"] = "left"

        # RIGHT side: players 5-9
        # Mirror frame horizontally to access right side using same coordinates
        mirrored_frame = cv2.flip(frame, 1)
        right_players = self._crop_side_players(mirrored_frame, player_info, individual_info)

        # Flip player_name, health, and armor back for right side players (text needs to be readable)
        # Also swap ability_1 and ability_3 to maintain consistent ordering
        for player in right_players:
            player["side"] = "right"

            # Flip player_name, health, and armor to make text readable
            if "player_name" in player and player["player_name"].size > 0:
                player["player_name"] = cv2.flip(player["player_name"], 1)
            if "health" in player and player['health'].size > 0:
                player['health'] = cv2.flip(player['health'], 1)
            if "armor" in player and player['armor'].size > 0:
                player['armor'] = cv2.flip(player['armor'], 1)

            # Swap abilities to maintain consistent order (ability_1 <-> ability_3)
            if "ability_1" in player and "ability_3" in player:
                player["ability_1"], player["ability_3"] = player["ability_3"], player["ability_1"]

        # Combine both sides
        return left_players + right_players

    def _crop_side_players(
        self,
        frame: np.ndarray,
        player_info: Dict[str, int],
        individual_info: Dict[str, Dict[str, int]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Crop player info for one side of the screen (5 players).

        Args:
            frame: Input frame or mirrored frame
            player_info: Player info region coordinates
            individual_info: Individual player element coordinates

        Returns:
            List of 5 player info dictionaries
        """
        # Crop main player_info region
        x, y = player_info["x"], player_info["y"]
        width, height = player_info["width"], player_info["height"]

        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + width)
        y2 = min(frame.shape[0], y + height)

        player_region = frame[y:y2, x:x2].copy()

        # Subdivide into individual player boxes
        individual_height = player_info["individual_height"]
        offset = player_info["offset"]

        players = []
        current_y = 0

        for i in range(5):
            # Check if we have enough space
            if current_y + individual_height > player_region.shape[0]:
                break

            # Crop individual player box
            player_box = player_region[current_y:current_y + individual_height, :].copy()

            # Crop sub-regions within player box
            player_crops = {}
            for element_name, coords in individual_info.items():
                ex, ey = coords["x"], coords["y"]
                ewidth, eheight = coords["width"], coords["height"]

                # Ensure coordinates are within player box bounds
                ex = max(0, ex)
                ey = max(0, ey)
                ex2 = min(player_box.shape[1], ex + ewidth)
                ey2 = min(player_box.shape[0], ey + eheight)

                if ex < ex2 and ey < ey2:
                    player_crops[element_name] = player_box[ey:ey2, ex:ex2].copy()
                else:
                    # Invalid crop, create empty array
                    player_crops[element_name] = np.array([])

            players.append(player_crops)

            # Move to next player position
            current_y += individual_height + offset

        return players

    def crop_player_info_preround(self, frame: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Crop all ten preround players, reusing results inside a frame scope."""
        return self._get_player_crops(frame, preround=True)

    def _crop_player_info_preround(self, frame: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Crop pre-round player info for all 10 players (5 per side).

        Process is identical to crop_player_info() but uses pre-round coordinates.

        LEFT side (players 0-4):
        1. Crop main player_info_preround region
        2. Subdivide into 5 player boxes
        3. For each box, crop all individual_player_info_preround sub-regions

        RIGHT side (players 5-9):
        1. Mirror frame horizontally
        2. Repeat cropping process
        3. Flip agent_icon and player_name back to match left side orientation

        Args:
            frame: Input frame (1080p)

        Returns:
            List of 10 player info dictionaries, each containing cropped sub-regions
        """
        player_info_preround = self.regions["player_info_preround"]
        individual_info_preround = self.regions["individual_player_info_preround"]

        # LEFT side: players 0-4
        left_players = self._crop_side_players(frame, player_info_preround, individual_info_preround)
        for player in left_players:
            player["side"] = "left"

        # RIGHT side: players 5-9
        # Mirror frame horizontally to access right side using same coordinates
        mirrored_frame = cv2.flip(frame, 1)
        right_players = self._crop_side_players(mirrored_frame, player_info_preround, individual_info_preround)

        # Flip player_name back for right side players (text needs to be readable)
        # Also swap ability_1 and ability_3 to maintain consistent ordering
        for player in right_players:
            player["side"] = "right"

            # Flip player_name to make text readable
            if "player_name" in player and player["player_name"].size > 0:
                player["player_name"] = cv2.flip(player["player_name"], 1)

            # Swap abilities to maintain consistent order (ability_1 <-> ability_3)
            if "ability_1" in player and "ability_3" in player:
                player["ability_1"], player["ability_3"] = player["ability_3"], player["ability_1"]

        # Combine both sides
        return left_players + right_players

    def crop_all_regions(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Crop all regions from a frame.

        Args:
            frame: Input frame (1080p)

        Returns:
            Dictionary containing all cropped regions:
            - Simple regions: round_number, team1_score, team2_score, round_timer, minimap
            - killfeed: List of 10 kill entry crops
            - player_info: List of 10 player info dictionaries
        """
        result = {}

        # Simple regions
        simple_regions = ["round_number", "team1_score", "team2_score", "round_timer", "minimap"]
        for region_name in simple_regions:
            result[region_name] = self.crop_simple_region(frame, region_name)

        # Killfeed
        result["killfeed"] = self.crop_killfeed(frame)

        # Player info
        result["player_info"] = self.crop_player_info(frame)

        return result

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the loaded configuration."""
        return {
            "name": self.config["name"],
            "simple_regions": ["round_number", "team1_score", "team2_score", "round_timer", "minimap"],
            "killfeed_entries": 10,
            "player_count": 10,
            "player_elements": list(self.regions["individual_player_info"].keys()),
            "player_elements_preround": list(self.regions["individual_player_info_preround"].keys()),
        }
