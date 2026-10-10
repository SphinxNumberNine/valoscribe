"""
Video reader for frame-by-frame processing.

This module provides a clean interface for reading video frames from local files
and is designed to be extensible for future streaming sources (YouTube, Twitch).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

import cv2
import numpy as np

from valoscribe.types.video import FrameInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class VideoSource(Protocol):
    """Protocol for video sources (files, streams, etc.)."""

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read next frame. Returns (success, frame)."""
        ...

    def get_position(self) -> int:
        """Get current frame position."""
        ...

    def set_position(self, frame_number: int) -> bool:
        """Seek to specific frame. Returns success."""
        ...

    def get_fps(self) -> float:
        """Get frames per second."""
        ...

    def get_frame_count(self) -> int:
        """Get total frame count (may be -1 for streams)."""
        ...

    def get_width(self) -> int:
        """Get frame width."""
        ...

    def get_height(self) -> int:
        """Get frame height."""
        ...

    def release(self) -> None:
        """Release video source."""
        ...


class FileVideoSource:
    """Video source for local video files."""

    def __init__(self, video_path: Path | str):
        """
        Initialize video file source.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        """Open video file."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read next frame."""
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def get_position(self) -> int:
        """Get current frame position."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def set_position(self, frame_number: int) -> bool:
        """Seek to specific frame."""
        if self._cap is None:
            return False
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_fps(self) -> float:
        """Get frames per second."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self) -> int:
        """Get total frame count."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_width(self) -> int:
        """Get frame width."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> int:
        """Get frame height."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self) -> None:
        """Release video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoReader:
    """
    Frame-by-frame video reader with filtering and sampling support.

    Example:
        reader = VideoReader("path/to/video.mp4")
        for frame_info in reader:
            # Process frame_info.frame
            print(f"Frame {frame_info.frame_number} at {frame_info.timestamp_sec}s")
    """

    def __init__(
        self,
        source: Path | str | VideoSource,
        *,
        fps_filter: Optional[float] = None,
        start_time_sec: Optional[float] = None,
        end_time_sec: Optional[float] = None,
        frame_filter: Optional[Callable[[np.ndarray], bool]] = None,
        progress_callback: Optional[Callable[[FrameInfo], None]] = None,
    ):
        """
        Initialize video reader.

        Args:
            source: Video file path or VideoSource instance
            fps_filter: If set, only process frames at this rate (e.g., 1.0 = 1 frame/sec)
            start_time_sec: Start reading from this timestamp (seconds)
            end_time_sec: Stop reading at this timestamp (seconds)
            frame_filter: Optional function to filter frames (returns True to keep)
            progress_callback: Optional callback called for each processed frame
        """
        # Initialize source
        if isinstance(source, (Path, str)):
            self._source: VideoSource = FileVideoSource(source)
        else:
            self._source = source

        self.fps_filter = fps_filter
        self.start_time_sec = start_time_sec
        self.end_time_sec = end_time_sec
        self.frame_filter = frame_filter
        self.progress_callback = progress_callback

        source_fps = self._source.get_fps()
        if not math.isfinite(source_fps) or source_fps <= 0:
            self._source.release()
            raise ValueError("Video FPS must be finite and greater than zero")
        if fps_filter is not None and (not math.isfinite(fps_filter) or fps_filter <= 0):
            self._source.release()
            raise ValueError("Sampling FPS must be finite and greater than zero")

        # Keep fractional intervals so 29.97 FPS sampled at 4 FPS does not
        # accidentally become 29.97 / 7 = 4.28 FPS. Requests above the source
        # rate yield every available frame rather than dividing by zero.
        self._frame_interval = max(1.0, source_fps / fps_filter) if fps_filter else 1.0

        # Seek to start position if specified
        if start_time_sec is not None:
            start_frame = int(start_time_sec * self._source.get_fps())
            log.info(f"Seeking to start frame {start_frame} (time: {start_time_sec}s)")
            self._source.set_position(start_frame)
            actual_position = self._source.get_position()
            log.info(
                f"After seek, actual position: {actual_position} "
                f"(time: {actual_position / source_fps:.2f}s)"
            )
            if abs(actual_position - start_frame) > 1:
                log.warning(
                    f"Seek imprecise: requested frame {start_frame}, got {actual_position} "
                    f"(diff: {actual_position - start_frame} frames, "
                    f"{abs(actual_position - start_frame) / self._source.get_fps():.2f}s)"
                )

        # Track current position
        self._current_frame = self._source.get_position()
        self._last_processed_frame = -1

        # Get video metadata
        self.fps = self._source.get_fps()
        self.frame_count = self._source.get_frame_count()
        self.width = self._source.get_width()
        self.height = self._source.get_height()
        self.duration_sec = (
            self.frame_count / self.fps if self.fps > 0 and self.frame_count > 0 else 0.0
        )

        log.info(
            f"Video reader initialized: {self.width}x{self.height}, "
            f"{self.fps:.2f} FPS, {self.duration_sec:.2f}s duration"
        )
        log.info(f"Starting at frame {self._current_frame}, frame_interval={self._frame_interval}")

        # Track for debug logging
        self._frames_yielded = 0

    def __iter__(self) -> Iterator[FrameInfo]:
        """Iterate over video frames."""
        return self

    def __next__(self) -> FrameInfo:
        """Get next frame."""
        while True:
            # Check end condition
            if self.end_time_sec is not None:
                current_time = self._current_frame / self.fps if self.fps > 0 else 0.0
                if current_time >= self.end_time_sec:
                    raise StopIteration

            # Read frame
            success, frame = self._source.read_frame()
            if not success or frame is None:
                raise StopIteration

            # Apply frame interval filtering
            # Select the first source frame at or after each sampling time.
            # Deriving the target from the absolute frame index avoids drift
            # and keeps the sampling grid stable across seeks and start ranges.
            sample_index = math.floor(self._current_frame / self._frame_interval + 1e-9)
            sample_frame = math.ceil(sample_index * self._frame_interval - 1e-9)
            if self._current_frame != sample_frame:
                self._current_frame += 1
                continue

            # Apply custom frame filter
            if self.frame_filter is not None:
                if not self.frame_filter(frame):
                    self._current_frame += 1
                    continue

            # Create frame info
            timestamp_sec = self._current_frame / self.fps if self.fps > 0 else 0.0
            timestamp_ms = timestamp_sec * 1000.0

            frame_info = FrameInfo(
                frame_number=self._current_frame,
                timestamp_ms=timestamp_ms,
                timestamp_sec=timestamp_sec,
                frame=frame.copy(),  # Copy to avoid modification issues
            )

            # Call progress callback
            if self.progress_callback:
                self.progress_callback(frame_info)

            # Debug logging for first few frames
            if self._frames_yielded < 3:
                log.debug(
                    f"Yielding frame {frame_info.frame_number} "
                    f"(time: {frame_info.timestamp_sec:.2f}s, "
                    f"actual video position: {self._source.get_position()})"
                )

            self._frames_yielded += 1
            self._current_frame += 1
            return frame_info

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def seek(self, timestamp_sec: float) -> bool:
        """
        Seek to specific timestamp.

        Args:
            timestamp_sec: Timestamp in seconds

        Returns:
            True if seek was successful
        """
        frame_number = int(timestamp_sec * self.fps)
        success = self._source.set_position(frame_number)
        if success:
            self._current_frame = frame_number
        return success

    def get_frame_at(self, timestamp_sec: float) -> Optional[FrameInfo]:
        """
        Get frame at specific timestamp without advancing iterator.

        Args:
            timestamp_sec: Timestamp in seconds

        Returns:
            FrameInfo if found, None otherwise
        """
        old_position = self._source.get_position()
        try:
            if self.seek(timestamp_sec):
                success, frame = self._source.read_frame()
                if success and frame is not None:
                    timestamp_ms = timestamp_sec * 1000.0
                    return FrameInfo(
                        frame_number=self._source.get_position(),
                        timestamp_ms=timestamp_ms,
                        timestamp_sec=timestamp_sec,
                        frame=frame.copy(),
                    )
        finally:
            self._source.set_position(old_position)
        return None

    def close(self) -> None:
        """Release video source."""
        self._source.release()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoReader(fps={self.fps:.2f}, "
            f"resolution={self.width}x{self.height}, "
            f"duration={self.duration_sec:.2f}s)"
        )


def read_video_frames(
    video_path: Path | str,
    *,
    fps_filter: Optional[float] = None,
    start_time_sec: Optional[float] = None,
    end_time_sec: Optional[float] = None,
) -> Iterator[FrameInfo]:
    """
    Convenience function to read video frames.

    Args:
        video_path: Path to video file
        fps_filter: Process frames at this rate (e.g., 1.0 = 1 frame/sec)
        start_time_sec: Start timestamp
        end_time_sec: End timestamp

    Yields:
        FrameInfo objects
    """
    with VideoReader(
        video_path,
        fps_filter=fps_filter,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
    ) as reader:
        yield from reader
