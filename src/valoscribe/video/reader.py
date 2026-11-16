"""
Video reader for frame-by-frame processing.

This module provides a clean interface for reading video frames from local files
and is designed to be extensible for future streaming sources (YouTube, Twitch).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

import cv2
import numpy as np

from valoscribe.types.video import FrameInfo
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)

FFMPEG_ENV_VAR = "VALOSCRIBE_FFMPEG"
FRAME_SOURCE_ENV_VAR = "VALOSCRIBE_FRAME_SOURCE"
FRAME_SOURCE_MODES = ("auto", "ffmpeg", "opencv")


def resolve_ffmpeg() -> Optional[str]:
    """Locate the ffmpeg executable, honouring an explicit override."""
    override = os.environ.get(FFMPEG_ENV_VAR)
    if override:
        candidate = Path(override)
        return str(candidate) if candidate.is_file() else None
    return shutil.which("ffmpeg")



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


class FFmpegFrameSource:
    """
    Video source that decodes through an ffmpeg subprocess.

    OpenCV decodes and converts every source frame to BGR, even though VideoReader
    only keeps one frame in ``source_fps / target_fps``. On 1080p60 broadcast files
    that wasted work dominates the pipeline. ffmpeg decodes the same frames roughly
    three times faster on the same machine, and its stdout gives us raw BGR frames
    without an intermediate copy.

    The sampling grid stays in VideoReader, so frame numbers and timestamps are
    unchanged; this class only replaces who turns the bitstream into pixels.
    """

    # Frames handed out are freshly allocated per read, so VideoReader need not copy.
    yields_owned_frames = True

    def __init__(self, video_path: Path | str, *, ffmpeg_path: Optional[str] = None):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._ffmpeg = ffmpeg_path or resolve_ffmpeg()
        if not self._ffmpeg:
            raise RuntimeError(
                "ffmpeg executable not found; install ffmpeg or set "
                f"{FFMPEG_ENV_VAR} to its path"
            )

        probe = cv2.VideoCapture(str(self.video_path))
        try:
            self._fps = float(probe.get(cv2.CAP_PROP_FPS))
            self._frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            self._width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            probe.release()

        if (
            not math.isfinite(self._fps)
            or self._fps <= 0
            or self._width <= 0
            or self._height <= 0
        ):
            raise RuntimeError(f"Could not read video metadata: {self.video_path}")

        self._frame_bytes = self._width * self._height * 3
        self._proc: Optional[subprocess.Popen] = None
        self._position = 0
        self._start_frame = 0
        self._spawn()

    def _command(self) -> list[str]:
        command = [str(self._ffmpeg), "-nostdin", "-v", "error"]
        if self._start_frame:
            # Input seeking is frame-accurate by default and skips the earlier GOPs.
            command += ["-ss", f"{self._start_frame / self._fps:.6f}"]
        command += [
            "-i",
            str(self.video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        return command

    def _spawn(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self._proc = subprocess.Popen(
            self._command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next decoded frame as a writable BGR array."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return False, None

        buffer = bytearray(self._frame_bytes)
        view = memoryview(buffer)
        filled = 0
        while filled < self._frame_bytes:
            chunk = proc.stdout.readinto(view[filled:])
            if not chunk:
                return False, None
            filled += chunk

        self._position += 1
        return True, np.frombuffer(buffer, dtype=np.uint8).reshape(
            self._height, self._width, 3
        )

    def get_position(self) -> int:
        """Frame index that the next read will return."""
        return self._position

    def set_position(self, frame_number: int) -> bool:
        """Restart the decoder at a frame, seeking inside ffmpeg."""
        frame_number = max(0, int(frame_number))
        self.release()
        self._start_frame = frame_number
        self._position = frame_number
        self._spawn()
        return True

    def get_fps(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def release(self) -> None:
        """Stop the decoder process and close the pipe."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


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
        frame_source: str = "opencv",
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
            frame_source: "opencv" (default, matches previous behaviour), "ffmpeg"
                (decode through ffmpeg, ~2-4x faster on 60 fps sources) or "auto"
                (ffmpeg when available). Only applies to file paths; the
                VALOSCRIBE_FRAME_SOURCE environment variable overrides this value.
        """
        # Initialize source
        if isinstance(source, (Path, str)):
            self._source: VideoSource = self._create_source(source, frame_source)
        else:
            self._source = source

        # Sources that hand out freshly allocated frames do not need another copy.
        self._copy_frames = not getattr(self._source, "yields_owned_frames", False)

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

    @staticmethod
    def _create_source(path: Path | str, frame_source: str) -> VideoSource:
        """Pick a decoder for a file path, falling back to OpenCV when needed."""
        mode = (os.environ.get(FRAME_SOURCE_ENV_VAR) or frame_source or "auto").lower()
        if mode not in FRAME_SOURCE_MODES:
            raise ValueError(
                "frame_source must be one of "
                f"{', '.join(FRAME_SOURCE_MODES)}; got {mode!r}"
            )

        if mode != "opencv" and resolve_ffmpeg():
            try:
                source = FFmpegFrameSource(path)
                log.info(f"Using ffmpeg decoding for {Path(path).name}")
                return source
            except (OSError, RuntimeError) as exc:
                if mode == "ffmpeg":
                    raise
                log.warning(f"Falling back to OpenCV decoding: {exc}")

        if mode == "ffmpeg":
            raise RuntimeError(
                "ffmpeg decoding requested but no ffmpeg executable was found; "
                f"install ffmpeg or set {FFMPEG_ENV_VAR}"
            )
        return FileVideoSource(path)

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
                frame=frame.copy() if self._copy_frames else frame,
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
        old_current_frame = self._current_frame
        try:
            if self.seek(timestamp_sec):
                frame_number = self._source.get_position()
                success, frame = self._source.read_frame()
                if success and frame is not None:
                    actual_timestamp = frame_number / self.fps if self.fps > 0 else 0.0
                    return FrameInfo(
                        frame_number=frame_number,
                        timestamp_ms=actual_timestamp * 1000.0,
                        timestamp_sec=actual_timestamp,
                        frame=frame.copy() if self._copy_frames else frame,
                    )
        finally:
            self._source.set_position(old_position)
            self._current_frame = old_current_frame
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
