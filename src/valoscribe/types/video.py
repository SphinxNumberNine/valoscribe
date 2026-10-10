"""Video-related Pydantic models and schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DownloadResult(BaseModel):
    """Result of a YouTube video download operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str = Field(..., description="Original YouTube URL")
    out_path: Path = Field(..., description="Final merged media file path")
    title: Optional[str] = Field(None, description="Video title")
    id: Optional[str] = Field(None, description="YouTube video ID")
    height: Optional[int] = Field(None, description="Video height in pixels")
    fps: Optional[float] = Field(None, description="Video frames per second")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    source_offset_sec: float = Field(
        0.0,
        ge=0,
        description="Requested clip start in the source VOD; stream-copy seek is approximate",
    )
    requested_duration_sec: Optional[float] = Field(
        None,
        gt=0,
        description="Requested clip duration; duration records the produced media length",
    )


class FrameInfo(BaseModel):
    """Metadata for a single video frame."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_number: int = Field(..., description="Frame index in video")
    timestamp_ms: float = Field(..., description="Timestamp in milliseconds from video start")
    timestamp_sec: float = Field(..., description="Timestamp in seconds from video start")
    frame: np.ndarray = Field(..., description="Frame data in BGR format (OpenCV default)")
