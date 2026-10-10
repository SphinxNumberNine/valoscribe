"""Sampling contracts independent of OpenCV decoding and gameplay inference."""

import math

import numpy as np
import pytest

from valoscribe.video.reader import VideoReader


class FrameSource:
    """Small seekable source with distinct frame contents."""

    def __init__(self, fps=29.97, count=300, position=0):
        self.fps = fps
        self.count = count
        self.position = position
        self.released = False

    def read_frame(self):
        if self.position >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.position % 256, dtype=np.uint8)
        self.position += 1
        return True, frame

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position
        return True

    def get_fps(self):
        return self.fps

    def get_frame_count(self):
        return self.count

    def get_width(self):
        return 2

    def get_height(self):
        return 2

    def release(self):
        self.released = True


@pytest.mark.parametrize("source_fps,target_fps", [(29.97, 4), (59.94, 4), (30, 7), (60, 4)])
def test_sampling_uses_absolute_target_times(source_fps, target_fps):
    with VideoReader(FrameSource(fps=source_fps), fps_filter=target_fps) as reader:
        frames = list(reader)
    expected = []
    sample = 0
    while (index := math.ceil(sample * source_fps / target_fps - 1e-9)) < 300:
        expected.append(index)
        sample += 1
    assert [frame.frame_number for frame in frames] == expected
    assert all(frame.timestamp_sec == frame.frame_number / source_fps for frame in frames)


def test_sampling_above_source_rate_yields_all_frames():
    with VideoReader(FrameSource(fps=30, count=10), fps_filter=120) as reader:
        assert [frame.frame_number for frame in reader] == list(range(10))


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("argument", ["source", "sampling"])
def test_invalid_rates_fail_before_reading(argument, rate):
    source = FrameSource(fps=rate if argument == "source" else 30)
    with pytest.raises(ValueError, match="FPS must be finite and greater than zero"):
        VideoReader(source, fps_filter=rate if argument == "sampling" else 4)
    assert source.position == 0
    assert source.released


def test_start_range_keeps_absolute_sampling_grid():
    source = FrameSource()
    with VideoReader(source, fps_filter=4, start_time_sec=1.1, end_time_sec=2) as reader:
        frames = list(reader)
    assert [frame.frame_number for frame in frames] == [38, 45, 53]


def test_seek_keeps_absolute_sampling_grid():
    with VideoReader(FrameSource(), fps_filter=4) as reader:
        next(reader)
        reader.seek(1.1)
        assert next(reader).frame_number == 38
