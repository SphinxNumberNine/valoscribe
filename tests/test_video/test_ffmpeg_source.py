"""Tests for the ffmpeg-backed frame source and decoder selection."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from valoscribe.video import reader as reader_module
from valoscribe.video.reader import FFmpegFrameSource, FileVideoSource, VideoReader

FPS = 60.0
WIDTH, HEIGHT = 64, 48
FRAME_BYTES = WIDTH * HEIGHT * 3


class FakeProcess:
    """Minimal subprocess stand-in with a readable stdout pipe."""

    def __init__(self, payload: bytes):
        self.stdout = io.BytesIO(payload)
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode or 0


@pytest.fixture
def probe_capture():
    """cv2.VideoCapture stand-in that only reports metadata."""
    with patch.object(reader_module.cv2, "VideoCapture") as factory:
        cap = MagicMock()
        cap.get.side_effect = lambda prop: {
            reader_module.cv2.CAP_PROP_FPS: FPS,
            reader_module.cv2.CAP_PROP_FRAME_COUNT: 120,
            reader_module.cv2.CAP_PROP_FRAME_WIDTH: WIDTH,
            reader_module.cv2.CAP_PROP_FRAME_HEIGHT: HEIGHT,
        }.get(prop, 0)
        factory.return_value = cap
        yield factory


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not-a-real-video")
    return path


def install_fake_process(monkeypatch, payload: bytes):
    processes: list[FakeProcess] = []

    def fake_spawn(self) -> None:
        process = FakeProcess(payload)
        processes.append(process)
        self._proc = process

    monkeypatch.setattr(FFmpegFrameSource, "_spawn", fake_spawn)
    return processes


def frame_payload(count: int) -> bytes:
    frames = [bytes([i % 256]) * FRAME_BYTES for i in range(count)]
    return b"".join(frames)


class TestFFmpegFrameSource:
    def test_reads_frames_until_eof(self, probe_capture, video_file, monkeypatch):
        install_fake_process(monkeypatch, frame_payload(3))
        source = FFmpegFrameSource(video_file, ffmpeg_path="ffmpeg")

        frames = []
        while True:
            ok, frame = source.read_frame()
            if not ok:
                break
            frames.append(frame)
        source.release()

        assert len(frames) == 3
        assert all(frame.shape == (HEIGHT, WIDTH, 3) for frame in frames)
        assert all(frame.flags.writeable for frame in frames)
        assert source.get_position() == 3

    def test_command_has_seek_after_set_position(self, probe_capture, video_file, monkeypatch):
        install_fake_process(monkeypatch, frame_payload(1))
        source = FFmpegFrameSource(video_file, ffmpeg_path="ffmpeg")

        assert "-ss" not in source._command()
        assert source.set_position(30) is True
        command = source._command()
        assert command[command.index("-ss") + 1] == "0.500000"
        assert source.get_position() == 30
        source.release()

    def test_release_terminates_process(self, probe_capture, video_file, monkeypatch):
        processes = install_fake_process(monkeypatch, frame_payload(1))
        source = FFmpegFrameSource(video_file, ffmpeg_path="ffmpeg")

        source.release()

        assert processes[0].terminated
        assert source.read_frame() == (False, None)

    def test_metadata_failure_raises(self, probe_capture, video_file, monkeypatch):
        install_fake_process(monkeypatch, b"")
        probe_capture.return_value.get.side_effect = lambda prop: 0

        with pytest.raises(RuntimeError, match="Could not read video metadata"):
            FFmpegFrameSource(video_file, ffmpeg_path="ffmpeg")


class TestDecoderSelection:
    def test_auto_uses_ffmpeg_when_available(self, probe_capture, video_file, monkeypatch):
        install_fake_process(monkeypatch, frame_payload(1))
        monkeypatch.setattr(reader_module, "resolve_ffmpeg", lambda: "ffmpeg")

        reader = VideoReader(video_file, frame_source="auto")

        assert isinstance(reader._source, FFmpegFrameSource)
        reader.close()

    def test_default_stays_on_opencv_even_with_ffmpeg(self, probe_capture, video_file, monkeypatch):
        monkeypatch.setattr(reader_module, "resolve_ffmpeg", lambda: "ffmpeg")

        reader = VideoReader(video_file)

        assert isinstance(reader._source, FileVideoSource)
        reader.close()

    def test_auto_falls_back_to_opencv(self, probe_capture, video_file, monkeypatch):
        monkeypatch.setattr(reader_module, "resolve_ffmpeg", lambda: None)

        with patch.object(reader_module, "FileVideoSource", wraps=FileVideoSource) as factory:
            reader = VideoReader(video_file)
            assert factory.called
        assert isinstance(reader._source, FileVideoSource)
        reader.close()

    def test_explicit_ffmpeg_without_binary_raises(self, video_file, monkeypatch):
        monkeypatch.setattr(reader_module, "resolve_ffmpeg", lambda: None)

        with pytest.raises(RuntimeError, match="no ffmpeg executable"):
            VideoReader(video_file, frame_source="ffmpeg")

    def test_unknown_mode_raises(self, video_file):
        with pytest.raises(ValueError, match="frame_source must be one of"):
            VideoReader(video_file, frame_source="magic")

    def test_env_var_overrides_argument(self, probe_capture, video_file, monkeypatch):
        monkeypatch.setenv(reader_module.FRAME_SOURCE_ENV_VAR, "opencv")
        monkeypatch.setattr(reader_module, "resolve_ffmpeg", lambda: "ffmpeg")

        reader = VideoReader(video_file, frame_source="ffmpeg")

        assert isinstance(reader._source, FileVideoSource)
        reader.close()


class OwnedFrameSource:
    """Source that already hands out writable frames, like the ffmpeg pipe."""

    yields_owned_frames = True

    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.read = False
        self.position = 0

    def read_frame(self):
        if self.read:
            return False, None
        self.read = True
        return True, self.frame

    def get_position(self):
        return self.position

    def set_position(self, frame_number):
        self.position = frame_number
        return True

    def get_fps(self):
        return 60.0

    def get_frame_count(self):
        return 1

    def get_width(self):
        return WIDTH

    def get_height(self):
        return HEIGHT

    def release(self):
        pass


def test_owned_frames_are_not_copied_again():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    reader = VideoReader(OwnedFrameSource(frame))

    frame_info = next(reader)

    assert frame_info.frame is frame
