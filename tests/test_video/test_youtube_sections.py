from unittest.mock import MagicMock, patch

import pytest

from valoscribe.video.youtube import download_youtube


@pytest.mark.parametrize(
    "start,duration",
    [(None, 120), (0, None), (-1, 120), (0, 0), (float("nan"), 120), (0, float("inf"))],
)
def test_section_requires_explicit_finite_bounds(tmp_path, start, duration):
    with pytest.raises(ValueError):
        download_youtube(
            "https://youtu.be/test",
            tmp_path,
            start_time=start,
            duration=duration,
            section_only=True,
        )


def test_section_uses_range_downloader_without_full_download(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    ydl = MagicMock()
    ydl.extract_info.return_value = {"title": "clip", "id": "test", "fps": 30, "height": 1080}
    ydl.prepare_filename.return_value = str(path)
    with patch("valoscribe.video.youtube.YoutubeDL") as factory:
        factory.return_value.__enter__.return_value = ydl
        result = download_youtube(
            "https://youtu.be/test", tmp_path, start_time=132, duration=120, section_only=True
        )
        options = factory.call_args.args[0]
        assert list(options["download_ranges"]({}, ydl)) == [{"start_time": 132, "end_time": 252}]
        assert options["force_keyframes_at_cuts"]
        assert options["noplaylist"]
        assert result.out_path == path
        assert result.duration == 120


def test_section_does_not_accept_a_missing_file(tmp_path):
    with patch("valoscribe.video.youtube.YoutubeDL") as factory:
        ydl = factory.return_value.__enter__.return_value
        ydl.extract_info.return_value = {}
        ydl.prepare_filename.return_value = str(tmp_path / "missing.mp4")
        with pytest.raises(RuntimeError):
            download_youtube(
                "https://youtu.be/test", tmp_path, start_time=0, duration=120, section_only=True
            )
