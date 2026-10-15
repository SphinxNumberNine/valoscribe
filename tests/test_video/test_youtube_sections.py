from pathlib import Path
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
    ydl = MagicMock()
    with (
        patch("valoscribe.video.youtube.YoutubeDL") as factory,
        patch(
            "valoscribe.video.youtube._probe_clip",
            return_value={"height": 1080, "fps": 30, "duration": 120.01},
        ),
    ):

        def fake_download(*args, **kwargs):
            Path(factory.call_args.args[0]["outtmpl"]).write_bytes(b"video")
            return {"title": "clip", "id": "test", "fps": 30, "height": 1080}

        ydl.extract_info.side_effect = fake_download
        factory.return_value.__enter__.return_value = ydl
        result = download_youtube(
            "https://youtu.be/test", tmp_path, start_time=132, duration=120, section_only=True
        )
        options = factory.call_args.args[0]
        assert list(options["download_ranges"]({}, ydl)) == [{"start_time": 132, "end_time": 252}]
        assert options["force_keyframes_at_cuts"]
        assert options["noplaylist"]
        assert result.out_path == Path(options["outtmpl"])
        assert result.duration == 120.01
        assert result.source_offset_sec == 132
        # Same bounds reuse provenance-checked media; another range downloads separately.
        again = download_youtube(
            "https://youtu.be/test", tmp_path, start_time=132, duration=120, section_only=True
        )
        assert again.out_path == result.out_path
        assert ydl.extract_info.call_count == 1


def test_section_does_not_accept_a_missing_file(tmp_path):
    with patch("valoscribe.video.youtube.YoutubeDL") as factory:
        ydl = factory.return_value.__enter__.return_value
        ydl.extract_info.return_value = {}
        ydl.prepare_filename.return_value = str(tmp_path / "missing.mp4")
        with pytest.raises(RuntimeError):
            download_youtube(
                "https://youtu.be/test", tmp_path, start_time=0, duration=120, section_only=True
            )
