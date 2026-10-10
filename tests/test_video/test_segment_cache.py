import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valoscribe.video.youtube import download_youtube

URL = "https://youtube.com/watch?v=abc123"


@pytest.fixture
def downloader(tmp_path):
    source = tmp_path / "abc123-h1080-f60-full.mp4"
    info = {
        "id": "abc123",
        "title": "A / renamed: title",
        "height": 1080,
        "fps": 60,
        "duration": 100,
    }
    client = MagicMock()
    client.prepare_filename.return_value = str(source)

    def extract(url, download):
        if download:
            source.write_bytes(b"full source")
        return dict(info)

    client.extract_info.side_effect = extract
    with patch("valoscribe.video.youtube.YoutubeDL") as factory:
        factory.return_value.__enter__.return_value = client
        yield client, source, info, factory


@pytest.fixture
def trim():
    def run(args, **kwargs):
        Path(args[-1]).write_bytes(b"trimmed clip")

    with (
        patch("valoscribe.video.youtube.subprocess.run", side_effect=run) as ffmpeg,
        patch(
            "valoscribe.video.youtube._probe_clip",
            return_value={
                "duration": 9.8,
                "height": 1080,
                "fps": 60,
            },
        ),
    ):
        yield ffmpeg


def test_segments_have_distinct_identity_and_preserve_full_source(tmp_path, downloader, trim):
    client, source, info, _ = downloader
    first = download_youtube(URL, tmp_path, start_time=10, duration=10)
    second = download_youtube(URL, tmp_path, start_time=20, duration=10)
    third = download_youtube(URL, tmp_path, start_time=10, duration=20)
    assert len({first.out_path, second.out_path, third.out_path, source}) == 4
    assert source.read_bytes() == b"full source"
    assert first.duration == 9.8
    assert first.source_offset_sec == 10
    assert first.requested_duration_sec == 10
    assert sum(call.kwargs["download"] for call in client.extract_info.call_args_list) == 1
    info["title"] = "New title"
    cached = download_youtube(URL + "&t=10s", tmp_path, duration=10)
    assert cached.out_path == first.out_path
    assert cached.duration == 9.8
    assert cached.url.endswith("t=10s")
    assert trim.call_count == 3
    full = download_youtube(URL, tmp_path)
    assert full.out_path == source
    assert full.duration == 100
    assert full.source_offset_sec == 0


def test_zero_start_and_duration_only_are_segments(tmp_path, downloader, trim):
    default = download_youtube(URL + "&t=0s", tmp_path)
    assert default.requested_duration_sec == 5400
    explicit = download_youtube(URL, tmp_path, duration=5)
    assert explicit.source_offset_sec == 0
    assert explicit.requested_duration_sec == 5
    assert explicit.out_path != default.out_path


@pytest.mark.parametrize(
    "start,duration",
    [
        (-1, 10),
        (float("nan"), 10),
        (float("inf"), 10),
        (0, 0),
        (0, -1),
        (0, float("inf")),
        (0, float("nan")),
    ],
)
def test_invalid_ranges_fail_before_network(tmp_path, start, duration):
    with patch("valoscribe.video.youtube.YoutubeDL") as factory:
        with pytest.raises(ValueError):
            download_youtube(URL, tmp_path, start_time=start, duration=duration)
        factory.assert_not_called()


def test_start_beyond_source_is_rejected(tmp_path, downloader):
    with pytest.raises(ValueError, match="before the end"):
        download_youtube(URL, tmp_path, start_time=100, duration=10)


@pytest.mark.parametrize(
    "failure", [subprocess.CalledProcessError(1, "ffmpeg"), FileNotFoundError("ffmpeg")]
)
def test_trim_failure_preserves_source_and_does_not_cache_clip(tmp_path, downloader, failure):
    with patch("valoscribe.video.youtube.subprocess.run", side_effect=failure):
        with pytest.raises(RuntimeError, match="requested video segment"):
            download_youtube(URL, tmp_path, start_time=10, duration=10)
    assert downloader[1].read_bytes() == b"full source"
    assert not list(tmp_path.glob("*segment*"))


def test_failed_overwrite_preserves_good_clip(tmp_path, downloader, trim):
    first = download_youtube(URL, tmp_path, start_time=10, duration=10)
    manifest = first.out_path.with_suffix(".mp4.valoscribe.json")
    original_manifest = manifest.read_bytes()
    with patch("valoscribe.video.youtube._probe_clip", side_effect=ValueError("bad media")):
        with pytest.raises(RuntimeError):
            download_youtube(URL, tmp_path, start_time=10, duration=10, overwrite=True)
    assert first.out_path.read_bytes() == b"trimmed clip"
    assert manifest.read_bytes() == original_manifest
    assert not list(tmp_path.glob("*.partial.mp4"))


@pytest.mark.parametrize("manifest", [None, "invalid", '{"version": 999}'])
def test_unverified_source_is_not_reused(tmp_path, downloader, manifest):
    client, source, _, _ = downloader
    source.write_bytes(b"legacy clip disguised as full source")
    if manifest is not None:
        source.with_suffix(".mp4.valoscribe.json").write_text(manifest)
    download_youtube(URL, tmp_path)
    assert source.read_bytes() == b"full source"
    assert client.extract_info.call_args.kwargs["download"] is True


def test_reported_postprocessed_path_is_used(tmp_path, downloader):
    actual = tmp_path / "actual.mkv"
    actual.write_bytes(b"media")
    downloader[2]["filepath"] = str(actual)
    result = download_youtube(URL, tmp_path)
    assert result.out_path == actual
    assert download_youtube(URL, tmp_path).out_path == actual


def test_clip_metadata_probe_uses_produced_duration(tmp_path):
    from valoscribe.video.youtube import _probe_clip

    with patch(
        "valoscribe.video.youtube.subprocess.run",
        return_value=MagicMock(
            stdout=json.dumps(
                {
                    "streams": [{"height": 720, "avg_frame_rate": "30000/1001"}],
                    "format": {"duration": "3.25"},
                }
            )
        ),
    ):
        media = _probe_clip(tmp_path / "clip.mp4")
    assert media["duration"] == 3.25
    assert media["fps"] == pytest.approx(30000 / 1001)
    assert media["height"] == 720


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg and ffprobe required for media integration test",
)
def test_real_trim_to_source_end_and_cache_metadata(tmp_path):
    source = tmp_path / "abc123-h1080-f60-full.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=10:d=5",
            "-c:v",
            "mpeg4",
            "-g",
            "1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    original = source.read_bytes()
    client = MagicMock()
    client.prepare_filename.return_value = str(source)
    client.extract_info.return_value = {
        "id": "abc123",
        "title": "Synthetic VOD",
        "duration": 5,
        "height": 64,
        "fps": 10,
        "filepath": str(source),
    }
    with patch("valoscribe.video.youtube.YoutubeDL") as factory:
        factory.return_value.__enter__.return_value = client
        clip = download_youtube(URL, tmp_path, start_time=4, duration=10)
        cached = download_youtube(URL, tmp_path, start_time=4, duration=10)
    assert source.read_bytes() == original
    assert clip == cached
    assert clip.duration == pytest.approx(1, abs=0.2)
    assert clip.height == 64
    assert clip.fps == 10
    assert clip.source_offset_sec == 4
    assert clip.requested_duration_sec == 10
