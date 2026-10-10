from __future__ import annotations

import hashlib
import json
import math
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from tqdm import tqdm
from yt_dlp import YoutubeDL

from valoscribe.types.video import DownloadResult
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


def _parse_timestamp(timestamp_str: str) -> Optional[float]:
    """
    Parse YouTube timestamp parameter to seconds.

    Handles formats:
    - "1234" or "1234s" -> 1234 seconds
    - "20m34s" -> 1234 seconds
    - "1h20m34s" -> 4834 seconds

    Returns:
        Seconds as float, or None if parsing failed
    """
    import re

    timestamp_str = timestamp_str.strip()

    # Try simple integer or "1234s" format
    if timestamp_str.endswith("s"):
        try:
            return float(timestamp_str[:-1])
        except ValueError:
            pass
    else:
        try:
            return float(timestamp_str)
        except ValueError:
            pass

    # Try HH:MM:SS or MM:SS format
    if ":" in timestamp_str:
        parts = timestamp_str.split(":")
        try:
            if len(parts) == 2:  # MM:SS
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass

    # Try YouTube format: "1h20m34s"
    pattern = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    match = re.match(pattern, timestamp_str)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        # Only return if at least one component was matched
        if match.group(1) or match.group(2) or match.group(3):
            return hours * 3600 + minutes * 60 + seconds

    return None


class _TqdmProgressHook:
    """Progress hook for yt-dlp that uses tqdm for clean progress bars."""

    def __init__(self) -> None:
        self.pbar: Optional[tqdm] = None

    def __call__(self, d: Dict[str, Any]) -> None:
        status = d.get("status")

        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)

            if self.pbar is None and total:
                # Initialize progress bar on first call with known total
                self.pbar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading",
                )

            if self.pbar:
                # Update to current position
                self.pbar.n = downloaded
                self.pbar.refresh()

        elif status == "finished":
            if self.pbar:
                self.pbar.close()
                self.pbar = None
            # Single log message after download completes
            log.info("Download complete, merging streams...")


def _cache_manifest(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".valoscribe.json")


def _read_cached_result(path: Path, request: dict, url: str) -> Optional[DownloadResult]:
    """Only reuse files whose provenance and request were recorded by this downloader."""
    try:
        manifest = json.loads(_cache_manifest(path).read_text(encoding="utf-8"))
        if manifest["version"] != 1 or manifest["request"] != request:
            return None
        result = DownloadResult.model_validate(manifest["result"])
        if not result.out_path.is_file() or result.out_path.stat().st_size == 0:
            return None
        return result.model_copy(update={"url": url})
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cached_result(path: Path, request: dict, result: DownloadResult) -> None:
    manifest = _cache_manifest(path)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": 1,
                "request": request,
                "result": result.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    temporary.replace(manifest)


def _probe_clip(path: Path) -> dict:
    """Read the produced media rather than reporting the source VOD's duration."""
    from fractions import Fraction

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    duration = float(metadata["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Trimmed video has no valid duration")
    rate = stream.get("avg_frame_rate", "0/1")
    fps = float(Fraction(rate)) if rate != "0/0" else None
    return {"duration": duration, "height": stream.get("height"), "fps": fps}


def download_youtube(
    url: str,
    out_dir: str | Path,
    *,
    prefer_height: int = 1080,
    prefer_fps: int = 60,
    prefer_ext: str = "mp4",
    overwrite: bool = False,
    rate_limit: Optional[str] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> DownloadResult:
    """Download a full VOD or a separately cached, stream-copied segment.

    A timestamped URL defaults to a 90-minute segment. Duration alone clips from
    zero. Full sources remain intact and can serve multiple segment requests.
    Cache identity includes video ID, format preferences, start and duration;
    legacy files without provenance are not reused because they may be trimmed.
    Segment metadata records the requested source offset and measured duration.
    Stream copy retains ffmpeg's keyframe-dependent seek precision.
    """
    if start_time is None:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        fragment = urllib.parse.parse_qs(parsed.fragment)
        value = query.get("t", fragment.get("t", [None]))[0]
        if value is not None:
            start_time = _parse_timestamp(value)
            if start_time is None:
                raise ValueError(f"Invalid URL timestamp: {value}")
    if duration is not None and start_time is None:
        start_time = 0.0
    if start_time is not None:
        start_time = float(start_time)
        duration = float(duration) if duration is not None else 5400.0
        if not math.isfinite(start_time) or start_time < 0:
            raise ValueError("start_time must be finite and nonnegative")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ID-only names remain stable when titles change and use yt-dlp sanitization.
    outtmpl = str(out_dir / f"%(id)s-h{prefer_height}-f{prefer_fps}-full.%(ext)s")
    fmt = (
        f"bestvideo[height<={prefer_height}][fps<={prefer_fps}][ext={prefer_ext}]"
        "+bestaudio[ext=m4a]/"
        f"best[height<={prefer_height}][fps<={prefer_fps}][ext={prefer_ext}]/best"
    )
    hooks = [_TqdmProgressHook()]
    if on_progress:
        hooks.append(on_progress)
    options = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "merge_output_format": prefer_ext,
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": hooks,
        "ratelimit": rate_limit,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 5,
        # An unverified existing file must not be silently accepted by yt-dlp.
        "overwrites": True,
    }
    with YoutubeDL(options) as probe:
        info = probe.extract_info(url, download=False)
        source_path = Path(probe.prepare_filename({**info, "ext": prefer_ext}))
    if start_time is not None and info.get("duration") is not None:
        if start_time >= info["duration"]:
            raise ValueError("start_time must be before the end of the source video")

    source_request = {
        "id": info["id"],
        "height": prefer_height,
        "fps": prefer_fps,
        "ext": prefer_ext,
        "start": None,
        "duration": None,
    }
    clip_path = None
    clip_request = None
    if start_time is not None:
        clip_request = {**source_request, "start": start_time, "duration": duration}
        digest = hashlib.sha256(json.dumps(clip_request, sort_keys=True).encode()).hexdigest()[:16]
        clip_path = source_path.with_name(f"{source_path.stem}-segment-{digest}.{prefer_ext}")
        if not overwrite:
            cached = _read_cached_result(clip_path, clip_request, url)
            if cached is not None:
                return cached

    # Overwriting one clip doesn't require downloading its full source again.
    source = (
        None
        if overwrite and clip_path is None
        else _read_cached_result(source_path, source_request, url)
    )
    if source is None:
        with YoutubeDL(options) as downloader:
            result = downloader.extract_info(url, download=True)
            # filepath describes the postprocessed file; prepare_filename is a fallback.
            reported_path = result.get("filepath")
            if reported_path:
                final_path = Path(reported_path)
            else:
                final_path = Path(downloader.prepare_filename(result))
                if not final_path.exists():
                    final_path = final_path.with_suffix(f".{prefer_ext}")
        if not final_path.is_file():
            raise RuntimeError(f"download appeared to succeed but file not found: {final_path}")
        source = DownloadResult(
            url=url,
            out_path=final_path,
            title=result.get("title") or info.get("title"),
            id=result.get("id") or info["id"],
            height=result.get("height"),
            fps=result.get("fps"),
            duration=result.get("duration"),
        )
        _write_cached_result(source_path, source_request, source)
    if clip_path is None:
        return source

    # Publish only after trimming and probing succeed. Keep any previous good clip.
    temporary = clip_path.with_name(f"{clip_path.stem}.partial{clip_path.suffix}")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-i",
                str(source.out_path),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        media = _probe_clip(temporary)
        temporary.replace(clip_path)
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not create requested video segment: {error}") from error
    clipped = source.model_copy(
        update={
            "out_path": clip_path,
            "source_offset_sec": start_time,
            "requested_duration_sec": duration,
            **media,
        }
    )
    _write_cached_result(clip_path, clip_request, clipped)
    return clipped
