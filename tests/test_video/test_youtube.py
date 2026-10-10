"""Unit tests for YouTube downloader module."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from valoscribe.video.youtube import download_youtube, _TqdmProgressHook
from valoscribe.types.video import DownloadResult


class TestDownloadYoutube:
    """Tests for download_youtube function."""

    @pytest.fixture
    def mock_video_info(self):
        """Sample video metadata."""
        return {
            "title": "VCT Champions 2024 Grand Finals",
            "id": "abc123",
            "height": 1080,
            "fps": 60.0,
            "duration": 3600.0,
        }

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Temporary output directory for tests."""
        return tmp_path / "videos"

    @patch("valoscribe.video.youtube.YoutubeDL")
    def test_successful_download(self, mock_ydl_class, mock_video_info, temp_output_dir):
        """Test successful video download with default parameters."""
        # Setup mocks
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info

        # Mock the final file path
        expected_filename = f"{mock_video_info['title']}-{mock_video_info['id']}.mp4"
        expected_path = temp_output_dir / expected_filename
        mock_ydl.prepare_filename.return_value = str(expected_path)

        # Create the file so it "exists" after download
        temp_output_dir.mkdir(parents=True)
        expected_path.touch()

        # Execute
        result = download_youtube(
            url="https://youtube.com/watch?v=abc123",
            out_dir=temp_output_dir,
        )

        # Assert
        assert isinstance(result, DownloadResult)
        assert result.url == "https://youtube.com/watch?v=abc123"
        assert result.out_path == expected_path
        assert result.title == mock_video_info["title"]
        assert result.id == mock_video_info["id"]
        assert result.height == 1080
        assert result.fps == 60.0
        assert result.duration == 3600.0

    @patch("valoscribe.video.youtube.YoutubeDL")
    def test_reuse_existing_file(self, mock_ydl_class, mock_video_info, temp_output_dir):
        """Only a recorded full download can be reused as a full source."""
        temp_output_dir.mkdir(parents=True)
        existing_file = temp_output_dir / "abc123-h1080-f60-full.mp4"
        existing_file.write_bytes(b"video")
        downloader = MagicMock()
        downloader.extract_info.return_value = mock_video_info
        downloader.prepare_filename.return_value = str(existing_file)
        mock_ydl_class.return_value.__enter__.return_value = downloader
        first = download_youtube("https://youtube.com/watch?v=abc123", temp_output_dir)
        downloader.extract_info.reset_mock()
        second = download_youtube("https://youtube.com/watch?v=abc123", temp_output_dir)
        assert second == first
        downloader.extract_info.assert_called_once_with(first.url, download=False)

    @patch("valoscribe.video.youtube.YoutubeDL")
    def test_overwrite_existing_file(self, mock_ydl_class, mock_video_info, temp_output_dir):
        """Test that existing file is overwritten when overwrite=True."""
        # Setup: create existing file
        temp_output_dir.mkdir(parents=True)
        existing_file = temp_output_dir / f"{mock_video_info['title']}-{mock_video_info['id']}.mp4"
        existing_file.write_text("old content")

        # Mock both probe and download YoutubeDL instances
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info
        mock_ydl.prepare_filename.return_value = str(existing_file)

        # Execute
        result = download_youtube(
            url="https://youtube.com/watch?v=abc123",
            out_dir=temp_output_dir,
            overwrite=True,
        )

        # Assert: download was called even though file existed
        # extract_info called twice: probe + download
        assert mock_ydl.extract_info.call_count == 2
        # Second call should have download=True
        assert mock_ydl.extract_info.call_args_list[1][1]["download"] is True

    @patch("valoscribe.video.youtube.YoutubeDL")
    def test_custom_parameters(self, mock_ydl_class, mock_video_info, temp_output_dir):
        """Test download with custom height, fps, and extension."""
        # Setup mocks
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info

        # Use mkv extension to match prefer_ext parameter
        expected_path = temp_output_dir / f"{mock_video_info['title']}-{mock_video_info['id']}.mkv"
        mock_ydl.prepare_filename.return_value = str(expected_path)

        temp_output_dir.mkdir(parents=True)
        expected_path.touch()  # Create with correct extension

        # Execute with custom parameters
        result = download_youtube(
            url="https://youtube.com/watch?v=abc123",
            out_dir=temp_output_dir,
            prefer_height=720,
            prefer_fps=30,
            prefer_ext="mkv",
            overwrite=True,  # Force download to test ydl_opts
        )

        # Assert: Check that YoutubeDL was configured with correct format string
        # YoutubeDL is called twice: once for probe, once for download
        # Download call has 'format' key in options
        assert len(mock_ydl_class.call_args_list) >= 2, "Expected at least 2 YoutubeDL instantiations"

        # Find the call with 'format' in the options dict
        download_call_opts = None
        for call in mock_ydl_class.call_args_list:
            # call.args[0] is the first positional argument (the options dict)
            if call.args and isinstance(call.args[0], dict) and "format" in call.args[0]:
                download_call_opts = call.args[0]
                break

        assert download_call_opts is not None, "Could not find download call with format"
        format_str = download_call_opts["format"]
        assert "height<=720" in format_str
        assert "fps<=30" in format_str
        assert "ext=mkv" in format_str

    @patch("valoscribe.video.youtube.YoutubeDL")
    def test_download_with_progress_callback(self, mock_ydl_class, mock_video_info, temp_output_dir):
        """Test that custom progress callback is registered."""
        # Setup
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info

        expected_path = temp_output_dir / f"{mock_video_info['title']}-{mock_video_info['id']}.mp4"
        mock_ydl.prepare_filename.return_value = str(expected_path)

        temp_output_dir.mkdir(parents=True)
        expected_path.touch()

        custom_callback = Mock()

        # Execute
        result = download_youtube(
            url="https://youtube.com/watch?v=abc123",
            out_dir=temp_output_dir,
            on_progress=custom_callback,
            overwrite=True,  # Force download to test progress hooks
        )

        # Assert: Check that progress_hooks includes both default and custom
        # Find the download call (has progress_hooks)
        download_call_opts = None
        for call in mock_ydl_class.call_args_list:
            if call.args and isinstance(call.args[0], dict) and "progress_hooks" in call.args[0]:
                download_call_opts = call.args[0]
                break

        assert download_call_opts is not None, "Could not find download call with progress_hooks"
        progress_hooks = download_call_opts["progress_hooks"]
        assert len(progress_hooks) == 2  # _TqdmProgressHook + custom callback

    @patch("valoscribe.video.youtube.YoutubeDL")
    def test_missing_file_after_download_raises_error(self, mock_ydl_class, mock_video_info, temp_output_dir):
        """Test that RuntimeError is raised if downloaded file doesn't exist."""
        # Setup: Don't create the file after download
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info

        nonexistent_path = temp_output_dir / f"{mock_video_info['title']}-{mock_video_info['id']}.mp4"
        mock_ydl.prepare_filename.return_value = str(nonexistent_path)

        temp_output_dir.mkdir(parents=True)
        # Deliberately don't create the file

        # Execute & Assert
        with pytest.raises(RuntimeError, match="download appeared to succeed but file not found"):
            download_youtube(
                url="https://youtube.com/watch?v=abc123",
                out_dir=temp_output_dir,
            )

    @patch("valoscribe.video.youtube.YoutubeDL")
    def test_output_directory_created_if_missing(self, mock_ydl_class, mock_video_info, temp_output_dir):
        """Test that output directory is created if it doesn't exist."""
        # Setup
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = mock_video_info

        expected_path = temp_output_dir / f"{mock_video_info['title']}-{mock_video_info['id']}.mp4"
        mock_ydl.prepare_filename.return_value = str(expected_path)

        # Don't create directory beforehand
        assert not temp_output_dir.exists()

        expected_path.parent.mkdir(parents=True)
        expected_path.touch()

        # Execute
        result = download_youtube(
            url="https://youtube.com/watch?v=abc123",
            out_dir=temp_output_dir,
        )

        # Assert
        assert temp_output_dir.exists()
        assert result.out_path.exists()


class TestTqdmProgressHook:
    """Tests for _TqdmProgressHook class."""

    def test_progress_hook_initialization(self):
        """Test that progress hook initializes with no progress bar."""
        hook = _TqdmProgressHook()
        assert hook.pbar is None

    @patch("valoscribe.video.youtube.tqdm")
    def test_progress_hook_creates_bar_on_download(self, mock_tqdm_class):
        """Test that progress bar is created when download starts."""
        hook = _TqdmProgressHook()
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        # Simulate download progress
        hook({
            "status": "downloading",
            "total_bytes": 1000000,
            "downloaded_bytes": 250000,
        })

        # Assert: tqdm was initialized
        mock_tqdm_class.assert_called_once()
        assert mock_pbar.n == 250000
        mock_pbar.refresh.assert_called_once()

    @patch("valoscribe.video.youtube.tqdm")
    def test_progress_hook_updates_existing_bar(self, mock_tqdm_class):
        """Test that progress bar updates on subsequent calls."""
        hook = _TqdmProgressHook()
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        # First call - creates bar
        hook({
            "status": "downloading",
            "total_bytes": 1000000,
            "downloaded_bytes": 250000,
        })

        # Second call - updates bar
        hook({
            "status": "downloading",
            "total_bytes": 1000000,
            "downloaded_bytes": 500000,
        })

        # Assert: Only created once, updated twice
        assert mock_tqdm_class.call_count == 1
        assert mock_pbar.n == 500000
        assert mock_pbar.refresh.call_count == 2

    @patch("valoscribe.video.youtube.tqdm")
    def test_progress_hook_closes_on_finish(self, mock_tqdm_class):
        """Test that progress bar closes when download finishes."""
        hook = _TqdmProgressHook()
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        # Start download
        hook({
            "status": "downloading",
            "total_bytes": 1000000,
            "downloaded_bytes": 500000,
        })

        # Finish download
        hook({"status": "finished"})

        # Assert: Progress bar was closed
        mock_pbar.close.assert_called_once()
        assert hook.pbar is None

    def test_progress_hook_handles_missing_total_bytes(self):
        """Test that hook handles cases where total_bytes is unavailable."""
        hook = _TqdmProgressHook()

        # Call with no total_bytes
        hook({
            "status": "downloading",
            "downloaded_bytes": 250000,
        })

        # Assert: No progress bar created without total
        assert hook.pbar is None

    @patch("valoscribe.video.youtube.tqdm")
    def test_progress_hook_handles_total_bytes_estimate(self, mock_tqdm_class):
        """Test that hook uses total_bytes_estimate if total_bytes unavailable."""
        hook = _TqdmProgressHook()
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        # Call with total_bytes_estimate instead of total_bytes
        hook({
            "status": "downloading",
            "total_bytes_estimate": 1000000,
            "downloaded_bytes": 250000,
        })

        # Assert: Progress bar created with estimate
        mock_tqdm_class.assert_called_once()
        call_kwargs = mock_tqdm_class.call_args[1]
        assert call_kwargs["total"] == 1000000
