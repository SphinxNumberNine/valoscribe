from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from valoscribe.commands.orchestrate import app
from valoscribe.orchestration.game_state_manager import GameStateManager


@pytest.mark.parametrize("failure", [None, StopIteration, RuntimeError, KeyboardInterrupt])
def test_process_video_closes_reader_and_writer(failure):
    manager = GameStateManager.__new__(GameStateManager)
    manager.video_path = Path("test.mp4")
    manager.fps = 4
    manager.frame_count = 0
    manager.output_writer = MagicMock()
    manager.process_frame = Mock(side_effect=failure)
    manager._finalize_events = Mock()
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.__iter__.return_value = iter([SimpleNamespace(timestamp_sec=1, frame=None)])

    with patch("valoscribe.orchestration.game_state_manager.VideoReader", return_value=reader):
        if failure in (RuntimeError, KeyboardInterrupt):
            with pytest.raises(failure):
                manager.process_video()
        else:
            manager.process_video()

    reader.__exit__.assert_called_once()
    manager.output_writer.__exit__.assert_called_once()
    assert manager._finalize_events.call_count == (
        0 if failure in (RuntimeError, KeyboardInterrupt) else 1
    )


@pytest.mark.parametrize("failure", [None, StopIteration, RuntimeError, KeyboardInterrupt])
def test_cli_closes_reader_writer_and_windows(tmp_path, failure):
    video = tmp_path / "test.mp4"
    video.touch()
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"teams": [], "players": []}')
    manager = MagicMock()
    manager.frame_count = 1
    manager.event_collector.get_all_events.return_value = []
    manager.event_collector.get_event_summary.return_value = {}
    manager.process_frame.side_effect = failure
    reader = MagicMock()
    reader.__enter__.return_value = reader
    reader.__iter__.return_value = iter([SimpleNamespace(timestamp_sec=1, frame=None)])
    module = "valoscribe.commands.orchestrate."
    with (
        patch(module + "GameStateManager", return_value=manager),
        patch(module + "VideoReader", return_value=reader),
        patch(module + "cv2.namedWindow"),
        patch(module + "cv2.resizeWindow"),
        patch(module + "cv2.imshow"),
        patch(module + "cv2.waitKey", return_value=ord("q")),
        patch(module + "_create_display_frame"),
        patch(module + "cv2.destroyAllWindows") as cleanup,
    ):
        result = CliRunner().invoke(app, [str(video), str(metadata), "--show", "--quiet"])
    expected_exit = (
        130 if failure is KeyboardInterrupt else (0 if failure in (None, StopIteration) else 1)
    )
    assert result.exit_code == expected_exit
    reader.__exit__.assert_called_once()
    manager.output_writer.__exit__.assert_called_once()
    cleanup.assert_called_once()
