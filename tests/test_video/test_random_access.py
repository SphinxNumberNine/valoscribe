"""Random frame inspection must leave sequential processing unchanged."""

from unittest.mock import Mock

import numpy as np
import pytest

from valoscribe.video.reader import VideoReader


@pytest.fixture
def source():
    source = Mock()
    source.position = 0
    source.get_fps.return_value = 30.0
    source.get_frame_count.return_value = 300
    source.get_width.return_value = 2
    source.get_height.return_value = 2
    source.get_position.side_effect = lambda: source.position

    def set_position(position):
        source.position = position
        return True

    def read_frame():
        frame = np.full((2, 2, 3), source.position, dtype=np.uint8)
        source.position += 1
        return True, frame

    source.set_position.side_effect = set_position
    source.read_frame.side_effect = read_frame
    return source


def test_inspection_does_not_advance_iterator(source):
    with VideoReader(source) as reader:
        assert next(reader).frame_number == 0
        inspected = reader.get_frame_at(2.0)
        assert inspected.frame_number == 60
        assert inspected.timestamp_sec == 2.0
        assert inspected.timestamp_ms == 2000.0
        assert np.all(inspected.frame == 60)
        following = next(reader)
        assert following.frame_number == 1
        assert np.all(following.frame == 1)


def test_inspection_reports_actual_frame_timestamp(source):
    with VideoReader(source) as reader:
        inspected = reader.get_frame_at(2.01)
    assert inspected.frame_number == 60
    assert inspected.timestamp_sec == 2.0


def test_inspection_reports_imprecise_seek_position(source):
    def seek_early(position):
        source.position = max(0, position - 1)
        return True

    source.set_position.side_effect = seek_early
    with VideoReader(source) as reader:
        inspected = reader.get_frame_at(2.0)
    assert inspected.frame_number == 59
    assert inspected.timestamp_sec == 59 / 30


@pytest.mark.parametrize("failure", ["read", "exception", "seek"])
def test_failed_inspection_restores_iterator(source, failure):
    with VideoReader(source) as reader:
        next(reader)
        original_read = source.read_frame.side_effect
        original_seek = source.set_position.side_effect
        if failure == "read":
            source.read_frame.side_effect = lambda: (False, None)
        elif failure == "exception":
            source.read_frame.side_effect = RuntimeError("decode failed")
        else:
            source.set_position.side_effect = lambda position: (
                False if position == 60 else original_seek(position)
            )
        if failure == "exception":
            with pytest.raises(RuntimeError, match="decode failed"):
                reader.get_frame_at(2.0)
        else:
            assert reader.get_frame_at(2.0) is None
        assert source.position == 1
        assert reader._current_frame == 1
        source.read_frame.side_effect = original_read
        following = next(reader)
        assert following.frame_number == 1
        assert np.all(following.frame == 1)
