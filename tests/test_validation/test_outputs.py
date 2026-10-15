import csv
import json

import pytest

from valoscribe.orchestration.output_writer import OutputWriter
from valoscribe.validation.outputs import main, validate_outputs


def write_output(path):
    path.mkdir(exist_ok=True)
    columns = OutputWriter(path)._get_frame_state_columns()
    with (path / "frame_states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i in range(4):
            writer.writerow(
                {
                    "timestamp": i / 4,
                    "frame_number": i + 1,
                    "phase": "ACTIVE_ROUND",
                    "score_team1": 0,
                    "score_team2": 0,
                }
            )
    (path / "event_log.jsonl").write_text(
        json.dumps({"type": "spike_plant", "timestamp": 0.5}), encoding="utf-8"
    )


def test_output_validation_and_clip_mode(tmp_path):
    write_output(tmp_path)
    assert validate_outputs(tmp_path, required_types=["spike_plant"])["frames"] == 4
    with pytest.raises(ValueError, match="full-map"):
        validate_outputs(tmp_path, full_map=True)
    with pytest.raises(ValueError, match="coverage"):
        validate_outputs(tmp_path, required_types=["kill"])
    with pytest.raises(ValueError, match="frames"):
        validate_outputs(tmp_path, min_frames=5)


@pytest.mark.parametrize(
    "content", ["", "timestamp,frame_number\n", "timestamp,frame_number\nNaN,1\n"]
)
def test_bad_csv_fails(tmp_path, content):
    write_output(tmp_path)
    (tmp_path / "frame_states.csv").write_text(content)
    assert main([str(tmp_path)]) == 1


def test_empty_events_and_out_of_bounds_fail(tmp_path):
    write_output(tmp_path)
    (tmp_path / "event_log.jsonl").write_text("")
    assert main([str(tmp_path)]) == 1
    (tmp_path / "event_log.jsonl").write_text('{"type":"kill","timestamp":99}')
    assert main([str(tmp_path)]) == 1


def test_bad_frame_order_fails(tmp_path):
    write_output(tmp_path)
    path = tmp_path / "frame_states.csv"
    rows = path.read_text().splitlines()
    path.write_text("\n".join([rows[0], rows[2], rows[1]]) + "\n")
    assert main([str(tmp_path)]) == 1


def test_full_map_score_and_lifecycle(tmp_path):
    write_output(tmp_path)
    events = [
        {"type": "match_start", "timestamp": 0, "team1": "NRG", "team2": "FNATIC"},
        {
            "type": "round_start",
            "timestamp": 0,
            "round_number": 16,
            "score_team1": 12,
            "score_team2": 3,
        },
        {
            "type": "round_end",
            "timestamp": 0.5,
            "round_number": 16,
            "score_team1": 13,
            "score_team2": 3,
        },
        {"type": "match_end", "timestamp": 0.5},
    ]
    path = tmp_path / "event_log.jsonl"
    path.write_text(json.dumps(events))
    assert validate_outputs(tmp_path, full_map=True)["passed"]
    events[2]["score_team1"] = 12
    path.write_text(json.dumps(events))
    assert main([str(tmp_path), "--full-map"]) == 1
