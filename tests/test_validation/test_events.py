"""Regression policy tests, including exhaustive matching checks."""

import itertools
import json
import math

import pytest

from valoscribe.validation.events import _match, compare_events, load_events, main


def event(frame=10, **payload):
    return {
        "type": "ability_used",
        "timestamp": frame / 4,
        "player": "s0m",
        "team": "NRG",
        "agent": "omen",
        "ability": "dark cover",
        "charges_used": 1,
        "remaining_charges": 0,
        **payload,
    }


@pytest.mark.parametrize("shift,score", [(-2, 0), (-1, 100), (0, 100), (1, 100), (2, 0)])
def test_three_frame_window(shift, score):
    assert compare_events([event()], [event(10 + shift)])["similarity_percent"] == score


def test_missing_extra_duplicates_and_semantic_changes():
    reference = [event(), event(20)]
    assert compare_events(reference, reference + [event(20)])["similarity_percent"] == 80
    assert compare_events(reference, [event()])["similarity_percent"] == pytest.approx(200 / 3)
    assert compare_events([event()], [event(player="other")])["similarity_percent"] == 0
    assert compare_events([event(charges=1)], [event(charges=1.0)])["similarity_percent"] == 0
    assert compare_events([event()], [event(new_semantic_field=True)])["similarity_percent"] == 0


def test_diagnostics_do_not_change_semantics():
    report = compare_events(
        [event(confidence=0.9, timers={"game_timer": 95})],
        [event(11, confidence=0.8, timers={"game_timer": 94})],
    )
    assert report["passed"]
    assert report["matches"][0]["frame_delta"] == 1
    assert "timers" in report["matches"][0]["diagnostic_differences"]


def test_same_frame_order_is_irrelevant():
    events = [event(player="a"), event(player="b")]
    assert compare_events(events, events[::-1])["passed"]


def test_fractional_sampling_interval_and_origin():
    interval = 7 / (30000 / 1001)
    reference = [event(timestamp=2 + interval * 10)]
    candidate = [event(timestamp=2 + interval * 11)]
    assert compare_events(reference, candidate, frame_interval=interval, origin=2)["passed"]
    with pytest.raises(ValueError, match="lattice"):
        compare_events(reference, candidate)


def test_current_fractional_source_sampling_grid():
    source_fps = 30000 / 1001
    def timestamp(index):
        return math.ceil(index * source_fps / 4 - 1e-8) / source_fps
    for index in range(1, 100):
        reference = [event(timestamp=timestamp(index))]
        assert compare_events(
            reference, [event(timestamp=timestamp(index + 1))], source_fps=source_fps
        )["passed"]
        assert not compare_events(
            reference, [event(timestamp=timestamp(index + 2))], source_fps=source_fps
        )["passed"]
    with pytest.raises(ValueError, match="lattice"):
        compare_events([event(timestamp=1 / source_fps)], [event()], source_fps=source_fps)


def test_matching_score_is_symmetric():
    reference = [event(1), event(3), event(4), event(7)]
    candidate = [event(0), event(2), event(5)]
    forward = compare_events(reference, candidate)
    backward = compare_events(candidate, reference)
    assert forward["similarity_percent"] == backward["similarity_percent"]
    assert forward["matched"] == backward["matched"]


def test_required_semantic_fields_cannot_disappear():
    invalid = event()
    invalid.pop("ability")
    with pytest.raises(ValueError, match="ability"):
        compare_events([invalid], [invalid])


def test_empty_and_invalid_inputs():
    assert compare_events([event()], [])["similarity_percent"] == 0
    assert compare_events([], [event()])["similarity_percent"] == 0
    with pytest.raises(ValueError, match="empty"):
        compare_events([], [])
    for invalid in (
        event(timestamp=float("nan")),
        event(timestamp=True),
        event(processed_frame_index=-1),
        event(timers={"game_timer": -1}),
        event(confidence=2),
        event(timestamp=0.123),
    ):
        with pytest.raises(ValueError):
            compare_events([invalid], [event()])


def test_threshold_uses_unrounded_score_and_critical_types():
    reference = [event(frame) for frame in range(100)]
    candidate = reference + [event(200), event(201)]
    assert not compare_events(reference, candidate, threshold=99.01)["passed"]
    assert compare_events(reference, candidate, threshold=99)["passed"]
    critical = {
        "type": "round_start",
        "timestamp": 0,
        "round_number": 1,
        "score_team1": 0,
        "score_team2": 0,
    }
    assert not compare_events([critical] + reference, reference, threshold=0)["passed"]
    assert not compare_events(reference, reference, type_thresholds={"kill": 90})["passed"]


def test_maximum_matching_against_exhaustive_oracle():
    def oracle(left, right):
        best = (0, 0)
        for count in range(min(len(left), len(right)) + 1):
            for chosen in itertools.combinations(left, count):
                for assigned in itertools.permutations(right, count):
                    distances = [abs(a - b) for a, b in zip(chosen, assigned)]
                    if all(d <= 1 for d in distances):
                        best = max(best, (count, -sum(distances)))
        return best

    groups = [
        g for size in range(4) for g in itertools.combinations_with_replacement(range(3), size)
    ]
    for left in groups:
        for right in groups:
            pairs = _match(
                list(zip(left, range(len(left)))), list(zip(right, range(len(right)))), 1
            )
            assert (len(pairs), -sum(abs(left[i] - right[j]) for i, j in pairs)) == oracle(
                left, right
            )


def test_cli_and_json_formats(tmp_path):
    reference, candidate, report = (
        tmp_path / name for name in ("a.json", "b.jsonl", "report.json")
    )
    reference.write_text(json.dumps([event()]), encoding="utf-8")
    candidate.write_text(json.dumps(event(11)) + "\n", encoding="utf-8")
    assert load_events(reference) == [event()]
    args = [str(reference), str(candidate), "--sample-fps", "4", "--report", str(report)]
    assert main(args) == 0
    assert json.loads(report.read_text())["similarity_percent"] == 100
    candidate.write_text(json.dumps(event(12)), encoding="utf-8")
    assert main(args) == 1
    candidate.write_text("{broken", encoding="utf-8")
    assert main(args) == 2
    assert main([str(reference), str(reference), "--sample-fps", "0"]) == 2


def test_log_schema_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    for content in (
        "",
        "[]",
        "42",
        "{}",
        '{"type":"x","timestamp":NaN}',
        '{"type":"round_start","timestamp":0}',
    ):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_events(path)
