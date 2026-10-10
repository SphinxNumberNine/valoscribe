"""Unit tests for OutputWriter."""

from __future__ import annotations
import pytest
import csv
import json
from pathlib import Path
import tempfile
import shutil

from valoscribe.orchestration.output_writer import OutputWriter
from valoscribe.orchestration.phase_detector import Phase


class TestOutputWriter:
    """Tests for OutputWriter class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def writer(self, temp_dir):
        """Create OutputWriter instance."""
        writer = OutputWriter(output_dir=temp_dir)
        yield writer
        writer.close()

    @pytest.fixture
    def sample_player_states(self):
        """Create sample player states."""
        return [
            {
                "metadata": {"name": f"player{i}", "team": "team1" if i < 5 else "team2", "agent": "sova"},
                "current_state": {
                    "alive": True,
                    "health": 100,
                    "armor": 50,
                    "ability_1": 2,
                    "ability_2": 1,
                    "ability_3": 1,
                    "ultimate": {"charges": 5, "is_full": False},
                },
            }
            for i in range(10)
        ]

    def test_init(self, temp_dir):
        """Test OutputWriter initialization."""
        writer = OutputWriter(output_dir=temp_dir)

        assert writer.output_dir == temp_dir
        assert writer.frame_states_path == temp_dir / "frame_states.csv"
        assert writer.event_log_path == temp_dir / "event_log.jsonl"
        assert not writer.frame_states_initialized
        assert not writer.event_log_initialized

    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates output directory."""
        nested_dir = temp_dir / "nested" / "output"
        writer = OutputWriter(output_dir=nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_write_frame_state_creates_file(self, writer, sample_player_states, temp_dir):
        """Test that writing frame state creates CSV file."""
        writer.write_frame_state(
            timestamp=0.0,
            frame_number=1,
            phase=Phase.ACTIVE_ROUND,
            round_number=1,
            scores=(0, 0),
            player_states=sample_player_states,
        )

        assert (temp_dir / "frame_states.csv").exists()
        assert writer.frame_states_initialized

    def test_write_frame_state_headers(self, writer, sample_player_states, temp_dir):
        """Test that frame_states.csv has correct headers."""
        writer.write_frame_state(
            timestamp=0.0,
            frame_number=1,
            phase=Phase.ACTIVE_ROUND,
            round_number=1,
            scores=(0, 0),
            player_states=sample_player_states,
        )
        writer.close()

        # Read and check headers
        with open(temp_dir / "frame_states.csv", "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Check basic columns
            assert "timestamp" in headers
            assert "frame_number" in headers
            assert "phase" in headers
            assert "round_number" in headers
            assert "score_team1" in headers
            assert "score_team2" in headers

            # Check player columns for first player
            assert "player_0_name" in headers
            assert "player_0_team" in headers
            assert "player_0_agent" in headers
            assert "player_0_alive" in headers
            assert "player_0_health" in headers
            assert "player_0_armor" in headers
            assert "player_0_ability_1" in headers
            assert "player_0_ability_2" in headers
            assert "player_0_ability_3" in headers
            assert "player_0_ultimate_charges" in headers
            assert "player_0_ultimate_full" in headers

            # Check last player exists
            assert "player_9_name" in headers

    def test_write_frame_state_data(self, writer, sample_player_states, temp_dir):
        """Test that frame state data is written correctly."""
        writer.write_frame_state(
            timestamp=10.5,
            frame_number=42,
            phase=Phase.ACTIVE_ROUND,
            round_number=3,
            scores=(5, 4),
            player_states=sample_player_states,
        )
        writer.close()

        # Read and verify data
        with open(temp_dir / "frame_states.csv", "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 1
            row = rows[0]

            assert row["timestamp"] == "10.500"
            assert row["frame_number"] == "42"
            assert row["phase"] == "ACTIVE_ROUND"
            assert row["round_number"] == "3"
            assert row["score_team1"] == "5"
            assert row["score_team2"] == "4"

            # Check player 0 data
            assert row["player_0_name"] == "player0"
            assert row["player_0_team"] == "team1"
            assert row["player_0_agent"] == "sova"
            assert row["player_0_alive"] == "True"
            assert row["player_0_health"] == "100"
            assert row["player_0_armor"] == "50"
            assert row["player_0_ability_1"] == "2"
            assert row["player_0_ultimate_charges"] == "5"
            assert row["player_0_ultimate_full"] == "False"

    def test_write_multiple_frame_states(self, writer, sample_player_states, temp_dir):
        """Test writing multiple frame states."""
        for i in range(5):
            writer.write_frame_state(
                timestamp=i * 0.25,
                frame_number=i + 1,
                phase=Phase.ACTIVE_ROUND,
                round_number=1,
                scores=(0, 0),
                player_states=sample_player_states,
            )
        writer.close()

        # Read and verify
        with open(temp_dir / "frame_states.csv", "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 5
            assert rows[0]["frame_number"] == "1"
            assert rows[4]["frame_number"] == "5"

    def test_write_event_creates_file(self, writer, temp_dir):
        """Test that writing event creates JSONL file."""
        event = {"type": "death", "timestamp": 10.0, "player": "brawk"}
        writer.write_event(event)

        assert (temp_dir / "event_log.jsonl").exists()
        assert writer.event_log_initialized

    def test_write_event_format(self, writer, temp_dir):
        """Test that event is written in correct JSONL format."""
        event = {"type": "death", "timestamp": 10.0, "player": "brawk"}
        writer.write_event(event)
        writer.close()

        # Read JSONL file
        with open(temp_dir / "event_log.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 1

            # Parse JSON
            parsed_event = json.loads(lines[0])
            assert parsed_event["type"] == "death"
            assert parsed_event["timestamp"] == 10.0
            assert parsed_event["player"] == "brawk"

    def test_write_event_data(self, writer, temp_dir):
        """Test that event data is written correctly."""
        event = {
            "type": "death",
            "timestamp": 10.5,
            "player": "brawk",
            "team": "NRG",
            "agent": "sova",
        }
        writer.write_event(event)
        writer.close()

        with open(temp_dir / "event_log.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 1

            parsed_event = json.loads(lines[0])
            assert parsed_event["timestamp"] == 10.5
            assert parsed_event["type"] == "death"
            assert parsed_event["player"] == "brawk"
            assert parsed_event["team"] == "NRG"
            assert parsed_event["agent"] == "sova"

    def test_write_multiple_events(self, writer, temp_dir):
        """Test writing multiple events."""
        events = [
            {"type": "death", "timestamp": 10.0, "player": "player1"},
            {"type": "kill", "timestamp": 10.0, "killer": "player2"},
            {"type": "ability_used", "timestamp": 12.0, "player": "player3"},
        ]

        writer.write_events(events)
        writer.close()

        with open(temp_dir / "event_log.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 3

            event1 = json.loads(lines[0])
            event2 = json.loads(lines[1])
            event3 = json.loads(lines[2])

            assert event1["type"] == "death"
            assert event2["type"] == "kill"
            assert event3["type"] == "ability_used"

    def test_write_event_with_complex_data(self, writer, temp_dir):
        """Test writing event with complex nested data."""
        event = {
            "type": "kill",
            "timestamp": 15.0,
            "killer": "jett",
            "victim": "sova",
            "weapon": {"name": "vandal", "headshot": True},
            "position": {"x": 100, "y": 200},
        }
        writer.write_event(event)
        writer.close()

        with open(temp_dir / "event_log.jsonl", "r") as f:
            lines = f.readlines()
            parsed_event = json.loads(lines[0])

            assert parsed_event["killer"] == "jett"
            assert parsed_event["weapon"]["name"] == "vandal"
            assert parsed_event["weapon"]["headshot"] is True
            assert parsed_event["position"]["x"] == 100
            assert parsed_event["position"]["y"] == 200

    def test_context_manager(self, temp_dir, sample_player_states):
        """Test using OutputWriter as context manager."""
        with OutputWriter(output_dir=temp_dir) as writer:
            writer.write_frame_state(
                timestamp=0.0,
                frame_number=1,
                phase=Phase.ACTIVE_ROUND,
                round_number=1,
                scores=(0, 0),
                player_states=sample_player_states,
            )
            writer.write_event({"type": "death", "timestamp": 10.0})

        # Files should be closed after context
        assert (temp_dir / "frame_states.csv").exists()
        assert (temp_dir / "event_log.jsonl").exists()

        # Should be readable
        with open(temp_dir / "frame_states.csv", "r") as f:
            reader = csv.DictReader(f)
            assert len(list(reader)) == 1

    def test_flush(self, writer, sample_player_states):
        """Test flushing buffers."""
        writer.write_frame_state(
            timestamp=0.0,
            frame_number=1,
            phase=Phase.ACTIVE_ROUND,
            round_number=1,
            scores=(0, 0),
            player_states=sample_player_states,
        )
        writer.write_event({"type": "death", "timestamp": 10.0})

        # Should not raise
        writer.flush()

    def test_close(self, writer, sample_player_states):
        """Test closing files."""
        writer.write_frame_state(
            timestamp=0.0,
            frame_number=1,
            phase=Phase.ACTIVE_ROUND,
            round_number=1,
            scores=(0, 0),
            player_states=sample_player_states,
        )

        writer.close()

        # File handles should be None
        assert writer.frame_states_file is None
        assert writer.event_log_file is None

    def test_repr(self, temp_dir):
        """Test string representation."""
        writer = OutputWriter(output_dir=temp_dir)
        repr_str = repr(writer)

        assert "OutputWriter" in repr_str
        assert str(temp_dir) in repr_str

    def test_frame_state_with_none_values(self, writer, temp_dir):
        """Test frame state with None values."""
        player_states = [
            {
                "metadata": {"name": f"player{i}", "team": "team1", "agent": "sova"},
                "current_state": {
                    "alive": True,
                    "health": None,  # Not detected yet
                    "armor": None,
                    "ability_1": None,
                    "ability_2": None,
                    "ability_3": None,
                    "ultimate": None,
                },
            }
            for i in range(10)
        ]

        writer.write_frame_state(
            timestamp=0.0,
            frame_number=1,
            phase=Phase.PREROUND,
            round_number=None,  # Round not started
            scores=(0, 0),
            player_states=player_states,
        )
        writer.close()

        # Should write empty strings for None values
        with open(temp_dir / "frame_states.csv", "r") as f:
            reader = csv.DictReader(f)
            row = list(reader)[0]

            assert row["round_number"] == ""
            assert row["player_0_health"] == ""
            assert row["player_0_armor"] == ""
            assert row["player_0_ultimate_charges"] == ""

    def test_realistic_scenario(self, temp_dir):
        """Test realistic scenario with mixed writes."""
        with OutputWriter(output_dir=temp_dir) as writer:
            # Write multiple frames
            for frame_num in range(1, 6):
                player_states = [
                    {
                        "metadata": {"name": f"player{i}", "team": "team1" if i < 5 else "team2", "agent": "sova"},
                        "current_state": {
                            "alive": True,
                            "health": 100 - frame_num * 10,
                            "armor": 50,
                            "ability_1": 2,
                            "ability_2": 1,
                            "ability_3": 1,
                            "ultimate": {"charges": frame_num, "is_full": False},
                        },
                    }
                    for i in range(10)
                ]

                writer.write_frame_state(
                    timestamp=frame_num * 0.25,
                    frame_number=frame_num,
                    phase=Phase.ACTIVE_ROUND,
                    round_number=1,
                    scores=(0, 0),
                    player_states=player_states,
                )

            # Write events
            events = [
                {"type": "death", "timestamp": 1.0, "player": "player0"},
                {"type": "kill", "timestamp": 1.0, "killer": "player5"},
                {"type": "round_end", "timestamp": 100.0, "winner": "team2"},
            ]
            writer.write_events(events)

        # Verify frame states
        with open(temp_dir / "frame_states.csv", "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert rows[0]["player_0_health"] == "90"
            assert rows[4]["player_0_health"] == "50"

        # Verify events
        with open(temp_dir / "event_log.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 3

            event1 = json.loads(lines[0])
            event3 = json.loads(lines[2])

            assert event1["type"] == "death"
            assert event3["type"] == "round_end"
