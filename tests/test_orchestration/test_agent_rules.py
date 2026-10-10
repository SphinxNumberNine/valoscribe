"""Agent rules are session-specific and respect preround purchases."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from valoscribe.orchestration.game_state_manager import GameStateManager
from valoscribe.orchestration.player_state_tracker import PlayerStateTracker
from valoscribe.orchestration.state_validator import StateValidator
from valoscribe.types.detections import AbilityInfo, AgentInfo


def make_tracker(**kwargs):
    return PlayerStateTracker(0, {"name": "player", "team": "team", "agent": "sova"}, **kwargs)


def charges(value):
    return {"ability_1": AbilityInfo(charges=value, total_blobs_detected=value)}


def test_default_rules_match_validator_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = make_tracker()
    validator = StateValidator()
    assert tracker.agent_config == validator.agent_config
    assert tracker.agent_config["sova"]["ability_1"]["max_charges"] == 1


def test_configurations_do_not_leak_between_trackers(tmp_path):
    configs = []
    for maximum in (1, 2):
        path = tmp_path / f"agents-{maximum}.json"
        path.write_text(json.dumps({"sova": {"ability_1": {
            "max_charges": maximum, "rechargeable": False,
        }}}))
        configs.append(path)
    first = make_tracker(config_path=configs[0])
    second = make_tracker(config_path=configs[1])
    first.update(charges(2), 0)
    second.update(charges(2), 0)
    assert first.current_state["ability_1"] is None
    assert second.current_state["ability_1"] == 2


def test_injected_rules_are_shared_without_reloading():
    rules = StateValidator().agent_config
    tracker = make_tracker(agent_config=rules)
    assert tracker.agent_config is rules


def test_missing_explicit_configuration_fails_instead_of_disabling_rules(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_tracker(config_path=tmp_path / "missing.json")


def test_charge_limits_apply_in_active_round():
    tracker = make_tracker()
    tracker.update(charges(1), 0)
    tracker.update(charges(2), 1)
    assert tracker.current_state["ability_1"] == 1


def test_nonrechargeable_charges_do_not_increase_in_active_round():
    tracker = make_tracker()
    tracker.update(charges(0), 0)
    tracker.update(charges(1), 1)
    assert tracker.current_state["ability_1"] == 0


def test_preround_allows_purchase_after_refund_but_keeps_maximum():
    tracker = make_tracker()
    tracker.update(charges(1), 0, allow_charge_increase=True)
    tracker.update(charges(0), 1, allow_charge_increase=True)
    tracker.update(charges(1), 2, allow_charge_increase=True)
    assert tracker.current_state["ability_1"] == 1
    tracker.update(charges(2), 3, allow_charge_increase=True)
    assert tracker.current_state["ability_1"] == 1


def test_rechargeable_ability_can_gain_charges_in_active_round():
    tracker = make_tracker()
    tracker.update({"ability_3": AbilityInfo(charges=0, total_blobs_detected=0)}, 0)
    tracker.update({"ability_3": AbilityInfo(charges=1, total_blobs_detected=1)}, 1)
    assert tracker.current_state["ability_3"] == 1


def test_manager_shares_rules_and_allows_preround_purchases(tmp_path):
    agents = ["sova", "sage", "omen", "jett", "raze"] * 2
    metadata = {
        "teams": [
            {"name": "left", "starting_side": "attack"},
            {"name": "right", "starting_side": "defense"},
        ],
        "players": [{
            "name": f"player{index}",
            "team": "left" if index < 5 else "right",
            "agent": agent,
        } for index, agent in enumerate(agents)],
    }
    manager = GameStateManager(tmp_path / "video.mp4", metadata, tmp_path / "output")
    sides = ["attack"] * 5 + ["defense"] * 5
    manager._initialize_player_trackers_from_agents(agents, sides)
    for tracker in manager.player_trackers:
        assert tracker.agent_config is manager.state_validator.agent_config
        tracker.current_state["ability_1"] = 0
    registry = manager.detector_registry
    registry.preround_agent_detector = Mock()
    registry.preround_agent_detector.detect.side_effect = lambda frame, index: AgentInfo(
        agent_name=agents[index], side=sides[index], confidence=1.0
    )
    registry.preround_ability_detector = Mock()
    registry.preround_ability_detector.detect_player_abilities.return_value = charges(1)
    registry.preround_ultimate_detector = Mock()
    registry.preround_ultimate_detector.detect_ultimate.return_value = None
    registry.score_detector = Mock()
    registry.score_detector.detect.return_value = None
    manager._process_preround(1.0, np.zeros((2, 2, 3), dtype=np.uint8))
    assert manager.player_trackers[0].current_state["ability_1"] == 1
    assert not manager.event_collector.get_all_events()
