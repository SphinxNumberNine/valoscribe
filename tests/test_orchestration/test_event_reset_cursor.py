from valoscribe.orchestration.event_collector import EventCollector


def test_reset_restarts_incremental_event_delivery():
    collector = EventCollector()
    collector.add_event("round_start", 0)
    collector.add_event("round_end", 10)
    assert len(collector.get_events_since_last_write()) == 2
    assert collector.get_events_since_last_write() == []

    collector.reset()
    collector.add_event("round_start", 20)
    assert collector.get_events_since_last_write() == [{"type": "round_start", "timestamp": 20}]
    assert collector.get_events_since_last_write() == []
    collector.reset()
    assert collector.get_events_since_last_write() == []
