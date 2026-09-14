from botcolosseo.demo.hierarchical_recording import viewer_event_label
from botcolosseo.envs.extraction_protocol import ExtractionEvent, ExtractionEventType


def event(kind, side):
    return ExtractionEvent(kind, side, 1, 10, 0, 1, 4)


def test_own_event_identity_and_ambiguous_loot_value():
    events = (event(ExtractionEventType.LOOT_PICKUP, "host"),
              event(ExtractionEventType.LOOT_PICKUP, "opponent"))
    assert viewer_event_label(events, side="host") == "LOOT PICKED UP"
    assert viewer_event_label(events[1:], side="host") == ""


def test_combat_events_keep_first_person_meaning():
    events = (event(ExtractionEventType.VALID_HIT, "opponent"),
              event(ExtractionEventType.DEATH, "opponent"))
    assert viewer_event_label(events, side="host") == "TAKING FIRE | ENEMY DOWN - CACHE"
