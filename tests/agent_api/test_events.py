import pytest

from evalscope.agent_api.events import (
    EventCursorAheadError,
    EventCursorExpiredError,
    JobEventStore,
)


def test_event_pages_are_stable_resumable_and_paginated(tmp_path):
    store = JobEventStore('eval_test', str(tmp_path))
    for index in range(5):
        store.append('progress.updated', f'event {index}', data={'index': index})

    first = store.page(after_cursor=0, limit=2)
    repeated = store.page(after_cursor=0, limit=2)
    second = store.page(after_cursor=first.next_cursor, limit=10)
    empty = store.page(after_cursor=second.next_cursor, limit=10)

    assert [event.sequence for event in first.events] == [1, 2]
    assert first == repeated
    assert first.next_cursor == 2
    assert first.has_more is True
    assert [event.sequence for event in second.events] == [3, 4, 5]
    assert second.has_more is False
    assert empty.events == []
    assert empty.next_cursor == 5


def test_event_store_reopens_existing_sequence(tmp_path):
    store = JobEventStore('eval_test', str(tmp_path))
    store.append('job.queued', 'queued')

    reopened = JobEventStore('eval_test', str(tmp_path))
    event = reopened.append('job.started', 'started')

    assert event.sequence == 2


def test_event_store_rejects_invalid_or_expired_cursors(tmp_path):
    store = JobEventStore('eval_test', str(tmp_path))
    store.append('job.queued', 'queued')

    with pytest.raises(EventCursorAheadError):
        store.page(after_cursor=2)

    # Simulate retention beginning at sequence 3 without adding a pruning API.
    path = tmp_path / '_agent' / 'events.jsonl'
    lines = path.read_text(encoding='utf-8').splitlines()
    payload = lines[0].replace('"sequence":1', '"sequence":3')
    path.write_text(payload + '\n', encoding='utf-8')
    retained = JobEventStore('eval_test', str(tmp_path))
    with pytest.raises(EventCursorExpiredError):
        retained.page(after_cursor=0)
