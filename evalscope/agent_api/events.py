"""Append-only run events for resumable Agent polling.

The store deliberately uses JSON Lines rather than a database.  A single
service worker owns each job today, while the on-disk format keeps the event
contract independent from the in-memory job registry and ready for a future
persistent scheduler.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


EventLevel = Literal['debug', 'info', 'warning', 'error']


class RunEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sequence: int = Field(ge=1)
    timestamp: datetime
    type: str
    level: EventLevel = 'info'
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class EventPage(BaseModel):
    model_config = ConfigDict(extra='forbid')

    job_id: str
    events: List[RunEvent]
    next_cursor: int = Field(ge=0)
    has_more: bool = False


class EventCursorExpiredError(ValueError):
    """Raised when a requested cursor predates retained history."""


class EventCursorAheadError(ValueError):
    """Raised when a requested cursor is newer than the event stream."""


class JobEventStore:
    """Thread-safe, append-only event storage for one logical job."""

    def __init__(self, job_id: str, run_root: str):
        self.job_id = job_id
        self.path = Path(run_root) / '_agent' / 'events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        events = self._read_unlocked()
        self._next_sequence = (events[-1].sequence + 1) if events else 1

    def _read_unlocked(self) -> List[RunEvent]:
        if not self.path.exists():
            return []
        events = []
        with self.path.open('r', encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    events.append(RunEvent.model_validate_json(line))
        return events

    def append(
        self,
        event_type: str,
        message: str,
        *,
        level: EventLevel = 'info',
        data: Optional[Dict[str, Any]] = None,
    ) -> RunEvent:
        with self._lock:
            event = RunEvent(
                sequence=self._next_sequence,
                timestamp=datetime.now(timezone.utc),
                type=event_type,
                level=level,
                message=message,
                data=data or {},
            )
            payload = event.model_dump_json() + '\n'
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, payload.encode('utf-8'))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._next_sequence += 1
            return event

    def page(self, after_cursor: int = 0, limit: int = 100) -> EventPage:
        if after_cursor < 0:
            raise ValueError('after_cursor must be non-negative')
        if limit < 1 or limit > 1000:
            raise ValueError('limit must be between 1 and 1000')
        with self._lock:
            events = self._read_unlocked()
        if not events:
            if after_cursor:
                raise EventCursorAheadError(f'cursor {after_cursor} is ahead of the empty event stream')
            return EventPage(job_id=self.job_id, events=[], next_cursor=after_cursor)

        earliest = events[0].sequence
        latest = events[-1].sequence
        if after_cursor < earliest - 1:
            raise EventCursorExpiredError(
                f'cursor {after_cursor} predates the earliest retained event {earliest}'
            )
        if after_cursor > latest:
            raise EventCursorAheadError(f'cursor {after_cursor} is ahead of latest event {latest}')

        remaining = [event for event in events if event.sequence > after_cursor]
        selected = remaining[:limit]
        next_cursor = selected[-1].sequence if selected else after_cursor
        return EventPage(
            job_id=self.job_id,
            events=selected,
            next_cursor=next_cursor,
            has_more=len(remaining) > len(selected),
        )
