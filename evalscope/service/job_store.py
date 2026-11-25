# Copyright (c) Alibaba, Inc. and its affiliates.
"""Persistent job metadata storage for the EvalScope service."""

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict

JobStatus = Literal['queued', 'running', 'completed', 'error', 'cancelling', 'cancelled', 'interrupted']

_VALID_STATUSES = frozenset({'queued', 'running', 'completed', 'error', 'cancelling', 'cancelled', 'interrupted'})
_TERMINAL_STATUSES = frozenset({'completed', 'error', 'cancelled', 'interrupted'})
_STATUS_TRANSITIONS = {
    'queued': frozenset({'running', 'error', 'cancelled'}),
    'running': frozenset({'completed', 'error', 'cancelling', 'cancelled', 'interrupted'}),
    'cancelling': frozenset({'error', 'cancelled', 'interrupted'}),
    'completed': frozenset(),
    'error': frozenset(),
    'cancelled': frozenset(),
    'interrupted': frozenset(),
}
_DEFAULT_DB_PATH = '~/.evalscope/jobs.db'
_SCHEMA_VERSION = 1


class JobStoreError(Exception):
    """Base exception for persistent job-store errors."""


class JobNotFoundError(JobStoreError):
    """Raised when a requested job does not exist."""


class JobAlreadyExistsError(JobStoreError):
    """Raised when a new job reuses an existing job identifier."""


class IdempotencyConflictError(JobStoreError):
    """Raised when an idempotency key is reused for a different request."""


class InvalidJobTransitionError(JobStoreError):
    """Raised when a job status transition is not allowed."""


class JobRecord(BaseModel):
    """Persisted service job metadata and its optional result."""

    model_config = ConfigDict(extra='forbid')

    job_id: str
    idempotency_key: str
    request_hash: str
    kind: str
    status: JobStatus
    request: dict[str, Any]
    result: Any = None
    error: str | None = None
    output_dir: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class CreateJobResult(BaseModel):
    """Result of an idempotent job creation attempt."""

    model_config = ConfigDict(extra='forbid')

    job: JobRecord
    created: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Job request and result values must be JSON serializable') from error


def _request_hash(kind: str, request_json: str) -> str:
    payload = f'{kind}\0{request_json}'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class JobStore:
    """Store asynchronous service jobs in a thread-safe SQLite database.

    Each operation opens its own connection so one store instance can be used
    from Flask request threads and background workers. Write operations use
    an immediate transaction to serialize idempotency checks with inserts.
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None, busy_timeout_seconds: float = 30.0) -> None:
        if busy_timeout_seconds <= 0:
            raise ValueError('busy_timeout_seconds must be positive')
        configured_path = db_path or os.environ.get('EVALSCOPE_JOB_DB', _DEFAULT_DB_PATH)
        self.db_path = Path(os.path.expandvars(os.path.expanduser(os.fspath(configured_path)))).resolve()
        self.busy_timeout_seconds = busy_timeout_seconds
        self._prepare_database()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f'PRAGMA busy_timeout = {int(self.busy_timeout_seconds * 1000)}')
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _prepare_database(self) -> None:
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute('PRAGMA journal_mode = WAL')
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    output_dir TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    CHECK (status IN ({','.join(repr(status) for status in sorted(_VALID_STATUSES))}))
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                PRAGMA user_version = {_SCHEMA_VERSION};
                """,
            )
        try:
            self.db_path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> JobRecord:
        result_json = row['result_json']
        return JobRecord(
            job_id=row['job_id'],
            idempotency_key=row['idempotency_key'],
            request_hash=row['request_hash'],
            kind=row['kind'],
            status=row['status'],
            request=json.loads(row['request_json']),
            result=json.loads(result_json) if result_json is not None else None,
            error=row['error'],
            output_dir=row['output_dir'],
            created_at=row['created_at'],
            started_at=row['started_at'],
            finished_at=row['finished_at'],
        )

    @staticmethod
    def _require_text(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{field} must be a non-empty string')
        return value.strip()

    def create_job(
        self,
        job_id: str,
        idempotency_key: str,
        kind: str,
        request: Mapping[str, Any],
        output_dir: str | None = None,
    ) -> CreateJobResult:
        """Create a queued job or return the existing idempotent job.

        Reusing an idempotency key with different request content raises
        IdempotencyConflictError.
        """
        job_id = self._require_text(job_id, 'job_id')
        idempotency_key = self._require_text(idempotency_key, 'idempotency_key')
        kind = self._require_text(kind, 'kind')
        if not isinstance(request, Mapping):
            raise ValueError('request must be a mapping')
        request_json = _json_dumps(dict(request))
        request_hash = _request_hash(kind, request_json)
        created_at = _utc_now()

        with self._write_transaction() as connection:
            existing = connection.execute(
                'SELECT * FROM jobs WHERE idempotency_key = ?',
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                record = self._record_from_row(existing)
                if record.request_hash != request_hash:
                    raise IdempotencyConflictError(
                        f'Idempotency key {idempotency_key!r} is already associated with a different request'
                    )
                return CreateJobResult(job=record, created=False)

            try:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, idempotency_key, request_hash, kind, status, request_json, output_dir, created_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (job_id, idempotency_key, request_hash, kind, request_json, output_dir, created_at),
                )
            except sqlite3.IntegrityError as error:
                raise JobAlreadyExistsError(f'Job {job_id!r} already exists') from error
            row = connection.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,)).fetchone()
            return CreateJobResult(job=self._record_from_row(row), created=True)

    def get_job(self, job_id: str) -> JobRecord | None:
        """Return a job by identifier, or None when it does not exist."""
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,)).fetchone()
        return self._record_from_row(row) if row is not None else None

    def get_by_idempotency_key(self, idempotency_key: str) -> JobRecord | None:
        """Return a job by idempotency key, or None when absent."""
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM jobs WHERE idempotency_key = ?',
                (idempotency_key,),
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def _require_job(self, connection: sqlite3.Connection, job_id: str) -> JobRecord:
        row = connection.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(f'Job {job_id!r} was not found')
        return self._record_from_row(row)

    @staticmethod
    def _validate_transition(current: JobStatus, target: JobStatus) -> None:
        if target not in _VALID_STATUSES:
            raise ValueError(f'Unknown job status: {target}')
        if current == target:
            return
        if target not in _STATUS_TRANSITIONS[current]:
            raise InvalidJobTransitionError(f'Cannot transition a job from {current!r} to {target!r}')

    def update_status(self, job_id: str, status: JobStatus, error: str | None = None) -> JobRecord:
        """Apply a valid status transition and return the updated job."""
        with self._write_transaction() as connection:
            current = self._require_job(connection, job_id)
            self._validate_transition(current.status, status)
            if current.status == status and error == current.error:
                return current
            started_at = current.started_at
            finished_at = current.finished_at
            now = _utc_now()
            if status == 'running' and started_at is None:
                started_at = now
            if status in _TERMINAL_STATUSES:
                finished_at = now
            connection.execute(
                'UPDATE jobs SET status = ?, error = ?, started_at = ?, finished_at = ? WHERE job_id = ?',
                (status, error, started_at, finished_at, job_id),
            )
            return self._require_job(connection, job_id)

    def complete_job(self, job_id: str, result: Any, output_dir: str | None = None) -> JobRecord:
        """Mark a running job completed and persist its JSON result."""
        result_json = _json_dumps(result)
        with self._write_transaction() as connection:
            current = self._require_job(connection, job_id)
            self._validate_transition(current.status, 'completed')
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed', result_json = ?, error = NULL, output_dir = ?, finished_at = ?
                WHERE job_id = ?
                """,
                (result_json, output_dir, _utc_now(), job_id),
            )
            return self._require_job(connection, job_id)

    def fail_job(self, job_id: str, error: str) -> JobRecord:
        """Mark a queued or running job failed with a safe error message."""
        error = self._require_text(error, 'error')
        return self.update_status(job_id, 'error', error=error)

    def mark_running_jobs_interrupted(self) -> int:
        """Mark jobs left running by a previous service process interrupted."""
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'interrupted', finished_at = ?
                WHERE status = 'running'
                """,
                (_utc_now(),),
            )
            return cursor.rowcount
