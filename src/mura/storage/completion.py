from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from mura.domain.models import PipelineResult
from mura.jobs import JobStatus
from mura.observability import ProcessingTraceEvent, persist_trace_events
from mura.storage.database import (
    Database,
    PipelineResultRow,
    ProcessingJobRow,
    utcnow,
)


class JobFinalizationError(RuntimeError):
    pass


def finalize_recording_job(
    session: Session,
    *,
    job_id: str,
    result: PipelineResult,
    trace_events: Sequence[ProcessingTraceEvent],
) -> None:
    """Persist the result and completed job state inside the caller's archive transaction."""
    now = utcnow()
    job = session.scalar(
        select(ProcessingJobRow).where(ProcessingJobRow.job_id == job_id).with_for_update()
    )
    if job is None:
        raise JobFinalizationError(f"unknown job: {job_id}")
    if job.status == JobStatus.FAILED.value:
        raise JobFinalizationError(f"cannot complete failed job: {job_id}")

    payload = result.model_dump(mode="json")
    stored = session.get(PipelineResultRow, job.recording_id)
    if stored is None:
        session.add(
            PipelineResultRow(
                recording_id=job.recording_id,
                payload=payload,
            )
        )
    else:
        stored.payload = payload
        stored.updated_at = now

    persist_trace_events(session, trace_events)
    job.status = JobStatus.COMPLETED.value
    job.stage = "completed"
    job.error_code = None
    job.error_detail = None
    job.completed_at = now
    job.updated_at = now


def defer_recording_job(
    database: Database,
    *,
    job_id: str,
    error_code: str,
    error_detail: str,
    retry_after_seconds: float,
    trace_events: Sequence[ProcessingTraceEvent],
) -> None:
    now = utcnow()
    with database.session_factory.begin() as session:
        job = session.scalar(
            select(ProcessingJobRow).where(ProcessingJobRow.job_id == job_id).with_for_update()
        )
        if job is None:
            raise JobFinalizationError(f"unknown job: {job_id}")
        persist_trace_events(session, trace_events)
        job.status = JobStatus.QUEUED.value
        job.stage = "waiting_for_asr"
        job.attempts += 1
        job.next_attempt_at = now + timedelta(seconds=retry_after_seconds)
        job.error_code = error_code
        job.error_detail = error_detail
        job.updated_at = now


def fail_recording_job(
    database: Database,
    *,
    job_id: str,
    error_code: str,
    error_detail: str,
    trace_events: Sequence[ProcessingTraceEvent],
) -> None:
    now = utcnow()
    with database.session_factory.begin() as session:
        job = session.scalar(
            select(ProcessingJobRow).where(ProcessingJobRow.job_id == job_id).with_for_update()
        )
        if job is None:
            raise JobFinalizationError(f"unknown job: {job_id}")
        persist_trace_events(session, trace_events)
        job.status = JobStatus.FAILED.value
        job.stage = "failed"
        job.error_code = error_code
        job.error_detail = error_detail
        job.completed_at = now
        job.updated_at = now
