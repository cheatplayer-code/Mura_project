from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import JSON

from mura.domain.models import PipelineResult
from mura.jobs import JobStatus

JSON_VALUE = JSON().with_variant(JSONB, "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class RecordingRow(Base):
    __tablename__ = "recordings"

    recording_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(128), index=True)
    speaker_id: Mapped[str] = mapped_column(String(128), index=True)
    speaker_name: Mapped[str] = mapped_column(String(256))
    original_filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    audio_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProcessingJobRow(Base):
    __tablename__ = "processing_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    recording_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("recordings.recording_id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=JobStatus.QUEUED.value,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        index=True,
    )
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )


class PipelineResultRow(Base):
    __tablename__ = "pipeline_results"

    recording_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("recordings.recording_id", ondelete="CASCADE"),
        primary_key=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )


class WorkerRegistrationRow(Base):
    __tablename__ = "worker_registrations"

    worker_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    url: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="ready")
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Database:
    def __init__(self, database_url: str) -> None:
        connect_args: dict[str, Any] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            future=True,
            connect_args=connect_args,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)


class RecordingRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create_recording_and_job(
        self,
        *,
        recording_id: str,
        job_id: str,
        family_id: str,
        speaker_id: str,
        speaker_name: str,
        original_filename: str,
        content_type: str | None,
        audio_path: Path,
    ) -> None:
        with self.database.session_factory.begin() as session:
            session.add(
                RecordingRow(
                    recording_id=recording_id,
                    family_id=family_id,
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    original_filename=original_filename,
                    content_type=content_type,
                    audio_path=str(audio_path),
                )
            )
            session.add(
                ProcessingJobRow(
                    job_id=job_id,
                    recording_id=recording_id,
                    status=JobStatus.QUEUED.value,
                    stage="queued",
                )
            )

    def get_recording(self, recording_id: str) -> RecordingRow | None:
        with self.database.session_factory() as session:
            return session.get(RecordingRow, recording_id)

    def get_job(self, job_id: str) -> ProcessingJobRow | None:
        with self.database.session_factory() as session:
            return session.get(ProcessingJobRow, job_id)

    def get_job_for_recording(self, recording_id: str) -> ProcessingJobRow | None:
        with self.database.session_factory() as session:
            statement = select(ProcessingJobRow).where(
                ProcessingJobRow.recording_id == recording_id
            )
            return session.scalar(statement)

    def get_pipeline_result(self, recording_id: str) -> PipelineResult | None:
        with self.database.session_factory() as session:
            row = session.get(PipelineResultRow, recording_id)
            if row is None:
                return None
            return PipelineResult.model_validate(row.payload)

    def claim_next_job(self) -> ProcessingJobRow | None:
        now = utcnow()
        with self.database.session_factory.begin() as session:
            statement = (
                select(ProcessingJobRow)
                .where(
                    ProcessingJobRow.status == JobStatus.QUEUED.value,
                    ProcessingJobRow.next_attempt_at <= now,
                )
                .order_by(ProcessingJobRow.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            job = session.scalar(statement)
            if job is None:
                return None
            job.status = JobStatus.TRANSCRIBING.value
            job.stage = "asr_transcription"
            job.started_at = job.started_at or now
            job.updated_at = now
            job.error_code = None
            job.error_detail = None
            session.flush()
            session.expunge(job)
            return job

    def update_job_stage(self, job_id: str, status: JobStatus, stage: str) -> None:
        with self.database.session_factory.begin() as session:
            job = session.get(ProcessingJobRow, job_id)
            if job is None:
                raise LookupError(f"unknown job: {job_id}")
            job.status = status.value
            job.stage = stage
            job.updated_at = utcnow()

    def defer_job(
        self,
        job_id: str,
        *,
        error_code: str,
        error_detail: str,
        retry_after_seconds: float,
    ) -> None:
        now = utcnow()
        with self.database.session_factory.begin() as session:
            job = session.get(ProcessingJobRow, job_id)
            if job is None:
                raise LookupError(f"unknown job: {job_id}")
            job.status = JobStatus.QUEUED.value
            job.stage = "waiting_for_asr"
            job.attempts += 1
            job.next_attempt_at = now + timedelta(seconds=retry_after_seconds)
            job.error_code = error_code
            job.error_detail = error_detail
            job.updated_at = now

    def fail_job(self, job_id: str, *, error_code: str, error_detail: str) -> None:
        now = utcnow()
        with self.database.session_factory.begin() as session:
            job = session.get(ProcessingJobRow, job_id)
            if job is None:
                raise LookupError(f"unknown job: {job_id}")
            job.status = JobStatus.FAILED.value
            job.stage = "failed"
            job.error_code = error_code
            job.error_detail = error_detail
            job.completed_at = now
            job.updated_at = now

    def complete_job(self, job_id: str, result: PipelineResult) -> None:
        now = utcnow()
        payload = result.model_dump(mode="json")
        with self.database.session_factory.begin() as session:
            job = session.get(ProcessingJobRow, job_id)
            if job is None:
                raise LookupError(f"unknown job: {job_id}")
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
            job.status = JobStatus.COMPLETED.value
            job.stage = "completed"
            job.error_code = None
            job.error_detail = None
            job.completed_at = now
            job.updated_at = now

    def register_worker(self, *, url: str, status: str) -> WorkerRegistrationRow:
        now = utcnow()
        with self.database.session_factory.begin() as session:
            row = session.get(WorkerRegistrationRow, "kaggle-asr")
            if row is None:
                row = WorkerRegistrationRow(
                    worker_name="kaggle-asr",
                    url=url.rstrip("/"),
                    status=status,
                    registered_at=now,
                )
                session.add(row)
            else:
                row.url = url.rstrip("/")
                row.status = status
                row.registered_at = now
            session.flush()
            session.expunge(row)
            return row

    def current_worker(self) -> WorkerRegistrationRow | None:
        with self.database.session_factory() as session:
            return session.get(WorkerRegistrationRow, "kaggle-asr")
