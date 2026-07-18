from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from mura.domain.models import PipelineResult, StrictModel


class JobStatus(StrEnum):
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    CLEANING = "cleaning"
    EXTRACTING = "extracting"
    RESOLVING = "resolving"
    COMPLETED = "completed"
    FAILED = "failed"


class RecordingAccepted(StrictModel):
    recording_id: str
    job_id: str
    status: JobStatus = JobStatus.QUEUED


class JobView(StrictModel):
    job_id: str
    recording_id: str
    status: JobStatus
    stage: str
    attempts: int = Field(ge=0)
    error_code: str | None = None
    error_detail: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime


class RecordingResultView(StrictModel):
    recording_id: str
    family_id: str
    speaker_id: str
    speaker_name: str
    job_id: str
    status: JobStatus
    result: PipelineResult


class ReviewItemsView(StrictModel):
    recording_id: str
    uncertain_fragments: list[dict[str, Any]] = Field(default_factory=list)
    detected_corrections: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_questions: list[dict[str, Any]] = Field(default_factory=list)
    extraction_issues: list[dict[str, Any]] = Field(default_factory=list)
    ambiguous_resolutions: list[dict[str, Any]] = Field(default_factory=list)
