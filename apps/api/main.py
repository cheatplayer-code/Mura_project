from __future__ import annotations

import atexit
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator

from mura.asr import RemoteASRClient
from mura.config import CoreSettings
from mura.deepseek import DeepSeekClient, DeepSeekError, DeepSeekPipelineService
from mura.domain.models import PipelineRequest, PipelineResult, ResolutionStatus
from mura.jobs import (
    JobStatus,
    JobView,
    RecordingAccepted,
    RecordingResultView,
    ReviewItemsView,
)
from mura.orchestration import AudioStorageError, LocalAudioStorage, RecordingJobWorker
from mura.pipeline import MuraPipeline
from mura.security import verify_bearer_token
from mura.storage.database import Database, ProcessingJobRow, RecordingRepository
from mura.validation import ContractValidationError

app = FastAPI(
    title="Mura Core API",
    version="0.2.0",
    description=(
        "Audio ingestion, asynchronous ASR orchestration, source-linked family-memory "
        "extraction, review items, and entity resolution."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_runtime_lock = Lock()
_runtime: CoreRuntime | None = None


class WorkerRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    status: str = "ready"

    @field_validator("url")
    @classmethod
    def require_https(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("worker URL must use HTTPS")
        return value


@dataclass
class CoreRuntime:
    settings: CoreSettings
    database: Database
    repository: RecordingRepository
    pipeline: MuraPipeline
    storage: LocalAudioStorage
    worker: RecordingJobWorker

    def stop(self) -> None:
        self.worker.stop()


def get_settings() -> CoreSettings:
    try:
        return CoreSettings()  # type: ignore[call-arg]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Core service is not configured: {exc}",
        ) from exc


def _build_pipeline(settings: CoreSettings) -> MuraPipeline:
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        primary_model=settings.deepseek_model,
        fallback_model=settings.deepseek_fallback_model,
    )
    return MuraPipeline(DeepSeekPipelineService(client))


def get_runtime(
    settings: Annotated[CoreSettings, Depends(get_settings)],
) -> CoreRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                database = Database(settings.database_url)
                if settings.database_auto_create:
                    database.create_schema()
                repository = RecordingRepository(database)
                pipeline = _build_pipeline(settings)
                storage = LocalAudioStorage(
                    settings.audio_storage_dir,
                    max_upload_bytes=settings.core_max_upload_mb * 1024 * 1024,
                )
                worker = RecordingJobWorker(
                    repository=repository,
                    pipeline=pipeline,
                    asr_client=RemoteASRClient(
                        api_key=settings.kaggle_asr_api_key,
                        timeout_seconds=settings.asr_request_timeout_seconds,
                    ),
                    poll_interval_seconds=settings.job_poll_interval_seconds,
                    asr_retry_seconds=settings.asr_retry_seconds,
                )
                runtime = CoreRuntime(
                    settings=settings,
                    database=database,
                    repository=repository,
                    pipeline=pipeline,
                    storage=storage,
                    worker=worker,
                )
                _runtime = runtime
                worker.start()
                atexit.register(runtime.stop)
    assert _runtime is not None
    return _runtime


def require_core_token(
    settings: Annotated[CoreSettings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    verify_bearer_token(authorization, expected_token=settings.core_api_key)


def require_worker_token(
    settings: Annotated[CoreSettings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    verify_bearer_token(
        authorization,
        expected_token=settings.worker_registration_token,
    )


@app.exception_handler(DeepSeekError)
async def handle_deepseek_error(_request: Request, exc: DeepSeekError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "deepseek_pipeline_failed",
            "detail": str(exc),
        },
    )


@app.exception_handler(ContractValidationError)
async def handle_contract_validation_error(
    _request: Request,
    exc: ContractValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "pipeline_output_invalid",
            "detail": str(exc),
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mura-core"}


@app.post(
    "/v1/recordings",
    response_model=RecordingAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_core_token)],
)
def create_recording(
    runtime: Annotated[CoreRuntime, Depends(get_runtime)],
    file: Annotated[UploadFile, File(...)],
    family_id: Annotated[str, Form(min_length=1, max_length=128)],
    speaker_id: Annotated[str, Form(min_length=1, max_length=128)],
    speaker_name: Annotated[str, Form(min_length=1, max_length=256)],
) -> RecordingAccepted:
    recording_id = f"rec_{uuid.uuid4().hex}"
    job_id = f"job_{uuid.uuid4().hex}"
    original_filename = Path(file.filename or "audio.bin").name

    try:
        file.file.seek(0)
        audio_path = runtime.storage.save(
            recording_id=recording_id,
            original_filename=original_filename,
            source=file.file,
        )
    except AudioStorageError as exc:
        message = str(exc)
        code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if "maximum upload size" in message
            else status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )
        raise HTTPException(status_code=code, detail=message) from exc

    try:
        runtime.repository.create_recording_and_job(
            recording_id=recording_id,
            job_id=job_id,
            family_id=family_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            original_filename=original_filename,
            content_type=file.content_type,
            audio_path=audio_path,
        )
    except Exception:
        audio_path.unlink(missing_ok=True)
        raise

    return RecordingAccepted(recording_id=recording_id, job_id=job_id)


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobView,
    dependencies=[Depends(require_core_token)],
)
def get_job(
    job_id: str,
    runtime: Annotated[CoreRuntime, Depends(get_runtime)],
) -> JobView:
    row = runtime.repository.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_view(row)


@app.get(
    "/v1/recordings/{recording_id}",
    response_model=RecordingResultView,
    dependencies=[Depends(require_core_token)],
)
def get_recording_result(
    recording_id: str,
    runtime: Annotated[CoreRuntime, Depends(get_runtime)],
) -> RecordingResultView:
    recording = runtime.repository.get_recording(recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="recording not found")
    job = runtime.repository.get_job_for_recording(recording_id)
    if job is None:
        raise HTTPException(status_code=500, detail="recording has no processing job")
    result = runtime.repository.get_pipeline_result(recording_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "recording processing is not completed",
                "job": _job_view(job).model_dump(mode="json"),
            },
        )
    return RecordingResultView(
        recording_id=recording.recording_id,
        family_id=recording.family_id,
        speaker_id=recording.speaker_id,
        speaker_name=recording.speaker_name,
        job_id=job.job_id,
        status=JobStatus(job.status),
        result=result,
    )


@app.get(
    "/v1/recordings/{recording_id}/review-items",
    response_model=ReviewItemsView,
    dependencies=[Depends(require_core_token)],
)
def get_review_items(
    recording_id: str,
    runtime: Annotated[CoreRuntime, Depends(get_runtime)],
) -> ReviewItemsView:
    result = runtime.repository.get_pipeline_result(recording_id)
    if result is None:
        raise HTTPException(status_code=404, detail="completed recording result not found")

    extractor_usage = result.processing.get("extractor_usage", {})
    extraction_issues = (
        extractor_usage.get("extraction_issues", []) if isinstance(extractor_usage, dict) else []
    )
    return ReviewItemsView(
        recording_id=recording_id,
        uncertain_fragments=[
            item.model_dump(mode="json") for item in result.cleaned_transcript.uncertain_fragments
        ],
        detected_corrections=[
            item.model_dump(mode="json") for item in result.cleaned_transcript.detected_corrections
        ],
        unresolved_questions=[
            item.model_dump(mode="json") for item in result.extraction.unresolved_questions
        ],
        extraction_issues=(extraction_issues if isinstance(extraction_issues, list) else []),
        ambiguous_resolutions=[
            item.model_dump(mode="json")
            for item in result.resolutions
            if item.status == ResolutionStatus.NEEDS_REVIEW
        ],
    )


@app.post(
    "/v1/process-transcript",
    response_model=PipelineResult,
    dependencies=[Depends(require_core_token)],
)
def process_transcript(
    request: PipelineRequest,
    runtime: Annotated[CoreRuntime, Depends(get_runtime)],
) -> PipelineResult:
    return runtime.pipeline.process(request)


@app.post(
    "/v1/workers/register",
    dependencies=[Depends(require_worker_token)],
)
def register_worker(
    registration: WorkerRegistration,
    runtime: Annotated[CoreRuntime, Depends(get_runtime)],
) -> dict[str, object]:
    row = runtime.repository.register_worker(
        url=str(registration.url),
        status=registration.status,
    )
    return {
        "accepted": True,
        "url": row.url,
        "status": row.status,
        "registered_at": row.registered_at.isoformat(),
    }


@app.get(
    "/v1/workers/current",
    dependencies=[Depends(require_worker_token)],
)
def current_worker(
    runtime: Annotated[CoreRuntime, Depends(get_runtime)],
) -> dict[str, object]:
    row = runtime.repository.current_worker()
    if row is None:
        return {"url": None, "status": "missing", "registered_at": None}
    return {
        "url": row.url,
        "status": row.status,
        "registered_at": row.registered_at.isoformat(),
    }


def _job_view(row: ProcessingJobRow) -> JobView:
    return JobView(
        job_id=row.job_id,
        recording_id=row.recording_id,
        status=JobStatus(row.status),
        stage=row.stage,
        attempts=row.attempts,
        error_code=row.error_code,
        error_detail=row.error_detail,
        created_at=_aware_required(row.created_at),
        started_at=_aware_optional(row.started_at),
        completed_at=_aware_optional(row.completed_at),
        updated_at=_aware_required(row.updated_at),
    )


def _aware_required(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _aware_optional(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return _aware_required(value)
