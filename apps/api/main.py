from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator

from mura.config import CoreSettings
from mura.deepseek import DeepSeekClient, DeepSeekError, DeepSeekPipelineService
from mura.domain.models import PipelineRequest, PipelineResult
from mura.pipeline import MuraPipeline
from mura.security import verify_bearer_token
from mura.validation import ContractValidationError

app = FastAPI(
    title="Mura Core API",
    version="0.1.0",
    description="Transcript cleaning, family-memory extraction, and entity resolution.",
)

_pipeline: MuraPipeline | None = None
_pipeline_lock = Lock()
_worker_lock = Lock()
_worker_state: dict[str, object] = {"url": None, "registered_at": None}


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


def get_settings() -> CoreSettings:
    try:
        return CoreSettings()  # type: ignore[call-arg]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Core service is not configured: {exc}",
        ) from exc


def get_pipeline(
    settings: Annotated[CoreSettings, Depends(get_settings)],
) -> MuraPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                client = DeepSeekClient(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                    primary_model=settings.deepseek_model,
                    fallback_model=settings.deepseek_fallback_model,
                )
                _pipeline = MuraPipeline(DeepSeekPipelineService(client))
    return _pipeline


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


@app.post("/v1/process-transcript", response_model=PipelineResult)
def process_transcript(
    request: PipelineRequest,
    pipeline: Annotated[MuraPipeline, Depends(get_pipeline)],
) -> PipelineResult:
    return pipeline.process(request)


@app.post("/v1/workers/register")
def register_worker(
    registration: WorkerRegistration,
    settings: Annotated[CoreSettings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    verify_bearer_token(
        authorization,
        expected_token=settings.worker_registration_token,
    )

    with _worker_lock:
        _worker_state["url"] = str(registration.url).rstrip("/")
        _worker_state["status"] = registration.status
        _worker_state["registered_at"] = datetime.now(UTC).isoformat()

    return {"accepted": True, **_worker_state}


@app.get("/v1/workers/current")
def current_worker(
    settings: Annotated[CoreSettings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    verify_bearer_token(
        authorization,
        expected_token=settings.worker_registration_token,
    )
    with _worker_lock:
        return dict(_worker_state)
