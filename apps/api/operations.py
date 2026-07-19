from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from fastapi import Depends, FastAPI, HTTPException, status

from mura.observability import JobTraceView, TraceRepository
from mura.storage.database import Database, RecordingRepository


class RuntimeWithDatabase(Protocol):
    database: Database
    repository: RecordingRepository


def register_operations_routes(
    app: FastAPI,
    *,
    get_runtime_dependency: Callable[..., object],
    core_token_dependency: Callable[..., None],
) -> None:
    dependencies = [Depends(core_token_dependency)]

    @app.get(
        "/v1/jobs/{job_id}/trace",
        response_model=JobTraceView,
        dependencies=dependencies,
    )
    def get_job_trace(
        job_id: str,
        runtime: object = Depends(get_runtime_dependency),
    ) -> JobTraceView:
        typed_runtime = cast(RuntimeWithDatabase, runtime)
        if typed_runtime.repository.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="job not found",
            )
        trace = TraceRepository(typed_runtime.database).get_job_trace(job_id=job_id)
        if trace is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="job trace is not available yet",
            )
        return trace
