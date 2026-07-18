from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import BinaryIO

from mura.asr import ASRClientError, RemoteASRClient
from mura.domain.models import PipelineRequest
from mura.jobs import JobStatus
from mura.pipeline import MuraPipeline
from mura.storage.database import ProcessingJobRow, RecordingRepository

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".ogg",
    ".opus",
    ".webm",
    ".flac",
}


class AudioStorageError(ValueError):
    pass


class LocalAudioStorage:
    def __init__(self, root: Path, *, max_upload_bytes: int) -> None:
        self.root = root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        recording_id: str,
        original_filename: str,
        source: BinaryIO,
    ) -> Path:
        safe_name = Path(original_filename or "audio.bin").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise AudioStorageError(f"unsupported audio extension: {suffix or '<none>'}")

        destination = self.root / f"{recording_id}{suffix}"
        temporary = self.root / f".{recording_id}{suffix}.uploading"
        total = 0
        try:
            with temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_upload_bytes:
                        raise AudioStorageError(
                            f"audio exceeds maximum upload size of "
                            f"{self.max_upload_bytes // (1024 * 1024)} MB"
                        )
                    output.write(chunk)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination


class RecordingJobWorker:
    def __init__(
        self,
        *,
        repository: RecordingRepository,
        pipeline: MuraPipeline,
        asr_client: RemoteASRClient,
        poll_interval_seconds: float = 1.0,
        asr_retry_seconds: float = 15.0,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self.asr_client = asr_client
        self.poll_interval_seconds = poll_interval_seconds
        self.asr_retry_seconds = asr_retry_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="mura-recording-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)

    def process_once(self) -> bool:
        job = self.repository.claim_next_job()
        if job is None:
            return False
        self._process_job(job)
        return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            processed = self.process_once()
            if not processed:
                self._stop_event.wait(self.poll_interval_seconds)

    def _process_job(self, job: ProcessingJobRow) -> None:
        recording = self.repository.get_recording(job.recording_id)
        if recording is None:
            self.repository.fail_job(
                job.job_id,
                error_code="recording_missing",
                error_detail=f"recording {job.recording_id} does not exist",
            )
            return

        worker = self.repository.current_worker()
        if worker is None or worker.status != "ready":
            self.repository.defer_job(
                job.job_id,
                error_code="asr_worker_unavailable",
                error_detail="no ready ASR worker is registered",
                retry_after_seconds=self.asr_retry_seconds,
            )
            return

        try:
            transcript = self.asr_client.transcribe(
                worker_url=worker.url,
                audio_path=Path(recording.audio_path),
                recording_id=recording.recording_id,
                content_type=recording.content_type,
            )
        except ASRClientError as exc:
            if exc.retryable:
                self.repository.defer_job(
                    job.job_id,
                    error_code="asr_temporarily_unavailable",
                    error_detail=str(exc),
                    retry_after_seconds=self.asr_retry_seconds,
                )
            else:
                self.repository.fail_job(
                    job.job_id,
                    error_code="asr_failed",
                    error_detail=str(exc),
                )
            return

        status_by_stage = {
            "cleaning": JobStatus.CLEANING,
            "extracting": JobStatus.EXTRACTING,
            "resolving": JobStatus.RESOLVING,
        }

        def report_stage(stage: str) -> None:
            status = status_by_stage.get(stage)
            if status is not None:
                self.repository.update_job_stage(job.job_id, status, stage)

        try:
            result = self.pipeline.process(
                PipelineRequest(
                    transcript=transcript,
                    speaker_id=recording.speaker_id,
                    speaker_name=recording.speaker_name,
                    known_people=[],
                ),
                stage_callback=report_stage,
            )
            self.repository.complete_job(job.job_id, result)
        except Exception as exc:
            self.repository.fail_job(
                job.job_id,
                error_code="pipeline_failed",
                error_detail=str(exc)[:4000],
            )

    def wait_until_idle(self, timeout_seconds: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not self.process_once():
                return True
        return False
