from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY", min_length=8)
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")
    deepseek_fallback_model: str = Field(
        default="deepseek-v4-pro",
        alias="DEEPSEEK_FALLBACK_MODEL",
    )
    core_api_key: str = Field(alias="CORE_API_KEY", min_length=32)
    worker_registration_token: str = Field(
        alias="WORKER_REGISTRATION_TOKEN",
        min_length=32,
    )
    kaggle_asr_api_key: str = Field(alias="KAGGLE_ASR_API_KEY", min_length=32)
    database_url: str = Field(alias="DATABASE_URL", min_length=1)
    database_auto_create: bool = Field(default=True, alias="DATABASE_AUTO_CREATE")
    audio_storage_dir: Path = Field(default=Path(".mura/audio"), alias="AUDIO_STORAGE_DIR")
    core_max_upload_mb: int = Field(default=25, alias="CORE_MAX_UPLOAD_MB", ge=1, le=200)
    job_poll_interval_seconds: float = Field(
        default=1.0,
        alias="JOB_POLL_INTERVAL_SECONDS",
        ge=0.1,
        le=60,
    )
    asr_retry_seconds: float = Field(default=15.0, alias="ASR_RETRY_SECONDS", ge=1, le=600)
    asr_request_timeout_seconds: float = Field(
        default=900.0,
        alias="ASR_REQUEST_TIMEOUT_SECONDS",
        ge=30,
        le=3600,
    )


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kaggle_asr_api_key: str = Field(alias="KAGGLE_ASR_API_KEY", min_length=32)
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    asr_device: str = Field(default="cuda:0", alias="ASR_DEVICE")
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB", ge=1, le=200)
    max_audio_seconds: int = Field(default=900, alias="MAX_AUDIO_SECONDS", ge=10, le=3600)
    core_backend_url: str | None = Field(default=None, alias="CORE_BACKEND_URL")
    worker_registration_token: str | None = Field(
        default=None,
        alias="WORKER_REGISTRATION_TOKEN",
        min_length=32,
    )

    @model_validator(mode="after")
    def require_registration_token_for_callback(self) -> WorkerSettings:
        if self.core_backend_url and not self.worker_registration_token:
            raise ValueError(
                "WORKER_REGISTRATION_TOKEN is required when CORE_BACKEND_URL is configured"
            )
        return self
