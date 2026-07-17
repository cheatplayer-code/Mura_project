from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY", min_length=8)
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")
    deepseek_fallback_model: str = Field(default="deepseek-v4-pro", alias="DEEPSEEK_FALLBACK_MODEL")
    worker_registration_token: str = Field(
        alias="WORKER_REGISTRATION_TOKEN",
        min_length=32,
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
