from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from mura.deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekUsage
from mura.deepseek.prompts import CLEANER_SYSTEM_PROMPT, EXTRACTOR_SYSTEM_PROMPT
from mura.domain.models import CleanerResult, ExtractionResult, TranscriptEnvelope
from mura.validation import validate_cleaner_result, validate_extraction_result

ModelT = TypeVar("ModelT", bound=BaseModel)


class DeepSeekPipelineService:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def clean(
        self,
        *,
        transcript: TranscriptEnvelope,
        speaker_id: str,
        speaker_name: str,
    ) -> tuple[CleanerResult, dict[str, Any]]:
        payload = {
            "recording_id": transcript.recording_id,
            "speaker": {"speaker_id": speaker_id, "speaker_name": speaker_name},
            "output_schema": CleanerResult.model_json_schema(),
            "segments": [segment.model_dump() for segment in transcript.segments],
        }
        raw, usage = self.client.request_json(
            system_prompt=CLEANER_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=12_000,
        )
        result = self._validate_model(CleanerResult, raw, "cleaner")
        validate_cleaner_result(transcript, result)
        return result, self._usage_dict(usage)

    def extract(
        self,
        *,
        transcript: TranscriptEnvelope,
        cleaned: CleanerResult,
        speaker_id: str,
        speaker_name: str,
        known_people: list[dict[str, Any]] | None = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        payload = {
            "recording_id": transcript.recording_id,
            "speaker": {"speaker_id": speaker_id, "speaker_name": speaker_name},
            "known_people": known_people or [],
            "output_schema": ExtractionResult.model_json_schema(),
            "raw_segments": [segment.model_dump() for segment in transcript.segments],
            "readable_segments": [segment.model_dump() for segment in cleaned.readable_segments],
            "detected_corrections": [
                correction.model_dump() for correction in cleaned.detected_corrections
            ],
            "uncertain_fragments": [
                fragment.model_dump() for fragment in cleaned.uncertain_fragments
            ],
            "full_readable_text": cleaned.full_readable_text,
        }
        raw, usage = self.client.request_json(
            system_prompt=EXTRACTOR_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=16_000,
        )
        result = self._validate_model(ExtractionResult, raw, "extractor")
        validate_extraction_result(transcript, result)
        return result, self._usage_dict(usage)

    @staticmethod
    def _validate_model(model_type: type[ModelT], raw: dict[str, Any], stage: str) -> ModelT:
        try:
            return model_type.model_validate(raw)
        except ValidationError as exc:
            raise DeepSeekError(f"{stage} JSON failed Pydantic validation: {exc}") from exc

    @staticmethod
    def _usage_dict(usage: DeepSeekUsage) -> dict[str, Any]:
        return asdict(usage)
