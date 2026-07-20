from __future__ import annotations

from typing import Any

from mura.deepseek.discourse_telemetry import discourse_link_counters
from mura.deepseek.service import DeepSeekPipelineService as _DeepSeekPipelineService
from mura.domain.models import CleanerResult, ExtractionResult, KnownPerson, TranscriptEnvelope


class DeepSeekPipelineService(_DeepSeekPipelineService):
    def extract(
        self,
        *,
        transcript: TranscriptEnvelope,
        cleaned: CleanerResult,
        speaker_id: str,
        speaker_name: str,
        known_people: list[KnownPerson] | None = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        result, usage = super().extract(
            transcript=transcript,
            cleaned=cleaned,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            known_people=known_people,
        )
        usage["coreference_metrics"] = discourse_link_counters(result)
        return result, usage
