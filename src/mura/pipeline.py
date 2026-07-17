from __future__ import annotations

import time

from mura.deepseek.service import DeepSeekPipelineService
from mura.domain.models import PipelineRequest, PipelineResult
from mura.resolution import resolve_mentions


class MuraPipeline:
    def __init__(self, deepseek: DeepSeekPipelineService) -> None:
        self.deepseek = deepseek

    def process(self, request: PipelineRequest) -> PipelineResult:
        started = time.perf_counter()
        cleaned, cleaner_usage = self.deepseek.clean(
            transcript=request.transcript,
            speaker_id=request.speaker_id,
            speaker_name=request.speaker_name,
        )
        extraction, extractor_usage = self.deepseek.extract(
            transcript=request.transcript,
            cleaned=cleaned,
            speaker_id=request.speaker_id,
            speaker_name=request.speaker_name,
            known_people=[person.model_dump() for person in request.known_people],
        )
        resolutions = resolve_mentions(extraction, request.known_people)

        return PipelineResult(
            transcript=request.transcript,
            cleaned_transcript=cleaned,
            extraction=extraction,
            resolutions=resolutions,
            processing={
                "total_seconds": round(time.perf_counter() - started, 3),
                "cleaner_usage": cleaner_usage,
                "extractor_usage": extractor_usage,
            },
        )
