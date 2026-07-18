from __future__ import annotations

import time
from collections.abc import Callable

from mura.deepseek.service import DeepSeekPipelineService
from mura.domain.models import PipelineRequest, PipelineResult
from mura.resolution import resolve_mentions
from mura.versioning import get_pipeline_versions

StageCallback = Callable[[str], None]


class MuraPipeline:
    def __init__(self, deepseek: DeepSeekPipelineService) -> None:
        self.deepseek = deepseek

    def process(
        self,
        request: PipelineRequest,
        *,
        stage_callback: StageCallback | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        self._report(stage_callback, "cleaning")
        cleaned, cleaner_usage = self.deepseek.clean(
            transcript=request.transcript,
            speaker_id=request.speaker_id,
            speaker_name=request.speaker_name,
        )
        self._report(stage_callback, "extracting")
        extraction, extractor_usage = self.deepseek.extract(
            transcript=request.transcript,
            cleaned=cleaned,
            speaker_id=request.speaker_id,
            speaker_name=request.speaker_name,
            known_people=request.known_people,
        )
        self._report(stage_callback, "resolving")
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
                "versions": get_pipeline_versions().model_dump(mode="json"),
            },
        )

    @staticmethod
    def _report(callback: StageCallback | None, stage: str) -> None:
        if callback is not None:
            callback(stage)
