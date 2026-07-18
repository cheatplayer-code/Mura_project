from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from mura.deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekUsage
from mura.deepseek.prompts import (
    CLEANER_REPAIR_SYSTEM_PROMPT,
    CLEANER_SYSTEM_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
)
from mura.domain.models import CleanerResult, ExtractionResult, TranscriptEnvelope
from mura.extraction_sanitizer import sanitize_extraction_output
from mura.validation import ContractValidationError, validate_cleaner_result

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
        initial_usage = self._usage_dict(usage)
        try:
            result = self._validate_model(CleanerResult, raw, "cleaner")
            validate_cleaner_result(transcript, result)
        except (DeepSeekError, ContractValidationError) as exc:
            result, repair_usage = self._repair_cleaner(
                transcript=transcript,
                invalid_output=raw,
                validation_error=str(exc),
            )
            return result, {
                **repair_usage,
                "repair_attempted": True,
                "initial_validation_error": str(exc),
                "initial_usage": initial_usage,
            }
        return result, {**initial_usage, "repair_attempted": False}

    def _repair_cleaner(
        self,
        *,
        transcript: TranscriptEnvelope,
        invalid_output: dict[str, Any],
        validation_error: str,
    ) -> tuple[CleanerResult, dict[str, Any]]:
        repair_payload = {
            "validation_error": validation_error,
            "output_schema": CleanerResult.model_json_schema(),
            "invalid_output": invalid_output,
            "allowed_segment_ids": [segment.segment_id for segment in transcript.segments],
            "raw_segments": [segment.model_dump() for segment in transcript.segments],
        }
        repaired_raw, repair_usage = self.client.request_json(
            system_prompt=CLEANER_REPAIR_SYSTEM_PROMPT,
            payload=repair_payload,
            max_tokens=12_000,
            attempts=2,
        )
        repaired = self._validate_model(CleanerResult, repaired_raw, "cleaner repair")
        validate_cleaner_result(transcript, repaired)
        return repaired, self._usage_dict(repair_usage)

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
        result, extraction_issues, evidence_closure_count = sanitize_extraction_output(
            raw=raw,
            transcript=transcript,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
        )
        raw_relationships = raw.get("relationship_claims", [])
        relationship_candidates = (
            len(raw_relationships) if isinstance(raw_relationships, list) else 0
        )
        quarantined_relationships = sum(
            issue.get("object_type") == "relationship" for issue in extraction_issues
        )
        accepted_relationships = len(result.relationship_claims)
        acceptance_rate = (
            accepted_relationships / relationship_candidates if relationship_candidates else None
        )
        evidence_class_counts = Counter(
            item.evidence_class.value
            for item in [
                *result.people_mentions,
                *result.relationship_claims,
                *result.events,
                *result.descriptions,
                *result.stories,
                *result.unresolved_questions,
            ]
        )
        return result, {
            **self._usage_dict(usage),
            "repair_attempted": False,
            "evidence_closure_relationships": evidence_closure_count,
            "quarantined_items": len(extraction_issues),
            "extraction_issues": extraction_issues,
            "relationship_metrics": {
                "candidates": relationship_candidates,
                "accepted": accepted_relationships,
                "quarantined": quarantined_relationships,
                "acceptance_rate": acceptance_rate,
            },
            "claim_contract": {
                "schema_version": result.schema_version,
                "evidence_spans": len(result.evidence_spans),
                "provenance_activities": len(result.provenance_activities),
                "coreference_links": len(result.coreference_links),
                "conflict_sets": len(result.conflict_sets),
                "evidence_class_counts": dict(sorted(evidence_class_counts.items())),
            },
        }

    @staticmethod
    def _validate_model(model_type: type[ModelT], raw: dict[str, Any], stage: str) -> ModelT:
        try:
            return model_type.model_validate(raw)
        except ValidationError as exc:
            raise DeepSeekError(f"{stage} JSON failed Pydantic validation: {exc}") from exc

    @staticmethod
    def _usage_dict(usage: DeepSeekUsage) -> dict[str, Any]:
        return asdict(usage)
