from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from mura.deepseek.anchor_prompts import (
    ANCHOR_CONSTRAINED_EXTRACTION_REPAIR_SYSTEM_PROMPT,
    ANCHOR_CONSTRAINED_EXTRACTOR_SYSTEM_PROMPT,
)
from mura.deepseek.anchors import ExtractionAnchorBundle, build_extraction_anchor_bundle
from mura.deepseek.client import DeepSeekClient, DeepSeekError, DeepSeekUsage
from mura.deepseek.prompts import CLEANER_REPAIR_SYSTEM_PROMPT, CLEANER_SYSTEM_PROMPT
from mura.domain.models import (
    CleanerResult,
    ExtractionResult,
    KnownPerson,
    ReadableSegment,
    TranscriptEnvelope,
)
from mura.evidence_recovery import EvidenceOffsetRecoveryMetrics, recover_evidence_offsets
from mura.extraction_sanitizer import sanitize_extraction_output
from mura.validation import ContractValidationError, validate_cleaner_result

ModelT = TypeVar("ModelT", bound=BaseModel)
_FATAL_EXTRACTION_COLLECTIONS = frozenset(
    {
        "people_mentions",
        "relationship_claims",
        "events",
        "descriptions",
        "stories",
        "unresolved_questions",
    }
)


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
            initial_validation_error = str(exc)
            try:
                result, repair_usage = self._repair_cleaner(
                    transcript=transcript,
                    invalid_output=raw,
                    validation_error=initial_validation_error,
                )
            except (DeepSeekError, ContractValidationError) as repair_exc:
                fallback = self._raw_preserving_cleaner_fallback(transcript)
                return fallback, {
                    **initial_usage,
                    "repair_attempted": True,
                    "fallback_used": True,
                    "fallback_strategy": "raw_transcript",
                    "initial_validation_error": initial_validation_error,
                    "repair_validation_error": str(repair_exc),
                    "initial_usage": initial_usage,
                }
            return result, {
                **repair_usage,
                "repair_attempted": True,
                "fallback_used": False,
                "initial_validation_error": initial_validation_error,
                "initial_usage": initial_usage,
            }
        return result, {
            **initial_usage,
            "repair_attempted": False,
            "fallback_used": False,
        }

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

    @staticmethod
    def _raw_preserving_cleaner_fallback(transcript: TranscriptEnvelope) -> CleanerResult:
        fallback = CleanerResult(
            readable_segments=[
                ReadableSegment(segment_id=segment.segment_id, text=segment.text)
                for segment in transcript.segments
            ],
            detected_corrections=[],
            uncertain_fragments=[],
            full_readable_text=" ".join(segment.text for segment in transcript.segments),
        )
        validate_cleaner_result(transcript, fallback)
        return fallback

    def extract(
        self,
        *,
        transcript: TranscriptEnvelope,
        cleaned: CleanerResult,
        speaker_id: str,
        speaker_name: str,
        known_people: list[KnownPerson] | None = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        resolved_known_people = known_people or []
        anchors = build_extraction_anchor_bundle(
            transcript=transcript,
            cleaned=cleaned,
            speaker_name=speaker_name,
            known_people=resolved_known_people,
        )
        payload = self._extraction_payload(
            transcript=transcript,
            cleaned=cleaned,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            known_people=resolved_known_people,
            anchors=anchors,
        )
        raw, usage = self.client.request_json(
            system_prompt=ANCHOR_CONSTRAINED_EXTRACTOR_SYSTEM_PROMPT,
            payload=payload,
            max_tokens=16_000,
        )
        initial_usage = self._usage_dict(usage)
        final_raw, evidence_recovery = recover_evidence_offsets(raw=raw, transcript=transcript)
        initial_evidence_recovery = evidence_recovery
        repair_attempted = False
        initial_validation_error: str | None = None
        try:
            result, extraction_issues, evidence_closure_count = sanitize_extraction_output(
                raw=final_raw,
                transcript=transcript,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
            )
        except (ValidationError, ContractValidationError) as exc:
            initial_validation_error = str(exc)
            (
                final_raw,
                result,
                extraction_issues,
                evidence_closure_count,
                evidence_recovery,
                usage,
            ) = self._repair_extraction(
                transcript=transcript,
                cleaned=cleaned,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                known_people=resolved_known_people,
                anchors=anchors,
                invalid_output=raw,
                validation_error=initial_validation_error,
            )
            repair_attempted = True
        else:
            if self._requires_extraction_repair(
                raw=final_raw,
                result=result,
                extraction_issues=extraction_issues,
            ):
                initial_validation_error = self._repair_reason(extraction_issues)
                (
                    final_raw,
                    result,
                    extraction_issues,
                    evidence_closure_count,
                    evidence_recovery,
                    usage,
                ) = self._repair_extraction(
                    transcript=transcript,
                    cleaned=cleaned,
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    known_people=resolved_known_people,
                    anchors=anchors,
                    invalid_output=raw,
                    validation_error=initial_validation_error,
                )
                repair_attempted = True

        usage_payload = self._extraction_usage(
            raw=final_raw,
            usage=usage,
            result=result,
            extraction_issues=extraction_issues,
            evidence_closure_count=evidence_closure_count,
            evidence_recovery=evidence_recovery,
            anchors=anchors,
            repair_attempted=repair_attempted,
        )
        if repair_attempted:
            usage_payload["initial_usage"] = initial_usage
            usage_payload["initial_validation_error"] = initial_validation_error
            usage_payload["initial_evidence_offset_recovery"] = initial_evidence_recovery.to_dict()
        return result, usage_payload

    @staticmethod
    def _extraction_payload(
        *,
        transcript: TranscriptEnvelope,
        cleaned: CleanerResult,
        speaker_id: str,
        speaker_name: str,
        known_people: list[KnownPerson],
        anchors: ExtractionAnchorBundle,
    ) -> dict[str, Any]:
        return {
            "recording_id": transcript.recording_id,
            "speaker": {"speaker_id": speaker_id, "speaker_name": speaker_name},
            "known_people": [person.model_dump(mode="json") for person in known_people],
            "anchor_contract": anchors.model_dump(mode="json"),
            "output_schema": ExtractionResult.model_json_schema(),
            "raw_segments": [segment.model_dump(mode="json") for segment in transcript.segments],
            "readable_segments": [
                segment.model_dump(mode="json") for segment in cleaned.readable_segments
            ],
            "detected_corrections": [
                correction.model_dump(mode="json") for correction in cleaned.detected_corrections
            ],
            "uncertain_fragments": [
                fragment.model_dump(mode="json") for fragment in cleaned.uncertain_fragments
            ],
            "full_readable_text": cleaned.full_readable_text,
        }

    def _repair_extraction(
        self,
        *,
        transcript: TranscriptEnvelope,
        cleaned: CleanerResult,
        speaker_id: str,
        speaker_name: str,
        known_people: list[KnownPerson],
        anchors: ExtractionAnchorBundle,
        invalid_output: dict[str, Any],
        validation_error: str,
    ) -> tuple[
        dict[str, Any],
        ExtractionResult,
        list[dict[str, Any]],
        int,
        EvidenceOffsetRecoveryMetrics,
        DeepSeekUsage,
    ]:
        repair_payload = {
            **self._extraction_payload(
                transcript=transcript,
                cleaned=cleaned,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                known_people=known_people,
                anchors=anchors,
            ),
            "validation_error": validation_error,
            "invalid_output": invalid_output,
        }
        repaired_raw, repair_usage = self.client.request_json(
            system_prompt=ANCHOR_CONSTRAINED_EXTRACTION_REPAIR_SYSTEM_PROMPT,
            payload=repair_payload,
            max_tokens=16_000,
            attempts=2,
        )
        recovered_raw, evidence_recovery = recover_evidence_offsets(
            raw=repaired_raw,
            transcript=transcript,
        )
        result, issues, closure_count = sanitize_extraction_output(
            raw=recovered_raw,
            transcript=transcript,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
        )
        return recovered_raw, result, issues, closure_count, evidence_recovery, repair_usage

    @staticmethod
    def _requires_extraction_repair(
        *,
        raw: dict[str, Any],
        result: ExtractionResult,
        extraction_issues: list[dict[str, Any]],
    ) -> bool:
        accepted_objects = sum(
            len(items)
            for items in (
                result.people_mentions,
                result.relationship_claims,
                result.events,
                result.descriptions,
                result.stories,
                result.unresolved_questions,
            )
        )
        if accepted_objects:
            return False
        fatal_schema_issue = any(
            issue.get("stage") == "schema"
            and issue.get("object_type") in _FATAL_EXTRACTION_COLLECTIONS
            and issue.get("object_id") is None
            for issue in extraction_issues
        )
        model_attempted_content = any(
            raw.get(key) not in (None, []) for key in _FATAL_EXTRACTION_COLLECTIONS
        )
        return fatal_schema_issue and model_attempted_content

    @staticmethod
    def _repair_reason(extraction_issues: list[dict[str, Any]]) -> str:
        details = [
            str(issue.get("detail"))
            for issue in extraction_issues
            if issue.get("stage") == "schema"
            and issue.get("object_type") in _FATAL_EXTRACTION_COLLECTIONS
        ]
        return "; ".join(details) or "fatal extraction collection schema failure"

    def _extraction_usage(
        self,
        *,
        raw: dict[str, Any],
        usage: DeepSeekUsage,
        result: ExtractionResult,
        extraction_issues: list[dict[str, Any]],
        evidence_closure_count: int,
        evidence_recovery: EvidenceOffsetRecoveryMetrics,
        anchors: ExtractionAnchorBundle,
        repair_attempted: bool,
    ) -> dict[str, Any]:
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
        return {
            **self._usage_dict(usage),
            "repair_attempted": repair_attempted,
            "evidence_closure_relationships": evidence_closure_count,
            "repaired_evidence_offsets": evidence_recovery.repaired_evidence_offsets,
            "evidence_offset_recovery": evidence_recovery.to_dict(),
            "quarantined_items": len(extraction_issues),
            "extraction_issues": extraction_issues,
            "relationship_metrics": {
                "candidates": relationship_candidates,
                "accepted": accepted_relationships,
                "quarantined": quarantined_relationships,
                "acceptance_rate": acceptance_rate,
            },
            "anchor_contract": {
                "schema_version": anchors.schema_version,
                "allowed_segments": len(anchors.allowed_segment_ids),
                "mention_anchors": len(anchors.mention_anchors),
                "lexical_annotations": len(anchors.lexical_annotations),
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
