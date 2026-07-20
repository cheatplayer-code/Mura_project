from __future__ import annotations

from typing import Any

from mura.domain.models import RawSegment, TranscriptEnvelope
from mura.extraction_sanitizer import sanitize_extraction_output


def transcript(*texts: str) -> TranscriptEnvelope:
    segments = [
        RawSegment(
            segment_id=f"seg_{index:03d}",
            start=float((index - 1) * 10),
            end=float(index * 10),
            text=text,
        )
        for index, text in enumerate(texts, start=1)
    ]
    return TranscriptEnvelope(
        recording_id="rec_coreference_v2",
        duration_seconds=float(len(segments) * 10),
        language_hints=["ru", "kk"],
        full_text=" ".join(texts),
        segments=segments,
        asr_model="fixture",
        asr_revision="v1",
        chunker_version="v1",
    )


def person(
    mention_id: str,
    name: str,
    *segment_ids: str,
    relation_to_speaker: str | None = None,
) -> dict[str, Any]:
    return {
        "mention_id": mention_id,
        "name": name,
        "category": "family_member",
        "relation_to_speaker": relation_to_speaker,
        "source_segment_ids": list(segment_ids),
        "confidence": 1.0,
    }


def relationship(
    relationship_id: str,
    relationship_type: str,
    subject_id: str,
    subject_role: str,
    object_id: str,
    object_role: str,
    *segment_ids: str,
) -> dict[str, Any]:
    return {
        "relationship_id": relationship_id,
        "relationship_type": relationship_type,
        "subject_mention_id": subject_id,
        "subject_role": subject_role,
        "object_mention_id": object_id,
        "object_role": object_role,
        "source_segment_ids": list(segment_ids),
        "confidence": 1.0,
    }


def raw(
    *,
    people: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "recording_id": "rec_coreference_v2",
        "speaker_id": "speaker_1",
        "speaker_name": "Narrator",
        "languages": ["ru", "kk"],
        "people_mentions": people,
        "relationship_claims": relationships or [],
        "events": [],
        "descriptions": [],
        "stories": [],
        "unresolved_questions": [],
        "coreference_links": [],
        "conflict_sets": [],
        "evidence_spans": [],
        "provenance_activities": [],
    }


def sanitize(
    transcript_value: TranscriptEnvelope,
    *,
    people: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
):
    return sanitize_extraction_output(
        raw=raw(people=people, relationships=relationships),
        transcript=transcript_value,
        speaker_id="speaker_1",
        speaker_name="Narrator",
    )
