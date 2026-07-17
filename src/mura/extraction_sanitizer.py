from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from mura.domain.models import (
    ExtractionResult,
    FamilyEvent,
    PersonDescription,
    PersonMention,
    RelationshipClaim,
    Story,
    TranscriptEnvelope,
    UnresolvedQuestion,
)
from mura.evidence import complete_relationship_evidence
from mura.validation import ContractValidationError, validate_extraction_result

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ExtractionIssue:
    object_type: str
    object_id: str | None
    stage: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _object_id(raw: object, field_name: str) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get(field_name)
    return value if isinstance(value, str) and value else None


def _list_value(
    raw: dict[str, Any], key: str, issues: list[ExtractionIssue]
) -> list[object]:
    value = raw.get(key, [])
    if isinstance(value, list):
        return value
    issues.append(
        ExtractionIssue(
            object_type=key,
            object_id=None,
            stage="schema",
            detail=f"top-level field {key!r} must be a list; received {type(value).__name__}",
        )
    )
    return []


def _parse_items(
    *,
    raw_items: list[object],
    model_type: type[ModelT],
    object_type: str,
    id_field: str,
    issues: list[ExtractionIssue],
) -> list[ModelT]:
    parsed: list[ModelT] = []
    seen_ids: set[str] = set()

    for raw_item in raw_items:
        object_id = _object_id(raw_item, id_field)
        try:
            item = model_type.model_validate(raw_item)
        except ValidationError as exc:
            issues.append(
                ExtractionIssue(
                    object_type=object_type,
                    object_id=object_id,
                    stage="schema",
                    detail=str(exc),
                )
            )
            continue

        resolved_id = str(getattr(item, id_field))
        if resolved_id in seen_ids:
            issues.append(
                ExtractionIssue(
                    object_type=object_type,
                    object_id=resolved_id,
                    stage="schema",
                    detail=f"duplicate {id_field}",
                )
            )
            continue

        seen_ids.add(resolved_id)
        parsed.append(item)

    return parsed


def _base_result(
    *,
    recording_id: str,
    speaker_id: str,
    speaker_name: str,
    languages: list[str],
    people: list[PersonMention] | None = None,
    relationships: list[RelationshipClaim] | None = None,
    events: list[FamilyEvent] | None = None,
    descriptions: list[PersonDescription] | None = None,
    stories: list[Story] | None = None,
    questions: list[UnresolvedQuestion] | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        recording_id=recording_id,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        languages=languages,
        people_mentions=people or [],
        relationship_claims=relationships or [],
        events=events or [],
        descriptions=descriptions or [],
        stories=stories or [],
        unresolved_questions=questions or [],
    )


def _semantic_filter(
    *,
    items: list[ModelT],
    object_type: str,
    id_field: str,
    build_candidate: Callable[[ModelT], ExtractionResult],
    transcript: TranscriptEnvelope,
    issues: list[ExtractionIssue],
) -> list[ModelT]:
    accepted: list[ModelT] = []
    for item in items:
        object_id = str(getattr(item, id_field))
        try:
            validate_extraction_result(transcript, build_candidate(item))
        except ContractValidationError as exc:
            issues.append(
                ExtractionIssue(
                    object_type=object_type,
                    object_id=object_id,
                    stage="semantic",
                    detail=str(exc),
                )
            )
            continue
        accepted.append(item)
    return accepted


def sanitize_extraction_output(
    *,
    raw: dict[str, Any],
    transcript: TranscriptEnvelope,
    speaker_id: str,
    speaker_name: str,
) -> tuple[ExtractionResult, list[dict[str, Any]], int]:
    """Return valid unreviewed claims while quarantining malformed individual objects."""
    issues: list[ExtractionIssue] = []

    for key, expected in (
        ("recording_id", transcript.recording_id),
        ("speaker_id", speaker_id),
        ("speaker_name", speaker_name),
    ):
        actual = raw.get(key)
        if actual != expected:
            issues.append(
                ExtractionIssue(
                    object_type="metadata",
                    object_id=key,
                    stage="schema",
                    detail=f"model returned {actual!r}; authoritative value {expected!r} was used",
                )
            )

    raw_languages = raw.get("languages", [])
    if isinstance(raw_languages, list) and all(
        isinstance(item, str) for item in raw_languages
    ):
        languages = list(dict.fromkeys(raw_languages))
    else:
        languages = []
        issues.append(
            ExtractionIssue(
                object_type="metadata",
                object_id="languages",
                stage="schema",
                detail="languages must be a list of strings",
            )
        )

    people = _parse_items(
        raw_items=_list_value(raw, "people_mentions", issues),
        model_type=PersonMention,
        object_type="person",
        id_field="mention_id",
        issues=issues,
    )
    relationships = _parse_items(
        raw_items=_list_value(raw, "relationship_claims", issues),
        model_type=RelationshipClaim,
        object_type="relationship",
        id_field="relationship_id",
        issues=issues,
    )
    events = _parse_items(
        raw_items=_list_value(raw, "events", issues),
        model_type=FamilyEvent,
        object_type="event",
        id_field="event_id",
        issues=issues,
    )
    descriptions = _parse_items(
        raw_items=_list_value(raw, "descriptions", issues),
        model_type=PersonDescription,
        object_type="description",
        id_field="description_id",
        issues=issues,
    )
    stories = _parse_items(
        raw_items=_list_value(raw, "stories", issues),
        model_type=Story,
        object_type="story",
        id_field="story_id",
        issues=issues,
    )
    questions = _parse_items(
        raw_items=_list_value(raw, "unresolved_questions", issues),
        model_type=UnresolvedQuestion,
        object_type="question",
        id_field="question_id",
        issues=issues,
    )

    valid_people = _semantic_filter(
        items=people,
        object_type="person",
        id_field="mention_id",
        build_candidate=lambda item: _base_result(
            recording_id=transcript.recording_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            languages=languages,
            people=[item],
        ),
        transcript=transcript,
        issues=issues,
    )

    preliminary = _base_result(
        recording_id=transcript.recording_id,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        languages=languages,
        people=valid_people,
        relationships=relationships,
        events=events,
        descriptions=descriptions,
        stories=stories,
        questions=questions,
    )
    preliminary, evidence_closure_count = complete_relationship_evidence(
        preliminary, transcript
    )
    relationships = preliminary.relationship_claims

    common = {
        "recording_id": transcript.recording_id,
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "languages": languages,
    }
    valid_relationships = _semantic_filter(
        items=relationships,
        object_type="relationship",
        id_field="relationship_id",
        build_candidate=lambda item: _base_result(
            **common, people=valid_people, relationships=[item]
        ),
        transcript=transcript,
        issues=issues,
    )
    valid_events = _semantic_filter(
        items=events,
        object_type="event",
        id_field="event_id",
        build_candidate=lambda item: _base_result(
            **common, people=valid_people, events=[item]
        ),
        transcript=transcript,
        issues=issues,
    )
    valid_descriptions = _semantic_filter(
        items=descriptions,
        object_type="description",
        id_field="description_id",
        build_candidate=lambda item: _base_result(
            **common, people=valid_people, descriptions=[item]
        ),
        transcript=transcript,
        issues=issues,
    )
    valid_stories = _semantic_filter(
        items=stories,
        object_type="story",
        id_field="story_id",
        build_candidate=lambda item: _base_result(
            **common, people=valid_people, events=valid_events, stories=[item]
        ),
        transcript=transcript,
        issues=issues,
    )
    valid_questions = _semantic_filter(
        items=questions,
        object_type="question",
        id_field="question_id",
        build_candidate=lambda item: _base_result(
            **common, people=valid_people, questions=[item]
        ),
        transcript=transcript,
        issues=issues,
    )

    result = _base_result(
        **common,
        people=valid_people,
        relationships=valid_relationships,
        events=valid_events,
        descriptions=valid_descriptions,
        stories=valid_stories,
        questions=valid_questions,
    )
    validate_extraction_result(transcript, result)
    return result, [issue.to_dict() for issue in issues], evidence_closure_count
