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
from mura.relationship_evidence import analyze_relationship_evidence
from mura.validation import ContractValidationError, validate_extraction_result

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ExtractionIssue:
    object_type: str
    object_id: str | None
    stage: str
    detail: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.context is None:
            result.pop("context")
        return result


def _object_id(raw: object, field_name: str) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get(field_name)
    return value if isinstance(value, str) and value else None


def _list_value(
    raw: dict[str, Any],
    key: str,
    issues: list[ExtractionIssue],
) -> list[object]:
    value = raw.get(key, [])
    if isinstance(value, list):
        return value
    issues.append(
        ExtractionIssue(
            object_type=key,
            object_id=None,
            stage="schema",
            detail=(f"top-level field {key!r} must be a list; received {type(value).__name__}"),
            context={"received_value": value},
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
                    context={"candidate": raw_item},
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
                    context={"candidate": item.model_dump(mode="json")},
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
    issue_context: Callable[[ModelT], dict[str, Any]] | None = None,
) -> list[ModelT]:
    accepted: list[ModelT] = []
    for item in items:
        object_id = str(getattr(item, id_field))
        try:
            validate_extraction_result(transcript, build_candidate(item))
        except ContractValidationError as exc:
            context: dict[str, Any] = {"candidate": item.model_dump(mode="json")}
            if issue_context is not None:
                context.update(issue_context(item))
            issues.append(
                ExtractionIssue(
                    object_type=object_type,
                    object_id=object_id,
                    stage="semantic",
                    detail=str(exc),
                    context=context,
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
                    detail=(
                        f"model returned {actual!r}; authoritative value {expected!r} was used"
                    ),
                    context={"model_value": actual, "authoritative_value": expected},
                )
            )

    raw_languages = raw.get("languages", [])
    if isinstance(raw_languages, list) and all(isinstance(item, str) for item in raw_languages):
        languages = list(dict.fromkeys(raw_languages))
    else:
        languages = []
        issues.append(
            ExtractionIssue(
                object_type="metadata",
                object_id="languages",
                stage="schema",
                detail="languages must be a list of strings",
                context={"received_value": raw_languages},
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
    original_relationships = {
        item.relationship_id: item.model_dump(mode="json") for item in relationships
    }
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

    def build_result(
        *,
        selected_people: list[PersonMention] | None = None,
        selected_relationships: list[RelationshipClaim] | None = None,
        selected_events: list[FamilyEvent] | None = None,
        selected_descriptions: list[PersonDescription] | None = None,
        selected_stories: list[Story] | None = None,
        selected_questions: list[UnresolvedQuestion] | None = None,
    ) -> ExtractionResult:
        return _base_result(
            recording_id=transcript.recording_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            languages=languages,
            people=selected_people,
            relationships=selected_relationships,
            events=selected_events,
            descriptions=selected_descriptions,
            stories=selected_stories,
            questions=selected_questions,
        )

    valid_people = _semantic_filter(
        items=people,
        object_type="person",
        id_field="mention_id",
        build_candidate=lambda item: build_result(selected_people=[item]),
        transcript=transcript,
        issues=issues,
    )

    preliminary = build_result(
        selected_people=valid_people,
        selected_relationships=relationships,
        selected_events=events,
        selected_descriptions=descriptions,
        selected_stories=stories,
        selected_questions=questions,
    )
    preliminary, evidence_closure_count = complete_relationship_evidence(
        preliminary,
        transcript,
    )
    relationships = preliminary.relationship_claims

    def relationship_issue_context(item: RelationshipClaim) -> dict[str, Any]:
        analysis = analyze_relationship_evidence(
            relationship=item,
            transcript=transcript,
            people=valid_people,
            speaker_name=speaker_name,
        )
        return {
            "original_candidate": original_relationships.get(item.relationship_id),
            "evidence_analysis": analysis.to_dict(),
        }

    valid_relationships = _semantic_filter(
        items=relationships,
        object_type="relationship",
        id_field="relationship_id",
        build_candidate=lambda item: build_result(
            selected_people=valid_people,
            selected_relationships=[item],
        ),
        transcript=transcript,
        issues=issues,
        issue_context=relationship_issue_context,
    )
    valid_events = _semantic_filter(
        items=events,
        object_type="event",
        id_field="event_id",
        build_candidate=lambda item: build_result(
            selected_people=valid_people,
            selected_events=[item],
        ),
        transcript=transcript,
        issues=issues,
    )
    valid_descriptions = _semantic_filter(
        items=descriptions,
        object_type="description",
        id_field="description_id",
        build_candidate=lambda item: build_result(
            selected_people=valid_people,
            selected_descriptions=[item],
        ),
        transcript=transcript,
        issues=issues,
    )
    valid_stories = _semantic_filter(
        items=stories,
        object_type="story",
        id_field="story_id",
        build_candidate=lambda item: build_result(
            selected_people=valid_people,
            selected_events=valid_events,
            selected_stories=[item],
        ),
        transcript=transcript,
        issues=issues,
    )
    valid_questions = _semantic_filter(
        items=questions,
        object_type="question",
        id_field="question_id",
        build_candidate=lambda item: build_result(
            selected_people=valid_people,
            selected_questions=[item],
        ),
        transcript=transcript,
        issues=issues,
    )

    result = build_result(
        selected_people=valid_people,
        selected_relationships=valid_relationships,
        selected_events=valid_events,
        selected_descriptions=valid_descriptions,
        selected_stories=valid_stories,
        selected_questions=valid_questions,
    )
    validate_extraction_result(transcript, result)
    return result, [issue.to_dict() for issue in issues], evidence_closure_count
