from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from mura.claim_model import validate_extraction_contract_v2
from mura.claim_semantics import (
    date_is_invalid_calendar_value,
    date_is_silently_exactified,
    infer_relationship_state,
    relationship_semantic_text,
)
from mura.domain.models import (
    ClaimObjectType,
    CleanerResult,
    ConflictStatus,
    CoreferenceStatus,
    CorrectionKind,
    EvidenceBackedObject,
    EvidenceClass,
    ExtractionResult,
    PersonMention,
    RelationshipClaim,
    RelationshipState,
    TemporalKind,
    TranscriptEnvelope,
    VerificationStatus,
)
from mura.linguistics.corrections import has_explicit_correction_cue
from mura.relationship_evidence import (
    analyze_relationship_evidence,
    contains_surface,
    has_first_person_reference,
    person_name_surfaces,
)


class ContractValidationError(ValueError):
    pass


def _ensure_known_segments(
    source_ids: list[str], valid_segment_ids: set[str], object_name: str
) -> None:
    if not source_ids:
        raise ContractValidationError(f"{object_name} has no evidence segments")
    unknown = set(source_ids) - valid_segment_ids
    if unknown:
        raise ContractValidationError(
            f"{object_name} references unknown segments: {sorted(unknown)}"
        )


def _normalize_evidence(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.replace("_", " ").split())


def _contains_evidence(haystack: str, needle: str) -> bool:
    normalized_haystack = _normalize_evidence(haystack)
    normalized_needle = _normalize_evidence(needle)
    if not normalized_needle:
        return False
    return f" {normalized_needle} " in f" {normalized_haystack} "


def _joined_segment_text(segment_ids: Iterable[str], segment_text_by_id: dict[str, str]) -> str:
    return " ".join(segment_text_by_id[segment_id] for segment_id in segment_ids)


def _ensure_evidence_text(
    *,
    evidence_text: str,
    source_ids: list[str],
    segment_text_by_id: dict[str, str],
    object_name: str,
) -> None:
    source_text = _joined_segment_text(source_ids, segment_text_by_id)
    if not _contains_evidence(source_text, evidence_text):
        raise ContractValidationError(
            f"{object_name} evidence text is not present in its cited segments"
        )


def _ensure_unique_ids(values: list[str], object_name: str) -> None:
    if len(values) != len(set(values)):
        raise ContractValidationError(f"extractor returned duplicate {object_name} IDs")


def _explicit_people_in_segments(
    source_ids: list[str],
    segment_text_by_id: dict[str, str],
    people: list[PersonMention],
) -> set[str]:
    source_text = _joined_segment_text(source_ids, segment_text_by_id)
    explicit_people: set[str] = set()
    for person in people:
        if any(
            _contains_evidence(source_text, surface) for surface in person_name_surfaces(person)
        ):
            explicit_people.add(person.mention_id)
    return explicit_people


def _ensure_person_evidence_overlap(
    *,
    source_ids: list[str],
    person: PersonMention,
    object_name: str,
) -> None:
    if not set(source_ids).intersection(person.source_segment_ids):
        raise ContractValidationError(
            f"{object_name} has no evidence overlap with {person.mention_id}"
        )


def _resolved_coreference_antecedents(
    relationship: RelationshipClaim,
    *,
    result: ExtractionResult,
    valid_segments: set[str],
    mention_set: set[str],
) -> set[str]:
    link_by_id = {item.coreference_id: item for item in result.coreference_links}
    antecedents: set[str] = set()
    for link_id in relationship.coreference_link_ids:
        link = link_by_id.get(link_id)
        if link is None or link.status is not CoreferenceStatus.RESOLVED:
            continue
        if set(link.source_segment_ids) - valid_segments:
            continue
        if not set(link.source_segment_ids).issubset(relationship.source_segment_ids):
            continue
        if set(link.antecedent_mention_ids) - mention_set:
            continue
        antecedents.update(link.antecedent_mention_ids)
    return antecedents


def validate_cleaner_result(transcript: TranscriptEnvelope, result: CleanerResult) -> None:
    raw_ids = [segment.segment_id for segment in transcript.segments]
    cleaned_ids = [segment.segment_id for segment in result.readable_segments]

    if len(cleaned_ids) != len(set(cleaned_ids)):
        raise ContractValidationError("cleaner returned duplicate segment IDs")
    if set(raw_ids) != set(cleaned_ids):
        missing = sorted(set(raw_ids) - set(cleaned_ids))
        invented = sorted(set(cleaned_ids) - set(raw_ids))
        raise ContractValidationError(
            f"cleaner segment coverage mismatch: missing={missing}, invented={invented}"
        )

    valid_ids = set(raw_ids)
    raw_text_by_id = {segment.segment_id: segment.text for segment in transcript.segments}
    readable_text_by_id = {segment.segment_id: segment.text for segment in result.readable_segments}

    joined_readable = " ".join(readable_text_by_id[segment_id] for segment_id in raw_ids)
    if _normalize_evidence(joined_readable) != _normalize_evidence(result.full_readable_text):
        raise ContractValidationError(
            "full_readable_text does not match the ordered readable segments"
        )

    normalized_correction_sources: list[str] = []
    for correction in result.detected_corrections:
        object_name = f"detected correction {correction.original_value!r}"
        _ensure_known_segments(correction.source_segment_ids, valid_ids, object_name)
        raw_source_text = _joined_segment_text(correction.source_segment_ids, raw_text_by_id)
        if (
            correction.kind is CorrectionKind.SPEAKER_SELF_CORRECTION
            and not has_explicit_correction_cue(raw_source_text)
        ):
            raise ContractValidationError(
                f"{object_name} is marked as speaker_self_correction without an explicit cue"
            )
        _ensure_evidence_text(
            evidence_text=correction.original_value,
            source_ids=correction.source_segment_ids,
            segment_text_by_id=raw_text_by_id,
            object_name=object_name,
        )
        _ensure_evidence_text(
            evidence_text=correction.corrected_value,
            source_ids=correction.source_segment_ids,
            segment_text_by_id=readable_text_by_id,
            object_name=f"{object_name} corrected value",
        )
        normalized_correction_sources.append(_normalize_evidence(correction.original_value))

    for fragment in result.uncertain_fragments:
        object_name = f"uncertain fragment {fragment.raw_text!r}"
        _ensure_known_segments(fragment.source_segment_ids, valid_ids, object_name)
        _ensure_evidence_text(
            evidence_text=fragment.raw_text,
            source_ids=fragment.source_segment_ids,
            segment_text_by_id=raw_text_by_id,
            object_name=object_name,
        )
        normalized_fragment = _normalize_evidence(fragment.raw_text)
        if normalized_fragment in normalized_correction_sources:
            raise ContractValidationError(
                f"{object_name} cannot also be returned as a detected correction"
            )

        _ensure_evidence_text(
            evidence_text=fragment.raw_text,
            source_ids=fragment.source_segment_ids,
            segment_text_by_id=readable_text_by_id,
            object_name=f"{object_name} readable preservation",
        )


def _open_conflicted_relationship_ids(result: ExtractionResult) -> set[str]:
    return {
        reference.object_id
        for conflict in result.conflict_sets
        if conflict.status is ConflictStatus.OPEN
        for reference in conflict.claim_refs
        if reference.object_type is ClaimObjectType.RELATIONSHIP
    }


def _claim_support_text(item: object, result: ExtractionResult) -> str:
    evidence_by_id = {evidence.evidence_id: evidence for evidence in result.evidence_spans}
    evidence_ids = getattr(item, "evidence_ids", [])
    return " ".join(
        evidence_by_id[evidence_id].text
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id
    )


def _significant_tokens(value: str) -> set[str]:
    normalized = _normalize_evidence(value)
    return {token for token in normalized.split() if len(token) >= 3 or token.isdigit()}


def _ensure_claim_text_supported(
    *,
    value: str | None,
    evidence_text: str,
    object_name: str,
    field_name: str,
) -> None:
    if value is None or not value.strip():
        return
    if _contains_evidence(evidence_text, value):
        return
    value_tokens = _significant_tokens(value)
    evidence_tokens = _significant_tokens(evidence_text)
    if value_tokens and value_tokens.issubset(evidence_tokens):
        return
    raise ContractValidationError(f"{object_name} has unsupported {field_name} text")


def _has_ambiguity_signal(value: str) -> bool:
    normalized = _normalize_evidence(value)
    signals = (
        "не знаю",
        "не помню",
        "кажется",
        "возможно",
        "кто",
        "чей",
        "чья",
        "чьи",
        "неясно",
        "білмеймін",
        "есімде жоқ",
        "кім",
        "мүмкін",
        "maybe",
        "not sure",
        "who",
    )
    return any(signal in normalized for signal in signals)


_NEGATION_MARKERS = frozenset(
    {
        "не",
        "ни",
        "нет",
        "никогда",
        "емес",
        "жоқ",
        "ешқашан",
        "not",
        "never",
        "no",
    }
)
_QUESTION_SCAFFOLD = frozenset(
    {
        "кто",
        "чей",
        "чья",
        "чьи",
        "какой",
        "какая",
        "когда",
        "где",
        "почему",
        "кім",
        "қашан",
        "қайда",
        "неге",
        "who",
        "whose",
        "when",
        "where",
        "why",
        "is",
        "was",
        "ли",
    }
)


def _resolved_antecedents_for_object(item: object, result: ExtractionResult) -> set[str]:
    link_by_id = {link.coreference_id: link for link in result.coreference_links}
    resolved: set[str] = set()
    for link_id in getattr(item, "coreference_link_ids", []):
        link = link_by_id.get(link_id)
        if link is not None and link.status is CoreferenceStatus.RESOLVED:
            resolved.update(link.antecedent_mention_ids)
    return resolved


def _description_drops_negation(description: str, evidence_text: str) -> bool:
    evidence_tokens = set(_normalize_evidence(evidence_text).split())
    description_tokens = set(_normalize_evidence(description).split())
    return bool(evidence_tokens.intersection(_NEGATION_MARKERS)) and not bool(
        description_tokens.intersection(_NEGATION_MARKERS)
    )


def _ensure_question_text_supported(question: str, evidence_text: str, object_name: str) -> None:
    question_tokens = _significant_tokens(question) - _QUESTION_SCAFFOLD
    evidence_tokens = _significant_tokens(evidence_text)
    if question_tokens and not question_tokens.issubset(evidence_tokens):
        raise ContractValidationError(f"{object_name} question adds unsupported facts")


def _validate_claim_uncertainty(
    item: EvidenceBackedObject,
    *,
    result: ExtractionResult,
    valid_segments: set[str],
    object_name: str,
) -> None:
    uncertainty = item.uncertainty
    if uncertainty is None:
        return
    if set(uncertainty.source_segment_ids) - valid_segments:
        raise ContractValidationError(f"{object_name} uncertainty references unknown segments")
    if not set(uncertainty.source_segment_ids).issubset(item.source_segment_ids):
        raise ContractValidationError(f"{object_name} uncertainty is outside claim scope")
    if set(uncertainty.evidence_ids) - set(item.evidence_ids):
        raise ContractValidationError(f"{object_name} uncertainty references unrelated evidence")
    evidence_by_id = {evidence.evidence_id: evidence for evidence in result.evidence_spans}
    cited_text = " ".join(
        evidence_by_id[evidence_id].text
        for evidence_id in uncertainty.evidence_ids
        if evidence_id in evidence_by_id
    )
    if uncertainty.markers and cited_text:
        if any(not _contains_evidence(cited_text, marker) for marker in uncertainty.markers):
            raise ContractValidationError(f"{object_name} uncertainty marker is unsupported")
    assertion_mode = getattr(item, "assertion_mode", None)
    if assertion_mode is not None and assertion_mode.value != "uncertain":
        raise ContractValidationError(
            f"{object_name} uncertainty is not reflected in assertion mode"
        )
    if item.provenance is not None and item.evidence_class is not EvidenceClass.U_UNCERTAIN:
        raise ContractValidationError(f"{object_name} uncertainty is auto-materializable")


def validate_extraction_result(
    transcript: TranscriptEnvelope,
    result: ExtractionResult,
    *,
    cleaned: CleanerResult | None = None,
) -> None:
    valid_segments = {segment.segment_id for segment in transcript.segments}
    segment_text_by_id = {segment.segment_id: segment.text for segment in transcript.segments}
    mention_ids = [person.mention_id for person in result.people_mentions]
    relationship_ids = [item.relationship_id for item in result.relationship_claims]
    event_ids = [event.event_id for event in result.events]
    description_ids = [item.description_id for item in result.descriptions]
    story_ids = [story.story_id for story in result.stories]
    question_ids = [item.question_id for item in result.unresolved_questions]

    _ensure_unique_ids(mention_ids, "mention")
    _ensure_unique_ids(relationship_ids, "relationship")
    _ensure_unique_ids(event_ids, "event")
    _ensure_unique_ids(description_ids, "description")
    _ensure_unique_ids(story_ids, "story")
    _ensure_unique_ids(question_ids, "question")

    mention_by_id = {person.mention_id: person for person in result.people_mentions}
    mention_set = set(mention_ids)
    event_set = set(event_ids)
    open_conflicted_relationship_ids = _open_conflicted_relationship_ids(result)

    for person in result.people_mentions:
        _ensure_known_segments(person.source_segment_ids, valid_segments, person.mention_id)
        if person.verification_status is not VerificationStatus.UNREVIEWED:
            raise ContractValidationError(f"{person.mention_id} is not unreviewed")
        _validate_claim_uncertainty(
            person,
            result=result,
            valid_segments=valid_segments,
            object_name=person.mention_id,
        )
        source_text = _joined_segment_text(person.source_segment_ids, segment_text_by_id)
        named = any(
            contains_surface(source_text, surface) for surface in person_name_surfaces(person)
        )
        speaker_anchored = _normalize_evidence(person.name) == _normalize_evidence(
            result.speaker_name
        ) and has_first_person_reference(source_text)
        if not named and not speaker_anchored:
            raise ContractValidationError(
                f"{person.mention_id} name is not supported by cited evidence"
            )

    for relationship in result.relationship_claims:
        object_name = f"relationship {relationship.relationship_id}"
        if relationship.verification_status is not VerificationStatus.UNREVIEWED:
            raise ContractValidationError(f"{relationship.relationship_id} is not unreviewed")
        _ensure_known_segments(relationship.source_segment_ids, valid_segments, object_name)
        _validate_claim_uncertainty(
            relationship,
            result=result,
            valid_segments=valid_segments,
            object_name=object_name,
        )
        relationship_text = relationship_semantic_text(
            relationship,
            evidence_spans=result.evidence_spans,
            people=result.people_mentions,
            fallback_text=_joined_segment_text(
                relationship.source_segment_ids,
                segment_text_by_id,
            ),
        )
        inferred_state = infer_relationship_state(relationship_text)
        if (
            inferred_state is not RelationshipState.CURRENT
            and relationship.relationship_state is RelationshipState.CURRENT
        ):
            raise ContractValidationError(
                f"{relationship.relationship_id} loses historical or negative relationship state"
            )
        if relationship.relationship_state is not RelationshipState.CURRENT:
            if (
                relationship.provenance is not None
                and relationship.evidence_class is not EvidenceClass.U_UNCERTAIN
            ):
                raise ContractValidationError(
                    f"{relationship.relationship_id} non-current state is auto-materializable"
                )
            if set(relationship.state_evidence_ids) - set(relationship.evidence_ids):
                raise ContractValidationError(
                    f"{relationship.relationship_id} state evidence is outside "
                    "relationship evidence"
                )
        if relationship.subject_mention_id not in mention_set:
            raise ContractValidationError(
                f"{relationship.relationship_id} has unknown subject mention"
            )
        if relationship.object_mention_id not in mention_set:
            raise ContractValidationError(
                f"{relationship.relationship_id} has unknown object mention"
            )

        subject = mention_by_id[relationship.subject_mention_id]
        object_person = mention_by_id[relationship.object_mention_id]
        _ensure_person_evidence_overlap(
            source_ids=relationship.source_segment_ids,
            person=subject,
            object_name=object_name,
        )
        _ensure_person_evidence_overlap(
            source_ids=relationship.source_segment_ids,
            person=object_person,
            object_name=object_name,
        )

        antecedents = _resolved_coreference_antecedents(
            relationship,
            result=result,
            valid_segments=valid_segments,
            mention_set=mention_set,
        )
        evidence = analyze_relationship_evidence(
            relationship=relationship,
            transcript=transcript,
            people=result.people_mentions,
            speaker_name=result.speaker_name,
            resolved_coreference_antecedent_ids=antecedents,
        )
        unsupported = set(evidence.unsupported_endpoint_ids)
        if unsupported and not unsupported.issubset(antecedents):
            raise ContractValidationError(
                f"{relationship.relationship_id} has unsupported relationship endpoints: "
                f"{evidence.unsupported_endpoint_ids}"
            )
        preserve_open_conflict = (
            relationship.relationship_id in open_conflicted_relationship_ids
            and evidence.grounding_decision == "insufficient_deterministic_signal"
        )
        if (
            relationship.relationship_state is RelationshipState.CURRENT
            and evidence.role_consistent is False
            and not preserve_open_conflict
        ):
            raise ContractValidationError(
                f"{relationship.relationship_id} contradicts deterministic multilingual "
                f"kinship evidence: {evidence.linguistic_relationship_signals}; "
                f"possessive_markers={evidence.third_person_possessive_markers}"
            )
        if (
            relationship.relationship_state is RelationshipState.CURRENT
            and evidence.evidence_class == EvidenceClass.C_SPEAKER_ANCHORED.value
            and evidence.role_consistent is not True
        ):
            raise ContractValidationError(
                f"{relationship.relationship_id} uses an implicit speaker endpoint without a "
                "deterministic kinship signal"
            )

    for event in result.events:
        _ensure_known_segments(event.source_segment_ids, valid_segments, event.event_id)
        if event.verification_status is not VerificationStatus.UNREVIEWED:
            raise ContractValidationError(f"{event.event_id} is not unreviewed")
        _validate_claim_uncertainty(
            event,
            result=result,
            valid_segments=valid_segments,
            object_name=event.event_id,
        )
        unknown = set(event.participant_mention_ids) - mention_set
        if unknown:
            raise ContractValidationError(
                f"{event.event_id} references unknown participants: {sorted(unknown)}"
            )
        evidence_text = _claim_support_text(event, result)
        if not evidence_text:
            raise ContractValidationError(f"{event.event_id} has no claim evidence")
        _ensure_claim_text_supported(
            value=event.title,
            evidence_text=evidence_text,
            object_name=event.event_id,
            field_name="title",
        )
        _ensure_claim_text_supported(
            value=event.description,
            evidence_text=evidence_text,
            object_name=event.event_id,
            field_name="description",
        )
        _ensure_claim_text_supported(
            value=event.location,
            evidence_text=evidence_text,
            object_name=event.event_id,
            field_name="location",
        )
        if event.date is not None:
            if event.date.verification_status is not VerificationStatus.UNREVIEWED:
                raise ContractValidationError(f"{event.event_id} temporal value is not unreviewed")
            if not event.date.original_expression:
                raise ContractValidationError(
                    f"{event.event_id} temporal value lost original expression"
                )
            _ensure_claim_text_supported(
                value=event.date.original_expression,
                evidence_text=evidence_text,
                object_name=event.event_id,
                field_name="date",
            )
            if set(event.date.source_evidence_ids) - set(event.evidence_ids):
                raise ContractValidationError(
                    f"{event.event_id} temporal evidence is outside event evidence"
                )
            if date_is_silently_exactified(event.date):
                raise ContractValidationError(
                    f"{event.event_id} silently exactifies an approximate date"
                )
            if date_is_invalid_calendar_value(event.date):
                raise ContractValidationError(f"{event.event_id} contains an invalid calendar date")
            if (
                event.date.kind is TemporalKind.RELATIVE
                and event.date.normalized_value is not None
                and event.date.anchor_event_id is None
            ):
                raise ContractValidationError(
                    f"{event.event_id} resolves a relative date without an anchor"
                )
        explicit_people = _explicit_people_in_segments(
            event.source_segment_ids, segment_text_by_id, result.people_mentions
        )
        resolved_people = _resolved_antecedents_for_object(event, result)
        grounded_people = explicit_people.union(resolved_people)
        if event.participant_mention_ids and not set(event.participant_mention_ids).issubset(
            grounded_people
        ):
            raise ContractValidationError(
                f"{event.event_id} participant is not grounded in event evidence"
            )
        if explicit_people and not event.participant_mention_ids:
            raise ContractValidationError(
                f"{event.event_id} omits a grounded participant from person-specific evidence"
            )

    for description in result.descriptions:
        object_name = f"description {description.description_id}"
        _ensure_known_segments(description.source_segment_ids, valid_segments, object_name)
        _validate_claim_uncertainty(
            description,
            result=result,
            valid_segments=valid_segments,
            object_name=object_name,
        )
        if description.person_mention_id not in mention_set:
            raise ContractValidationError(
                f"{description.description_id} references an unknown person"
            )

        target = mention_by_id[description.person_mention_id]
        _ensure_person_evidence_overlap(
            source_ids=description.source_segment_ids,
            person=target,
            object_name=object_name,
        )
        explicit_people = _explicit_people_in_segments(
            description.source_segment_ids,
            segment_text_by_id,
            result.people_mentions,
        )
        resolved_people = _resolved_antecedents_for_object(description, result)
        if target.mention_id not in explicit_people.union(resolved_people):
            raise ContractValidationError(
                f"{description.description_id} is assigned to a person not grounded in its evidence"
            )
        if description.verification_status is not VerificationStatus.UNREVIEWED:
            raise ContractValidationError(f"{description.description_id} is not unreviewed")
        evidence_text = _claim_support_text(description, result)
        if not evidence_text:
            raise ContractValidationError(f"{description.description_id} has no claim evidence")
        _ensure_claim_text_supported(
            value=description.description,
            evidence_text=evidence_text,
            object_name=description.description_id,
            field_name="description",
        )
        if _description_drops_negation(description.description, evidence_text):
            raise ContractValidationError(f"{description.description_id} drops source negation")
        if _normalize_evidence(description.perspective) != _normalize_evidence(
            result.speaker_name
        ) and not _contains_evidence(evidence_text, description.perspective):
            raise ContractValidationError(
                f"{description.description_id} has unsupported perspective"
            )

    for story in result.stories:
        _ensure_known_segments(story.source_segment_ids, valid_segments, story.story_id)
        unknown_people = set(story.person_mention_ids) - mention_set
        unknown_events = set(story.event_ids) - event_set
        if unknown_people or unknown_events:
            raise ContractValidationError(
                f"{story.story_id} has broken references: "
                f"people={sorted(unknown_people)}, events={sorted(unknown_events)}"
            )
        if story.privacy.value != "private":
            raise ContractValidationError(f"{story.story_id} privacy is not private")
        evidence_text = _claim_support_text(story, result)
        if not evidence_text:
            raise ContractValidationError(f"{story.story_id} has no claim evidence")
        _ensure_claim_text_supported(
            value=story.title,
            evidence_text=evidence_text,
            object_name=story.story_id,
            field_name="title",
        )
        _ensure_claim_text_supported(
            value=story.summary,
            evidence_text=evidence_text,
            object_name=story.story_id,
            field_name="summary",
        )
        explicit_story_people = _explicit_people_in_segments(
            story.source_segment_ids, segment_text_by_id, result.people_mentions
        )
        resolved_story_people = _resolved_antecedents_for_object(story, result)
        for mention_id in story.person_mention_ids:
            if not set(story.source_segment_ids).intersection(
                mention_by_id[mention_id].source_segment_ids
            ):
                raise ContractValidationError(
                    f"{story.story_id} person reference is outside story evidence"
                )
            if mention_id not in explicit_story_people.union(resolved_story_people):
                raise ContractValidationError(
                    f"{story.story_id} person reference is not grounded in the episode"
                )
        event_by_id = {event.event_id: event for event in result.events}
        for event_id in story.event_ids:
            if not set(story.source_segment_ids).intersection(
                event_by_id[event_id].source_segment_ids
            ):
                raise ContractValidationError(
                    f"{story.story_id} event reference is outside story evidence"
                )

    for question in result.unresolved_questions:
        _ensure_known_segments(question.source_segment_ids, valid_segments, question.question_id)
        unknown = set(question.related_mention_ids) - mention_set
        if unknown:
            raise ContractValidationError(
                f"{question.question_id} references unknown mentions: {sorted(unknown)}"
            )
        evidence_text = _claim_support_text(question, result)
        if not evidence_text:
            raise ContractValidationError(f"{question.question_id} has no ambiguity evidence")
        if not _has_ambiguity_signal(evidence_text) and len(question.related_mention_ids) < 2:
            raise ContractValidationError(
                f"{question.question_id} is not grounded in a real ambiguity"
            )
        _ensure_question_text_supported(question.question, evidence_text, question.question_id)

    try:
        validate_extraction_contract_v2(transcript, result, cleaned=cleaned)
    except ValueError as exc:
        raise ContractValidationError(f"evidence/claim v2 contract failed: {exc}") from exc
