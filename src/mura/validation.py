from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from mura.claim_model import validate_extraction_contract_v2
from mura.domain.models import (
    CleanerResult,
    CoreferenceStatus,
    CorrectionKind,
    EvidenceClass,
    ExtractionResult,
    PersonMention,
    RelationshipClaim,
    TranscriptEnvelope,
)
from mura.linguistics.corrections import has_explicit_correction_cue
from mura.relationship_evidence import (
    analyze_relationship_evidence,
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


def validate_extraction_result(transcript: TranscriptEnvelope, result: ExtractionResult) -> None:
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

    for person in result.people_mentions:
        _ensure_known_segments(person.source_segment_ids, valid_segments, person.mention_id)

    for relationship in result.relationship_claims:
        object_name = f"relationship {relationship.relationship_id}"
        _ensure_known_segments(relationship.source_segment_ids, valid_segments, object_name)
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

        evidence = analyze_relationship_evidence(
            relationship=relationship,
            transcript=transcript,
            people=result.people_mentions,
            speaker_name=result.speaker_name,
        )
        unsupported = set(evidence.unsupported_endpoint_ids)
        if unsupported:
            antecedents = _resolved_coreference_antecedents(
                relationship,
                result=result,
                valid_segments=valid_segments,
                mention_set=mention_set,
            )
            if not unsupported.issubset(antecedents):
                raise ContractValidationError(
                    f"{relationship.relationship_id} has unsupported relationship endpoints: "
                    f"{evidence.unsupported_endpoint_ids}"
                )
        if evidence.role_consistent is False:
            raise ContractValidationError(
                f"{relationship.relationship_id} contradicts deterministic multilingual "
                f"kinship evidence: {evidence.linguistic_relationship_signals}; "
                f"possessive_markers={evidence.third_person_possessive_markers}"
            )
        if (
            evidence.evidence_class == EvidenceClass.C_SPEAKER_ANCHORED.value
            and evidence.role_consistent is not True
        ):
            raise ContractValidationError(
                f"{relationship.relationship_id} uses an implicit speaker endpoint without a "
                "deterministic kinship signal"
            )

    for event in result.events:
        _ensure_known_segments(event.source_segment_ids, valid_segments, event.event_id)
        unknown = set(event.participant_mention_ids) - mention_set
        if unknown:
            raise ContractValidationError(
                f"{event.event_id} references unknown participants: {sorted(unknown)}"
            )

    for description in result.descriptions:
        object_name = f"description {description.description_id}"
        _ensure_known_segments(description.source_segment_ids, valid_segments, object_name)
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
        if explicit_people and target.mention_id not in explicit_people:
            raise ContractValidationError(
                f"{description.description_id} is assigned to a person not named in its evidence"
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

    for question in result.unresolved_questions:
        _ensure_known_segments(question.source_segment_ids, valid_segments, question.question_id)
        unknown = set(question.related_mention_ids) - mention_set
        if unknown:
            raise ContractValidationError(
                f"{question.question_id} references unknown mentions: {sorted(unknown)}"
            )

    try:
        validate_extraction_contract_v2(transcript, result)
    except ValueError as exc:
        raise ContractValidationError(f"evidence/claim v2 contract failed: {exc}") from exc
