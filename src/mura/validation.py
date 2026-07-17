from __future__ import annotations

from mura.domain.models import CleanerResult, ExtractionResult, TranscriptEnvelope


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


def validate_cleaner_result(
    transcript: TranscriptEnvelope, result: CleanerResult
) -> None:
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
    for correction in result.detected_corrections:
        _ensure_known_segments(
            correction.source_segment_ids, valid_ids, "detected correction"
        )
    for fragment in result.uncertain_fragments:
        _ensure_known_segments(fragment.source_segment_ids, valid_ids, "uncertain fragment")


def validate_extraction_result(
    transcript: TranscriptEnvelope, result: ExtractionResult
) -> None:
    valid_segments = {segment.segment_id for segment in transcript.segments}
    mention_ids = [person.mention_id for person in result.people_mentions]
    event_ids = [event.event_id for event in result.events]

    if len(mention_ids) != len(set(mention_ids)):
        raise ContractValidationError("extractor returned duplicate mention IDs")
    if len(event_ids) != len(set(event_ids)):
        raise ContractValidationError("extractor returned duplicate event IDs")

    mention_set = set(mention_ids)
    event_set = set(event_ids)

    for person in result.people_mentions:
        _ensure_known_segments(person.source_segment_ids, valid_segments, person.mention_id)

    for relationship in result.relationship_claims:
        _ensure_known_segments(
            relationship.source_segment_ids, valid_segments, relationship.relationship_id
        )
        if relationship.subject_mention_id not in mention_set:
            raise ContractValidationError(
                f"{relationship.relationship_id} has unknown subject mention"
            )
        if relationship.object_mention_id not in mention_set:
            raise ContractValidationError(
                f"{relationship.relationship_id} has unknown object mention"
            )
        if relationship.subject_mention_id == relationship.object_mention_id:
            raise ContractValidationError(
                f"{relationship.relationship_id} creates a self relationship"
            )

    for event in result.events:
        _ensure_known_segments(event.source_segment_ids, valid_segments, event.event_id)
        unknown = set(event.participant_mention_ids) - mention_set
        if unknown:
            raise ContractValidationError(
                f"{event.event_id} references unknown participants: {sorted(unknown)}"
            )

    for description in result.descriptions:
        _ensure_known_segments(
            description.source_segment_ids, valid_segments, description.description_id
        )
        if description.person_mention_id not in mention_set:
            raise ContractValidationError(
                f"{description.description_id} references an unknown person"
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
        _ensure_known_segments(
            question.source_segment_ids, valid_segments, question.question_id
        )
        unknown = set(question.related_mention_ids) - mention_set
        if unknown:
            raise ContractValidationError(
                f"{question.question_id} references unknown mentions: {sorted(unknown)}"
            )
