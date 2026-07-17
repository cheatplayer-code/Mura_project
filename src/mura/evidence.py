from __future__ import annotations

from mura.domain.models import ExtractionResult, RelationshipClaim, TranscriptEnvelope


def _ordered_unique_segment_ids(
    segment_ids: list[str], transcript: TranscriptEnvelope
) -> list[str]:
    """Return unique IDs in transcript order, preserving unknown IDs at the end."""
    requested = set(segment_ids)
    ordered = [
        segment.segment_id
        for segment in transcript.segments
        if segment.segment_id in requested
    ]

    seen = set(ordered)
    ordered.extend(
        segment_id
        for segment_id in segment_ids
        if segment_id not in seen and not seen.add(segment_id)
    )
    return ordered


def _complete_relationship(
    relationship: RelationshipClaim,
    *,
    mention_sources: dict[str, list[str]],
    transcript: TranscriptEnvelope,
) -> tuple[RelationshipClaim, bool]:
    source_ids = list(relationship.source_segment_ids)
    source_set = set(source_ids)
    changed = False

    for mention_id in (
        relationship.subject_mention_id,
        relationship.object_mention_id,
    ):
        person_source_ids = mention_sources.get(mention_id, [])
        if source_set.intersection(person_source_ids):
            continue

        source_ids.extend(person_source_ids)
        source_set.update(person_source_ids)
        changed = changed or bool(person_source_ids)

    if not changed:
        return relationship, False

    return relationship.model_copy(
        update={
            "source_segment_ids": _ordered_unique_segment_ids(source_ids, transcript)
        }
    ), True


def complete_relationship_evidence(
    result: ExtractionResult,
    transcript: TranscriptEnvelope,
) -> tuple[ExtractionResult, int]:
    """Add missing endpoint identity evidence without changing relationship semantics.

    DeepSeek sometimes cites the segment containing the kinship statement but omits the
    earlier segment that establishes a pronoun's canonical person mention. The endpoint
    mention already carries that source evidence, so this function closes the evidence
    bundle deterministically before strict semantic validation.

    It never changes endpoint IDs, roles, relationship type, confidence, or status.
    Unknown endpoint IDs are left untouched for the validator to reject.
    """
    mention_sources = {
        person.mention_id: person.source_segment_ids for person in result.people_mentions
    }
    completed: list[RelationshipClaim] = []
    changed_count = 0

    for relationship in result.relationship_claims:
        updated, changed = _complete_relationship(
            relationship,
            mention_sources=mention_sources,
            transcript=transcript,
        )
        completed.append(updated)
        changed_count += int(changed)

    if not changed_count:
        return result, 0

    return result.model_copy(update={"relationship_claims": completed}), changed_count
