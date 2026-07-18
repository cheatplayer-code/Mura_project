from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from mura.domain.models import PersonMention, RelationshipClaim, TranscriptEnvelope

_FIRST_PERSON_TOKENS = {
    "я",
    "мы",
    "мой",
    "моя",
    "моё",
    "мои",
    "моего",
    "моей",
    "наш",
    "наша",
    "наше",
    "наши",
    "нашего",
    "нашей",
    "мен",
    "менің",
    "біз",
    "біздің",
}


def normalize_evidence(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.replace("_", " ").split())


def contains_surface(text: str, surface: str) -> bool:
    normalized_text = normalize_evidence(text)
    normalized_surface = normalize_evidence(surface)
    if not normalized_surface:
        return False
    return f" {normalized_surface} " in f" {normalized_text} "


def has_first_person_reference(text: str) -> bool:
    tokens = set(normalize_evidence(text).split())
    return bool(tokens.intersection(_FIRST_PERSON_TOKENS))


def joined_segment_text(segment_ids: list[str], transcript: TranscriptEnvelope) -> str:
    text_by_id = {segment.segment_id: segment.text for segment in transcript.segments}
    return " ".join(text_by_id[segment_id] for segment_id in segment_ids if segment_id in text_by_id)


def explicitly_named_people(
    source_text: str,
    people: list[PersonMention],
) -> list[PersonMention]:
    return [
        person
        for person in people
        if any(contains_surface(source_text, surface) for surface in [person.name, *person.aliases])
    ]


def speaker_mentions(people: list[PersonMention], speaker_name: str) -> list[PersonMention]:
    normalized_speaker = normalize_evidence(speaker_name)
    return [person for person in people if normalize_evidence(person.name) == normalized_speaker]


@dataclass(frozen=True)
class RelationshipEvidenceAnalysis:
    relationship_id: str
    source_segment_ids: list[str]
    source_text: str
    subject_mention_id: str
    subject_name: str | None
    object_mention_id: str
    object_name: str | None
    explicit_people: list[dict[str, str]]
    speaker_mention_ids: list[str]
    first_person_reference: bool
    supported_endpoint_ids: list[str]
    unsupported_endpoint_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_relationship_evidence(
    *,
    relationship: RelationshipClaim,
    transcript: TranscriptEnvelope,
    people: list[PersonMention],
    speaker_name: str,
) -> RelationshipEvidenceAnalysis:
    mention_by_id = {person.mention_id: person for person in people}
    source_text = joined_segment_text(relationship.source_segment_ids, transcript)
    explicit = explicitly_named_people(source_text, people)
    explicit_ids = {person.mention_id for person in explicit}
    speakers = speaker_mentions(people, speaker_name)
    speaker_ids = {person.mention_id for person in speakers}
    first_person = has_first_person_reference(source_text)

    endpoint_ids = [relationship.subject_mention_id, relationship.object_mention_id]
    supported = [
        mention_id
        for mention_id in endpoint_ids
        if mention_id in explicit_ids or (first_person and mention_id in speaker_ids)
    ]
    unsupported = [mention_id for mention_id in endpoint_ids if mention_id not in supported]

    subject = mention_by_id.get(relationship.subject_mention_id)
    object_person = mention_by_id.get(relationship.object_mention_id)
    return RelationshipEvidenceAnalysis(
        relationship_id=relationship.relationship_id,
        source_segment_ids=list(relationship.source_segment_ids),
        source_text=source_text,
        subject_mention_id=relationship.subject_mention_id,
        subject_name=subject.name if subject else None,
        object_mention_id=relationship.object_mention_id,
        object_name=object_person.name if object_person else None,
        explicit_people=[
            {"mention_id": person.mention_id, "name": person.name} for person in explicit
        ],
        speaker_mention_ids=sorted(speaker_ids),
        first_person_reference=first_person,
        supported_endpoint_ids=supported,
        unsupported_endpoint_ids=unsupported,
    )
