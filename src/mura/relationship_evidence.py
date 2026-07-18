from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mura.domain.models import (
    EvidenceClass,
    PersonMention,
    RelationshipClaim,
    TranscriptEnvelope,
)
from mura.linguistics.common import normalize_text
from mura.linguistics.kazakh import (
    contains_known_name_surface,
    find_known_name_matches,
    find_relationship_signals,
    find_speaker_anchor_matches,
    find_uncertainty_markers,
    signal_matches_relationship,
)

_NON_KAZAKH_FIRST_PERSON_TOKENS = {
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
}

_AUTO_ACCEPTABLE_CLASSES = {
    EvidenceClass.A_EXPLICIT,
    EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT,
    EvidenceClass.C_SPEAKER_ANCHORED,
}


def normalize_evidence(value: str) -> str:
    return normalize_text(value)


def contains_exact_surface(text: str, surface: str) -> bool:
    normalized_text = normalize_evidence(text)
    normalized_surface = normalize_evidence(surface)
    if not normalized_surface:
        return False
    return f" {normalized_surface} " in f" {normalized_text} "


def contains_surface(text: str, surface: str) -> bool:
    if contains_exact_surface(text, surface):
        return True
    return contains_known_name_surface(text, surface)


def person_name_surfaces(person: PersonMention) -> list[str]:
    values = [person.name, *person.aliases]
    values.extend(variant.surface for variant in person.name_variants)
    return list(dict.fromkeys(value for value in values if value))


def has_first_person_reference(text: str) -> bool:
    if find_speaker_anchor_matches(text):
        return True
    tokens = set(normalize_evidence(text).split())
    return bool(tokens.intersection(_NON_KAZAKH_FIRST_PERSON_TOKENS))


def joined_segment_text(segment_ids: list[str], transcript: TranscriptEnvelope) -> str:
    text_by_id = {segment.segment_id: segment.text for segment in transcript.segments}
    return " ".join(
        text_by_id[segment_id] for segment_id in segment_ids if segment_id in text_by_id
    )


def explicitly_named_people(
    source_text: str,
    people: list[PersonMention],
) -> list[PersonMention]:
    return [
        person
        for person in people
        if any(contains_surface(source_text, surface) for surface in person_name_surfaces(person))
    ]


def exactly_named_people(
    source_text: str,
    people: list[PersonMention],
) -> list[PersonMention]:
    return [
        person
        for person in people
        if any(
            contains_exact_surface(source_text, surface) for surface in person_name_surfaces(person)
        )
    ]


def speaker_mentions(people: list[PersonMention], speaker_name: str) -> list[PersonMention]:
    normalized_speaker = normalize_evidence(speaker_name)
    return [
        person
        for person in people
        if any(
            normalize_evidence(surface) == normalized_speaker
            for surface in person_name_surfaces(person)
        )
    ]


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
    exact_people: list[dict[str, str]]
    morphological_people: list[dict[str, str]]
    speaker_mention_ids: list[str]
    first_person_reference: bool
    speaker_anchor_matches: list[dict[str, Any]]
    supported_endpoint_ids: list[str]
    unsupported_endpoint_ids: list[str]
    evidence_class: str
    auto_accept_eligible: bool
    coreference_link_ids: list[str]
    kazakh_relationship_signals: list[dict[str, str]]
    role_consistent: bool | None
    uncertainty_markers: list[dict[str, str]]
    linguistic_rule_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _name_rule_ids(source_text: str, people: list[PersonMention]) -> list[str]:
    rule_ids = {
        match.rule_id
        for person in people
        for surface in person_name_surfaces(person)
        for match in find_known_name_matches(source_text, surface)
    }
    return sorted(rule_ids)


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
    exact = exactly_named_people(source_text, people)
    explicit_ids = {person.mention_id for person in explicit}
    exact_ids = {person.mention_id for person in exact}
    morphological = [person for person in explicit if person.mention_id not in exact_ids]
    speakers = speaker_mentions(people, speaker_name)
    speaker_ids = {person.mention_id for person in speakers}
    speaker_anchors = find_speaker_anchor_matches(source_text)
    first_person = bool(speaker_anchors) or has_first_person_reference(source_text)

    endpoint_ids = [relationship.subject_mention_id, relationship.object_mention_id]
    supported = [
        mention_id
        for mention_id in endpoint_ids
        if mention_id in explicit_ids or (first_person and mention_id in speaker_ids)
    ]
    unsupported = [mention_id for mention_id in endpoint_ids if mention_id not in supported]

    endpoint_set = set(endpoint_ids)
    if endpoint_set.issubset(exact_ids):
        evidence_class = EvidenceClass.A_EXPLICIT
    elif not unsupported and first_person and endpoint_set.intersection(speaker_ids):
        evidence_class = EvidenceClass.C_SPEAKER_ANCHORED
    elif not unsupported:
        evidence_class = EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT
    elif relationship.coreference_link_ids:
        evidence_class = EvidenceClass.D_CONTEXT_RESOLVED
    elif relationship.assertion_mode.value == "inferred":
        evidence_class = EvidenceClass.E_INFERRED
    else:
        evidence_class = EvidenceClass.U_UNCERTAIN

    signals = find_relationship_signals(source_text, people, speaker_name)
    endpoint_signals = [
        signal
        for signal in signals
        if {
            signal.subject_mention_id,
            signal.object_mention_id,
        }
        == endpoint_set
    ]
    role_consistent = (
        any(signal_matches_relationship(signal, relationship) for signal in endpoint_signals)
        if endpoint_signals
        else None
    )
    uncertainty_markers = find_uncertainty_markers(source_text)

    rule_ids = set(_name_rule_ids(source_text, people))
    rule_ids.update(anchor.rule_id for anchor in speaker_anchors)
    rule_ids.update(signal.rule_id for signal in signals)
    rule_ids.update(marker.rule_id for marker in uncertainty_markers)

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
        exact_people=[{"mention_id": person.mention_id, "name": person.name} for person in exact],
        morphological_people=[
            {"mention_id": person.mention_id, "name": person.name} for person in morphological
        ],
        speaker_mention_ids=sorted(speaker_ids),
        first_person_reference=first_person,
        speaker_anchor_matches=[anchor.to_dict() for anchor in speaker_anchors],
        supported_endpoint_ids=supported,
        unsupported_endpoint_ids=unsupported,
        evidence_class=evidence_class.value,
        auto_accept_eligible=(
            evidence_class in _AUTO_ACCEPTABLE_CLASSES and role_consistent is not False
        ),
        coreference_link_ids=list(relationship.coreference_link_ids),
        kazakh_relationship_signals=[signal.to_dict() for signal in signals],
        role_consistent=role_consistent,
        uncertainty_markers=[marker.to_dict() for marker in uncertainty_markers],
        linguistic_rule_ids=sorted(rule_ids),
    )
