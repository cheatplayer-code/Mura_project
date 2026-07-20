from __future__ import annotations

from mura import _relationship_grounding_impl as _impl
from mura._relationship_grounding_impl import GroundingContext
from mura.domain.models import (
    PersonMention,
    RelationshipClaim,
    RelationshipRole,
    TranscriptEnvelope,
)
from mura.explicit_pair_grounding import find_explicit_pair_matches
from mura.linguistics.common import normalize_text
from mura.linguistics.multilingual import LinguisticRelationshipSignal

_MAX_CONTEXT_CHARS = _impl._MAX_CONTEXT_CHARS
_MAX_CONTEXT_SENTENCES = _impl._MAX_CONTEXT_SENTENCES
_split_units = _impl._split_units
grounding_rule_family = _impl.grounding_rule_family
supported_endpoint_ids = _impl.supported_endpoint_ids
_EXPLICIT_PAIR_RULE_IDS = frozenset(
    {
        "ru.relationship.explicit_spouse_coordination.v2",
        "kk.relationship.explicit_spouse_coordination.v2",
        "en.relationship.explicit_spouse_coordination.v1",
    }
)


def _pair_signals(text: str, people: list[PersonMention]) -> list[LinguisticRelationshipSignal]:
    return [
        LinguisticRelationshipSignal(
            language=match.language,
            relationship_type=match.relationship_type,
            subject_mention_id=match.subject_mention_id,
            subject_role=RelationshipRole.SPOUSE,
            object_mention_id=match.object_mention_id,
            object_role=RelationshipRole.SPOUSE,
            source_surface=match.source_surface,
            rule_id=match.rule_id,
        )
        for match in find_explicit_pair_matches(text, people)
    ]


def find_bounded_relationship_signals(
    *,
    contexts: list[GroundingContext],
    people: list[PersonMention],
    speaker_name: str,
) -> list[LinguisticRelationshipSignal]:
    legacy = _impl.find_bounded_relationship_signals(
        contexts=contexts,
        people=people,
        speaker_name=speaker_name,
    )
    signals = [signal for signal in legacy if signal.rule_id not in _EXPLICIT_PAIR_RULE_IDS]
    for context in contexts:
        for unit in _split_units(context.text):
            signals.extend(_pair_signals(unit, people))

    unique: dict[tuple[str, str, str, str, str], LinguisticRelationshipSignal] = {}
    for signal in signals:
        key = (
            signal.relationship_type.value,
            signal.subject_mention_id,
            signal.subject_role.value,
            signal.object_mention_id,
            signal.object_role.value,
        )
        unique.setdefault(key, signal)
    return list(unique.values())


def _source_units(
    relationship: RelationshipClaim,
    transcript: TranscriptEnvelope,
) -> list[str]:
    requested = set(relationship.source_segment_ids)
    units: list[str] = []
    for segment in transcript.segments:
        if segment.segment_id not in requested:
            continue
        units.extend(_split_units(segment.text))
    return units


def _windows(units: list[str]) -> list[GroundingContext]:
    values: dict[str, GroundingContext] = {}
    for start in range(len(units)):
        for count in range(1, _MAX_CONTEXT_SENTENCES + 1):
            selected = units[start : start + count]
            if len(selected) != count:
                break
            combined = " ".join(selected)
            if len(combined) > _MAX_CONTEXT_CHARS:
                break
            values.setdefault(
                normalize_text(combined),
                GroundingContext(combined, count),
            )
    return list(values.values())


def select_relationship_grounding_contexts(
    *,
    relationship: RelationshipClaim,
    transcript: TranscriptEnvelope,
    people: list[PersonMention],
    speaker_name: str,
    resolved_antecedent_ids: set[str],
) -> list[GroundingContext]:
    del resolved_antecedent_ids
    mention_by_id = {person.mention_id: person for person in people}
    endpoint_ids = {
        relationship.subject_mention_id,
        relationship.object_mention_id,
    }
    endpoints = [mention_by_id[item] for item in endpoint_ids if item in mention_by_id]
    if len(endpoints) != 2:
        return []

    eligible: list[GroundingContext] = []
    for window in _windows(_source_units(relationship, transcript)):
        supported = supported_endpoint_ids(
            contexts=[window],
            people=endpoints,
            speaker_name=speaker_name,
            resolved_antecedent_ids=set(),
        )
        if endpoint_ids.issubset(supported):
            eligible.append(window)
    eligible.sort(key=lambda item: (item.sentence_count, len(item.text), item.text))
    return eligible[:6]
