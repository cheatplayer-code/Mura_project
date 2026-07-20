from __future__ import annotations

from mura import _relationship_grounding_impl as _impl
from mura._relationship_grounding_impl import GroundingContext
from mura.domain.models import (
    PersonMention,
    RelationshipClaim,
    RelationshipRole,
    RelationshipType,
    TranscriptEnvelope,
)
from mura.explicit_pair_grounding import find_explicit_pair_matches
from mura.linguistics.common import normalize_text, tokenize
from mura.linguistics.multilingual import (
    LinguisticRelationshipSignal,
    find_known_name_matches,
    find_relationship_signals,
)

_MAX_CONTEXT_CHARS = _impl._MAX_CONTEXT_CHARS
_MAX_CONTEXT_SENTENCES = _impl._MAX_CONTEXT_SENTENCES
_split_units = _impl._split_units
grounding_rule_family = _impl.grounding_rule_family
supported_endpoint_ids = _impl.supported_endpoint_ids
_COORDINATORS = frozenset({"и", "and"})
_SIBLING_CUES = {
    ("брат", "и", "сестра"): ("ru", "ru.relationship.explicit_sibling_coordination.v2"),
    ("сестра", "и", "брат"): ("ru", "ru.relationship.explicit_sibling_coordination.v2"),
    ("brother", "and", "sister"): ("en", "en.relationship.explicit_sibling_coordination.v1"),
    ("sister", "and", "brother"): ("en", "en.relationship.explicit_sibling_coordination.v1"),
}
_SIBLING_BRIDGE_TOKENS = frozenset({"это", "are", "were"})


def _surfaces(person: PersonMention) -> list[str]:
    values = [person.name, *person.aliases]
    values.extend(item.surface for item in person.name_variants)
    return list(dict.fromkeys(value for value in values if value))


def _name_occurrences(text: str, person: PersonMention) -> list[tuple[int, int]]:
    return sorted(
        {
            (match.start, match.end)
            for surface in _surfaces(person)
            for match in find_known_name_matches(text, surface)
            if match.start >= 0
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


def _sibling_pair_signals(
    text: str, people: list[PersonMention]
) -> list[LinguisticRelationshipSignal]:
    if len(people) != 2:
        return []
    people_by_id = {person.mention_id: person for person in people}
    if len(people_by_id) != 2:
        return []
    normalized_surfaces = [
        {normalize_text(surface) for surface in _surfaces(person) if normalize_text(surface)}
        for person in people
    ]
    if normalized_surfaces[0].intersection(normalized_surfaces[1]):
        return []

    tokens = tokenize(text)
    if any(token.normalized.startswith("двоюрод") for token in tokens):
        return []
    endpoint_ids = sorted(people_by_id)
    occurrences = {
        mention_id: _name_occurrences(text, people_by_id[mention_id])
        for mention_id in endpoint_ids
    }
    signals: list[LinguisticRelationshipSignal] = []
    for first in occurrences[endpoint_ids[0]]:
        for second in occurrences[endpoint_ids[1]]:
            left, right = sorted((first, second), key=lambda item: (item[0], item[1]))
            between_names = [
                token for token in tokens if left[1] <= token.start and token.end <= right[0]
            ]
            if len(between_names) != 1 or between_names[0].normalized not in _COORDINATORS:
                continue
            following = [token for token in tokens if token.start >= right[1]]
            for index in range(len(following) - 2):
                cue_tokens = following[index : index + 3]
                cue = tuple(token.normalized for token in cue_tokens)
                cue_spec = _SIBLING_CUES.get(cue)
                if cue_spec is None:
                    continue
                bridge = following[:index]
                if cue_tokens[0].start - right[1] > 80 or any(
                    token.normalized not in _SIBLING_BRIDGE_TOKENS for token in bridge
                ):
                    continue
                language, rule_id = cue_spec
                signals.append(
                    LinguisticRelationshipSignal(
                        language=language,
                        relationship_type=RelationshipType.SIBLING,
                        subject_mention_id=endpoint_ids[0],
                        subject_role=RelationshipRole.SIBLING,
                        object_mention_id=endpoint_ids[1],
                        object_role=RelationshipRole.SIBLING,
                        source_surface=text[left[0] : cue_tokens[-1].end],
                        rule_id=rule_id,
                    )
                )
                break
    return signals


def find_bounded_relationship_signals(
    *,
    contexts: list[GroundingContext],
    people: list[PersonMention],
    speaker_name: str,
) -> list[LinguisticRelationshipSignal]:
    signals: list[LinguisticRelationshipSignal] = []
    for context in contexts:
        text = _impl._masked(context.text)
        signals.extend(find_relationship_signals(text, people, speaker_name))
        signals.extend(_impl._speaker_signals(text, people, speaker_name))
        signals.extend(_impl._named_signals(text, people))
        for unit in _split_units(text):
            signals.extend(_pair_signals(unit, people))
            signals.extend(_sibling_pair_signals(unit, people))

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
