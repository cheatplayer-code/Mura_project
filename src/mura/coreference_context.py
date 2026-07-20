from __future__ import annotations

from dataclasses import dataclass

from mura.coreference_language import (
    AnaphorOccurrence,
    KinshipOccurrence,
    find_anaphors,
    nearest_kinship,
)
from mura.coreference_units import (
    NameOccurrence,
    TextUnit,
    candidate_context_unit,
    candidate_ids,
    context_units,
    ordered_units,
    person_occurrences,
    target_ids,
    unit_index_for_anaphor,
)
from mura.domain.models import GrammaticalNumber, PersonMention, TranscriptEnvelope
from mura.linguistics.common import normalize_text, tokenize


@dataclass(frozen=True)
class BoundedCoreferenceContext:
    segment_id: str
    anaphor: AnaphorOccurrence
    kinship: KinshipOccurrence
    candidate_ids: list[str]
    target_ids: list[str]
    candidate_context: TextUnit
    resolved: bool


_COORDINATORS = frozenset({"мен", "және", "и", "and"})
_PAIR_CUES = (
    "үйленді",
    "үйленген",
    "ерлі зайыпты",
    "жұбайлар",
    "поженились",
    "женаты",
    "супруги",
    "married",
    "spouses",
    "couple",
)


def _has_coordinator_between(
    segment_text: str,
    occurrences: list[NameOccurrence],
    candidates: list[str],
) -> bool:
    candidate_set = set(candidates)
    selected = [item for item in occurrences if item.mention_id in candidate_set]
    if len({item.mention_id for item in selected}) != 2:
        return False
    selected.sort(key=lambda item: (item.start, item.end))
    first = selected[0]
    second = selected[-1]
    between = segment_text[first.end : second.start]
    return bool({token.normalized for token in tokenize(between)}.intersection(_COORDINATORS))


def _has_pair_cue(text: str) -> bool:
    normalized = normalize_text(text)
    return any(f" {normalize_text(cue)} " in f" {normalized} " for cue in _PAIR_CUES)


def _is_explicit_pair(
    *,
    candidates: list[str],
    candidate_context: TextUnit,
    segment_text_by_id: dict[str, str],
    occurrences_by_segment: dict[str, list[NameOccurrence]],
) -> bool:
    if len(candidates) != 2:
        return False
    segment_text = segment_text_by_id[candidate_context.segment_id]
    occurrences = [
        item
        for item in occurrences_by_segment[candidate_context.segment_id]
        if candidate_context.start <= item.start and item.end <= candidate_context.end
    ]
    return _has_coordinator_between(
        segment_text,
        occurrences,
        candidates,
    ) and _has_pair_cue(candidate_context.text)


def bounded_coreference_contexts(
    *,
    people: list[PersonMention],
    transcript: TranscriptEnvelope,
) -> list[BoundedCoreferenceContext]:
    segment_text_by_id = {segment.segment_id: segment.text for segment in transcript.segments}
    units = ordered_units(transcript)
    occurrences_by_segment = {
        segment.segment_id: person_occurrences(
            segment.text,
            people,
            segment_id=segment.segment_id,
        )
        for segment in transcript.segments
    }
    contexts: list[BoundedCoreferenceContext] = []
    for segment in transcript.segments:
        occurrences = occurrences_by_segment[segment.segment_id]
        for anaphor in find_anaphors(segment.text):
            unit_index = unit_index_for_anaphor(
                units,
                segment_id=segment.segment_id,
                anaphor=anaphor,
            )
            if unit_index is None:
                continue
            current_unit = units[unit_index]
            kinship = nearest_kinship(segment.text, anaphor)
            if kinship is None or not (
                current_unit.start <= kinship.start and kinship.end <= current_unit.end
            ):
                continue
            targets = target_ids(
                occurrences,
                kinship_end=kinship.end,
                unit_end=current_unit.end,
            )
            selected_units = context_units(units, current_index=unit_index)
            candidates = candidate_ids(
                selected_units=selected_units,
                current_unit=current_unit,
                anaphor=anaphor,
                excluded_ids=set(targets),
                occurrences_by_segment=occurrences_by_segment,
            )
            if not candidates:
                continue
            candidate_context = candidate_context_unit(
                selected_units=selected_units,
                candidates=candidates,
                occurrences_by_segment=occurrences_by_segment,
            )
            if candidate_context is None:
                continue
            resolved = (
                len(candidates) == 1
                if anaphor.grammatical_number is GrammaticalNumber.SINGULAR
                else _is_explicit_pair(
                    candidates=candidates,
                    candidate_context=candidate_context,
                    segment_text_by_id=segment_text_by_id,
                    occurrences_by_segment=occurrences_by_segment,
                )
            )
            contexts.append(
                BoundedCoreferenceContext(
                    segment_id=segment.segment_id,
                    anaphor=anaphor,
                    kinship=kinship,
                    candidate_ids=candidates,
                    target_ids=targets,
                    candidate_context=candidate_context,
                    resolved=resolved,
                )
            )
    return contexts
