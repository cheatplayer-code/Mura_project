from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from mura.domain.models import (
    CoreferenceLink,
    CoreferenceMethod,
    CoreferenceStatus,
    EvidenceClass,
    EvidencePurpose,
    EvidenceSpan,
    ExtractionResult,
    GrammaticalNumber,
    PersonMention,
    RelationshipClaim,
    RelationshipRole,
    RelationshipType,
    TranscriptEnvelope,
)
from mura.linguistics import english, kazakh, russian
from mura.linguistics.common import normalize_text, tokenize
from mura.linguistics.multilingual import find_known_name_matches
from mura.relationship_evidence import person_name_surfaces


class _KinshipFrame(Protocol):
    @property
    def relationship_type(self) -> RelationshipType: ...

    @property
    def possessor_role(self) -> RelationshipRole: ...

    @property
    def relative_role(self) -> RelationshipRole: ...


@dataclass(frozen=True)
class _NameOccurrence:
    mention_id: str
    start: int
    end: int


@dataclass(frozen=True)
class _AnaphorOccurrence:
    surface: str
    start: int
    end: int
    language: str
    grammatical_number: GrammaticalNumber


@dataclass(frozen=True)
class _KinshipOccurrence:
    surface: str
    start: int
    end: int
    language: str
    frame: _KinshipFrame


@dataclass(frozen=True)
class CoreferenceAugmentation:
    result: ExtractionResult
    changed_relationship_count: int
    generated_link_count: int


_ANAPHORS: dict[str, tuple[str, GrammaticalNumber]] = {
    "оның": ("kk", GrammaticalNumber.SINGULAR),
    "олардың": ("kk", GrammaticalNumber.PLURAL),
    "его": ("ru", GrammaticalNumber.SINGULAR),
    "ее": ("ru", GrammaticalNumber.SINGULAR),
    "её": ("ru", GrammaticalNumber.SINGULAR),
    "их": ("ru", GrammaticalNumber.PLURAL),
    "his": ("en", GrammaticalNumber.SINGULAR),
    "her": ("en", GrammaticalNumber.SINGULAR),
    "their": ("en", GrammaticalNumber.PLURAL),
}

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
_TRUSTED_RESOLVED_METHODS = {
    CoreferenceMethod.DETERMINISTIC_DISCOURSE,
    CoreferenceMethod.HUMAN_REVIEW,
}
_SENTENCE_WINDOW_CHARS = 220
_TARGET_WINDOW_CHARS = 90
_KINSHIP_WINDOW_CHARS = 28


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_")
    return normalized or "unknown"


def _ordered_segment_ids(segment_ids: list[str], transcript: TranscriptEnvelope) -> list[str]:
    requested = set(segment_ids)
    ordered = [
        segment.segment_id for segment in transcript.segments if segment.segment_id in requested
    ]
    seen = set(ordered)
    ordered.extend(item for item in segment_ids if item not in seen)
    return list(dict.fromkeys(ordered))


def _person_occurrences(text: str, people: list[PersonMention]) -> list[_NameOccurrence]:
    occurrences: dict[tuple[str, int, int], _NameOccurrence] = {}
    for person in people:
        for surface in person_name_surfaces(person):
            for match in find_known_name_matches(text, surface):
                if match.start < 0:
                    continue
                key = (person.mention_id, match.start, match.end)
                occurrences.setdefault(
                    key,
                    _NameOccurrence(
                        mention_id=person.mention_id,
                        start=match.start,
                        end=match.end,
                    ),
                )
    return sorted(occurrences.values(), key=lambda item: (item.start, item.end, item.mention_id))


def _anaphors(text: str) -> list[_AnaphorOccurrence]:
    matches: list[_AnaphorOccurrence] = []
    for token in tokenize(text):
        specification = _ANAPHORS.get(token.normalized)
        if specification is None:
            continue
        language, number = specification
        matches.append(
            _AnaphorOccurrence(
                surface=token.surface,
                start=token.start,
                end=token.end,
                language=language,
                grammatical_number=number,
            )
        )
    return matches


def _kazakh_kinship_matches(text: str) -> list[_KinshipOccurrence]:
    frames = getattr(kazakh, "_NAMED_POSSESSOR_FRAMES")
    return [
        _KinshipOccurrence(
            surface=token.surface,
            start=token.start,
            end=token.end,
            language="kk",
            frame=frames[token.normalized],
        )
        for token in tokenize(text)
        if token.normalized in frames
    ]


def _kinship_matches(text: str) -> list[_KinshipOccurrence]:
    matches = [
        *_kazakh_kinship_matches(text),
        *(
            _KinshipOccurrence(
                surface=item.surface,
                start=item.start,
                end=item.end,
                language="ru",
                frame=item.frame,
            )
            for item in russian.find_kinship_matches(text)
        ),
        *(
            _KinshipOccurrence(
                surface=item.surface,
                start=item.start,
                end=item.end,
                language="en",
                frame=item.frame,
            )
            for item in english.find_kinship_matches(text)
        ),
    ]
    unique: dict[tuple[int, int, str], _KinshipOccurrence] = {}
    for match in matches:
        key = (match.start, match.end, match.surface)
        unique.setdefault(key, match)
    return list(unique.values())


def _nearest_kinship(
    text: str,
    anaphor: _AnaphorOccurrence,
) -> _KinshipOccurrence | None:
    candidates = [
        match
        for match in _kinship_matches(text)
        if match.start >= anaphor.end and match.start - anaphor.end <= _KINSHIP_WINDOW_CHARS
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item.start - anaphor.end,
            -(item.end - item.start),
        )
    )
    return candidates[0]


def _unique_target(
    occurrences: list[_NameOccurrence],
    *,
    kinship_end: int,
) -> str | None:
    candidates = {
        item.mention_id
        for item in occurrences
        if item.start >= kinship_end and item.start - kinship_end <= _TARGET_WINDOW_CHARS
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _preceding_candidates(
    *,
    segment_index: int,
    anaphor: _AnaphorOccurrence,
    target_id: str,
    transcript: TranscriptEnvelope,
    occurrences_by_segment: dict[str, list[_NameOccurrence]],
) -> tuple[list[str], str | None]:
    segment = transcript.segments[segment_index]
    lower_bound = max(0, anaphor.start - _SENTENCE_WINDOW_CHARS)
    local = {
        item.mention_id
        for item in occurrences_by_segment[segment.segment_id]
        if item.end <= anaphor.start and item.start >= lower_bound and item.mention_id != target_id
    }
    if local:
        return sorted(local), segment.segment_id
    if segment_index == 0:
        return [], None
    previous = transcript.segments[segment_index - 1]
    candidates = {
        item.mention_id
        for item in occurrences_by_segment[previous.segment_id]
        if item.mention_id != target_id
    }
    return sorted(candidates), previous.segment_id


def _has_coordinator_between(
    text: str,
    occurrences: list[_NameOccurrence],
    candidate_ids: list[str],
) -> bool:
    candidate_set = set(candidate_ids)
    selected = [item for item in occurrences if item.mention_id in candidate_set]
    if len({item.mention_id for item in selected}) != 2:
        return False
    selected.sort(key=lambda item: (item.start, item.end))
    first = selected[0]
    second = selected[-1]
    between = text[first.end : second.start]
    return bool({token.normalized for token in tokenize(between)}.intersection(_COORDINATORS))


def _has_pair_cue(text: str) -> bool:
    normalized = normalize_text(text)
    return any(f" {normalize_text(cue)} " in f" {normalized} " for cue in _PAIR_CUES)


def _has_spouse_claim(
    candidate_ids: list[str],
    context_segment_id: str,
    relationships: list[RelationshipClaim],
) -> bool:
    if len(candidate_ids) != 2:
        return False
    candidate_set = set(candidate_ids)
    return any(
        relationship.relationship_type is RelationshipType.SPOUSE
        and {
            relationship.subject_mention_id,
            relationship.object_mention_id,
        }
        == candidate_set
        and context_segment_id in relationship.source_segment_ids
        for relationship in relationships
    )


def _is_explicit_pair(
    *,
    candidate_ids: list[str],
    context_segment_id: str | None,
    transcript: TranscriptEnvelope,
    occurrences_by_segment: dict[str, list[_NameOccurrence]],
    relationships: list[RelationshipClaim],
) -> bool:
    if len(candidate_ids) != 2 or context_segment_id is None:
        return False
    segment = next(
        (item for item in transcript.segments if item.segment_id == context_segment_id),
        None,
    )
    if segment is None:
        return False
    coordinated = _has_coordinator_between(
        segment.text,
        occurrences_by_segment[context_segment_id],
        candidate_ids,
    )
    if not coordinated:
        return False
    return _has_pair_cue(segment.text) or _has_spouse_claim(
        candidate_ids,
        context_segment_id,
        relationships,
    )


def _expected_edge(
    possessor_id: str,
    target_id: str,
    frame: _KinshipFrame,
) -> tuple[RelationshipType, str, RelationshipRole, str, RelationshipRole]:
    if frame.relationship_type is RelationshipType.PARENT_CHILD:
        if frame.possessor_role is RelationshipRole.PARENT:
            return (
                frame.relationship_type,
                possessor_id,
                RelationshipRole.PARENT,
                target_id,
                RelationshipRole.CHILD,
            )
        return (
            frame.relationship_type,
            target_id,
            RelationshipRole.PARENT,
            possessor_id,
            RelationshipRole.CHILD,
        )
    if frame.relationship_type is RelationshipType.SIBLING:
        if frame.possessor_role is RelationshipRole.OLDER_SIBLING:
            return (
                frame.relationship_type,
                possessor_id,
                RelationshipRole.OLDER_SIBLING,
                target_id,
                RelationshipRole.YOUNGER_SIBLING,
            )
        if frame.relative_role is RelationshipRole.OLDER_SIBLING:
            return (
                frame.relationship_type,
                target_id,
                RelationshipRole.OLDER_SIBLING,
                possessor_id,
                RelationshipRole.YOUNGER_SIBLING,
            )
        return (
            frame.relationship_type,
            possessor_id,
            RelationshipRole.SIBLING,
            target_id,
            RelationshipRole.SIBLING,
        )
    return (
        frame.relationship_type,
        possessor_id,
        RelationshipRole.SPOUSE,
        target_id,
        RelationshipRole.SPOUSE,
    )


def _relationship_matches_edge(
    relationship: RelationshipClaim,
    edge: tuple[RelationshipType, str, RelationshipRole, str, RelationshipRole],
) -> bool:
    relationship_type, subject_id, subject_role, object_id, object_role = edge
    if relationship.relationship_type is not relationship_type:
        return False
    if relationship_type is RelationshipType.SPOUSE:
        return {
            relationship.subject_mention_id,
            relationship.object_mention_id,
        } == {subject_id, object_id}
    return (
        relationship.subject_mention_id == subject_id
        and relationship.subject_role is subject_role
        and relationship.object_mention_id == object_id
        and relationship.object_role is object_role
    )


def _matching_relationships(
    *,
    relationships: list[RelationshipClaim],
    candidate_ids: list[str],
    target_id: str,
    frame: _KinshipFrame,
    anaphor_segment_id: str,
) -> list[RelationshipClaim]:
    edges = [_expected_edge(candidate_id, target_id, frame) for candidate_id in candidate_ids]
    return [
        relationship
        for relationship in relationships
        if anaphor_segment_id in relationship.source_segment_ids
        and any(_relationship_matches_edge(relationship, edge) for edge in edges)
    ]


def _trusted_existing_link_ids(result: ExtractionResult) -> set[str]:
    return {
        link.coreference_id
        for link in result.coreference_links
        if not (
            link.status is CoreferenceStatus.RESOLVED
            and link.method not in _TRUSTED_RESOLVED_METHODS
        )
    }


def _clean_relationship_links(
    relationships: list[RelationshipClaim],
    trusted_link_ids: set[str],
) -> list[RelationshipClaim]:
    return [
        relationship.model_copy(
            update={
                "coreference_link_ids": [
                    link_id
                    for link_id in relationship.coreference_link_ids
                    if link_id in trusted_link_ids
                ]
            }
        )
        for relationship in relationships
    ]


def _build_evidence(
    *,
    link_id: str,
    context_segment_id: str,
    anaphor_segment_id: str,
    anaphor: _AnaphorOccurrence,
    mention_ids: list[str],
    evidence_class: EvidenceClass,
    transcript: TranscriptEnvelope,
    confidence: float,
) -> list[EvidenceSpan]:
    segment_by_id = {segment.segment_id: segment for segment in transcript.segments}
    context_segment = segment_by_id[context_segment_id]
    context_evidence_id = f"evidence_{_safe_id(link_id)}_context"
    anaphor_evidence_id = f"evidence_{_safe_id(link_id)}_anaphor"
    return [
        EvidenceSpan(
            evidence_id=context_evidence_id,
            segment_id=context_segment_id,
            text=context_segment.text,
            evidence_class=evidence_class,
            purposes=[EvidencePurpose.CONTEXT, EvidencePurpose.COREFERENCE],
            mention_ids=list(dict.fromkeys(mention_ids)),
            coreference_link_ids=[link_id],
            confidence=confidence,
        ),
        EvidenceSpan(
            evidence_id=anaphor_evidence_id,
            segment_id=anaphor_segment_id,
            text=anaphor.surface,
            start_char=anaphor.start,
            end_char=anaphor.end,
            evidence_class=evidence_class,
            purposes=[EvidencePurpose.COREFERENCE, EvidencePurpose.CLAIM],
            mention_ids=list(dict.fromkeys(mention_ids)),
            coreference_link_ids=[link_id],
            confidence=confidence,
        ),
    ]


def augment_bounded_coreference(
    result: ExtractionResult,
    transcript: TranscriptEnvelope,
) -> CoreferenceAugmentation:
    """Create bounded deterministic discourse links without guessing across long context.

    The resolver looks only inside the current segment and, when needed, one immediately
    preceding segment. Singular anaphors require one candidate. Plural anaphors require one
    explicitly coordinated pair with a spouse/marriage cue. Competing candidates are preserved
    as an ambiguous link and never authorize a relationship.
    """

    occurrences_by_segment = {
        segment.segment_id: _person_occurrences(segment.text, result.people_mentions)
        for segment in transcript.segments
    }
    trusted_link_ids = _trusted_existing_link_ids(result)
    relationships = _clean_relationship_links(list(result.relationship_claims), trusted_link_ids)
    links = list(result.coreference_links)
    evidence = list(result.evidence_spans)
    existing_link_ids = {link.coreference_id for link in links}
    existing_evidence_ids = {item.evidence_id for item in evidence}
    changed_relationship_ids: set[str] = set()
    generated_link_count = 0

    for segment_index, segment in enumerate(transcript.segments):
        occurrences = occurrences_by_segment[segment.segment_id]
        for anaphor in _anaphors(segment.text):
            kinship = _nearest_kinship(segment.text, anaphor)
            if kinship is None:
                continue
            target_id = _unique_target(occurrences, kinship_end=kinship.end)
            if target_id is None:
                continue

            candidate_ids, context_segment_id = _preceding_candidates(
                segment_index=segment_index,
                anaphor=anaphor,
                target_id=target_id,
                transcript=transcript,
                occurrences_by_segment=occurrences_by_segment,
            )
            if not candidate_ids or context_segment_id is None:
                continue

            if anaphor.grammatical_number is GrammaticalNumber.SINGULAR:
                resolved = len(candidate_ids) == 1
            else:
                resolved = _is_explicit_pair(
                    candidate_ids=candidate_ids,
                    context_segment_id=context_segment_id,
                    transcript=transcript,
                    occurrences_by_segment=occurrences_by_segment,
                    relationships=relationships,
                )

            matching = _matching_relationships(
                relationships=relationships,
                candidate_ids=candidate_ids,
                target_id=target_id,
                frame=kinship.frame,
                anaphor_segment_id=segment.segment_id,
            )
            if not matching:
                continue

            if resolved:
                status = CoreferenceStatus.RESOLVED
            elif len(candidate_ids) >= 2:
                status = CoreferenceStatus.AMBIGUOUS
            else:
                status = CoreferenceStatus.UNRESOLVED
            antecedents = candidate_ids if resolved else []
            evidence_class = (
                EvidenceClass.D_CONTEXT_RESOLVED if resolved else EvidenceClass.U_UNCERTAIN
            )
            confidence = (
                1.0
                if resolved
                else 0.5
                if status is CoreferenceStatus.AMBIGUOUS
                else 0.0
            )
            link_id = (
                f"coreference_{_safe_id(segment.segment_id)}_{anaphor.start}_"
                f"{anaphor.grammatical_number.value}"
            )
            if link_id in existing_link_ids:
                continue

            rule_id = (
                "discourse.plural.explicit_pair.v1"
                if resolved and anaphor.grammatical_number is GrammaticalNumber.PLURAL
                else "discourse.singular.unique_antecedent.v1"
                if resolved
                else "discourse.ambiguous_competing_antecedents.v1"
                if status is CoreferenceStatus.AMBIGUOUS
                else "discourse.unresolved.insufficient_candidates.v1"
            )
            source_segment_ids = _ordered_segment_ids(
                [context_segment_id, segment.segment_id],
                transcript,
            )
            generated_evidence = _build_evidence(
                link_id=link_id,
                context_segment_id=context_segment_id,
                anaphor_segment_id=segment.segment_id,
                anaphor=anaphor,
                mention_ids=[*candidate_ids, target_id],
                evidence_class=evidence_class,
                transcript=transcript,
                confidence=confidence,
            )
            if any(item.evidence_id in existing_evidence_ids for item in generated_evidence):
                continue

            link = CoreferenceLink(
                coreference_id=link_id,
                anaphor_text=anaphor.surface,
                source_segment_ids=source_segment_ids,
                evidence_ids=[item.evidence_id for item in generated_evidence],
                status=status,
                method=CoreferenceMethod.DETERMINISTIC_DISCOURSE,
                grammatical_number=anaphor.grammatical_number,
                antecedent_mention_ids=antecedents,
                candidate_mention_ids=candidate_ids,
                evidence_class=evidence_class,
                confidence=confidence,
                reason=(
                    f"{rule_id}: bounded to the current segment and one preceding segment; "
                    f"candidates={candidate_ids}"
                ),
            )
            links.append(link)
            evidence.extend(generated_evidence)
            existing_link_ids.add(link_id)
            existing_evidence_ids.update(item.evidence_id for item in generated_evidence)
            generated_link_count += 1

            matching_ids = {item.relationship_id for item in matching}
            generated_evidence_ids = [item.evidence_id for item in generated_evidence]
            updated_relationships: list[RelationshipClaim] = []
            for relationship in relationships:
                if relationship.relationship_id not in matching_ids:
                    updated_relationships.append(relationship)
                    continue
                updated_relationships.append(
                    relationship.model_copy(
                        update={
                            "source_segment_ids": _ordered_segment_ids(
                                [*relationship.source_segment_ids, *source_segment_ids],
                                transcript,
                            ),
                            "evidence_ids": list(
                                dict.fromkeys([*relationship.evidence_ids, *generated_evidence_ids])
                            ),
                            "coreference_link_ids": list(
                                dict.fromkeys([*relationship.coreference_link_ids, link_id])
                            ),
                        }
                    )
                )
                changed_relationship_ids.add(relationship.relationship_id)
            relationships = updated_relationships

    updated = result.model_copy(
        update={
            "evidence_spans": evidence,
            "coreference_links": links,
            "relationship_claims": relationships,
        }
    )
    return CoreferenceAugmentation(
        result=updated,
        changed_relationship_count=len(changed_relationship_ids),
        generated_link_count=generated_link_count,
    )
