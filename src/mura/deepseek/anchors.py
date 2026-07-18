from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from mura.domain.models import (
    CleanerResult,
    EvidenceSourceLayer,
    KnownPerson,
    StrictModel,
    TranscriptEnvelope,
)
from mura.linguistics import english, russian
from mura.linguistics.common import normalize_text, tokenize
from mura.linguistics.multilingual import find_known_name_matches, find_speaker_anchor_matches


class MentionAnchorKind(StrEnum):
    SPEAKER = "speaker"
    KNOWN_PERSON = "known_person"
    NAME_CANDIDATE = "name_candidate"


class LexicalAnnotationType(StrEnum):
    SPEAKER_ANCHOR = "speaker_anchor"
    KINSHIP_LEXEME = "kinship_lexeme"


class ExtractionMentionAnchor(StrictModel):
    anchor_id: str = Field(min_length=1)
    surface: str = Field(min_length=1)
    normalized: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    source_layer: EvidenceSourceLayer
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, gt=0)
    anchor_kind: MentionAnchorKind
    known_person_id: str | None = None
    rule_ids: list[str] = Field(min_length=1)


class ExtractionLexicalAnnotation(StrictModel):
    annotation_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    surface: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    annotation_type: LexicalAnnotationType
    language: str = Field(min_length=2)
    rule_id: str = Field(min_length=1)


class ExtractionAnchorBundle(StrictModel):
    schema_version: str = "extraction-anchors-v1"
    allowed_segment_ids: list[str] = Field(min_length=1)
    mention_anchors: list[ExtractionMentionAnchor] = Field(default_factory=list)
    lexical_annotations: list[ExtractionLexicalAnnotation] = Field(default_factory=list)


@dataclass(frozen=True)
class _AnchorCandidate:
    surface: str
    normalized: str
    segment_id: str
    source_layer: EvidenceSourceLayer
    start_char: int | None
    end_char: int | None
    anchor_kind: MentionAnchorKind
    known_person_id: str | None
    rule_ids: tuple[str, ...]


_KAZAKH_KINSHIP_LEXEMES = frozenset(
    (
        "әкем анам шешем ұлым ұлымыз қызым қызымыз балам баламыз ағам әпкем інім "
        "сіңлім қарындасым әйелім күйеуім әкесі анасы шешесі ұлы қызы баласы ағасы "
        "әпкесі інісі сіңлісі қарындасы әйелі күйеуі"
    ).split()
)
_NAME_CANDIDATE_EXCLUSIONS = frozenset(
    (
        "мен менің біз біздің ол оның олар олардың я мы мой моя мою наш наша его ее её их "
        "i we my our he she his her they their"
    ).split()
).union(_KAZAKH_KINSHIP_LEXEMES)


def _known_person_surfaces(person: KnownPerson) -> list[str]:
    return list(dict.fromkeys([person.canonical_name, *person.aliases]))


def _known_speaker_person_id(known_people: list[KnownPerson], speaker_name: str) -> str | None:
    normalized_speaker = normalize_text(speaker_name)
    matches = [
        person.person_id
        for person in known_people
        if any(
            normalize_text(surface) == normalized_speaker
            for surface in _known_person_surfaces(person)
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _lexical_annotations(transcript: TranscriptEnvelope) -> list[ExtractionLexicalAnnotation]:
    raw: list[tuple[str, str, int, int, LexicalAnnotationType, str, str]] = []
    for segment in transcript.segments:
        for match in find_speaker_anchor_matches(segment.text):
            if match.start < 0 or match.end <= match.start:
                continue
            raw.append(
                (
                    segment.segment_id,
                    match.surface,
                    match.start,
                    match.end,
                    LexicalAnnotationType.SPEAKER_ANCHOR,
                    match.language,
                    match.rule_id,
                )
            )
        for language, matches in (
            ("ru", russian.find_kinship_matches(segment.text)),
            ("en", english.find_kinship_matches(segment.text)),
        ):
            for match in matches:
                raw.append(
                    (
                        segment.segment_id,
                        match.surface,
                        match.start,
                        match.end,
                        LexicalAnnotationType.KINSHIP_LEXEME,
                        language,
                        match.rule_id,
                    )
                )
        for token in tokenize(segment.text):
            if token.normalized not in _KAZAKH_KINSHIP_LEXEMES:
                continue
            raw.append(
                (
                    segment.segment_id,
                    token.surface,
                    token.start,
                    token.end,
                    LexicalAnnotationType.KINSHIP_LEXEME,
                    "kk",
                    "kk.extraction_anchor.audited_kinship_lexeme.v1",
                )
            )

    unique = list(dict.fromkeys(raw))
    annotations: list[ExtractionLexicalAnnotation] = []
    for index, item in enumerate(unique, start=1):
        segment_id, surface, start, end, annotation_type, language, rule_id = item
        annotations.append(
            ExtractionLexicalAnnotation(
                annotation_id=f"annotation_{index:03d}",
                segment_id=segment_id,
                surface=surface,
                start_char=start,
                end_char=end,
                annotation_type=annotation_type,
                language=language,
                rule_id=rule_id,
            )
        )
    return annotations


def _known_person_anchors(
    *,
    transcript: TranscriptEnvelope,
    cleaned: CleanerResult,
    known_people: list[KnownPerson],
) -> list[_AnchorCandidate]:
    readable_by_id = {segment.segment_id: segment.text for segment in cleaned.readable_segments}
    anchors: list[_AnchorCandidate] = []
    for person in known_people:
        for segment in transcript.segments:
            for source_layer, text in (
                (EvidenceSourceLayer.RAW_TRANSCRIPT, segment.text),
                (EvidenceSourceLayer.READABLE_TRANSCRIPT, readable_by_id[segment.segment_id]),
            ):
                for surface in _known_person_surfaces(person):
                    for match in find_known_name_matches(text, surface):
                        anchors.append(
                            _AnchorCandidate(
                                surface=match.token,
                                normalized=normalize_text(match.token),
                                segment_id=segment.segment_id,
                                source_layer=source_layer,
                                start_char=match.start if match.start >= 0 else None,
                                end_char=match.end if match.end > 0 else None,
                                anchor_kind=MentionAnchorKind.KNOWN_PERSON,
                                known_person_id=person.person_id,
                                rule_ids=(match.rule_id,),
                            )
                        )
    return anchors


def _speaker_anchors(
    *,
    speaker_name: str,
    known_people: list[KnownPerson],
    annotations: list[ExtractionLexicalAnnotation],
) -> list[_AnchorCandidate]:
    speaker_person_id = _known_speaker_person_id(known_people, speaker_name)
    segment_ids = list(
        dict.fromkeys(
            annotation.segment_id
            for annotation in annotations
            if annotation.annotation_type is LexicalAnnotationType.SPEAKER_ANCHOR
        )
    )
    return [
        _AnchorCandidate(
            surface=speaker_name,
            normalized=normalize_text(speaker_name),
            segment_id=segment_id,
            source_layer=EvidenceSourceLayer.RAW_TRANSCRIPT,
            start_char=None,
            end_char=None,
            anchor_kind=MentionAnchorKind.SPEAKER,
            known_person_id=speaker_person_id,
            rule_ids=("extraction.anchor.supplied_speaker.v1",),
        )
        for segment_id in segment_ids
    ]


def _name_candidate_anchors(
    *,
    cleaned: CleanerResult,
    annotations: list[ExtractionLexicalAnnotation],
) -> list[_AnchorCandidate]:
    kinship_segments = {
        annotation.segment_id
        for annotation in annotations
        if annotation.annotation_type is LexicalAnnotationType.KINSHIP_LEXEME
    }
    anchors: list[_AnchorCandidate] = []
    for segment in cleaned.readable_segments:
        if segment.segment_id not in kinship_segments:
            continue
        tokens = tokenize(segment.text)
        kinship_indexes = {
            index
            for index, token in enumerate(tokens)
            if token.normalized in _KAZAKH_KINSHIP_LEXEMES
            or russian.find_kinship_matches(token.surface)
            or english.find_kinship_matches(token.surface)
        }
        for index, token in enumerate(tokens):
            if not token.surface[:1].isupper():
                continue
            if len(token.normalized) < 2 or token.normalized in _NAME_CANDIDATE_EXCLUSIONS:
                continue
            nearest_kinship = min(
                (abs(index - kinship_index) for kinship_index in kinship_indexes),
                default=999,
            )
            if nearest_kinship > 4:
                continue
            anchors.append(
                _AnchorCandidate(
                    surface=token.surface,
                    normalized=token.normalized,
                    segment_id=segment.segment_id,
                    source_layer=EvidenceSourceLayer.READABLE_TRANSCRIPT,
                    start_char=token.start,
                    end_char=token.end,
                    anchor_kind=MentionAnchorKind.NAME_CANDIDATE,
                    known_person_id=None,
                    rule_ids=("extraction.anchor.capitalized_near_kinship.v1",),
                )
            )
    return anchors


def build_extraction_anchor_bundle(
    *,
    transcript: TranscriptEnvelope,
    cleaned: CleanerResult,
    speaker_name: str,
    known_people: list[KnownPerson],
) -> ExtractionAnchorBundle:
    annotations = _lexical_annotations(transcript)
    raw_anchors = [
        *_known_person_anchors(
            transcript=transcript,
            cleaned=cleaned,
            known_people=known_people,
        ),
        *_speaker_anchors(
            speaker_name=speaker_name,
            known_people=known_people,
            annotations=annotations,
        ),
        *_name_candidate_anchors(cleaned=cleaned, annotations=annotations),
    ]

    unique: dict[tuple[str, str, str, str | None], _AnchorCandidate] = {}
    for candidate in raw_anchors:
        key = (
            candidate.anchor_kind.value,
            candidate.segment_id,
            candidate.normalized,
            candidate.known_person_id,
        )
        unique.setdefault(key, candidate)

    mention_anchors = [
        ExtractionMentionAnchor(
            anchor_id=f"anchor_{index:03d}",
            surface=candidate.surface,
            normalized=candidate.normalized,
            segment_id=candidate.segment_id,
            source_layer=candidate.source_layer,
            start_char=candidate.start_char,
            end_char=candidate.end_char,
            anchor_kind=candidate.anchor_kind,
            known_person_id=candidate.known_person_id,
            rule_ids=list(candidate.rule_ids),
        )
        for index, candidate in enumerate(unique.values(), start=1)
    ]
    return ExtractionAnchorBundle(
        allowed_segment_ids=[segment.segment_id for segment in transcript.segments],
        mention_anchors=mention_anchors,
        lexical_annotations=annotations,
    )
