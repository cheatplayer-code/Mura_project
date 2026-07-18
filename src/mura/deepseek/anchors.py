from __future__ import annotations

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
        if any(normalize_text(surface) == normalized_speaker for surface in _known_person_surfaces(person))
    ]
    return matches[0] if len(matches) == 1 else None


def _offsets(text: str, token: str) -> tuple[int | None, int | None]:
    start = text.casefold().find(token.casefold())
    if start < 0:
        return None, None
    return start, start + len(token)


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
    return [
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
        for index, (segment_id, surface, start, end, annotation_type, language, rule_id) in enumerate(
            unique, start=1
        )
    ]


def _known_person_anchors(
    *,
    transcript: TranscriptEnvelope,
    cleaned: CleanerResult,
    known_people: list[KnownPerson],
) -> list[tuple[str, str, str, EvidenceSourceLayer, int | None, int | None, str, list[str]]]:
    readable_by_id = {segment.segment_id: segment.text for segment in cleaned.readable_segments}
    anchors: list[
        tuple[str, str, str, EvidenceSourceLayer, int | None, int | None, str, list[str]]
    ] = []
    for person in known_people:
        for segment in transcript.segments:
            for source_layer, text in (
                (EvidenceSourceLayer.RAW_TRANSCRIPT, segment.text),
                (EvidenceSourceLayer.READABLE_TRANSCRIPT, readable_by_id[segment.segment_id]),
            ):
                for surface in _known_person_surfaces(person):
                    for match in find_known_name_matches(text, surface):
                        start = match.start if match.start >= 0 else None
                        end = match.end if match.end > 0 else None
                        anchors.append(
                            (
                                match.token,
                                normalize_text(match.token),
                                segment.segment_id,
                                source_layer,
                                start,
                                end,
                                person.person_id,
                                [match.rule_id],
                            )
                        )
    return anchors


def _speaker_anchors(
    *,
    transcript: TranscriptEnvelope,
    speaker_name: str,
    known_people: list[KnownPerson],
    annotations: list[ExtractionLexicalAnnotation],
) -> list[tuple[str, str, str, EvidenceSourceLayer, int | None, int | None, str | None, list[str]]]:
    speaker_person_id = _known_speaker_person_id(known_people, speaker_name)
    segment_ids = list(
        dict.fromkeys(
            annotation.segment_id
            for annotation in annotations
            if annotation.annotation_type is LexicalAnnotationType.SPEAKER_ANCHOR
        )
    )
    if not segment_ids:
        return []
    return [
        (
            speaker_name,
            normalize_text(speaker_name),
            segment_id,
            EvidenceSourceLayer.RAW_TRANSCRIPT,
            None,
            None,
            speaker_person_id,
            ["extraction.anchor.supplied_speaker.v1"],
        )
        for segment_id in segment_ids
    ]


def _name_candidate_anchors(
    *,
    cleaned: CleanerResult,
    annotations: list[ExtractionLexicalAnnotation],
) -> list[tuple[str, str, str, EvidenceSourceLayer, int, int, None, list[str]]]:
    kinship_segments = {
        annotation.segment_id
        for annotation in annotations
        if annotation.annotation_type is LexicalAnnotationType.KINSHIP_LEXEME
    }
    anchors: list[tuple[str, str, str, EvidenceSourceLayer, int, int, None, list[str]]] = []
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
            if not kinship_indexes or min(abs(index - kinship_index) for kinship_index in kinship_indexes) > 4:
                continue
            anchors.append(
                (
                    token.surface,
                    token.normalized,
                    segment.segment_id,
                    EvidenceSourceLayer.READABLE_TRANSCRIPT,
                    token.start,
                    token.end,
                    None,
                    ["extraction.anchor.capitalized_near_kinship.v1"],
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
            transcript=transcript,
            speaker_name=speaker_name,
            known_people=known_people,
            annotations=annotations,
        ),
        *_name_candidate_anchors(cleaned=cleaned, annotations=annotations),
    ]

    unique: dict[tuple[str, str, str, str | None], tuple[object, ...]] = {}
    for item in raw_anchors:
        surface, normalized, segment_id, source_layer, start, end, known_person_id, rule_ids = item
        anchor_kind = (
            MentionAnchorKind.SPEAKER
            if rule_ids == ["extraction.anchor.supplied_speaker.v1"]
            else MentionAnchorKind.KNOWN_PERSON
            if known_person_id is not None
            else MentionAnchorKind.NAME_CANDIDATE
        )
        key = (anchor_kind.value, str(segment_id), str(normalized), known_person_id)
        unique.setdefault(
            key,
            (
                surface,
                normalized,
                segment_id,
                source_layer,
                start,
                end,
                anchor_kind,
                known_person_id,
                rule_ids,
            ),
        )

    mention_anchors = [
        ExtractionMentionAnchor(
            anchor_id=f"anchor_{index:03d}",
            surface=str(surface),
            normalized=str(normalized),
            segment_id=str(segment_id),
            source_layer=source_layer,
            start_char=start,
            end_char=end,
            anchor_kind=anchor_kind,
            known_person_id=known_person_id,
            rule_ids=rule_ids,
        )
        for index, (
            surface,
            normalized,
            segment_id,
            source_layer,
            start,
            end,
            anchor_kind,
            known_person_id,
            rule_ids,
        ) in enumerate(unique.values(), start=1)
    ]
    return ExtractionAnchorBundle(
        allowed_segment_ids=[segment.segment_id for segment in transcript.segments],
        mention_anchors=mention_anchors,
        lexical_annotations=annotations,
    )
