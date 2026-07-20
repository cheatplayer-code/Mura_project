from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mura.domain.models import GrammaticalNumber, RelationshipRole, RelationshipType
from mura.linguistics import english, kazakh, russian
from mura.linguistics.common import tokenize


class KinshipFrame(Protocol):
    @property
    def relationship_type(self) -> RelationshipType: ...

    @property
    def possessor_role(self) -> RelationshipRole: ...

    @property
    def relative_role(self) -> RelationshipRole: ...


@dataclass(frozen=True)
class AnaphorOccurrence:
    surface: str
    start: int
    end: int
    language: str
    grammatical_number: GrammaticalNumber


@dataclass(frozen=True)
class KinshipOccurrence:
    surface: str
    start: int
    end: int
    language: str
    frame: KinshipFrame


_SINGULAR_ANAPHORS: dict[str, str] = {
    "оның": "kk",
    "ол": "kk",
    "его": "ru",
    "ее": "ru",
    "её": "ru",
    "он": "ru",
    "она": "ru",
    "his": "en",
    "her": "en",
}
_PLURAL_ANAPHORS: dict[str, str] = {
    "олардың": "kk",
    "олар": "kk",
    "их": "ru",
    "они": "ru",
    "their": "en",
}
_RU_U_ANAPHORS: dict[str, GrammaticalNumber] = {
    "него": GrammaticalNumber.SINGULAR,
    "нее": GrammaticalNumber.SINGULAR,
    "неё": GrammaticalNumber.SINGULAR,
    "них": GrammaticalNumber.PLURAL,
}
_KINSHIP_WINDOW_CHARS = 32
_PARENT = RelationshipRole.PARENT
_CHILD = RelationshipRole.CHILD
_PARENT_CHILD = RelationshipType.PARENT_CHILD
_ADDITIONAL_RU_KINSHIP_FRAMES: dict[str, russian.KinshipFrame] = {
    "дети": russian.KinshipFrame(_PARENT_CHILD, _PARENT, _CHILD),
    "детей": russian.KinshipFrame(_PARENT_CHILD, _PARENT, _CHILD),
    "сыновей": russian.KinshipFrame(_PARENT_CHILD, _PARENT, _CHILD),
}


def find_anaphors(text: str) -> list[AnaphorOccurrence]:
    tokens = tokenize(text)
    matches: list[AnaphorOccurrence] = []
    for index, token in enumerate(tokens):
        language = _SINGULAR_ANAPHORS.get(token.normalized)
        if language is not None:
            matches.append(
                AnaphorOccurrence(
                    surface=token.surface,
                    start=token.start,
                    end=token.end,
                    language=language,
                    grammatical_number=GrammaticalNumber.SINGULAR,
                )
            )
            continue
        language = _PLURAL_ANAPHORS.get(token.normalized)
        if language is not None:
            matches.append(
                AnaphorOccurrence(
                    surface=token.surface,
                    start=token.start,
                    end=token.end,
                    language=language,
                    grammatical_number=GrammaticalNumber.PLURAL,
                )
            )
            continue
        number = _RU_U_ANAPHORS.get(token.normalized)
        if number is None or index == 0 or tokens[index - 1].normalized != "у":
            continue
        prefix = tokens[index - 1]
        matches.append(
            AnaphorOccurrence(
                surface=text[prefix.start : token.end],
                start=prefix.start,
                end=token.end,
                language="ru",
                grammatical_number=number,
            )
        )
    unique: dict[tuple[int, int, str], AnaphorOccurrence] = {}
    for match in matches:
        unique.setdefault((match.start, match.end, match.surface), match)
    return list(unique.values())


def _kazakh_kinship_matches(text: str) -> list[KinshipOccurrence]:
    frames = getattr(kazakh, "_NAMED_POSSESSOR_FRAMES")
    return [
        KinshipOccurrence(
            surface=token.surface,
            start=token.start,
            end=token.end,
            language="kk",
            frame=frames[token.normalized],
        )
        for token in tokenize(text)
        if token.normalized in frames
    ]


def _additional_russian_kinship_matches(text: str) -> list[KinshipOccurrence]:
    return [
        KinshipOccurrence(
            surface=token.surface,
            start=token.start,
            end=token.end,
            language="ru",
            frame=_ADDITIONAL_RU_KINSHIP_FRAMES[token.normalized],
        )
        for token in tokenize(text)
        if token.normalized in _ADDITIONAL_RU_KINSHIP_FRAMES
    ]


def _kinship_matches(text: str) -> list[KinshipOccurrence]:
    matches = [
        *_kazakh_kinship_matches(text),
        *_additional_russian_kinship_matches(text),
        *(
            KinshipOccurrence(
                surface=item.surface,
                start=item.start,
                end=item.end,
                language="ru",
                frame=item.frame,
            )
            for item in russian.find_kinship_matches(text)
        ),
        *(
            KinshipOccurrence(
                surface=item.surface,
                start=item.start,
                end=item.end,
                language="en",
                frame=item.frame,
            )
            for item in english.find_kinship_matches(text)
        ),
    ]
    unique: dict[tuple[int, int, str], KinshipOccurrence] = {}
    for match in matches:
        unique.setdefault((match.start, match.end, match.surface), match)
    return list(unique.values())


def nearest_kinship(text: str, anaphor: AnaphorOccurrence) -> KinshipOccurrence | None:
    candidates = [
        match
        for match in _kinship_matches(text)
        if match.start >= anaphor.end and match.start - anaphor.end <= _KINSHIP_WINDOW_CHARS
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.start - anaphor.end, -(item.end - item.start)))
    return candidates[0]
