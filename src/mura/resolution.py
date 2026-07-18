from __future__ import annotations

import re
import unicodedata

from mura.domain.models import (
    ExtractionResult,
    KnownPerson,
    MentionResolution,
    ResolutionStatus,
)
from mura.relationship_evidence import person_name_surfaces


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)


def resolve_mentions(
    extraction: ExtractionResult, known_people: list[KnownPerson]
) -> list[MentionResolution]:
    index: dict[str, list[KnownPerson]] = {}
    for person in known_people:
        for name in [person.canonical_name, *person.aliases]:
            key = normalize_name(name)
            if key:
                index.setdefault(key, []).append(person)

    resolutions: list[MentionResolution] = []
    for mention in extraction.people_mentions:
        candidate_map: dict[str, KnownPerson] = {}
        for surface in person_name_surfaces(mention):
            for candidate in index.get(normalize_name(surface), []):
                candidate_map[candidate.person_id] = candidate

        candidates = list(candidate_map.values())
        if len(candidates) == 1:
            candidate = candidates[0]
            relation_matches = (
                not mention.relation_to_speaker
                or not candidate.relation_to_speaker
                or mention.relation_to_speaker.casefold()
                == candidate.relation_to_speaker.casefold()
            )
            if relation_matches:
                resolutions.append(
                    MentionResolution(
                        mention_id=mention.mention_id,
                        status=ResolutionStatus.RESOLVED,
                        person_id=candidate.person_id,
                        candidate_person_ids=[candidate.person_id],
                        reason="exact canonical-name, alias, or structured name-variant match",
                    )
                )
                continue

        if candidates:
            resolutions.append(
                MentionResolution(
                    mention_id=mention.mention_id,
                    status=ResolutionStatus.NEEDS_REVIEW,
                    candidate_person_ids=sorted(candidate_map),
                    reason="name matched, but multiple candidates or relation context conflicts",
                )
            )
        else:
            resolutions.append(
                MentionResolution(
                    mention_id=mention.mention_id,
                    status=ResolutionStatus.NEW_PERSON,
                    reason="no exact canonical-name, alias, or name-variant candidate",
                )
            )

    return resolutions
