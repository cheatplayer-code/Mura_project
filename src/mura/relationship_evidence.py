from __future__ import annotations

from dataclasses import replace
from typing import Any

from mura._relationship_evidence_core import (
    RelationshipEvidenceAnalysis,
    contains_exact_surface,
    contains_surface,
    exactly_named_people,
    explicitly_named_people,
    has_first_person_reference,
    joined_segment_text,
    normalize_evidence,
    person_name_surfaces,
    speaker_mentions,
)
from mura._relationship_evidence_core import (
    analyze_relationship_evidence as _analyze_relationship_evidence,
)
from mura.domain.models import EvidenceClass, PersonMention, RelationshipClaim, TranscriptEnvelope

__all__ = [
    "RelationshipEvidenceAnalysis",
    "analyze_relationship_evidence",
    "contains_exact_surface",
    "contains_surface",
    "exactly_named_people",
    "explicitly_named_people",
    "has_first_person_reference",
    "joined_segment_text",
    "normalize_evidence",
    "person_name_surfaces",
    "speaker_mentions",
]


def _resolved_coreference_signal(
    *,
    relationship: RelationshipClaim,
    transcript: TranscriptEnvelope,
    people: list[PersonMention],
    resolved_antecedent_ids: set[str],
) -> tuple[dict[str, str] | None, bool | None]:
    if not resolved_antecedent_ids:
        return None, None

    # Imported lazily because coreference itself imports person_name_surfaces from this module.
    import mura.coreference as coreference

    source_text = joined_segment_text(relationship.source_segment_ids, transcript)
    endpoint_ids = {relationship.subject_mention_id, relationship.object_mention_id}
    occurrences = coreference._person_occurrences(source_text, people)
    first_conflict: dict[str, str] | None = None

    for anaphor in coreference._anaphors(source_text):
        kinship = coreference._nearest_kinship(source_text, anaphor)
        if kinship is None:
            continue
        target_id = coreference._unique_target(occurrences, kinship_end=kinship.end)
        if target_id is None or target_id not in endpoint_ids:
            continue

        possessor_ids = endpoint_ids.intersection(resolved_antecedent_ids) - {target_id}
        for possessor_id in possessor_ids:
            edge = coreference._expected_edge(possessor_id, target_id, kinship.frame)
            relationship_type, subject_id, subject_role, object_id, object_role = edge
            signal = {
                "language": kinship.language,
                "relationship_type": relationship_type.value,
                "subject_mention_id": subject_id,
                "subject_role": subject_role.value,
                "object_mention_id": object_id,
                "object_role": object_role.value,
                "source_surface": f"{anaphor.surface} {kinship.surface}",
                "rule_id": (f"{kinship.language}.relationship.resolved_possessive_coreference.v1"),
            }
            if coreference._relationship_matches_edge(relationship, edge):
                return signal, True
            first_conflict = first_conflict or signal

    if first_conflict is not None:
        return first_conflict, False
    return None, None


def _coordinated_english_sibling_signal(
    *,
    relationship: RelationshipClaim,
    transcript: TranscriptEnvelope,
    analysis: RelationshipEvidenceAnalysis,
) -> dict[str, str] | None:
    if relationship.relationship_type.value != "sibling":
        return None
    if relationship.subject_role.value != "sibling" or relationship.object_role.value != "sibling":
        return None
    exact_ids = {item["mention_id"] for item in analysis.exact_people}
    endpoint_ids = {relationship.subject_mention_id, relationship.object_mention_id}
    if not endpoint_ids.issubset(exact_ids):
        return None

    source_text = normalize_evidence(joined_segment_text(relationship.source_segment_ids, transcript))
    if not any(
        phrase in f" {source_text} "
        for phrase in (" are sisters ", " are brothers ", " are siblings ")
    ):
        return None
    return {
        "language": "en",
        "relationship_type": "sibling",
        "subject_mention_id": relationship.subject_mention_id,
        "subject_role": "sibling",
        "object_mention_id": relationship.object_mention_id,
        "object_role": "sibling",
        "source_surface": "are siblings",
        "rule_id": "en.relationship.explicit_sibling_pair.v1",
    }


def _append_signal(
    values: list[dict[str, str]],
    signal: dict[str, str],
) -> list[dict[str, str]]:
    key = (
        signal["language"],
        signal["relationship_type"],
        signal["subject_mention_id"],
        signal["subject_role"],
        signal["object_mention_id"],
        signal["object_role"],
    )
    existing = {
        (
            item["language"],
            item["relationship_type"],
            item["subject_mention_id"],
            item["subject_role"],
            item["object_mention_id"],
            item["object_role"],
        )
        for item in values
    }
    return values if key in existing else [*values, signal]


def _apply_signal(
    analysis: RelationshipEvidenceAnalysis,
    *,
    signal: dict[str, str],
    evidence_class: EvidenceClass,
    auto_accept_eligible: bool,
    role_consistent: bool,
) -> RelationshipEvidenceAnalysis:
    all_signals = _append_signal(analysis.linguistic_relationship_signals, signal)
    language_signals: dict[str, list[dict[str, str]]] = {
        "kk": analysis.kazakh_relationship_signals,
        "ru": analysis.russian_relationship_signals,
        "en": analysis.english_relationship_signals,
        "mixed": analysis.code_switching_relationship_signals,
    }
    language_signals[signal["language"]] = _append_signal(
        language_signals.get(signal["language"], []),
        signal,
    )
    return replace(
        analysis,
        linguistic_relationship_signals=all_signals,
        kazakh_relationship_signals=language_signals["kk"],
        russian_relationship_signals=language_signals["ru"],
        english_relationship_signals=language_signals["en"],
        code_switching_relationship_signals=language_signals["mixed"],
        role_consistent=role_consistent,
        linguistic_rule_ids=sorted({*analysis.linguistic_rule_ids, signal["rule_id"]}),
        evidence_class=evidence_class.value,
        auto_accept_eligible=auto_accept_eligible,
    )


def analyze_relationship_evidence(
    *,
    relationship: RelationshipClaim,
    transcript: TranscriptEnvelope,
    people: list[PersonMention],
    speaker_name: str,
    resolved_coreference_antecedent_ids: set[str] | None = None,
) -> RelationshipEvidenceAnalysis:
    resolved = resolved_coreference_antecedent_ids or set()
    analysis = _analyze_relationship_evidence(
        relationship=relationship,
        transcript=transcript,
        people=people,
        speaker_name=speaker_name,
        resolved_coreference_antecedent_ids=resolved,
    )

    direct_signal = _coordinated_english_sibling_signal(
        relationship=relationship,
        transcript=transcript,
        analysis=analysis,
    )
    if direct_signal is not None:
        return _apply_signal(
            analysis,
            signal=direct_signal,
            evidence_class=EvidenceClass.A_EXPLICIT,
            auto_accept_eligible=True,
            role_consistent=True,
        )

    signal, matches = _resolved_coreference_signal(
        relationship=relationship,
        transcript=transcript,
        people=people,
        resolved_antecedent_ids=resolved,
    )
    if signal is None or matches is None:
        return analysis
    return _apply_signal(
        analysis,
        signal=signal,
        evidence_class=(
            EvidenceClass.D_CONTEXT_RESOLVED if matches else EvidenceClass.U_UNCERTAIN
        ),
        auto_accept_eligible=False,
        role_consistent=matches,
    )
