from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mura.domain.models import TranscriptEnvelope


@dataclass(frozen=True)
class EvidenceOffsetRecoveryMetrics:
    exact_offsets_accepted: int = 0
    unique_match_repaired: int = 0
    nearest_match_repaired: int = 0
    ambiguous_offsets_removed: int = 0
    missing_text_quarantined: int = 0
    unknown_segment_quarantined: int = 0

    @property
    def repaired_evidence_offsets(self) -> int:
        return self.unique_match_repaired + self.nearest_match_repaired

    def to_dict(self) -> dict[str, int]:
        return {
            "repaired_evidence_offsets": self.repaired_evidence_offsets,
            "exact_offsets_accepted": self.exact_offsets_accepted,
            "unique_match_repaired": self.unique_match_repaired,
            "nearest_match_repaired": self.nearest_match_repaired,
            "ambiguous_offsets_removed": self.ambiguous_offsets_removed,
            "missing_text_quarantined": self.missing_text_quarantined,
            "unknown_segment_quarantined": self.unknown_segment_quarantined,
        }


def _all_occurrences(text: str, substring: str) -> list[int]:
    occurrences: list[int] = []
    cursor = 0
    while True:
        position = text.find(substring, cursor)
        if position < 0:
            return occurrences
        occurrences.append(position)
        cursor = position + 1


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _offsets_match(
    *,
    segment_text: str,
    evidence_text: str,
    start_char: object,
    end_char: object,
) -> bool:
    if not _is_integer(start_char) or not _is_integer(end_char):
        return False
    return (
        0 <= start_char < end_char <= len(segment_text)
        and segment_text[start_char:end_char] == evidence_text
    )


def recover_evidence_offsets(
    *,
    raw: dict[str, Any],
    transcript: TranscriptEnvelope,
) -> tuple[dict[str, Any], EvidenceOffsetRecoveryMetrics]:
    """Repair exact raw-transcript offsets without mutating model output or transcript."""

    segment_text_by_id = {segment.segment_id: segment.text for segment in transcript.segments}
    raw_evidence = raw.get("evidence_spans")
    if not isinstance(raw_evidence, list):
        return dict(raw), EvidenceOffsetRecoveryMetrics()

    exact_offsets_accepted = 0
    unique_match_repaired = 0
    nearest_match_repaired = 0
    ambiguous_offsets_removed = 0
    missing_text_quarantined = 0
    unknown_segment_quarantined = 0
    recovered_evidence: list[object] = []

    for candidate in raw_evidence:
        if not isinstance(candidate, dict):
            recovered_evidence.append(candidate)
            continue

        recovered_candidate = dict(candidate)
        segment_id = candidate.get("segment_id")
        evidence_text = candidate.get("text")
        if not isinstance(segment_id, str) or segment_id not in segment_text_by_id:
            unknown_segment_quarantined += 1
            recovered_evidence.append(recovered_candidate)
            continue
        if not isinstance(evidence_text, str) or not evidence_text:
            recovered_evidence.append(recovered_candidate)
            continue

        segment_text = segment_text_by_id[segment_id]
        proposed_start = candidate.get("start_char")
        proposed_end = candidate.get("end_char")
        if _offsets_match(
            segment_text=segment_text,
            evidence_text=evidence_text,
            start_char=proposed_start,
            end_char=proposed_end,
        ):
            exact_offsets_accepted += 1
            recovered_evidence.append(recovered_candidate)
            continue

        occurrences = _all_occurrences(segment_text, evidence_text)
        if not occurrences:
            missing_text_quarantined += 1
            recovered_evidence.append(recovered_candidate)
            continue

        if len(occurrences) == 1:
            start_char = occurrences[0]
            recovered_candidate["start_char"] = start_char
            recovered_candidate["end_char"] = start_char + len(evidence_text)
            unique_match_repaired += 1
            recovered_evidence.append(recovered_candidate)
            continue

        if _is_integer(proposed_start):
            minimum_distance = min(abs(position - proposed_start) for position in occurrences)
            nearest = [
                position
                for position in occurrences
                if abs(position - proposed_start) == minimum_distance
            ]
            if len(nearest) == 1:
                start_char = nearest[0]
                recovered_candidate["start_char"] = start_char
                recovered_candidate["end_char"] = start_char + len(evidence_text)
                nearest_match_repaired += 1
                recovered_evidence.append(recovered_candidate)
                continue

        recovered_candidate["start_char"] = None
        recovered_candidate["end_char"] = None
        ambiguous_offsets_removed += 1
        recovered_evidence.append(recovered_candidate)

    recovered_raw = dict(raw)
    recovered_raw["evidence_spans"] = recovered_evidence
    metrics = EvidenceOffsetRecoveryMetrics(
        exact_offsets_accepted=exact_offsets_accepted,
        unique_match_repaired=unique_match_repaired,
        nearest_match_repaired=nearest_match_repaired,
        ambiguous_offsets_removed=ambiguous_offsets_removed,
        missing_text_quarantined=missing_text_quarantined,
        unknown_segment_quarantined=unknown_segment_quarantined,
    )
    return recovered_raw, metrics
