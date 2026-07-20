from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mura.domain.models import ExtractionResult, TranscriptEnvelope
from mura.relationship_evidence import analyze_relationship_evidence

_INSTALL_MARKER = "_mura_relationship_telemetry_installed"
_ACCEPTED_COUNTERS = {
    "speaker_anchor": "speaker_anchor_accepted",
    "named_possessor": "named_possessor_accepted",
    "explicit_spouse": "explicit_spouse_accepted",
    "explicit_parent_child": "explicit_parent_child_accepted",
    "explicit_sibling": "explicit_sibling_accepted",
}
_REJECTED_DECISIONS = {
    "unsupported_endpoints",
    "insufficient_deterministic_signal",
    "conflicting_deterministic_signal",
}


def relationship_grounding_counters(
    *,
    result: ExtractionResult,
    transcript: TranscriptEnvelope,
    extraction_issues: list[dict[str, Any]],
) -> dict[str, int]:
    counters = {
        "speaker_anchor_accepted": 0,
        "named_possessor_accepted": 0,
        "explicit_spouse_accepted": 0,
        "explicit_parent_child_accepted": 0,
        "explicit_sibling_accepted": 0,
        "ambiguous_grounding_rejected": 0,
        "conflict_linked_preserved": 0,
    }
    accepted_ids = {relationship.relationship_id for relationship in result.relationship_claims}
    for relationship in result.relationship_claims:
        analysis = analyze_relationship_evidence(
            relationship=relationship,
            transcript=transcript,
            people=result.people_mentions,
            speaker_name=result.speaker_name,
        )
        if (
            relationship.conflict_ids
            and analysis.grounding_decision == "insufficient_deterministic_signal"
        ):
            counters["conflict_linked_preserved"] += 1
            continue
        if not analysis.auto_accept_eligible:
            continue
        counter = _ACCEPTED_COUNTERS.get(analysis.grounding_decision)
        if counter is not None:
            counters[counter] += 1

    rejected_ids: set[str] = set()
    for issue in extraction_issues:
        if issue.get("object_type") != "relationship":
            continue
        object_id = issue.get("object_id")
        if not isinstance(object_id, str) or object_id in accepted_ids:
            continue
        context = issue.get("context")
        if not isinstance(context, dict):
            continue
        issue_analysis = context.get("evidence_analysis")
        if not isinstance(issue_analysis, dict):
            continue
        if issue_analysis.get("grounding_decision") in _REJECTED_DECISIONS:
            rejected_ids.add(object_id)
    counters["ambiguous_grounding_rejected"] = len(rejected_ids)
    return counters


def _unique_quarantined_relationship_ids(
    *,
    result: ExtractionResult,
    extraction_issues: list[dict[str, Any]],
) -> set[str]:
    accepted_ids = {relationship.relationship_id for relationship in result.relationship_claims}
    return {
        object_id
        for issue in extraction_issues
        if issue.get("object_type") == "relationship"
        and isinstance((object_id := issue.get("object_id")), str)
        and object_id not in accepted_ids
    }


def install_relationship_telemetry(service_type: type[Any]) -> None:
    if getattr(service_type, _INSTALL_MARKER, False):
        return

    original: Callable[..., tuple[ExtractionResult, dict[str, Any]]] = service_type.extract

    def extract(
        self: Any,
        *,
        transcript: TranscriptEnvelope,
        cleaned: Any,
        speaker_id: str,
        speaker_name: str,
        known_people: Any = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        result, payload = original(
            self,
            transcript=transcript,
            cleaned=cleaned,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            known_people=known_people,
        )
        extraction_issues = payload.get("extraction_issues", [])
        if not isinstance(extraction_issues, list):
            extraction_issues = []
        relationship_metrics = payload.get("relationship_metrics")
        if isinstance(relationship_metrics, dict):
            relationship_metrics["quarantined"] = len(
                _unique_quarantined_relationship_ids(
                    result=result,
                    extraction_issues=extraction_issues,
                )
            )
        payload["relationship_grounding_metrics"] = relationship_grounding_counters(
            result=result,
            transcript=transcript,
            extraction_issues=extraction_issues,
        )
        return result, payload

    service_type.extract = extract
    setattr(service_type, _INSTALL_MARKER, True)
