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
    accepted_ids = {
        relationship.relationship_id
        for relationship in result.relationship_claims
    }
    for relationship in result.relationship_claims:
        analysis = analyze_relationship_evidence(
            relationship=relationship,
            transcript=transcript,
            people=result.people_mentions,
            speaker_name=result.speaker_name,
        )
        if (
            relationship.conflict_ids
            and analysis.grounding_decision
            == "insufficient_deterministic_signal"
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
        analysis = context.get("evidence_analysis")
        if not isinstance(analysis, dict):
            continue
        if analysis.get("grounding_decision") in _REJECTED_DECISIONS:
            rejected_ids.add(object_id)
    counters["ambiguous_grounding_rejected"] = len(rejected_ids)
    return counters


def _unique_quarantined_relationship_ids(
    *,
    result: ExtractionResult,
    extraction_issues: list[dict[str, Any]],
) -> set[str]:
    accepted_ids = {
        relationship.relationship_id
        for relationship in result.relationship_claims
    }
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

    original: Callable[..., dict[str, Any]] = service_type._extraction_usage

    def extraction_usage(
        self: Any,
        *,
        raw: dict[str, Any],
        usage: Any,
        result: ExtractionResult,
        extraction_issues: list[dict[str, Any]],
        evidence_closure_count: int,
        evidence_recovery: Any,
        anchors: Any,
        repair_attempted: bool,
    ) -> dict[str, Any]:
        payload = original(
            self,
            raw=raw,
            usage=usage,
            result=result,
            extraction_issues=extraction_issues,
            evidence_closure_count=evidence_closure_count,
            evidence_recovery=evidence_recovery,
            anchors=anchors,
            repair_attempted=repair_attempted,
        )
        relationship_metrics = payload.get("relationship_metrics")
        if isinstance(relationship_metrics, dict):
            relationship_metrics["quarantined"] = len(
                _unique_quarantined_relationship_ids(
                    result=result,
                    extraction_issues=extraction_issues,
                )
            )
        transcript = getattr(anchors, "transcript", None)
        if not isinstance(transcript, TranscriptEnvelope):
            transcript = getattr(self, "_relationship_metrics_transcript", None)
        if isinstance(transcript, TranscriptEnvelope):
            payload["relationship_grounding_metrics"] = (
                relationship_grounding_counters(
                    result=result,
                    transcript=transcript,
                    extraction_issues=extraction_issues,
                )
            )
        return payload

    service_type._extraction_usage = extraction_usage
    setattr(service_type, _INSTALL_MARKER, True)
