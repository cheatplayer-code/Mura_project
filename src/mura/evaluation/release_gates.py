from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from mura.domain.models import StrictModel
from mura.evaluation.models import BenchmarkReport


class ReleaseGateConfig(StrictModel):
    schema_version: str = "release-gates-v1"
    minimum_case_count: int = Field(ge=1)
    minimum_adversarial_case_count: int = Field(ge=1)
    minimum_person_f1: float = Field(ge=0, le=1)
    minimum_relationship_precision: float = Field(ge=0, le=1)
    minimum_relationship_recall: float = Field(ge=0, le=1)
    minimum_quarantine_recall: float = Field(ge=0, le=1)
    minimum_direction_accuracy: float = Field(ge=0, le=1)
    minimum_provenance_completeness: float = Field(ge=0, le=1)
    maximum_unknown_segment_references: int = Field(ge=0)
    maximum_self_relationships: int = Field(ge=0)
    maximum_adversarial_relationship_false_positives: int = Field(ge=0)
    maximum_adversarial_quarantine_false_negatives: int = Field(ge=0)


class ReleaseGateResult(StrictModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    measurements: dict[str, int | float] = Field(default_factory=dict)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_release_gate_config(path: str | Path) -> ReleaseGateConfig:
    return ReleaseGateConfig.model_validate(_read_json(path))


def evaluate_release_gates(
    report: BenchmarkReport,
    config: ReleaseGateConfig,
) -> ReleaseGateResult:
    adversarial = [case for case in report.cases if "adversarial" in case.construction_tags]
    adversarial_relationship_false_positives = sum(
        case.relationships.false_positive for case in adversarial
    )
    adversarial_quarantine_false_negatives = sum(
        case.quarantined_relationships.false_negative for case in adversarial
    )

    measurements: dict[str, int | float] = {
        "case_count": report.summary.case_count,
        "adversarial_case_count": len(adversarial),
        "person_f1": report.summary.person_mentions.f1,
        "relationship_precision": report.summary.relationships.precision,
        "relationship_recall": report.summary.relationships.recall,
        "quarantine_recall": report.summary.quarantined_relationships.recall,
        "direction_accuracy": report.summary.relationship_direction_accuracy.value,
        "provenance_completeness": report.summary.provenance_completeness.value,
        "unknown_segment_references": report.summary.unknown_segment_references,
        "self_relationships": report.summary.self_relationships,
        "adversarial_relationship_false_positives": adversarial_relationship_false_positives,
        "adversarial_quarantine_false_negatives": adversarial_quarantine_false_negatives,
    }

    failures: list[str] = []

    def require_minimum(name: str, threshold: int | float) -> None:
        value = measurements[name]
        if value < threshold:
            failures.append(f"{name}={value} is below required minimum {threshold}")

    def require_maximum(name: str, threshold: int | float) -> None:
        value = measurements[name]
        if value > threshold:
            failures.append(f"{name}={value} exceeds allowed maximum {threshold}")

    require_minimum("case_count", config.minimum_case_count)
    require_minimum("adversarial_case_count", config.minimum_adversarial_case_count)
    require_minimum("person_f1", config.minimum_person_f1)
    require_minimum("relationship_precision", config.minimum_relationship_precision)
    require_minimum("relationship_recall", config.minimum_relationship_recall)
    require_minimum("quarantine_recall", config.minimum_quarantine_recall)
    require_minimum("direction_accuracy", config.minimum_direction_accuracy)
    require_minimum("provenance_completeness", config.minimum_provenance_completeness)
    require_maximum(
        "unknown_segment_references",
        config.maximum_unknown_segment_references,
    )
    require_maximum("self_relationships", config.maximum_self_relationships)
    require_maximum(
        "adversarial_relationship_false_positives",
        config.maximum_adversarial_relationship_false_positives,
    )
    require_maximum(
        "adversarial_quarantine_false_negatives",
        config.maximum_adversarial_quarantine_false_negatives,
    )
    return ReleaseGateResult(passed=not failures, failures=failures, measurements=measurements)


def render_release_gate_result(result: ReleaseGateResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [f"# Mura Release Gate: {status}", ""]
    for name, value in sorted(result.measurements.items()):
        lines.append(f"- {name}: `{value}`")
    if result.failures:
        lines.extend(["", "## Failures"])
        lines.extend(f"- {failure}" for failure in result.failures)
    return "\n".join(lines)
