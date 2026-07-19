from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from mura.domain.models import StrictModel
from mura.evaluation.entity_resolution import EntityResolutionBenchmarkReport
from mura.evaluation.models import (
    BenchmarkReport,
    BenchmarkSummary,
    DatasetLayer,
    LanguageBucket,
)


class GateProfile(StrEnum):
    PULL_REQUEST = "pull_request"
    PRODUCTION = "production"


class GateThresholds(StrictModel):
    min_relationship_precision: float = Field(default=0.97, ge=0, le=1)
    min_relationship_recall: float = Field(default=0.90, ge=0, le=1)
    min_direction_accuracy: float = Field(default=0.99, ge=0, le=1)
    min_provenance_completeness: float = Field(default=1.0, ge=0, le=1)
    max_unsupported_acceptance_rate: float = Field(default=0.01, ge=0, le=1)
    max_unknown_segment_references: int = Field(default=0, ge=0)
    max_self_relationships: int = Field(default=0, ge=0)
    max_accepted_claims_without_evidence: int = Field(default=0, ge=0)
    max_critical_graph_violations: int = Field(default=0, ge=0)
    max_false_merges: int = Field(default=0, ge=0)
    max_false_splits: int = Field(default=0, ge=0)
    min_entity_identity_accuracy: float = Field(default=1.0, ge=0, le=1)
    min_entity_review_routing_accuracy: float = Field(default=1.0, ge=0, le=1)
    required_languages: list[LanguageBucket] = Field(
        default_factory=lambda: list(LanguageBucket)
    )
    min_adversarial_cases: int = Field(default=8, ge=0)
    min_approved_real_narrators: int = Field(default=0, ge=0)
    max_common_case_f1_regression: float = Field(default=0.0, ge=0, le=1)


class ReleaseGateConfig(StrictModel):
    schema_version: str = "release-gates-v1"
    profiles: dict[str, GateThresholds]

    def thresholds_for(self, profile: GateProfile) -> GateThresholds:
        thresholds = self.profiles.get(profile.value)
        if thresholds is None:
            raise ValueError(f"release gate profile {profile.value!r} is not configured")
        return thresholds


class GateCheck(StrictModel):
    check_id: str
    category: str
    passed: bool
    actual: int | float | str | bool
    comparator: str
    threshold: int | float | str | bool
    detail: str


class ReleaseGateResult(StrictModel):
    schema_version: str = "release-gate-result-v1"
    profile: GateProfile
    passed: bool
    checks: list[GateCheck]
    production_eligible: bool
    failed_check_ids: list[str] = Field(default_factory=list)


def load_release_gate_config(path: str | Path) -> ReleaseGateConfig:
    return ReleaseGateConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_benchmark_report(path: str | Path) -> BenchmarkReport:
    return BenchmarkReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _summary_for_language(
    report: BenchmarkReport,
    language: LanguageBucket,
) -> BenchmarkSummary | None:
    return next(
        (
            item.summary
            for item in report.slices
            if item.dimension == "language" and item.key == language.value
        ),
        None,
    )


def _check(
    checks: list[GateCheck],
    *,
    check_id: str,
    category: str,
    passed: bool,
    actual: int | float | str | bool,
    comparator: str,
    threshold: int | float | str | bool,
    detail: str,
) -> None:
    checks.append(
        GateCheck(
            check_id=check_id,
            category=category,
            passed=passed,
            actual=actual,
            comparator=comparator,
            threshold=threshold,
            detail=detail,
        )
    )


def _minimum(
    checks: list[GateCheck],
    *,
    check_id: str,
    category: str,
    actual: float,
    threshold: float,
    detail: str,
) -> None:
    _check(
        checks,
        check_id=check_id,
        category=category,
        passed=actual >= threshold,
        actual=round(actual, 6),
        comparator=">=",
        threshold=threshold,
        detail=detail,
    )


def _maximum(
    checks: list[GateCheck],
    *,
    check_id: str,
    category: str,
    actual: int | float,
    threshold: int | float,
    detail: str,
) -> None:
    _check(
        checks,
        check_id=check_id,
        category=category,
        passed=actual <= threshold,
        actual=round(actual, 6) if isinstance(actual, float) else actual,
        comparator="<=",
        threshold=threshold,
        detail=detail,
    )


def _add_summary_checks(
    checks: list[GateCheck],
    *,
    prefix: str,
    category: str,
    summary: BenchmarkSummary,
    thresholds: GateThresholds,
    enforce_quality: bool,
) -> None:
    _minimum(
        checks,
        check_id=f"{prefix}.provenance_completeness",
        category=category,
        actual=summary.provenance_completeness.value,
        threshold=thresholds.min_provenance_completeness,
        detail="Every accepted benchmark object must retain source provenance.",
    )
    _maximum(
        checks,
        check_id=f"{prefix}.unknown_segment_references",
        category=category,
        actual=summary.unknown_segment_references,
        threshold=thresholds.max_unknown_segment_references,
        detail="Accepted objects may reference only transcript segments present in the fixture.",
    )
    _maximum(
        checks,
        check_id=f"{prefix}.self_relationships",
        category=category,
        actual=summary.self_relationships,
        threshold=thresholds.max_self_relationships,
        detail="A relationship may not connect a mention to itself.",
    )
    _maximum(
        checks,
        check_id=f"{prefix}.accepted_claims_without_evidence",
        category=category,
        actual=summary.accepted_claims_without_evidence,
        threshold=thresholds.max_accepted_claims_without_evidence,
        detail="Accepted claims require materialized evidence IDs, not only model output.",
    )
    _maximum(
        checks,
        check_id=f"{prefix}.critical_graph_violations",
        category=category,
        actual=summary.critical_graph_violations,
        threshold=thresholds.max_critical_graph_violations,
        detail="Critical graph safety invariants must remain at zero.",
    )
    _maximum(
        checks,
        check_id=f"{prefix}.unsupported_relationship_acceptance",
        category=category,
        actual=summary.unsupported_relationship_acceptance.value,
        threshold=thresholds.max_unsupported_acceptance_rate,
        detail="False positive accepted relationships are bounded by the hard safety rate.",
    )

    has_relationship_scope = (
        summary.relationships.true_positive
        + summary.relationships.false_positive
        + summary.relationships.false_negative
        > 0
    )
    if enforce_quality and has_relationship_scope:
        _minimum(
            checks,
            check_id=f"{prefix}.relationship_precision",
            category=category,
            actual=summary.relationships.precision,
            threshold=thresholds.min_relationship_precision,
            detail="Accepted relationships must stay high precision.",
        )
        _minimum(
            checks,
            check_id=f"{prefix}.relationship_recall",
            category=category,
            actual=summary.relationships.recall,
            threshold=thresholds.min_relationship_recall,
            detail="The release set must retain supported relationships.",
        )
        _minimum(
            checks,
            check_id=f"{prefix}.relationship_direction_accuracy",
            category=category,
            actual=summary.relationship_direction_accuracy.value,
            threshold=thresholds.min_direction_accuracy,
            detail="Parent, child, and ordered sibling direction must remain correct.",
        )


def _add_baseline_checks(
    checks: list[GateCheck],
    *,
    report: BenchmarkReport,
    baseline: BenchmarkReport,
    thresholds: GateThresholds,
) -> None:
    current_by_id = {case.case_id: case for case in report.cases}
    baseline_by_id = {case.case_id: case for case in baseline.cases}
    missing = sorted(set(baseline_by_id) - set(current_by_id))
    _check(
        checks,
        check_id="regression.baseline_case_coverage",
        category="regression",
        passed=not missing,
        actual=len(baseline_by_id) - len(missing),
        comparator="==",
        threshold=len(baseline_by_id),
        detail=f"All approved baseline cases must remain present; missing={missing}.",
    )
    for case_id in sorted(set(current_by_id).intersection(baseline_by_id)):
        current = current_by_id[case_id]
        previous = baseline_by_id[case_id]
        floor = previous.relationships.f1 - thresholds.max_common_case_f1_regression
        _minimum(
            checks,
            check_id=f"regression.{case_id}.relationship_f1",
            category="regression",
            actual=current.relationships.f1,
            threshold=max(0.0, floor),
            detail=(
                "A common baseline case may not lose relationship F1 beyond the "
                "configured budget."
            ),
        )
        _minimum(
            checks,
            check_id=f"regression.{case_id}.quarantine_f1",
            category="regression",
            actual=current.quarantined_relationships.f1,
            threshold=previous.quarantined_relationships.f1,
            detail="Expected quarantine behavior may not regress on an approved case.",
        )
        _maximum(
            checks,
            check_id=f"regression.{case_id}.critical_graph_violations",
            category="regression",
            actual=current.critical_graph_violations,
            threshold=previous.critical_graph_violations,
            detail="A common case may not introduce a new critical graph violation.",
        )


def evaluate_release_gates(
    *,
    report: BenchmarkReport,
    entity_report: EntityResolutionBenchmarkReport,
    config: ReleaseGateConfig,
    profile: GateProfile,
    baseline: BenchmarkReport | None = None,
) -> ReleaseGateResult:
    thresholds = config.thresholds_for(profile)
    checks: list[GateCheck] = []
    _add_summary_checks(
        checks,
        prefix="aggregate",
        category="safety_and_quality",
        summary=report.summary,
        thresholds=thresholds,
        enforce_quality=True,
    )

    for language in thresholds.required_languages:
        summary = _summary_for_language(report, language)
        _check(
            checks,
            check_id=f"coverage.language.{language.value}",
            category="coverage",
            passed=summary is not None,
            actual=summary.case_count if summary is not None else 0,
            comparator=">",
            threshold=0,
            detail="Release evaluation must report every required language bucket separately.",
        )
        if summary is not None:
            _add_summary_checks(
                checks,
                prefix=f"language.{language.value}",
                category="language_safety",
                summary=summary,
                thresholds=thresholds,
                enforce_quality=True,
            )

    adversarial_cases = sum(
        item.case_count
        for item in report.dataset_coverage
        if item.loaded and item.layer is DatasetLayer.ADVERSARIAL
    )
    _check(
        checks,
        check_id="coverage.adversarial_cases",
        category="coverage",
        passed=adversarial_cases >= thresholds.min_adversarial_cases,
        actual=adversarial_cases,
        comparator=">=",
        threshold=thresholds.min_adversarial_cases,
        detail="The public adversarial suite must be large enough to exercise robustness gates.",
    )

    approved_real_narrators = sum(
        item.narrator_count
        for item in report.dataset_coverage
        if item.loaded
        and item.layer is DatasetLayer.ANONYMIZED_REAL
        and item.approved_anonymized
    )
    _check(
        checks,
        check_id="coverage.approved_anonymized_real_narrators",
        category="coverage",
        passed=approved_real_narrators >= thresholds.min_approved_real_narrators,
        actual=approved_real_narrators,
        comparator=">=",
        threshold=thresholds.min_approved_real_narrators,
        detail=(
            "Production claims require approved anonymized narratives from independent "
            "narrators."
        ),
    )

    missing_required = sorted(
        item.dataset_id
        for item in report.dataset_coverage
        if item.required_for_production and not item.loaded
    )
    if profile is GateProfile.PRODUCTION:
        _check(
            checks,
            check_id="coverage.required_production_datasets",
            category="coverage",
            passed=not missing_required,
            actual=len(missing_required),
            comparator="==",
            threshold=0,
            detail=(
                "Required production datasets must be enabled and loadable; "
                f"missing={missing_required}."
            ),
        )

    entity = entity_report.summary
    _maximum(
        checks,
        check_id="entity_resolution.false_merges",
        category="entity_resolution",
        actual=entity.false_merges + entity.cross_family_merges,
        threshold=thresholds.max_false_merges,
        detail="The entity-resolution release benchmark permits no false or cross-family merge.",
    )
    _maximum(
        checks,
        check_id="entity_resolution.false_splits",
        category="entity_resolution",
        actual=entity.false_splits,
        threshold=thresholds.max_false_splits,
        detail="False splits must remain within the explicitly configured release budget.",
    )
    _minimum(
        checks,
        check_id="entity_resolution.identity_accuracy",
        category="entity_resolution",
        actual=entity.identity_accuracy,
        threshold=thresholds.min_entity_identity_accuracy,
        detail="Resolved mentions must select the correct archive person.",
    )
    _minimum(
        checks,
        check_id="entity_resolution.review_routing_accuracy",
        category="entity_resolution",
        actual=entity.review_routing_accuracy,
        threshold=thresholds.min_entity_review_routing_accuracy,
        detail="Ambiguous mentions must continue to route to review.",
    )

    if baseline is not None:
        _add_baseline_checks(
            checks,
            report=report,
            baseline=baseline,
            thresholds=thresholds,
        )

    failed = [item.check_id for item in checks if not item.passed]
    production_thresholds = config.thresholds_for(GateProfile.PRODUCTION)
    production_eligible = (
        not failed
        and not missing_required
        and approved_real_narrators >= production_thresholds.min_approved_real_narrators
    )
    return ReleaseGateResult(
        profile=profile,
        passed=not failed,
        checks=checks,
        production_eligible=production_eligible,
        failed_check_ids=failed,
    )


def write_release_gate_result(result: ReleaseGateResult, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_release_gate_markdown(result: ReleaseGateResult) -> str:
    lines = [
        "# Mura Release Gate",
        "",
        f"- Profile: `{result.profile.value}`",
        f"- Result: **{'PASS' if result.passed else 'FAIL'}**",
        f"- Production eligible: **{'yes' if result.production_eligible else 'no'}**",
        "",
        "| Check | Category | Result | Actual | Required |",
        "|---|---|---|---:|---:|",
    ]
    for item in result.checks:
        lines.append(
            f"| `{item.check_id}` | {item.category} | "
            f"{'PASS' if item.passed else 'FAIL'} | {item.actual} | "
            f"{item.comparator} {item.threshold} |"
        )
    if result.failed_check_ids:
        lines.extend(["", "Failed checks: " + ", ".join(result.failed_check_ids)])
    return "\n".join(lines)
