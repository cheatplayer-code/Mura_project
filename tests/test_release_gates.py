from pathlib import Path

from mura.evaluation.entity_resolution import run_entity_resolution_benchmark
from mura.evaluation.models import RatioMetric
from mura.evaluation.release_gates import (
    GateProfile,
    evaluate_release_gates,
    load_benchmark_report,
    load_release_gate_config,
)
from mura.evaluation.runner import run_benchmark

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifest.json"
GATES = ROOT / "benchmarks" / "release_gates.json"
BASELINE = ROOT / "docs" / "baselines" / "current_main.json"
ENTITY = ROOT / "benchmarks" / "entity_resolution_v1.json"


def _inputs():
    return (
        run_benchmark(MANIFEST),
        run_entity_resolution_benchmark(ENTITY),
        load_release_gate_config(GATES),
        load_benchmark_report(BASELINE),
    )


def test_pull_request_release_gate_passes_public_suites() -> None:
    report, entity, config, baseline = _inputs()

    result = evaluate_release_gates(
        report=report,
        entity_report=entity,
        config=config,
        profile=GateProfile.PULL_REQUEST,
        baseline=baseline,
    )

    assert result.passed is True
    assert result.failed_check_ids == []
    assert any(item.check_id == "coverage.language.mixed" for item in result.checks)
    assert any(item.check_id == "coverage.adversarial_cases" for item in result.checks)


def test_production_gate_fails_closed_without_approved_private_narratives() -> None:
    report, entity, config, baseline = _inputs()

    result = evaluate_release_gates(
        report=report,
        entity_report=entity,
        config=config,
        profile=GateProfile.PRODUCTION,
        baseline=baseline,
    )

    assert result.passed is False
    assert result.production_eligible is False
    assert "coverage.approved_anonymized_real_narrators" in result.failed_check_ids
    assert "coverage.required_production_datasets" in result.failed_check_ids


def test_unsupported_relationship_regression_blocks_release() -> None:
    report, entity, config, baseline = _inputs()
    unsafe_summary = report.summary.model_copy(
        update={
            "unsupported_relationship_acceptance": RatioMetric(
                numerator=1,
                denominator=1,
                value=1.0,
            )
        }
    )
    unsafe_report = report.model_copy(update={"summary": unsafe_summary})

    result = evaluate_release_gates(
        report=unsafe_report,
        entity_report=entity,
        config=config,
        profile=GateProfile.PULL_REQUEST,
        baseline=baseline,
    )

    assert result.passed is False
    assert "aggregate.unsupported_relationship_acceptance" in result.failed_check_ids


def test_missing_baseline_case_blocks_release() -> None:
    report, entity, config, baseline = _inputs()
    reduced_report = report.model_copy(update={"cases": report.cases[1:]})

    result = evaluate_release_gates(
        report=reduced_report,
        entity_report=entity,
        config=config,
        profile=GateProfile.PULL_REQUEST,
        baseline=baseline,
    )

    assert result.passed is False
    assert "regression.baseline_case_coverage" in result.failed_check_ids
