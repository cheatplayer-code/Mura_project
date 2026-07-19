from pathlib import Path

from mura.evaluation.models import BenchmarkReport, DatasetLayer
from mura.evaluation.reporting import render_markdown_report
from mura.evaluation.runner import load_manifest, run_benchmark

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MANIFEST = ROOT / "benchmarks" / "baseline_manifest.json"
RELEASE_MANIFEST = ROOT / "benchmarks" / "manifest.json"
FROZEN_BASELINE = ROOT / "docs" / "baselines" / "current_main.json"


def test_baseline_manifest_loads_only_approved_validation_dataset() -> None:
    manifest = load_manifest(BASELINE_MANIFEST)

    assert manifest.schema_version == "benchmark-manifest-v2"
    assert len(manifest.datasets) == 1
    assert manifest.datasets[0].dataset_id == "synthetic_current_main"
    assert manifest.datasets[0].split.value == "validation"
    assert manifest.datasets[0].layer is DatasetLayer.DETERMINISTIC


def test_current_main_baseline_is_reproducible() -> None:
    report = run_benchmark(BASELINE_MANIFEST)
    summary = report.summary

    assert summary.case_count == 6

    assert summary.person_mentions.true_positive == 14
    assert summary.person_mentions.false_positive == 0
    assert summary.person_mentions.false_negative == 0
    assert summary.person_mentions.f1 == 1.0

    assert summary.relationships.true_positive == 6
    assert summary.relationships.false_positive == 0
    assert summary.relationships.false_negative == 0
    assert summary.relationships.precision == 1.0
    assert summary.relationships.recall == 1.0
    assert summary.relationships.f1 == 1.0

    assert summary.quarantined_relationships.true_positive == 1
    assert summary.quarantined_relationships.false_positive == 0
    assert summary.quarantined_relationships.false_negative == 0
    assert summary.quarantined_relationships.precision == 1.0
    assert summary.quarantined_relationships.recall == 1.0
    assert summary.quarantined_relationships.f1 == 1.0

    assert summary.relationship_direction_accuracy.value == 1.0
    assert summary.relationship_direction_accuracy.denominator == 6
    assert summary.provenance_completeness.value == 1.0
    assert summary.provenance_completeness.denominator == 20
    assert summary.unsupported_relationship_acceptance.value == 0.0
    assert summary.unknown_segment_references == 0
    assert summary.self_relationships == 0
    assert summary.accepted_claims_without_evidence == 0
    assert summary.critical_graph_violations == 0


def test_generated_cases_match_frozen_approved_case_metrics() -> None:
    generated = run_benchmark(BASELINE_MANIFEST)
    frozen = BenchmarkReport.model_validate_json(FROZEN_BASELINE.read_text(encoding="utf-8"))
    generated_by_id = {case.case_id: case for case in generated.cases}
    frozen_by_id = {case.case_id: case for case in frozen.cases}

    assert generated_by_id.keys() == frozen_by_id.keys()
    for case_id, previous in frozen_by_id.items():
        current = generated_by_id[case_id]
        assert current.person_mentions == previous.person_mentions
        assert current.relationships == previous.relationships
        assert current.quarantined_relationships == previous.quarantined_relationships
        assert current.relationship_direction_accuracy == previous.relationship_direction_accuracy
        assert current.provenance_completeness == previous.provenance_completeness
        assert current.unknown_segment_references == previous.unknown_segment_references
        assert current.self_relationships == previous.self_relationships
        assert current.accepted_relationship_ids == previous.accepted_relationship_ids
        assert current.quarantined_relationship_ids == previous.quarantined_relationship_ids


def test_baseline_records_bounded_coreference_and_ambiguity_safety() -> None:
    report = run_benchmark(BASELINE_MANIFEST)
    cases = {case.case_id: case for case in report.cases}

    russian_speaker = cases["ru_inflected_speaker_anchor"]
    assert russian_speaker.accepted_relationship_ids == ["relationship_001"]
    assert russian_speaker.quarantined_relationship_ids == []
    assert russian_speaker.relationships.true_positive == 1

    ambiguous_pronoun = cases["ru_ambiguous_third_person"]
    assert ambiguous_pronoun.accepted_relationship_ids == []
    assert ambiguous_pronoun.quarantined_relationship_ids == ["relationship_001"]
    assert ambiguous_pronoun.relationships.false_positive == 0

    plural_antecedent = cases["kk_plural_antecedent_children"]
    assert plural_antecedent.accepted_relationship_ids == [
        "relationship_001",
        "relationship_002",
    ]
    assert plural_antecedent.quarantined_relationship_ids == []
    assert plural_antecedent.relationships.true_positive == 2
    assert plural_antecedent.relationships.false_negative == 0


def test_markdown_report_contains_versions_coverage_and_limitations() -> None:
    report = run_benchmark(RELEASE_MANIFEST)
    markdown = render_markdown_report(report)

    assert "# Mura ML Core Baseline" in markdown
    assert "mura-core-v0.9.0" in markdown
    assert "archive-claim-ledger-v1+conflict-decisions-v1+generic-claims-v1" in markdown
    assert "family-materializer-v3-graph-and-profiles" in markdown
    assert "extractor-v3-anchor-constrained" in markdown
    assert "core-evaluator-v2-release-gates+entity-resolution-v1" in markdown
    assert "benchmark-v2-adversarial+entity-resolution-benchmark-v1" in markdown
    assert "approved_anonymized_real" in markdown
    assert "does not measure live DeepSeek candidate generation" in markdown
