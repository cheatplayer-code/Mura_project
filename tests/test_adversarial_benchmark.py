from pathlib import Path

from mura.evaluation.models import DatasetLayer, LanguageBucket
from mura.evaluation.reporting import render_markdown_report
from mura.evaluation.runner import load_manifest, run_benchmark

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifest.json"


def test_manifest_declares_public_adversarial_and_private_real_layers() -> None:
    manifest = load_manifest(MANIFEST)
    entries = {item.dataset_id: item for item in manifest.datasets}

    assert manifest.schema_version == "benchmark-manifest-v2"
    assert entries["adversarial_public_v1"].enabled is True
    assert entries["adversarial_public_v1"].layer is DatasetLayer.ADVERSARIAL
    assert entries["approved_anonymized_real"].enabled is False
    assert entries["approved_anonymized_real"].required_for_production is True
    assert entries["approved_anonymized_real"].approved_anonymized is False


def test_adversarial_suite_is_reported_by_language_and_layer() -> None:
    report = run_benchmark(MANIFEST)
    coverage = {item.dataset_id: item for item in report.dataset_coverage}

    assert report.report_schema_version == "evaluation-report-v2"
    assert coverage["adversarial_public_v1"].case_count == 12
    assert coverage["approved_anonymized_real"].loaded is False
    assert report.summary.case_count == 18
    assert report.summary.provenance_completeness.value == 1.0
    assert report.summary.accepted_claims_without_evidence == 0
    assert report.summary.critical_graph_violations == 0
    assert report.summary.unsupported_relationship_acceptance.value == 0.0

    language_slices = {
        item.key: item.summary
        for item in report.slices
        if item.dimension == "language"
    }
    assert set(language_slices) == {item.value for item in LanguageBucket}
    assert all(summary.provenance_completeness.value == 1.0 for summary in language_slices.values())

    layer_slices = {
        item.key: item.summary
        for item in report.slices
        if item.dimension == "layer"
    }
    assert layer_slices["adversarial"].case_count == 12
    assert layer_slices["deterministic"].case_count == 6


def test_adversarial_partial_failures_do_not_destroy_valid_objects() -> None:
    report = run_benchmark(MANIFEST)
    cases = {item.case_id: item for item in report.cases}

    partial = cases["adv_en_partial_malformed_json"]
    assert partial.accepted_relationship_ids == ["relationship_001"]
    assert partial.quarantined_relationship_ids == ["relationship_002"]
    assert partial.relationships.f1 == 1.0
    assert partial.quarantined_relationships.f1 == 1.0

    unknown_segment = cases["adv_ru_unknown_segment"]
    assert unknown_segment.accepted_relationship_ids == []
    assert unknown_segment.quarantined_relationship_ids == ["relationship_001"]

    self_edge = cases["adv_en_self_relationship"]
    assert self_edge.accepted_relationship_ids == []
    assert self_edge.quarantined_relationship_ids == ["relationship_001"]
    assert self_edge.self_relationships == 0


def test_markdown_exposes_coverage_and_safety_rates() -> None:
    markdown = render_markdown_report(run_benchmark(MANIFEST))

    assert "## Dataset coverage" in markdown
    assert "## Language breakdown" in markdown
    assert "## Layer breakdown" in markdown
    assert "Unsupported relationship acceptance" in markdown
    assert "approved_anonymized_real" in markdown
