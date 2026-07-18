from pathlib import Path

from mura.evaluation.reporting import render_markdown_report
from mura.evaluation.runner import load_manifest, run_benchmark

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifest.json"


def test_manifest_loads_versioned_validation_dataset() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.schema_version == "benchmark-manifest-v1"
    assert len(manifest.datasets) == 1
    assert manifest.datasets[0].dataset_id == "synthetic_current_main"
    assert manifest.datasets[0].split.value == "validation"


def test_current_main_baseline_is_reproducible() -> None:
    report = run_benchmark(MANIFEST)
    summary = report.summary

    assert summary.case_count == 6

    assert summary.person_mentions.true_positive == 14
    assert summary.person_mentions.false_positive == 0
    assert summary.person_mentions.false_negative == 0
    assert summary.person_mentions.f1 == 1.0

    assert summary.relationships.true_positive == 3
    assert summary.relationships.false_positive == 1
    assert summary.relationships.false_negative == 3
    assert summary.relationships.precision == 0.75
    assert summary.relationships.recall == 0.5
    assert summary.relationships.f1 == 0.6

    assert summary.quarantined_relationships.true_positive == 3
    assert summary.quarantined_relationships.false_positive == 0
    assert summary.quarantined_relationships.false_negative == 1
    assert summary.quarantined_relationships.precision == 1.0
    assert summary.quarantined_relationships.recall == 0.75

    assert summary.relationship_direction_accuracy.value == 1.0
    assert summary.relationship_direction_accuracy.denominator == 3
    assert summary.provenance_completeness.value == 1.0
    assert summary.unknown_segment_references == 0
    assert summary.self_relationships == 0


def test_baseline_exposes_known_precision_and_recall_failures() -> None:
    report = run_benchmark(MANIFEST)
    cases = {case.case_id: case for case in report.cases}

    russian_speaker = cases["ru_inflected_speaker_anchor"]
    assert russian_speaker.accepted_relationship_ids == []
    assert russian_speaker.quarantined_relationship_ids == ["relationship_001"]

    ambiguous_pronoun = cases["ru_ambiguous_third_person"]
    assert ambiguous_pronoun.accepted_relationship_ids == ["relationship_001"]
    assert ambiguous_pronoun.quarantined_relationship_ids == []
    assert ambiguous_pronoun.relationships.false_positive == 1

    plural_antecedent = cases["kk_plural_antecedent_children"]
    assert plural_antecedent.accepted_relationship_ids == []
    assert plural_antecedent.quarantined_relationship_ids == [
        "relationship_001",
        "relationship_002",
    ]
    assert plural_antecedent.relationships.false_negative == 2


def test_markdown_report_contains_versions_and_limitations() -> None:
    report = run_benchmark(MANIFEST)
    markdown = render_markdown_report(report)

    assert "# Mura ML Core Baseline" in markdown
    assert "mura-core-v0.3.0" in markdown
    assert "does not measure live DeepSeek candidate generation" in markdown
