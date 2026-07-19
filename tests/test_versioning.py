from mura.versioning import CURRENT_PIPELINE_VERSIONS, get_pipeline_versions


def test_pipeline_versions_are_explicit_and_copy_safe() -> None:
    versions = get_pipeline_versions()

    assert versions.pipeline == "mura-core-v0.10.0"
    assert versions.domain_schema == "domain-v2"
    assert versions.cleaner_prompt == "cleaner-v1"
    assert versions.extractor_prompt == "extractor-v3-anchor-constrained"
    assert versions.extractor_repair_prompt == "extractor-repair-v1-anchor-constrained"
    assert versions.evidence_rules == "claim-evidence-v2+bounded-coreference-v1"
    assert versions.resolver == "mention-resolver-v2-cross-recording"
    assert versions.archive_schema == "archive-claim-ledger-v1"
    assert versions.materializer == "family-graph-materializer-v1"
    assert versions.evaluator == "core-evaluator-v1+entity-resolution-v1"
    assert versions.benchmark_schema == "benchmark-v1+entity-resolution-benchmark-v1"
    assert versions is not CURRENT_PIPELINE_VERSIONS
