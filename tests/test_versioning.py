from mura.versioning import CURRENT_PIPELINE_VERSIONS, get_pipeline_versions


def test_pipeline_versions_are_explicit_and_copy_safe() -> None:
    versions = get_pipeline_versions()

    assert versions.pipeline == "mura-core-v0.4.0"
    assert versions.domain_schema == "domain-v2"
    assert versions.cleaner_prompt == "cleaner-v1"
    assert versions.extractor_prompt == "extractor-v2"
    assert versions.evidence_rules == "claim-evidence-v2"
    assert versions.resolver == "mention-resolver-v1"
    assert versions.evaluator == "core-evaluator-v1"
    assert versions.benchmark_schema == "benchmark-v1"
    assert versions is not CURRENT_PIPELINE_VERSIONS
