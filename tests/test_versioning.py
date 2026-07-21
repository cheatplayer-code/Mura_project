from mura.versioning import CURRENT_PIPELINE_VERSIONS, get_pipeline_versions


def test_pipeline_versions_are_explicit_and_copy_safe() -> None:
    versions = get_pipeline_versions()

    assert versions.pipeline == "mura-core-v0.11.0"
    assert versions.domain_schema == "domain-v3-claim-semantics"
    assert versions.cleaner_prompt == "cleaner-v3-self-correction-semantics"
    assert versions.extractor_prompt == "extractor-v5-claim-semantics"
    assert versions.extractor_repair_prompt == "extractor-repair-v3-semantic-preservation"
    assert versions.evidence_rules == "claim-evidence-v3-layered-provenance+bounded-coreference-v2"
    assert versions.claim_semantics == "claim-semantics-v1"
    assert versions.temporal_rules == "temporal-normalizer-v1"
    assert versions.relationship_state_rules == "relationship-state-v1"
    assert versions.resolver == "mention-resolver-v2-cross-recording"
    assert (
        versions.archive_schema == "archive-claim-ledger-v1+conflict-decisions-v1+generic-claims-v1"
    )
    assert versions.materializer == "family-materializer-v4-active-state-guard"
    assert versions.evaluator == "core-evaluator-v3-claim-semantics+entity-resolution-v1"
    assert (
        versions.benchmark_schema == "benchmark-v3-claim-semantics+entity-resolution-benchmark-v1"
    )
    assert versions is not CURRENT_PIPELINE_VERSIONS
