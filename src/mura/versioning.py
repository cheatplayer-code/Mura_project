from __future__ import annotations

from mura.domain.models import StrictModel


class PipelineVersions(StrictModel):
    """Immutable identifiers for every behavior-changing ML-core component."""

    pipeline: str
    domain_schema: str
    cleaner_prompt: str
    extractor_prompt: str
    extractor_repair_prompt: str
    evidence_rules: str
    resolver: str
    archive_schema: str
    materializer: str
    evaluator: str
    benchmark_schema: str


CURRENT_PIPELINE_VERSIONS = PipelineVersions(
    pipeline="mura-core-v0.10.0",
    domain_schema="domain-v2",
    cleaner_prompt="cleaner-v1",
    extractor_prompt="extractor-v3-anchor-constrained",
    extractor_repair_prompt="extractor-repair-v1-anchor-constrained",
    evidence_rules="claim-evidence-v2+bounded-coreference-v1",
    resolver="mention-resolver-v2-cross-recording",
    archive_schema="archive-claim-ledger-v1",
    materializer="family-graph-materializer-v1",
    evaluator="core-evaluator-v1+entity-resolution-v1",
    benchmark_schema="benchmark-v1+entity-resolution-benchmark-v1",
)


def get_pipeline_versions() -> PipelineVersions:
    """Return a defensive copy suitable for traces, reports, and persisted results."""

    return CURRENT_PIPELINE_VERSIONS.model_copy(deep=True)
