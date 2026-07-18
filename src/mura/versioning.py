from __future__ import annotations

from mura.domain.models import StrictModel


class PipelineVersions(StrictModel):
    """Immutable identifiers for every behavior-changing ML-core component."""

    pipeline: str
    schema: str
    cleaner_prompt: str
    extractor_prompt: str
    evidence_rules: str
    resolver: str
    evaluator: str
    benchmark_schema: str


CURRENT_PIPELINE_VERSIONS = PipelineVersions(
    pipeline="mura-core-v0.3.0",
    schema="domain-v1",
    cleaner_prompt="cleaner-v1",
    extractor_prompt="extractor-v1",
    evidence_rules="relationship-evidence-v1",
    resolver="mention-resolver-v1",
    evaluator="core-evaluator-v1",
    benchmark_schema="benchmark-v1",
)


def get_pipeline_versions() -> PipelineVersions:
    """Return a defensive copy suitable for traces, reports, and persisted results."""

    return CURRENT_PIPELINE_VERSIONS.model_copy(deep=True)
