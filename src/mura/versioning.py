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
    extraction_orchestration: str
    narrative_rules: str
    claim_semantics: str
    temporal_rules: str
    relationship_state_rules: str
    resolver: str
    archive_schema: str
    materializer: str
    evaluator: str
    benchmark_schema: str


CURRENT_PIPELINE_VERSIONS = PipelineVersions(
    pipeline="mura-core-v0.13.0",
    domain_schema="domain-v5-identity-safety",
    cleaner_prompt="cleaner-v3-self-correction-semantics",
    extractor_prompt="extractor-v6-focused-passes",
    extractor_repair_prompt="extractor-repair-v4-focused-pass",
    evidence_rules="claim-evidence-v5-ordered-factual-support+bounded-coreference-v3",
    extraction_orchestration="focused-extraction-v1-three-pass",
    narrative_rules="event-story-grounding-v1",
    claim_semantics="claim-semantics-v1",
    temporal_rules="temporal-normalizer-v1",
    relationship_state_rules="relationship-state-v1",
    resolver="mention-resolver-v3-collision-safe",
    archive_schema="archive-claim-ledger-v1+conflict-decisions-v1+generic-claims-v1",
    materializer="family-materializer-v4-active-state-guard",
    evaluator="core-evaluator-v5-identity-safety",
    benchmark_schema="benchmark-v5-identity-safety+entity-resolution-benchmark-v2",
)


def get_pipeline_versions() -> PipelineVersions:
    """Return a defensive copy suitable for traces, reports, and persisted results."""
    return CURRENT_PIPELINE_VERSIONS.model_copy(deep=True)
