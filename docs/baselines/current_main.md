# Mura ML Core Baseline

Manifest: `benchmarks/manifest.json`

## Pipeline versions

| Component | Version |
|---|---|
| archive_schema | `archive-claim-ledger-v1+conflict-decisions-v1+generic-claims-v1` |
| benchmark_schema | `benchmark-v1+entity-resolution-benchmark-v1` |
| cleaner_prompt | `cleaner-v1` |
| domain_schema | `domain-v2` |
| evaluator | `core-evaluator-v1+entity-resolution-v1` |
| evidence_rules | `claim-evidence-v2+bounded-coreference-v1` |
| extractor_prompt | `extractor-v3-anchor-constrained` |
| extractor_repair_prompt | `extractor-repair-v1-anchor-constrained` |
| materializer | `family-materializer-v3-graph-and-profiles` |
| pipeline | `mura-core-v0.9.0` |
| resolver | `mention-resolver-v2-cross-recording` |

## Aggregate reasoning metrics

- Cases: **6**
- Person mentions: P=1.000, R=1.000, F1=1.000 (TP=14, FP=0, FN=0)
- Relationships: P=1.000, R=1.000, F1=1.000 (TP=6, FP=0, FN=0)
- Expected quarantine: P=1.000, R=1.000, F1=1.000 (TP=1, FP=0, FN=0)
- Relationship direction accuracy: 1.000 (6/6)
- Provenance completeness: 1.000 (20/20)
- Unknown segment references: **0**
- Self relationships: **0**

## Generic claims and profile materialization

1. Validated aliases, descriptions, and event facets are projected into immutable typed archive claims.
2. Birth and death dates are single-valued profile facets; incompatible grounded values create temporal conflicts and remain absent from the materialized profile until review.
3. Locations, professions, education, descriptions, and general events remain multi-valued because changes over time are not contradictions by default.
4. A grounded alias assigned to multiple archive people creates an identity conflict instead of silently attaching the alias to every profile.
5. Existing conflict review endpoints now adjudicate relationship, temporal, and identity conflicts without changing the partner API contract.
6. A new competing generic claim automatically reopens a previous decision and removes the stale preferred profile value.
7. Materialized person profiles are family-scoped, source-linked, and exposed through authenticated profile endpoints.
8. Relationship claims and generic profile claims are persisted within the same archive transaction.

## Scope and limitations

The six-case baseline above still measures the deterministic relationship-validation layer against fixed extraction candidates. It does not measure live DeepSeek candidate generation or ASR quality.

The entity-resolution release benchmark remains synthetic. Generic profile projection currently consumes already validated aliases, descriptions, and events. Location, profession, education, descriptions, and event history are intentionally multi-valued and are not temporally ranked. Child-count claims, arbitrary typed facts, event-identity overlap, multi-reviewer consensus, and cryptographically authenticated reviewer identities remain outside this release. The `reviewer_reference` field is client-asserted metadata under the shared core API credential, not an independently verified user identity.
