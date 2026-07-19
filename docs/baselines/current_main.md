# Mura ML Core Baseline

Manifest: `benchmarks/manifest.json`

## Pipeline versions

| Component | Version |
|---|---|
| archive_schema | `archive-claim-ledger-v1` |
| benchmark_schema | `benchmark-v1+entity-resolution-benchmark-v1` |
| cleaner_prompt | `cleaner-v1` |
| domain_schema | `domain-v2` |
| evaluator | `core-evaluator-v1+entity-resolution-v1` |
| evidence_rules | `claim-evidence-v2+bounded-coreference-v1` |
| extractor_prompt | `extractor-v3-anchor-constrained` |
| extractor_repair_prompt | `extractor-repair-v1-anchor-constrained` |
| materializer | `family-graph-materializer-v1` |
| pipeline | `mura-core-v0.10.0` |
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

## PR 20 persistent claims and graph materialization changes

1. Every accepted extraction object is persisted as an immutable archive claim with recording, evidence, verification, assertion-mode, and derivation metadata.
2. New people receive deterministic archive IDs, while existing people are resolved from a family-scoped archive context before each recording is processed.
3. Reprocessing identical content is idempotent; changed claim payloads receive distinct digest-based claim IDs instead of overwriting history.
4. Speaker self-corrections are preserved as separate archive correction records.
5. Grounded A-C relationship claims with resolved endpoints may materialize family-graph edges.
6. Incompatible grounded relationships for the same person pair remain as competing claims, create an open conflict, and remove the disputed edge from the materialized graph.
7. D, E, and U claims are retained for review but cannot create graph edges or block an already grounded edge.
8. Materialized parent, child, spouse, and sibling neighbours are fed back into the next recording's entity-resolution context.

## Scope and limitations

The six-case baseline above still measures the deterministic relationship-validation layer against fixed extraction candidates. It does not measure live DeepSeek candidate generation or ASR quality.

The entity-resolution release benchmark remains synthetic. Archive conflict detection in this version covers incompatible relationships for the same resolved person pair; dates, locations, descriptions, event overlap, human conflict resolution, and probabilistic ranking are not yet materialized. Missing graph edges remain unknown rather than contradictory because family archives are incomplete.
