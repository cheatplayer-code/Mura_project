# Mura ML Core Baseline

Manifest: `benchmarks/manifest.json`

## Pipeline versions

| Component | Version |
|---|---|
| archive_schema | `archive-claim-ledger-v1+conflict-decisions-v1` |
| benchmark_schema | `benchmark-v1+entity-resolution-benchmark-v1` |
| cleaner_prompt | `cleaner-v1` |
| domain_schema | `domain-v2` |
| evaluator | `core-evaluator-v1+entity-resolution-v1` |
| evidence_rules | `claim-evidence-v2+bounded-coreference-v1` |
| extractor_prompt | `extractor-v3-anchor-constrained` |
| extractor_repair_prompt | `extractor-repair-v1-anchor-constrained` |
| materializer | `family-graph-materializer-v2-human-review` |
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

## Human conflict review changes

1. Reviewers may resolve a conflict by selecting one grounded preferred claim; only that claim may rematerialize an edge.
2. Dismissing a conflict rejects every participating claim from automatic graph materialization.
3. Reopening a conflict returns every participating claim to disputed state and removes its graph edge.
4. Every resolve, dismiss, reopen, and automatic reopen is stored in an immutable decision audit table.
5. A later unseen competing claim automatically reopens a previously resolved or dismissed conflict instead of inheriting an outdated human decision.
6. Conflict reads and mutations are scoped by `family_id` and protected by the core API bearer token.
7. Legacy deterministic conflict IDs from the prior archive schema remain reviewable and are reused when their claim pair is reconciled.

## Scope and limitations

The six-case baseline above still measures the deterministic relationship-validation layer against fixed extraction candidates. It does not measure live DeepSeek candidate generation or ASR quality.

The entity-resolution release benchmark remains synthetic. Human review currently applies to incompatible relationship claims for the same resolved person pair. Dates, locations, descriptions, event overlap, identity conflicts, multi-reviewer consensus, and cryptographically authenticated reviewer identities are not yet materialized. The `reviewer_reference` field is client-asserted metadata under the shared core API credential, not an independently verified user identity.
