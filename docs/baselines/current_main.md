# Mura ML Core Baseline

Manifest: `benchmarks/manifest.json`

## Pipeline versions

| Component | Version |
|---|---|
| benchmark_schema | `benchmark-v1+entity-resolution-benchmark-v1` |
| cleaner_prompt | `cleaner-v1` |
| domain_schema | `domain-v2` |
| evaluator | `core-evaluator-v1+entity-resolution-v1` |
| evidence_rules | `claim-evidence-v2+bounded-coreference-v1` |
| extractor_prompt | `extractor-v3-anchor-constrained` |
| extractor_repair_prompt | `extractor-repair-v1-anchor-constrained` |
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

## PR 19 cross-recording entity resolution changes

1. Identity decisions are scoped to one `family_id`; foreign-family profiles are rejected before candidate generation.
2. Exact canonical-name equality alone now routes to `needs_review` instead of silently merging people.
3. Automatic resolution requires an established or evidence-backed alias, relation plus generation agreement, or a validated graph-neighbour match.
4. Multiple same-name candidates and family-versus-non-family conflicts remain reviewable.
5. Resolution traces preserve supporting and conflicting rule IDs, while every resolved identity stays `unreviewed`.
6. A separate eight-case entity-resolution release benchmark gates false merges, false splits, review routing, and new-person routing.

## Scope and limitations

The six-case baseline above still measures the deterministic relationship-validation layer against fixed extraction candidates. It does not measure live DeepSeek candidate generation or ASR quality.

The entity-resolution release benchmark is synthetic and archive-scoped. It does not yet use dates, locations, long-term event overlap, probabilistic ranking, or persistent conflict history. Missing graph edges are treated as unknown rather than contradictory because family archives are incomplete.
