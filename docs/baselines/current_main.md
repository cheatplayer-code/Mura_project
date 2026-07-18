# Mura ML Core Baseline

Manifest: `benchmarks/manifest.json`

## Pipeline versions

| Component | Version |
|---|---|
| benchmark_schema | `benchmark-v1` |
| cleaner_prompt | `cleaner-v1` |
| domain_schema | `domain-v2` |
| evaluator | `core-evaluator-v1` |
| evidence_rules | `claim-evidence-v2+bounded-coreference-v1` |
| extractor_prompt | `extractor-v2` |
| pipeline | `mura-core-v0.7.0` |
| resolver | `mention-resolver-v1+bounded-coreference-v1` |

## Aggregate metrics

- Cases: **6**
- Person mentions: P=1.000, R=1.000, F1=1.000 (TP=14, FP=0, FN=0)
- Relationships: P=1.000, R=1.000, F1=1.000 (TP=6, FP=0, FN=0)
- Expected quarantine: P=1.000, R=1.000, F1=1.000 (TP=1, FP=0, FN=0)
- Relationship direction accuracy: 1.000 (6/6)
- Provenance completeness: 1.000 (20/20)
- Unknown segment references: **0**
- Self relationships: **0**

## PR 17 bounded-coreference improvements

1. Singular `оның / его / his` relationships may be retained only when the bounded discourse window contains exactly one compatible antecedent.
2. Kazakh plural `олардың` resolves to an explicitly coordinated married pair and retains both parent-child claims.
3. Competing candidates such as `Ерлан встретил Болата. Его сын Нурлан.` still produce reviewable ambiguity instead of an accepted edge.
4. Context-resolved relationships retain evidence class D and remain ineligible for automatic graph materialization.
5. Model-proposed resolved links cannot authorize a claim unless deterministic discourse or human review independently supports the antecedent.

## Scope and limitations

This baseline measures the deterministic validation layer against fixed extraction candidates. It does not measure live DeepSeek candidate generation or ASR quality.

The perfect score applies only to the six declared regression cases. The resolver is intentionally bounded to the current segment and one immediately preceding segment; it does not solve arbitrary long-document coreference, implicit event participants, or cross-recording identity resolution.
