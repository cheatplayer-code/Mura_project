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
| extractor_prompt | `extractor-v3-anchor-constrained` |
| extractor_repair_prompt | `extractor-repair-v1-anchor-constrained` |
| pipeline | `mura-core-v0.8.0` |
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

## PR 18 anchor-constrained extraction changes

1. DeepSeek receives a typed, versioned anchor contract containing the complete allowed segment set, speaker anchors, known-person surfaces, bounded name candidates, and audited kinship annotations.
2. Anchors remain candidate-generation hints and cannot become evidence, prove identity, authorize a merge, or determine relationship direction.
3. Fatal top-level collection-shape failures may receive one bounded extraction-repair invocation using the exact same anchor contract.
4. Valid empty outputs and isolated invalid objects remain sanitizer outcomes and do not trigger a second semantic extraction call.
5. The deterministic reasoning benchmark metrics remain unchanged because this phase modifies candidate-generation inputs rather than sanitizer rules.

## Scope and limitations

This baseline measures the deterministic validation layer against fixed extraction candidates. It does not measure live DeepSeek candidate generation or ASR quality.

The perfect score applies only to the six declared regression cases. The resolver is intentionally bounded to the current segment and one immediately preceding segment; it does not solve arbitrary long-document coreference, implicit event participants, or cross-recording identity resolution.
