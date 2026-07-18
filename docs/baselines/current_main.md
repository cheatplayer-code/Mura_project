# Mura ML Core Baseline

Manifest: `benchmarks/manifest.json`

## Pipeline versions

| Component | Version |
|---|---|
| benchmark_schema | `benchmark-v1` |
| cleaner_prompt | `cleaner-v1` |
| domain_schema | `domain-v1` |
| evaluator | `core-evaluator-v1` |
| evidence_rules | `relationship-evidence-v1` |
| extractor_prompt | `extractor-v1` |
| pipeline | `mura-core-v0.3.0` |
| resolver | `mention-resolver-v1` |

## Aggregate metrics

- Cases: **6**
- Person mentions: P=1.000, R=1.000, F1=1.000 (TP=14, FP=0, FN=0)
- Relationships: P=0.750, R=0.500, F1=0.600 (TP=3, FP=1, FN=3)
- Expected quarantine: P=1.000, R=0.750, F1=0.857 (TP=3, FP=0, FN=1)
- Relationship direction accuracy: 1.000 (3/3)
- Provenance completeness: 1.000 (18/18)
- Unknown segment references: **0**
- Self relationships: **0**

## Known baseline failures

1. Russian inflected first-person possessive `мою` is not recognized as a speaker anchor, so a valid parent-child relationship is quarantined.
2. A relationship proposed from the ambiguous phrase `Его сын` is accepted when both endpoint names occur somewhere in the same segment, even though the pronoun antecedent is unresolved.
3. Kazakh plural antecedent `олардың` is not linked to the coordinated pair from the previous segment, so two valid parent-child relationships are quarantined.

## Scope

This baseline measures the deterministic validation layer against fixed extraction candidates. It intentionally does not call GigaAM or DeepSeek, so it does not measure ASR quality or live candidate-generation quality. The fixed candidates make changes to evidence logic measurable and reproducible.
