# Mura ML Core Baseline

Manifest: `benchmarks/manifest.json`

## Pipeline versions

| Component | Version |
|---|---|
| benchmark_schema | `benchmark-v1` |
| cleaner_prompt | `cleaner-v1` |
| domain_schema | `domain-v2` |
| evaluator | `core-evaluator-v1` |
| evidence_rules | `claim-evidence-v2+multilingual-v1` |
| extractor_prompt | `extractor-v2` |
| pipeline | `mura-core-v0.6.0` |
| resolver | `mention-resolver-v1` |

## Aggregate metrics

- Cases: **6**
- Person mentions: P=1.000, R=1.000, F1=1.000 (TP=14, FP=0, FN=0)
- Relationships: P=1.000, R=0.667, F1=0.800 (TP=4, FP=0, FN=2)
- Expected quarantine: P=1.000, R=0.750, F1=0.857 (TP=3, FP=0, FN=1)
- Relationship direction accuracy: 1.000 (4/4)
- Provenance completeness: 1.000 (18/18)
- Unknown segment references: **0**
- Self relationships: **0**

## PR 16 improvements

1. Russian inflected first-person possessive `мою` now produces a bounded speaker-anchored parent-child signal.
2. The ambiguous phrase `Его сын` is quarantined unless a valid `CoreferenceLink` is supplied.
3. Relationship precision rises from 0.750 to 1.000 while the fixed benchmark recall rises from 0.500 to 0.667.

## Remaining baseline failure

Kazakh plural antecedent `олардың` is still not linked to the coordinated pair from the previous segment, so two valid parent-child relationships remain quarantined. This is intentionally reserved for bounded discourse and coreference in PR 17.

## Scope

This baseline measures the deterministic validation layer against fixed extraction candidates. It intentionally does not call GigaAM or DeepSeek, so it does not measure ASR quality or live candidate-generation quality. PR 16 adds bounded Russian, English, and code-switching rules without resolving third-person antecedents.
