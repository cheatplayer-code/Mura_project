from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from mura.domain.models import ExtractionResult, RelationshipClaim, RelationshipType
from mura.evaluation.models import (
    BenchmarkCase,
    BenchmarkGold,
    BenchmarkSummary,
    CaseEvaluation,
    DatasetLayer,
    DatasetSplit,
    GoldRelationship,
    PrecisionRecallF1,
    RatioMetric,
)
from mura.relationship_evidence import normalize_evidence

RelationshipSemanticKey = tuple[str, str, str, str, str]
RelationshipBaseKey = tuple[str, str, str]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _safe_error_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> PrecisionRecallF1:
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PrecisionRecallF1(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
    )


def ratio_metric(numerator: int, denominator: int) -> RatioMetric:
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=round(_safe_ratio(numerator, denominator), 6),
    )


def error_rate_metric(numerator: int, denominator: int) -> RatioMetric:
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=round(_safe_error_rate(numerator, denominator), 6),
    )


def _gold_surface_index(gold: BenchmarkGold) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for person in gold.people:
        for surface in person.accepted_surfaces:
            normalized = normalize_evidence(surface)
            if normalized:
                index.setdefault(normalized, set()).add(person.person_key)
    return index


def _mention_to_gold_keys(
    extraction: ExtractionResult,
    gold: BenchmarkGold,
) -> dict[str, str | None]:
    surface_index = _gold_surface_index(gold)
    result: dict[str, str | None] = {}
    for mention in extraction.people_mentions:
        candidates: set[str] = set()
        for surface in [mention.name, *mention.aliases]:
            candidates.update(surface_index.get(normalize_evidence(surface), set()))
        result[mention.mention_id] = next(iter(candidates)) if len(candidates) == 1 else None
    return result


def _canonical_relationship_key(
    *,
    relationship_type: RelationshipType,
    subject_key: str,
    subject_role: str,
    object_key: str,
    object_role: str,
) -> RelationshipSemanticKey:
    if relationship_type is RelationshipType.SPOUSE:
        first, second = sorted((subject_key, object_key))
        return (relationship_type.value, first, "spouse", second, "spouse")

    if (
        relationship_type is RelationshipType.SIBLING
        and subject_role == "sibling"
        and object_role == "sibling"
    ):
        first, second = sorted((subject_key, object_key))
        return (relationship_type.value, first, "sibling", second, "sibling")

    return (
        relationship_type.value,
        subject_key,
        subject_role,
        object_key,
        object_role,
    )


def _actual_relationship_key(
    relationship: RelationshipClaim,
    mention_to_gold: dict[str, str | None],
) -> RelationshipSemanticKey:
    subject_key = mention_to_gold.get(relationship.subject_mention_id)
    object_key = mention_to_gold.get(relationship.object_mention_id)
    if subject_key is None:
        subject_key = f"__unmatched__:{relationship.subject_mention_id}"
    if object_key is None:
        object_key = f"__unmatched__:{relationship.object_mention_id}"
    return _canonical_relationship_key(
        relationship_type=relationship.relationship_type,
        subject_key=subject_key,
        subject_role=relationship.subject_role.value,
        object_key=object_key,
        object_role=relationship.object_role.value,
    )


def _gold_relationship_key(relationship: GoldRelationship) -> RelationshipSemanticKey:
    return _canonical_relationship_key(
        relationship_type=relationship.relationship_type,
        subject_key=relationship.subject_person_key,
        subject_role=relationship.subject_role.value,
        object_key=relationship.object_person_key,
        object_role=relationship.object_role.value,
    )


def _base_relationship_key(key: RelationshipSemanticKey) -> RelationshipBaseKey:
    relationship_type, subject_key, _, object_key, _ = key
    first, second = sorted((subject_key, object_key))
    return (relationship_type, first, second)


def _multiset_prf(
    actual: Iterable[RelationshipSemanticKey],
    expected: Iterable[RelationshipSemanticKey],
) -> PrecisionRecallF1:
    actual_counter = Counter(actual)
    expected_counter = Counter(expected)
    true_positive = sum((actual_counter & expected_counter).values())
    false_positive = sum(actual_counter.values()) - true_positive
    false_negative = sum(expected_counter.values()) - true_positive
    return precision_recall_f1(true_positive, false_positive, false_negative)


def _set_prf(actual: set[str], expected: set[str]) -> PrecisionRecallF1:
    true_positive = len(actual.intersection(expected))
    return precision_recall_f1(
        true_positive,
        len(actual - expected),
        len(expected - actual),
    )


def _person_metrics(
    extraction: ExtractionResult,
    gold: BenchmarkGold,
    mention_to_gold: dict[str, str | None],
) -> PrecisionRecallF1:
    expected_keys = {person.person_key for person in gold.people}
    mapped_keys = [
        person_key
        for mention_id in (person.mention_id for person in extraction.people_mentions)
        if (person_key := mention_to_gold.get(mention_id)) is not None
    ]
    true_positive = len(set(mapped_keys).intersection(expected_keys))
    false_positive = len(extraction.people_mentions) - true_positive
    false_negative = len(expected_keys) - true_positive
    return precision_recall_f1(true_positive, false_positive, false_negative)


def _objects(extraction: ExtractionResult) -> list[Any]:
    return [
        *extraction.people_mentions,
        *extraction.relationship_claims,
        *extraction.events,
        *extraction.descriptions,
        *extraction.stories,
        *extraction.unresolved_questions,
    ]


def _provenance_counts(extraction: ExtractionResult) -> tuple[int, int]:
    objects = _objects(extraction)
    complete = sum(bool(getattr(item, "source_segment_ids", [])) for item in objects)
    return complete, len(objects)


def _accepted_claims_without_evidence(extraction: ExtractionResult) -> int:
    return sum(not bool(getattr(item, "evidence_ids", [])) for item in _objects(extraction))


def _unknown_segment_reference_count(
    extraction: ExtractionResult,
    valid_segment_ids: set[str],
) -> int:
    return sum(
        len(set(getattr(item, "source_segment_ids", [])) - valid_segment_ids)
        for item in _objects(extraction)
    )


def score_case(
    *,
    case: BenchmarkCase,
    dataset_id: str,
    split: DatasetSplit,
    dataset_layer: DatasetLayer,
    extraction: ExtractionResult,
    issues: list[dict[str, Any]],
    evidence_closure_relationships: int,
) -> CaseEvaluation:
    mention_to_gold = _mention_to_gold_keys(extraction, case.gold)
    actual_relationship_keys = [
        _actual_relationship_key(item, mention_to_gold) for item in extraction.relationship_claims
    ]
    gold_relationship_keys = [_gold_relationship_key(item) for item in case.gold.relationships]
    relationship_metrics = _multiset_prf(actual_relationship_keys, gold_relationship_keys)

    gold_base_keys = {_base_relationship_key(item) for item in gold_relationship_keys}
    direction_denominator = 0
    direction_numerator = 0
    gold_exact_keys = Counter(gold_relationship_keys)
    for key in actual_relationship_keys:
        if _base_relationship_key(key) not in gold_base_keys:
            continue
        direction_denominator += 1
        if gold_exact_keys[key] > 0:
            direction_numerator += 1

    quarantined_relationship_ids = {
        str(issue["object_id"])
        for issue in issues
        if issue.get("object_type") == "relationship" and issue.get("object_id")
    }
    expected_quarantine = set(case.gold.quarantined_relationship_ids)
    provenance_complete, provenance_total = _provenance_counts(extraction)
    valid_segment_ids = {segment.segment_id for segment in case.transcript.segments}
    unknown_segment_references = _unknown_segment_reference_count(
        extraction,
        valid_segment_ids,
    )
    self_relationships = sum(
        item.subject_mention_id == item.object_mention_id
        for item in extraction.relationship_claims
    )
    claims_without_evidence = _accepted_claims_without_evidence(extraction)

    return CaseEvaluation(
        case_id=case.case_id,
        dataset_id=dataset_id,
        split=split,
        dataset_layer=dataset_layer,
        language=case.language,
        construction_tags=case.construction_tags,
        person_mentions=_person_metrics(extraction, case.gold, mention_to_gold),
        relationships=relationship_metrics,
        quarantined_relationships=_set_prf(
            quarantined_relationship_ids,
            expected_quarantine,
        ),
        relationship_direction_accuracy=ratio_metric(
            direction_numerator,
            direction_denominator,
        ),
        provenance_completeness=ratio_metric(provenance_complete, provenance_total),
        unsupported_relationship_acceptance=error_rate_metric(
            relationship_metrics.false_positive,
            len(actual_relationship_keys),
        ),
        unknown_segment_references=unknown_segment_references,
        self_relationships=self_relationships,
        accepted_claims_without_evidence=claims_without_evidence,
        critical_graph_violations=unknown_segment_references + self_relationships,
        accepted_relationship_ids=sorted(
            item.relationship_id for item in extraction.relationship_claims
        ),
        quarantined_relationship_ids=sorted(quarantined_relationship_ids),
        extraction_issue_count=len(issues),
        evidence_closure_relationships=evidence_closure_relationships,
    )


def aggregate_case_metrics(cases: list[CaseEvaluation]) -> BenchmarkSummary:
    def aggregate_prf(attribute: str) -> PrecisionRecallF1:
        metrics = [getattr(case, attribute) for case in cases]
        return precision_recall_f1(
            sum(metric.true_positive for metric in metrics),
            sum(metric.false_positive for metric in metrics),
            sum(metric.false_negative for metric in metrics),
        )

    direction_numerator = sum(case.relationship_direction_accuracy.numerator for case in cases)
    direction_denominator = sum(case.relationship_direction_accuracy.denominator for case in cases)
    provenance_numerator = sum(case.provenance_completeness.numerator for case in cases)
    provenance_denominator = sum(case.provenance_completeness.denominator for case in cases)
    unsupported_numerator = sum(
        case.unsupported_relationship_acceptance.numerator for case in cases
    )
    unsupported_denominator = sum(
        case.unsupported_relationship_acceptance.denominator for case in cases
    )
    unknown_segment_references = sum(case.unknown_segment_references for case in cases)
    self_relationships = sum(case.self_relationships for case in cases)

    return BenchmarkSummary(
        case_count=len(cases),
        person_mentions=aggregate_prf("person_mentions"),
        relationships=aggregate_prf("relationships"),
        quarantined_relationships=aggregate_prf("quarantined_relationships"),
        relationship_direction_accuracy=ratio_metric(
            direction_numerator,
            direction_denominator,
        ),
        provenance_completeness=ratio_metric(
            provenance_numerator,
            provenance_denominator,
        ),
        unsupported_relationship_acceptance=error_rate_metric(
            unsupported_numerator,
            unsupported_denominator,
        ),
        unknown_segment_references=unknown_segment_references,
        self_relationships=self_relationships,
        accepted_claims_without_evidence=sum(
            case.accepted_claims_without_evidence for case in cases
        ),
        critical_graph_violations=sum(case.critical_graph_violations for case in cases),
    )
