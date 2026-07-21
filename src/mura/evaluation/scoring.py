from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from mura.claim_semantics import (
    date_is_invalid_calendar_value,
    date_is_silently_exactified,
    infer_relationship_state,
    relationship_is_active_candidate,
    relationship_semantic_text,
)
from mura.domain.models import (
    EvidenceSourceLayer,
    ExtractionResult,
    PersonMention,
    RelationshipClaim,
    RelationshipState,
    RelationshipType,
    TemporalKind,
    VerificationStatus,
)
from mura.evaluation.models import (
    BenchmarkCase,
    BenchmarkGold,
    BenchmarkSummary,
    CaseEvaluation,
    DatasetSplit,
    GoldRelationship,
    PrecisionRecallF1,
    RatioMetric,
)
from mura.evidence_recovery import EvidenceOffsetRecoveryMetrics
from mura.extraction_issues import ExtractionIssueCode
from mura.relationship_evidence import normalize_evidence

RelationshipSemanticKey = tuple[str, str, str, str, str]
RelationshipBaseKey = tuple[str, str, str]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
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


def _invalid_evidence_span_count(
    extraction: ExtractionResult,
    *,
    raw_text_by_id: dict[str, str],
) -> int:
    activity_ids = {activity.activity_id for activity in extraction.provenance_activities}
    evidence_ids = {evidence.evidence_id for evidence in extraction.evidence_spans}
    invalid = 0
    for evidence in extraction.evidence_spans:
        source_text = raw_text_by_id.get(evidence.segment_id)
        if evidence.source_layer is not EvidenceSourceLayer.RAW_TRANSCRIPT:
            invalid += 1
            continue
        if source_text is None:
            invalid += 1
            continue
        if evidence.start_char is None or evidence.end_char is None:
            invalid += 1
            continue
        if source_text[evidence.start_char : evidence.end_char] != evidence.text:
            invalid += 1
            continue
        if evidence.created_by_activity_id not in activity_ids:
            invalid += 1
            continue
        if set(evidence.derived_from_evidence_ids) - evidence_ids:
            invalid += 1
    return invalid


def _object_has_closed_provenance(
    item: Any,
    *,
    extraction: ExtractionResult,
    evidence_by_id: dict[str, Any],
    activity_ids: set[str],
) -> bool:
    source_ids = set(getattr(item, "source_segment_ids", []))
    item_evidence_ids = list(getattr(item, "evidence_ids", []))
    provenance = getattr(item, "provenance", None)
    if not source_ids or not item_evidence_ids or provenance is None:
        return False
    if set(item_evidence_ids) - set(evidence_by_id):
        return False
    if any(
        evidence_by_id[evidence_id].segment_id not in source_ids
        for evidence_id in item_evidence_ids
    ):
        return False
    if provenance.evidence_ids != item_evidence_ids:
        return False
    if provenance.generated_by_activity_id not in activity_ids:
        return False
    if set(provenance.validated_by_activity_ids) - activity_ids:
        return False
    if provenance.recording_id != extraction.recording_id:
        return False
    if (
        provenance.speaker_id != extraction.speaker_id
        or provenance.speaker_name != extraction.speaker_name
    ):
        return False
    if (
        getattr(item, "verification_status", VerificationStatus.UNREVIEWED)
        is not VerificationStatus.UNREVIEWED
    ):
        return False
    if isinstance(item, PersonMention):
        if not item.name_variants:
            return False
        if any(set(variant.evidence_ids) - set(evidence_by_id) for variant in item.name_variants):
            return False
    return True


def _provenance_counts(extraction: ExtractionResult) -> tuple[int, int]:
    objects = _objects(extraction)
    evidence_by_id = {evidence.evidence_id: evidence for evidence in extraction.evidence_spans}
    activity_ids = {activity.activity_id for activity in extraction.provenance_activities}
    complete = sum(
        _object_has_closed_provenance(
            item,
            extraction=extraction,
            evidence_by_id=evidence_by_id,
            activity_ids=activity_ids,
        )
        for item in objects
    )
    return complete, len(objects)


def _objects_without_evidence(extraction: ExtractionResult) -> int:
    evidence_ids = {evidence.evidence_id for evidence in extraction.evidence_spans}
    return sum(
        not getattr(item, "evidence_ids", [])
        or bool(set(getattr(item, "evidence_ids", [])) - evidence_ids)
        for item in _objects(extraction)
    )


def _unsafe_verification_status_count(extraction: ExtractionResult) -> int:
    count = sum(
        getattr(item, "verification_status", VerificationStatus.UNREVIEWED)
        is not VerificationStatus.UNREVIEWED
        for item in _objects(extraction)
    )
    count += sum(
        link.verification_status is not VerificationStatus.UNREVIEWED
        for link in extraction.coreference_links
    )
    count += sum(
        conflict.verification_status is not VerificationStatus.UNREVIEWED
        for conflict in extraction.conflict_sets
    )
    return count


def _unknown_segment_reference_count(
    extraction: ExtractionResult,
    valid_segment_ids: set[str],
) -> int:
    objects: list[Any] = [
        *extraction.people_mentions,
        *extraction.relationship_claims,
        *extraction.events,
        *extraction.descriptions,
        *extraction.stories,
        *extraction.unresolved_questions,
    ]
    return sum(
        len(set(getattr(item, "source_segment_ids", [])) - valid_segment_ids) for item in objects
    )


def _uncertain_object_refs(extraction: ExtractionResult) -> set[str]:
    refs: set[str] = set()
    for prefix, items, id_field in (
        ("person", extraction.people_mentions, "mention_id"),
        ("relationship", extraction.relationship_claims, "relationship_id"),
        ("event", extraction.events, "event_id"),
        ("description", extraction.descriptions, "description_id"),
        ("story", extraction.stories, "story_id"),
        ("question", extraction.unresolved_questions, "question_id"),
    ):
        refs.update(
            f"{prefix}:{getattr(item, id_field)}"
            for item in items
            if getattr(item, "uncertainty", None) is not None
        )
    return refs


def _ratio_for_mapping(actual: Mapping[str, str], expected: Mapping[str, object]) -> RatioMetric:
    numerator = sum(
        actual.get(key) == getattr(value, "value", value) for key, value in expected.items()
    )
    return ratio_metric(numerator, len(expected))


def score_case(
    *,
    case: BenchmarkCase,
    dataset_id: str,
    split: DatasetSplit,
    extraction: ExtractionResult,
    issues: list[dict[str, Any]],
    evidence_closure_relationships: int,
    evidence_recovery: EvidenceOffsetRecoveryMetrics,
) -> CaseEvaluation:
    mention_to_gold = _mention_to_gold_keys(extraction, case.gold)
    actual_relationship_keys = [
        _actual_relationship_key(item, mention_to_gold) for item in extraction.relationship_claims
    ]
    gold_relationship_keys = [_gold_relationship_key(item) for item in case.gold.relationships]

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
        if issue.get("object_type") == "relationship"
        and issue.get("object_id")
        and issue.get("severity") in {"error", "fatal"}
    }
    expected_quarantine = set(case.gold.quarantined_relationship_ids)
    provenance_complete, provenance_total = _provenance_counts(extraction)
    valid_segment_ids = {segment.segment_id for segment in case.transcript.segments}
    raw_text_by_id = {segment.segment_id: segment.text for segment in case.transcript.segments}
    accepted_issue_codes = {code.value for code in ExtractionIssueCode}
    actual_issue_codes = {
        str(issue.get("code")) for issue in issues if isinstance(issue.get("code"), str)
    }
    accepted_object_refs = {
        *(f"person:{item.mention_id}" for item in extraction.people_mentions),
        *(f"person_mention:{item.mention_id}" for item in extraction.people_mentions),
        *(f"relationship:{item.relationship_id}" for item in extraction.relationship_claims),
        *(f"event:{item.event_id}" for item in extraction.events),
        *(f"description:{item.description_id}" for item in extraction.descriptions),
        *(f"story:{item.story_id}" for item in extraction.stories),
        *(f"question:{item.question_id}" for item in extraction.unresolved_questions),
        *(f"evidence:{item.evidence_id}" for item in extraction.evidence_spans),
        *(f"coreference:{item.coreference_id}" for item in extraction.coreference_links),
        *(f"conflict:{item.conflict_id}" for item in extraction.conflict_sets),
    }
    quarantined_objects = {
        reference
        for issue in issues
        if issue.get("severity") in {"error", "fatal"}
        and issue.get("object_id") is not None
        and (reference := f"{issue.get('object_type')}:{issue.get('object_id')}")
        not in accepted_object_refs
    }
    expected_quarantined_objects = set(case.gold.quarantined_object_ids).union(
        f"relationship:{relationship_id}"
        for relationship_id in case.gold.quarantined_relationship_ids
    )
    uncertain_actual = _uncertain_object_refs(extraction)
    uncertainty_expected = set(case.gold.uncertain_object_ids)
    uncertainty_correct = len(uncertain_actual.intersection(uncertainty_expected))
    uncertainty_denominator = len(uncertainty_expected.union(uncertain_actual))
    temporal_actual = {
        event.event_id: event.date.kind.value
        for event in extraction.events
        if event.date is not None
    }
    relationship_state_actual = {
        item.relationship_id: item.relationship_state.value
        for item in extraction.relationship_claims
    }
    segment_text = {segment.segment_id: segment.text for segment in case.transcript.segments}
    inferred_states = {
        item.relationship_id: infer_relationship_state(
            relationship_semantic_text(
                item,
                evidence_spans=extraction.evidence_spans,
                people=extraction.people_mentions,
                fallback_text=" ".join(
                    segment_text[sid] for sid in item.source_segment_ids if sid in segment_text
                ),
            )
        )
        for item in extraction.relationship_claims
    }

    return CaseEvaluation(
        case_id=case.case_id,
        dataset_id=dataset_id,
        split=split,
        language=case.language,
        construction_tags=case.construction_tags,
        person_mentions=_person_metrics(extraction, case.gold, mention_to_gold),
        relationships=_multiset_prf(actual_relationship_keys, gold_relationship_keys),
        quarantined_relationships=_set_prf(
            quarantined_relationship_ids,
            expected_quarantine,
        ),
        quarantined_objects=_set_prf(
            quarantined_objects,
            expected_quarantined_objects,
        ),
        relationship_direction_accuracy=ratio_metric(
            direction_numerator,
            direction_denominator,
        ),
        provenance_completeness=ratio_metric(provenance_complete, provenance_total),
        unknown_segment_references=_unknown_segment_reference_count(
            extraction,
            valid_segment_ids,
        ),
        self_relationships=sum(
            item.subject_mention_id == item.object_mention_id
            for item in extraction.relationship_claims
        ),
        accepted_relationship_ids=sorted(
            item.relationship_id for item in extraction.relationship_claims
        ),
        quarantined_relationship_ids=sorted(quarantined_relationship_ids),
        extraction_issue_count=len(issues),
        evidence_closure_relationships=evidence_closure_relationships,
        provenance_violations=provenance_total - provenance_complete,
        objects_without_evidence=_objects_without_evidence(extraction),
        invalid_evidence_spans=_invalid_evidence_span_count(
            extraction,
            raw_text_by_id=raw_text_by_id,
        ),
        unsafe_verification_statuses=_unsafe_verification_status_count(extraction),
        unsafe_story_privacy=sum(story.privacy.value != "private" for story in extraction.stories),
        unknown_issue_codes=len(actual_issue_codes - accepted_issue_codes),
        missing_required_issue_codes=len(set(case.gold.required_issue_codes) - actual_issue_codes),
        fatal_contract_failures=sum(
            issue.get("code") == ExtractionIssueCode.FINAL_CONTRACT_INVALID.value
            or issue.get("severity") == "fatal"
            for issue in issues
        ),
        evidence_recovery_counts=evidence_recovery.to_dict(),
        uncertainty_scope_accuracy=ratio_metric(uncertainty_correct, uncertainty_denominator),
        temporal_kind_accuracy=_ratio_for_mapping(temporal_actual, case.gold.temporal_kinds),
        relationship_state_accuracy=_ratio_for_mapping(
            relationship_state_actual,
            case.gold.relationship_states,
        ),
        approximate_dates_exactified=sum(
            date_is_silently_exactified(event.date) for event in extraction.events
        ),
        invalid_calendar_dates_accepted=sum(
            date_is_invalid_calendar_value(event.date) for event in extraction.events
        ),
        unresolved_relative_dates_absolutized=sum(
            event.date is not None
            and event.date.kind is TemporalKind.RELATIVE
            and event.date.normalized_value is not None
            and event.date.anchor_event_id is None
            for event in extraction.events
        ),
        negated_relationship_false_positives=sum(
            inferred_states[item.relationship_id] is RelationshipState.NEGATED
            and relationship_is_active_candidate(item)
            for item in extraction.relationship_claims
        ),
        figurative_relationship_false_positives=sum(
            inferred_states[item.relationship_id] is RelationshipState.FIGURATIVE
            and relationship_is_active_candidate(item)
            for item in extraction.relationship_claims
        ),
        former_relationships_active=sum(
            inferred_states[item.relationship_id] is RelationshipState.FORMER
            and relationship_is_active_candidate(item)
            for item in extraction.relationship_claims
        ),
        ended_relationships_active=sum(
            inferred_states[item.relationship_id] is RelationshipState.ENDED
            and relationship_is_active_candidate(item)
            for item in extraction.relationship_claims
        ),
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
    uncertainty_numerator = sum(case.uncertainty_scope_accuracy.numerator for case in cases)
    uncertainty_denominator = sum(case.uncertainty_scope_accuracy.denominator for case in cases)
    temporal_numerator = sum(case.temporal_kind_accuracy.numerator for case in cases)
    temporal_denominator = sum(case.temporal_kind_accuracy.denominator for case in cases)
    state_numerator = sum(case.relationship_state_accuracy.numerator for case in cases)
    state_denominator = sum(case.relationship_state_accuracy.denominator for case in cases)

    return BenchmarkSummary(
        case_count=len(cases),
        person_mentions=aggregate_prf("person_mentions"),
        relationships=aggregate_prf("relationships"),
        quarantined_relationships=aggregate_prf("quarantined_relationships"),
        quarantined_objects=aggregate_prf("quarantined_objects"),
        relationship_direction_accuracy=ratio_metric(
            direction_numerator,
            direction_denominator,
        ),
        provenance_completeness=ratio_metric(
            provenance_numerator,
            provenance_denominator,
        ),
        unknown_segment_references=sum(case.unknown_segment_references for case in cases),
        self_relationships=sum(case.self_relationships for case in cases),
        provenance_violations=sum(case.provenance_violations for case in cases),
        objects_without_evidence=sum(case.objects_without_evidence for case in cases),
        invalid_evidence_spans=sum(case.invalid_evidence_spans for case in cases),
        unsafe_verification_statuses=sum(case.unsafe_verification_statuses for case in cases),
        unsafe_story_privacy=sum(case.unsafe_story_privacy for case in cases),
        unknown_issue_codes=sum(case.unknown_issue_codes for case in cases),
        missing_required_issue_codes=sum(case.missing_required_issue_codes for case in cases),
        fatal_contract_failures=sum(case.fatal_contract_failures for case in cases),
        uncertainty_scope_accuracy=ratio_metric(uncertainty_numerator, uncertainty_denominator),
        temporal_kind_accuracy=ratio_metric(temporal_numerator, temporal_denominator),
        relationship_state_accuracy=ratio_metric(state_numerator, state_denominator),
        approximate_dates_exactified=sum(case.approximate_dates_exactified for case in cases),
        invalid_calendar_dates_accepted=sum(case.invalid_calendar_dates_accepted for case in cases),
        unresolved_relative_dates_absolutized=sum(
            case.unresolved_relative_dates_absolutized for case in cases
        ),
        negated_relationship_false_positives=sum(
            case.negated_relationship_false_positives for case in cases
        ),
        figurative_relationship_false_positives=sum(
            case.figurative_relationship_false_positives for case in cases
        ),
        former_relationships_active=sum(case.former_relationships_active for case in cases),
        ended_relationships_active=sum(case.ended_relationships_active for case in cases),
    )
