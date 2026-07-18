from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, TypeVar, cast

from mura.domain.models import (
    AssertionMode,
    ClaimObjectType,
    ClaimProvenance,
    ConflictSet,
    CoreferenceLink,
    CoreferenceStatus,
    EvidenceBackedObject,
    EvidenceClass,
    EvidencePurpose,
    EvidenceSourceLayer,
    EvidenceSpan,
    ExtractionResult,
    FamilyEvent,
    NameVariant,
    NameVariantType,
    PersonDescription,
    PersonMention,
    ProvenanceActivity,
    ProvenanceStage,
    RelationshipClaim,
    Story,
    TranscriptEnvelope,
    UnresolvedQuestion,
)
from mura.relationship_evidence import (
    analyze_relationship_evidence,
    contains_exact_surface,
    contains_surface,
    has_first_person_reference,
    joined_segment_text,
    normalize_evidence,
    person_name_surfaces,
)
from mura.versioning import get_pipeline_versions

ObjectT = TypeVar("ObjectT", bound=EvidenceBackedObject)

_AUTO_ACCEPTABLE_EVIDENCE_CLASSES = {
    EvidenceClass.A_EXPLICIT,
    EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT,
    EvidenceClass.C_SPEAKER_ANCHORED,
}

_EVIDENCE_CLASS_RANK = {
    EvidenceClass.A_EXPLICIT: 0,
    EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT: 1,
    EvidenceClass.C_SPEAKER_ANCHORED: 2,
    EvidenceClass.D_CONTEXT_RESOLVED: 3,
    EvidenceClass.E_INFERRED: 4,
    EvidenceClass.U_UNCERTAIN: 5,
}


@dataclass(frozen=True)
class ClaimModelIssue:
    object_type: str
    object_id: str | None
    stage: str
    detail: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.context is None:
            result.pop("context")
        return result


def is_auto_acceptable_evidence_class(evidence_class: EvidenceClass) -> bool:
    return evidence_class in _AUTO_ACCEPTABLE_EVIDENCE_CLASSES


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_")
    return normalized or "unknown"


def _ordered_segment_ids(segment_ids: list[str], transcript: TranscriptEnvelope) -> list[str]:
    requested = set(segment_ids)
    return [
        segment.segment_id for segment in transcript.segments if segment.segment_id in requested
    ]


def _object_identity(item: EvidenceBackedObject) -> tuple[ClaimObjectType, str]:
    for object_type, field_name in (
        (ClaimObjectType.PERSON_MENTION, "mention_id"),
        (ClaimObjectType.RELATIONSHIP, "relationship_id"),
        (ClaimObjectType.EVENT, "event_id"),
        (ClaimObjectType.DESCRIPTION, "description_id"),
        (ClaimObjectType.STORY, "story_id"),
        (ClaimObjectType.QUESTION, "question_id"),
    ):
        value = getattr(item, field_name, None)
        if isinstance(value, str) and value:
            return object_type, value
    raise ValueError(f"unsupported evidence-backed object {type(item).__name__}")


def _objects(result: ExtractionResult) -> list[EvidenceBackedObject]:
    return [
        *result.people_mentions,
        *result.relationship_claims,
        *result.events,
        *result.descriptions,
        *result.stories,
        *result.unresolved_questions,
    ]


def _authoritative_activities(
    transcript: TranscriptEnvelope,
) -> tuple[list[ProvenanceActivity], str, str]:
    suffix = _safe_id(transcript.recording_id)
    versions = get_pipeline_versions()
    extractor_activity_id = f"activity_extractor_{suffix}"
    sanitizer_activity_id = f"activity_sanitizer_{suffix}"
    return (
        [
            ProvenanceActivity(
                activity_id=f"activity_asr_{suffix}",
                stage=ProvenanceStage.ASR,
                system=transcript.asr_model,
                version=transcript.asr_revision,
                model_name=transcript.asr_model,
                metadata={"chunker_version": transcript.chunker_version},
            ),
            ProvenanceActivity(
                activity_id=extractor_activity_id,
                stage=ProvenanceStage.EXTRACTOR,
                system="deepseek",
                version=versions.extractor_prompt,
                prompt_version=versions.extractor_prompt,
                metadata={"pipeline": versions.pipeline},
            ),
            ProvenanceActivity(
                activity_id=sanitizer_activity_id,
                stage=ProvenanceStage.SANITIZER,
                system="mura",
                version=versions.evidence_rules,
                metadata={"domain_schema": versions.domain_schema},
            ),
        ],
        extractor_activity_id,
        sanitizer_activity_id,
    )


def _validate_candidate_evidence(
    evidence: EvidenceSpan,
    *,
    transcript: TranscriptEnvelope,
    mention_ids: set[str],
) -> str | None:
    segment_by_id = {segment.segment_id: segment for segment in transcript.segments}
    segment = segment_by_id.get(evidence.segment_id)
    if segment is None:
        return "references an unknown segment"
    if evidence.source_layer is not EvidenceSourceLayer.RAW_TRANSCRIPT:
        return "claim evidence must be anchored to the immutable raw transcript"
    if not contains_exact_surface(segment.text, evidence.text):
        return "text is not present in the cited raw segment"
    if evidence.start_char is not None and evidence.end_char is not None:
        if segment.text[evidence.start_char : evidence.end_char] != evidence.text:
            return "character offsets do not match the cited raw segment"
    unknown_mentions = set(evidence.mention_ids) - mention_ids
    if unknown_mentions:
        return f"references unknown mentions: {sorted(unknown_mentions)}"
    return None


def _infer_person_evidence_class(
    person: PersonMention,
    *,
    transcript: TranscriptEnvelope,
    speaker_name: str,
) -> EvidenceClass:
    source_text = joined_segment_text(person.source_segment_ids, transcript)
    surfaces = person_name_surfaces(person)
    if any(contains_exact_surface(source_text, surface) for surface in surfaces):
        return EvidenceClass.A_EXPLICIT
    if any(contains_surface(source_text, surface) for surface in surfaces):
        return EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT
    if normalize_evidence(person.name) == normalize_evidence(speaker_name):
        if has_first_person_reference(source_text):
            return EvidenceClass.C_SPEAKER_ANCHORED
    if person.assertion_mode is AssertionMode.INFERRED:
        return EvidenceClass.E_INFERRED
    return EvidenceClass.U_UNCERTAIN


def _infer_relationship_evidence_class(
    relationship: RelationshipClaim,
    *,
    transcript: TranscriptEnvelope,
    people: list[PersonMention],
    speaker_name: str,
    coreference_by_id: dict[str, CoreferenceLink],
) -> EvidenceClass:
    analysis = analyze_relationship_evidence(
        relationship=relationship,
        transcript=transcript,
        people=people,
        speaker_name=speaker_name,
    )
    if analysis.evidence_class in {
        EvidenceClass.A_EXPLICIT.value,
        EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT.value,
        EvidenceClass.C_SPEAKER_ANCHORED.value,
    }:
        return EvidenceClass(analysis.evidence_class)

    resolved_antecedents = {
        mention_id
        for coreference_id in relationship.coreference_link_ids
        if (link := coreference_by_id.get(coreference_id)) is not None
        and link.status is CoreferenceStatus.RESOLVED
        for mention_id in link.antecedent_mention_ids
    }
    if analysis.unsupported_endpoint_ids and set(analysis.unsupported_endpoint_ids).issubset(
        resolved_antecedents
    ):
        return EvidenceClass.D_CONTEXT_RESOLVED
    if relationship.assertion_mode is AssertionMode.INFERRED:
        return EvidenceClass.E_INFERRED
    return EvidenceClass.U_UNCERTAIN


def _infer_generic_evidence_class(item: EvidenceBackedObject) -> EvidenceClass:
    assertion_mode = getattr(item, "assertion_mode", None)
    if assertion_mode is AssertionMode.EXPLICIT:
        return EvidenceClass.A_EXPLICIT
    if assertion_mode is AssertionMode.INFERRED:
        return EvidenceClass.E_INFERRED
    return EvidenceClass.U_UNCERTAIN


def _mention_ids_for_object(
    object_type: ClaimObjectType,
    object_id: str,
    item: EvidenceBackedObject,
) -> list[str]:
    if object_type is ClaimObjectType.PERSON_MENTION:
        return [object_id]
    if isinstance(item, RelationshipClaim):
        return [item.subject_mention_id, item.object_mention_id]
    if isinstance(item, FamilyEvent):
        return list(item.participant_mention_ids)
    if isinstance(item, PersonDescription):
        return [item.person_mention_id]
    if isinstance(item, Story):
        return list(item.person_mention_ids)
    if isinstance(item, UnresolvedQuestion):
        return list(item.related_mention_ids)
    raise ValueError(f"unsupported evidence-backed object {type(item).__name__}")


def _generated_evidence(
    *,
    object_type: ClaimObjectType,
    object_id: str,
    item: EvidenceBackedObject,
    evidence_class: EvidenceClass,
    transcript: TranscriptEnvelope,
    sanitizer_activity_id: str,
) -> list[EvidenceSpan]:
    segment_by_id = {segment.segment_id: segment for segment in transcript.segments}
    purpose = (
        EvidencePurpose.IDENTITY
        if object_type is ClaimObjectType.PERSON_MENTION
        else EvidencePurpose.CLAIM
    )
    mention_ids = _mention_ids_for_object(object_type, object_id, item)
    generated: list[EvidenceSpan] = []
    for segment_id in _ordered_segment_ids(item.source_segment_ids, transcript):
        segment = segment_by_id[segment_id]
        generated.append(
            EvidenceSpan(
                evidence_id=(
                    f"evidence_{object_type.value}_{_safe_id(object_id)}_{_safe_id(segment_id)}"
                ),
                segment_id=segment_id,
                text=segment.text,
                evidence_class=evidence_class,
                purposes=[purpose],
                mention_ids=list(dict.fromkeys(mention_ids)),
                created_by_activity_id=sanitizer_activity_id,
                confidence=cast(float, getattr(item, "confidence", 1.0)),
            )
        )
    return generated


def _weakest_evidence_class(
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceSpan],
    fallback: EvidenceClass,
) -> EvidenceClass:
    classes = [
        evidence_by_id[item].evidence_class for item in evidence_ids if item in evidence_by_id
    ]
    if not classes:
        return fallback
    return max(classes, key=_EVIDENCE_CLASS_RANK.__getitem__)


def _materialize_name_variants(
    person: PersonMention,
    *,
    evidence_ids: list[str],
    valid_evidence_ids: set[str],
) -> list[NameVariant]:
    variants: list[NameVariant] = []
    seen: set[tuple[str, NameVariantType]] = set()
    person_segments = set(person.source_segment_ids)
    for candidate in person.name_variants:
        candidate_segments = [
            item for item in candidate.source_segment_ids if item in person_segments
        ]
        if not candidate_segments:
            continue
        candidate_evidence = [item for item in candidate.evidence_ids if item in valid_evidence_ids]
        updated = candidate.model_copy(
            update={
                "source_segment_ids": candidate_segments,
                "evidence_ids": candidate_evidence,
            }
        )
        key = (updated.normalized, updated.variant_type)
        if key in seen:
            continue
        seen.add(key)
        variants.append(updated)

    surfaces = [(person.name, NameVariantType.PRIMARY)]
    surfaces.extend((alias, NameVariantType.EXPLICIT_ALIAS) for alias in person.aliases)
    for index, (surface, variant_type) in enumerate(surfaces, start=1):
        normalized = normalize_evidence(surface)
        key = (normalized, variant_type)
        if not normalized or key in seen:
            continue
        seen.add(key)
        variants.append(
            NameVariant(
                variant_id=f"variant_{_safe_id(person.mention_id)}_{index:03d}",
                surface=surface,
                normalized=normalized,
                variant_type=variant_type,
                source_segment_ids=list(person.source_segment_ids),
                evidence_ids=list(evidence_ids),
                confidence=person.confidence,
                verification_status=person.verification_status,
            )
        )
    return variants


def _filter_coreference_links(
    links: list[CoreferenceLink],
    *,
    transcript: TranscriptEnvelope,
    mention_ids: set[str],
    evidence_by_id: dict[str, EvidenceSpan],
) -> tuple[list[CoreferenceLink], list[ClaimModelIssue]]:
    valid_segments = {segment.segment_id for segment in transcript.segments}
    accepted: list[CoreferenceLink] = []
    issues: list[ClaimModelIssue] = []
    for link in links:
        unknown_segments = set(link.source_segment_ids) - valid_segments
        unknown_evidence = set(link.evidence_ids) - set(evidence_by_id)
        unknown_mentions = (
            set(link.antecedent_mention_ids).union(link.candidate_mention_ids) - mention_ids
        )
        source_text = joined_segment_text(link.source_segment_ids, transcript)
        detail: str | None = None
        if unknown_segments:
            detail = f"references unknown segments: {sorted(unknown_segments)}"
        elif unknown_evidence:
            detail = f"references unknown evidence: {sorted(unknown_evidence)}"
        elif unknown_mentions:
            detail = f"references unknown mentions: {sorted(unknown_mentions)}"
        elif not contains_exact_surface(source_text, link.anaphor_text):
            detail = "anaphor text is not present in the cited segments"
        if detail is not None:
            issues.append(
                ClaimModelIssue(
                    object_type="coreference",
                    object_id=link.coreference_id,
                    stage="semantic",
                    detail=detail,
                    context={"candidate": link.model_dump(mode="json")},
                )
            )
            continue
        accepted.append(link)
    return accepted, issues


def _materialize_item(
    item: ObjectT,
    *,
    evidence_class: EvidenceClass,
    transcript: TranscriptEnvelope,
    speaker_id: str,
    speaker_name: str,
    evidence_by_id: dict[str, EvidenceSpan],
    coreference_by_id: dict[str, CoreferenceLink],
    extractor_activity_id: str,
    sanitizer_activity_id: str,
) -> tuple[ObjectT, list[ClaimModelIssue]]:
    object_type, object_id = _object_identity(item)
    issues: list[ClaimModelIssue] = []
    valid_evidence_ids = [
        evidence_id
        for evidence_id in item.evidence_ids
        if evidence_id in evidence_by_id
        and evidence_by_id[evidence_id].segment_id in item.source_segment_ids
    ]
    invalid_evidence_ids = sorted(set(item.evidence_ids) - set(valid_evidence_ids))
    if invalid_evidence_ids:
        issues.append(
            ClaimModelIssue(
                object_type=object_type.value,
                object_id=object_id,
                stage="provenance",
                detail=(
                    f"unknown or out-of-scope evidence IDs were removed: {invalid_evidence_ids}"
                ),
            )
        )

    if not valid_evidence_ids:
        generated = _generated_evidence(
            object_type=object_type,
            object_id=object_id,
            item=item,
            evidence_class=evidence_class,
            transcript=transcript,
            sanitizer_activity_id=sanitizer_activity_id,
        )
        valid_evidence_ids = [evidence.evidence_id for evidence in generated]
        evidence_by_id.update({evidence.evidence_id: evidence for evidence in generated})

    valid_coreference_ids = [
        link_id for link_id in item.coreference_link_ids if link_id in coreference_by_id
    ]
    invalid_coreference_ids = sorted(set(item.coreference_link_ids) - set(valid_coreference_ids))
    if invalid_coreference_ids:
        issues.append(
            ClaimModelIssue(
                object_type=object_type.value,
                object_id=object_id,
                stage="coreference",
                detail=f"unknown coreference link IDs were removed: {invalid_coreference_ids}",
            )
        )

    final_class = _weakest_evidence_class(valid_evidence_ids, evidence_by_id, evidence_class)
    provenance = ClaimProvenance(
        recording_id=transcript.recording_id,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        generated_by_activity_id=extractor_activity_id,
        validated_by_activity_ids=[sanitizer_activity_id],
        evidence_ids=list(valid_evidence_ids),
        derived_from_claim_ids=(
            list(item.provenance.derived_from_claim_ids) if item.provenance is not None else []
        ),
        pipeline_versions=get_pipeline_versions().model_dump(mode="json"),
    )
    update: dict[str, Any] = {
        "evidence_ids": valid_evidence_ids,
        "evidence_class": final_class,
        "coreference_link_ids": valid_coreference_ids,
        "provenance": provenance,
    }
    if isinstance(item, PersonMention):
        update["name_variants"] = _materialize_name_variants(
            item,
            evidence_ids=valid_evidence_ids,
            valid_evidence_ids=set(evidence_by_id),
        )
    return item.model_copy(update=update), issues


def _claim_ref_index(
    result: ExtractionResult,
) -> dict[tuple[ClaimObjectType, str], EvidenceBackedObject]:
    return {_object_identity(item): item for item in _objects(result)}


def _filter_conflicts(
    conflicts: list[ConflictSet],
    *,
    result: ExtractionResult,
    evidence_ids: set[str],
) -> tuple[list[ConflictSet], list[ClaimModelIssue]]:
    claim_index = _claim_ref_index(result)
    accepted: list[ConflictSet] = []
    issues: list[ClaimModelIssue] = []
    for conflict in conflicts:
        unknown_claims = [
            ref.model_dump(mode="json")
            for ref in conflict.claim_refs
            if (ref.object_type, ref.object_id) not in claim_index
        ]
        unknown_evidence = sorted(set(conflict.evidence_ids) - evidence_ids)
        detail: str | None = None
        if unknown_claims:
            detail = f"references unknown claims: {unknown_claims}"
        elif unknown_evidence:
            detail = f"references unknown evidence: {unknown_evidence}"
        if detail is not None:
            issues.append(
                ClaimModelIssue(
                    object_type="conflict",
                    object_id=conflict.conflict_id,
                    stage="semantic",
                    detail=detail,
                    context={"candidate": conflict.model_dump(mode="json")},
                )
            )
            continue
        accepted.append(conflict)
    return accepted, issues


def _attach_conflict_ids(
    items: list[ObjectT],
    mapping: dict[tuple[ClaimObjectType, str], list[str]],
) -> list[ObjectT]:
    return [
        item.model_copy(update={"conflict_ids": mapping.get(_object_identity(item), [])})
        for item in items
    ]


def _sync_conflict_ids(
    result: ExtractionResult,
    conflicts: list[ConflictSet],
) -> ExtractionResult:
    conflict_ids_by_claim: dict[tuple[ClaimObjectType, str], list[str]] = {}
    for conflict in conflicts:
        for ref in conflict.claim_refs:
            conflict_ids_by_claim.setdefault((ref.object_type, ref.object_id), []).append(
                conflict.conflict_id
            )
    return result.model_copy(
        update={
            "people_mentions": _attach_conflict_ids(
                list(result.people_mentions), conflict_ids_by_claim
            ),
            "relationship_claims": _attach_conflict_ids(
                list(result.relationship_claims), conflict_ids_by_claim
            ),
            "events": _attach_conflict_ids(list(result.events), conflict_ids_by_claim),
            "descriptions": _attach_conflict_ids(list(result.descriptions), conflict_ids_by_claim),
            "stories": _attach_conflict_ids(list(result.stories), conflict_ids_by_claim),
            "unresolved_questions": _attach_conflict_ids(
                list(result.unresolved_questions), conflict_ids_by_claim
            ),
            "conflict_sets": conflicts,
        }
    )


def _materialize_generic_items(
    items: list[ObjectT],
    *,
    transcript: TranscriptEnvelope,
    result: ExtractionResult,
    evidence_by_id: dict[str, EvidenceSpan],
    coreference_by_id: dict[str, CoreferenceLink],
    extractor_activity_id: str,
    sanitizer_activity_id: str,
) -> tuple[list[ObjectT], list[ClaimModelIssue]]:
    materialized: list[ObjectT] = []
    issues: list[ClaimModelIssue] = []
    for candidate in items:
        item, item_issues = _materialize_item(
            candidate,
            evidence_class=_infer_generic_evidence_class(candidate),
            transcript=transcript,
            speaker_id=result.speaker_id,
            speaker_name=result.speaker_name,
            evidence_by_id=evidence_by_id,
            coreference_by_id=coreference_by_id,
            extractor_activity_id=extractor_activity_id,
            sanitizer_activity_id=sanitizer_activity_id,
        )
        materialized.append(item)
        issues.extend(item_issues)
    return materialized, issues


def materialize_extraction_contract_v2(
    result: ExtractionResult,
    transcript: TranscriptEnvelope,
) -> tuple[ExtractionResult, list[ClaimModelIssue]]:
    """Attach authoritative provenance while preserving valid model-proposed metadata."""

    issues: list[ClaimModelIssue] = []
    if result.provenance_activities:
        issues.append(
            ClaimModelIssue(
                object_type="provenance_activity",
                object_id=None,
                stage="provenance",
                detail="model-provided provenance activities were replaced by authoritative data",
            )
        )

    mention_ids = {person.mention_id for person in result.people_mentions}
    activities, extractor_activity_id, sanitizer_activity_id = _authoritative_activities(transcript)
    evidence_by_id: dict[str, EvidenceSpan] = {}
    for evidence in result.evidence_spans:
        detail = _validate_candidate_evidence(
            evidence,
            transcript=transcript,
            mention_ids=mention_ids,
        )
        if detail is not None:
            issues.append(
                ClaimModelIssue(
                    object_type="evidence",
                    object_id=evidence.evidence_id,
                    stage="semantic",
                    detail=detail,
                    context={"candidate": evidence.model_dump(mode="json")},
                )
            )
            continue
        evidence_by_id[evidence.evidence_id] = evidence.model_copy(
            update={"created_by_activity_id": extractor_activity_id}
        )

    coreference_links, coreference_issues = _filter_coreference_links(
        result.coreference_links,
        transcript=transcript,
        mention_ids=mention_ids,
        evidence_by_id=evidence_by_id,
    )
    issues.extend(coreference_issues)
    coreference_by_id = {item.coreference_id: item for item in coreference_links}

    people: list[PersonMention] = []
    for person in result.people_mentions:
        materialized_person, item_issues = _materialize_item(
            person,
            evidence_class=_infer_person_evidence_class(
                person,
                transcript=transcript,
                speaker_name=result.speaker_name,
            ),
            transcript=transcript,
            speaker_id=result.speaker_id,
            speaker_name=result.speaker_name,
            evidence_by_id=evidence_by_id,
            coreference_by_id=coreference_by_id,
            extractor_activity_id=extractor_activity_id,
            sanitizer_activity_id=sanitizer_activity_id,
        )
        people.append(materialized_person)
        issues.extend(item_issues)

    relationships: list[RelationshipClaim] = []
    for relationship in result.relationship_claims:
        materialized_relationship, item_issues = _materialize_item(
            relationship,
            evidence_class=_infer_relationship_evidence_class(
                relationship,
                transcript=transcript,
                people=people,
                speaker_name=result.speaker_name,
                coreference_by_id=coreference_by_id,
            ),
            transcript=transcript,
            speaker_id=result.speaker_id,
            speaker_name=result.speaker_name,
            evidence_by_id=evidence_by_id,
            coreference_by_id=coreference_by_id,
            extractor_activity_id=extractor_activity_id,
            sanitizer_activity_id=sanitizer_activity_id,
        )
        relationships.append(materialized_relationship)
        issues.extend(item_issues)

    events, event_issues = _materialize_generic_items(
        list(result.events),
        transcript=transcript,
        result=result,
        evidence_by_id=evidence_by_id,
        coreference_by_id=coreference_by_id,
        extractor_activity_id=extractor_activity_id,
        sanitizer_activity_id=sanitizer_activity_id,
    )
    descriptions, description_issues = _materialize_generic_items(
        list(result.descriptions),
        transcript=transcript,
        result=result,
        evidence_by_id=evidence_by_id,
        coreference_by_id=coreference_by_id,
        extractor_activity_id=extractor_activity_id,
        sanitizer_activity_id=sanitizer_activity_id,
    )
    stories, story_issues = _materialize_generic_items(
        list(result.stories),
        transcript=transcript,
        result=result,
        evidence_by_id=evidence_by_id,
        coreference_by_id=coreference_by_id,
        extractor_activity_id=extractor_activity_id,
        sanitizer_activity_id=sanitizer_activity_id,
    )
    questions, question_issues = _materialize_generic_items(
        list(result.unresolved_questions),
        transcript=transcript,
        result=result,
        evidence_by_id=evidence_by_id,
        coreference_by_id=coreference_by_id,
        extractor_activity_id=extractor_activity_id,
        sanitizer_activity_id=sanitizer_activity_id,
    )
    issues.extend(event_issues)
    issues.extend(description_issues)
    issues.extend(story_issues)
    issues.extend(question_issues)

    updated = result.model_copy(
        update={
            "schema_version": "extraction-v2",
            "provenance_activities": activities,
            "evidence_spans": list(evidence_by_id.values()),
            "coreference_links": coreference_links,
            "people_mentions": people,
            "relationship_claims": relationships,
            "events": events,
            "descriptions": descriptions,
            "stories": stories,
            "unresolved_questions": questions,
        }
    )
    conflicts, conflict_issues = _filter_conflicts(
        result.conflict_sets,
        result=updated,
        evidence_ids=set(evidence_by_id),
    )
    issues.extend(conflict_issues)
    return _sync_conflict_ids(updated, conflicts), issues


def validate_extraction_contract_v2(
    transcript: TranscriptEnvelope,
    result: ExtractionResult,
) -> None:
    if result.schema_version != "extraction-v2":
        return

    activity_ids = {activity.activity_id for activity in result.provenance_activities}
    evidence_by_id = {evidence.evidence_id: evidence for evidence in result.evidence_spans}
    coreference_by_id = {item.coreference_id: item for item in result.coreference_links}
    conflict_by_id = {item.conflict_id: item for item in result.conflict_sets}
    mention_ids = {person.mention_id for person in result.people_mentions}

    if not activity_ids:
        raise ValueError("extraction-v2 requires provenance activities")
    if not evidence_by_id and _objects(result):
        raise ValueError("extraction-v2 objects require evidence spans")

    for evidence in result.evidence_spans:
        detail = _validate_candidate_evidence(
            evidence,
            transcript=transcript,
            mention_ids=mention_ids,
        )
        if detail is not None:
            raise ValueError(f"evidence {evidence.evidence_id} {detail}")
        if evidence.created_by_activity_id not in activity_ids:
            raise ValueError(
                f"evidence {evidence.evidence_id} references an unknown provenance activity"
            )

    for link in result.coreference_links:
        unknown_evidence = set(link.evidence_ids) - set(evidence_by_id)
        unknown_mentions = (
            set(link.antecedent_mention_ids).union(link.candidate_mention_ids) - mention_ids
        )
        if unknown_evidence:
            raise ValueError(
                f"coreference {link.coreference_id} references unknown evidence: "
                f"{sorted(unknown_evidence)}"
            )
        if unknown_mentions:
            raise ValueError(
                f"coreference {link.coreference_id} references unknown mentions: "
                f"{sorted(unknown_mentions)}"
            )

    claim_index = _claim_ref_index(result)
    for conflict in result.conflict_sets:
        for ref in conflict.claim_refs:
            if (ref.object_type, ref.object_id) not in claim_index:
                raise ValueError(
                    f"conflict {conflict.conflict_id} references unknown claim "
                    f"{ref.object_type.value}:{ref.object_id}"
                )
        unknown_evidence = set(conflict.evidence_ids) - set(evidence_by_id)
        if unknown_evidence:
            raise ValueError(
                f"conflict {conflict.conflict_id} references unknown evidence: "
                f"{sorted(unknown_evidence)}"
            )

    for item in _objects(result):
        object_type, object_id = _object_identity(item)
        if not item.evidence_ids:
            raise ValueError(f"{object_type.value} {object_id} has no evidence IDs")
        unknown_evidence = set(item.evidence_ids) - set(evidence_by_id)
        if unknown_evidence:
            raise ValueError(
                f"{object_type.value} {object_id} references unknown evidence: "
                f"{sorted(unknown_evidence)}"
            )
        evidence_segments = {evidence_by_id[item_id].segment_id for item_id in item.evidence_ids}
        if not evidence_segments.issubset(item.source_segment_ids):
            raise ValueError(
                f"{object_type.value} {object_id} evidence is outside source_segment_ids"
            )
        expected_class = _weakest_evidence_class(
            item.evidence_ids,
            evidence_by_id,
            item.evidence_class,
        )
        if item.evidence_class is not expected_class:
            raise ValueError(f"{object_type.value} {object_id} has inconsistent evidence class")
        if item.provenance is None:
            raise ValueError(f"{object_type.value} {object_id} has no provenance record")
        provenance = item.provenance
        if provenance.recording_id != result.recording_id:
            raise ValueError(f"{object_type.value} {object_id} has wrong provenance recording")
        narrator_mismatch = (
            provenance.speaker_id != result.speaker_id
            or provenance.speaker_name != result.speaker_name
        )
        if narrator_mismatch:
            raise ValueError(f"{object_type.value} {object_id} has wrong narrator provenance")
        if provenance.generated_by_activity_id not in activity_ids:
            raise ValueError(f"{object_type.value} {object_id} has unknown generation activity")
        if set(provenance.validated_by_activity_ids) - activity_ids:
            raise ValueError(f"{object_type.value} {object_id} has unknown validation activity")
        if provenance.evidence_ids != item.evidence_ids:
            raise ValueError(
                f"{object_type.value} {object_id} provenance evidence does not match claim"
            )
        if set(item.coreference_link_ids) - set(coreference_by_id):
            raise ValueError(
                f"{object_type.value} {object_id} references unknown coreference links"
            )
        if set(item.conflict_ids) - set(conflict_by_id):
            raise ValueError(f"{object_type.value} {object_id} references unknown conflicts")
        for conflict_id in item.conflict_ids:
            conflict_keys = {
                (candidate.object_type, candidate.object_id)
                for candidate in conflict_by_id[conflict_id].claim_refs
            }
            if (object_type, object_id) not in conflict_keys:
                raise ValueError(
                    f"{object_type.value} {object_id} is not included in conflict {conflict_id}"
                )

    for person in result.people_mentions:
        if not person.name_variants:
            raise ValueError(f"person mention {person.mention_id} has no name variants")
        variant_ids = [variant.variant_id for variant in person.name_variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError(f"person mention {person.mention_id} has duplicate variant IDs")
        primary = [
            variant
            for variant in person.name_variants
            if variant.variant_type is NameVariantType.PRIMARY
            and variant.normalized == normalize_evidence(person.name)
        ]
        if not primary:
            raise ValueError(f"person mention {person.mention_id} has no primary name variant")
        for variant in person.name_variants:
            if not set(variant.source_segment_ids).issubset(person.source_segment_ids):
                raise ValueError(
                    f"name variant {variant.variant_id} is outside person source segments"
                )
            if set(variant.evidence_ids) - set(evidence_by_id):
                raise ValueError(f"name variant {variant.variant_id} references unknown evidence")
