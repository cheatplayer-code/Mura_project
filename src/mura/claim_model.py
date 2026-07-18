from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from mura.domain.models import (
    AssertionMode,
    ClaimObjectType,
    ClaimProvenance,
    ClaimReference,
    ConflictSet,
    CoreferenceLink,
    CoreferenceStatus,
    EvidenceBackedObject,
    EvidenceClass,
    EvidencePurpose,
    EvidenceSourceLayer,
    EvidenceSpan,
    ExtractionResult,
    NameVariant,
    NameVariantType,
    PersonMention,
    ProvenanceActivity,
    ProvenanceStage,
    RelationshipClaim,
    TranscriptEnvelope,
)
from mura.relationship_evidence import (
    analyze_relationship_evidence,
    contains_surface,
    has_first_person_reference,
    joined_segment_text,
    normalize_evidence,
)
from mura.versioning import get_pipeline_versions

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


def _exact_surface(text: str, surface: str) -> bool:
    normalized_text = normalize_evidence(text)
    normalized_surface = normalize_evidence(surface)
    if not normalized_surface:
        return False
    return f" {normalized_surface} " in f" {normalized_text} "


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
) -> tuple[list[ProvenanceActivity], str, str, str]:
    suffix = _safe_id(transcript.recording_id)
    asr_activity_id = f"activity_asr_{suffix}"
    extractor_activity_id = f"activity_extractor_{suffix}"
    sanitizer_activity_id = f"activity_sanitizer_{suffix}"
    versions = get_pipeline_versions()
    activities = [
        ProvenanceActivity(
            activity_id=asr_activity_id,
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
    ]
    return activities, asr_activity_id, extractor_activity_id, sanitizer_activity_id


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
    if not _exact_surface(segment.text, evidence.text):
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
    surfaces = [person.name, *person.aliases]
    if any(_exact_surface(source_text, surface) for surface in surfaces):
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
    people_by_id = {person.mention_id: person for person in people}
    exact_endpoint_ids: set[str] = set()
    for mention_id in (relationship.subject_mention_id, relationship.object_mention_id):
        person = people_by_id.get(mention_id)
        if person is None:
            continue
        if any(_exact_surface(analysis.source_text, surface) for surface in [person.name, *person.aliases]):
            exact_endpoint_ids.add(mention_id)

    endpoint_ids = {relationship.subject_mention_id, relationship.object_mention_id}
    if exact_endpoint_ids == endpoint_ids:
        return EvidenceClass.A_EXPLICIT
    if not analysis.unsupported_endpoint_ids:
        speaker_endpoint = bool(
            endpoint_ids.intersection(analysis.speaker_mention_ids)
            and analysis.first_person_reference
        )
        if speaker_endpoint:
            return EvidenceClass.C_SPEAKER_ANCHORED
        return EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT

    resolved_antecedents = {
        mention_id
        for coreference_id in relationship.coreference_link_ids
        if (link := coreference_by_id.get(coreference_id)) is not None
        and link.status is CoreferenceStatus.RESOLVED
        for mention_id in link.antecedent_mention_ids
    }
    if set(analysis.unsupported_endpoint_ids).issubset(resolved_antecedents):
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
    mention_ids: list[str] = []
    if object_type is ClaimObjectType.PERSON_MENTION:
        mention_ids = [object_id]
    elif object_type is ClaimObjectType.RELATIONSHIP:
        mention_ids = [
            getattr(item, "subject_mention_id"),
            getattr(item, "object_mention_id"),
        ]
    elif object_type is ClaimObjectType.EVENT:
        mention_ids = list(getattr(item, "participant_mention_ids"))
    elif object_type is ClaimObjectType.DESCRIPTION:
        mention_ids = [getattr(item, "person_mention_id")]
    elif object_type is ClaimObjectType.STORY:
        mention_ids = list(getattr(item, "person_mention_ids"))
    elif object_type is ClaimObjectType.QUESTION:
        mention_ids = list(getattr(item, "related_mention_ids"))

    purpose = (
        EvidencePurpose.IDENTITY
        if object_type is ClaimObjectType.PERSON_MENTION
        else EvidencePurpose.CLAIM
    )
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
                confidence=getattr(item, "confidence", 1.0),
            )
        )
    return generated


def _weakest_evidence_class(
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceSpan],
    fallback: EvidenceClass,
) -> EvidenceClass:
    classes = [evidence_by_id[item].evidence_class for item in evidence_ids if item in evidence_by_id]
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
    for candidate in person.name_variants:
        candidate_evidence = [item for item in candidate.evidence_ids if item in valid_evidence_ids]
        updated = candidate.model_copy(update={"evidence_ids": candidate_evidence})
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


def _materialize_object(
    item: EvidenceBackedObject,
    *,
    transcript: TranscriptEnvelope,
    speaker_id: str,
    speaker_name: str,
    evidence_by_id: dict[str, EvidenceSpan],
    coreference_by_id: dict[str, CoreferenceLink],
    extractor_activity_id: str,
    sanitizer_activity_id: str,
) -> tuple[EvidenceBackedObject, list[EvidenceSpan], list[ClaimModelIssue]]:
    object_type, object_id = _object_identity(item)
    issues: list[ClaimModelIssue] = []
    valid_evidence_ids = [item_id for item_id in item.evidence_ids if item_id in evidence_by_id]
    invalid_evidence_ids = sorted(set(item.evidence_ids) - set(valid_evidence_ids))
    if invalid_evidence_ids:
        issues.append(
            ClaimModelIssue(
                object_type=object_type.value,
                object_id=object_id,
                stage="provenance",
                detail=f"unknown evidence IDs were removed: {invalid_evidence_ids}",
            )
        )

    if object_type is ClaimObjectType.PERSON_MENTION:
        inferred_class = _infer_person_evidence_class(
            item,
            transcript=transcript,
            speaker_name=speaker_name,
        )
    elif object_type is ClaimObjectType.RELATIONSHIP:
        inferred_class = _infer_relationship_evidence_class(
            item,
            transcript=transcript,
            people=[],
            speaker_name=speaker_name,
            coreference_by_id=coreference_by_id,
        )
    else:
        inferred_class = _infer_generic_evidence_class(item)

    generated: list[EvidenceSpan] = []
    if not valid_evidence_ids:
        generated = _generated_evidence(
            object_type=object_type,
            object_id=object_id,
            item=item,
            evidence_class=inferred_class,
            transcript=transcript,
            sanitizer_activity_id=sanitizer_activity_id,
        )
        valid_evidence_ids = [evidence.evidence_id for evidence in generated]
        evidence_by_id.update({evidence.evidence_id: evidence for evidence in generated})

    evidence_class = _weakest_evidence_class(
        valid_evidence_ids,
        evidence_by_id,
        inferred_class,
    )
    valid_coreference_ids = [
        item_id for item_id in item.coreference_link_ids if item_id in coreference_by_id
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

    versions = get_pipeline_versions().model_dump(mode="json")
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
        pipeline_versions=versions,
    )
    update: dict[str, Any] = {
        "evidence_ids": valid_evidence_ids,
        "evidence_class": evidence_class,
        "coreference_link_ids": valid_coreference_ids,
        "provenance": provenance,
    }
    if isinstance(item, PersonMention):
        update["name_variants"] = _materialize_name_variants(
            item,
            evidence_ids=valid_evidence_ids,
            valid_evidence_ids=set(evidence_by_id),
        )
    return item.model_copy(update=update), generated, issues


def _claim_ref_index(result: ExtractionResult) -> dict[tuple[ClaimObjectType, str], EvidenceBackedObject]:
    index: dict[tuple[ClaimObjectType, str], EvidenceBackedObject] = {}
    for item in _objects(result):
        index[_object_identity(item)] = item
    return index


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
        elif not _exact_surface(source_text, link.anaphor_text):
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

    def update_items(items: list[EvidenceBackedObject]) -> list[EvidenceBackedObject]:
        return [
            item.model_copy(
                update={"conflict_ids": conflict_ids_by_claim.get(_object_identity(item), [])}
            )
            for item in items
        ]

    return result.model_copy(
        update={
            "people_mentions": update_items(list(result.people_mentions)),
            "relationship_claims": update_items(list(result.relationship_claims)),
            "events": update_items(list(result.events)),
            "descriptions": update_items(list(result.descriptions)),
            "stories": update_items(list(result.stories)),
            "unresolved_questions": update_items(list(result.unresolved_questions)),
            "conflict_sets": conflicts,
        }
    )


def materialize_extraction_contract_v2(
    result: ExtractionResult,
    transcript: TranscriptEnvelope,
) -> tuple[ExtractionResult, list[ClaimModelIssue]]:
    """Attach authoritative provenance while preserving valid model-proposed evidence metadata."""

    issues: list[ClaimModelIssue] = []
    mention_ids = {person.mention_id for person in result.people_mentions}
    activities, _, extractor_activity_id, sanitizer_activity_id = _authoritative_activities(
        transcript
    )

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
    generated_evidence: list[EvidenceSpan] = []
    for person in result.people_mentions:
        inferred_class = _infer_person_evidence_class(
            person,
            transcript=transcript,
            speaker_name=result.speaker_name,
        )
        item, generated, item_issues = _materialize_object_with_class(
            person,
            evidence_class=inferred_class,
            transcript=transcript,
            speaker_id=result.speaker_id,
            speaker_name=result.speaker_name,
            evidence_by_id=evidence_by_id,
            coreference_by_id=coreference_by_id,
            extractor_activity_id=extractor_activity_id,
            sanitizer_activity_id=sanitizer_activity_id,
        )
        people.append(item)
        generated_evidence.extend(generated)
        issues.extend(item_issues)

    relationship_items: list[RelationshipClaim] = []
    for relationship in result.relationship_claims:
        inferred_class = _infer_relationship_evidence_class(
            relationship,
            transcript=transcript,
            people=people,
            speaker_name=result.speaker_name,
            coreference_by_id=coreference_by_id,
        )
        item, generated, item_issues = _materialize_object_with_class(
            relationship,
            evidence_class=inferred_class,
            transcript=transcript,
            speaker_id=result.speaker_id,
            speaker_name=result.speaker_name,
            evidence_by_id=evidence_by_id,
            coreference_by_id=coreference_by_id,
            extractor_activity_id=extractor_activity_id,
            sanitizer_activity_id=sanitizer_activity_id,
        )
        relationship_items.append(item)
        generated_evidence.extend(generated)
        issues.extend(item_issues)

    def materialize_generic(items: list[EvidenceBackedObject]) -> list[EvidenceBackedObject]:
        materialized: list[EvidenceBackedObject] = []
        for candidate in items:
            item, generated, item_issues = _materialize_object_with_class(
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
            generated_evidence.extend(generated)
            issues.extend(item_issues)
        return materialized

    updated = result.model_copy(
        update={
            "schema_version": "extraction-v2",
            "provenance_activities": activities,
            "evidence_spans": [*evidence_by_id.values()],
            "coreference_links": coreference_links,
            "people_mentions": people,
            "relationship_claims": relationship_items,
            "events": materialize_generic(list(result.events)),
            "descriptions": materialize_generic(list(result.descriptions)),
            "stories": materialize_generic(list(result.stories)),
            "unresolved_questions": materialize_generic(list(result.unresolved_questions)),
        }
    )
    conflicts, conflict_issues = _filter_conflicts(
        result.conflict_sets,
        result=updated,
        evidence_ids=set(evidence_by_id),
    )
    issues.extend(conflict_issues)
    updated = _sync_conflict_ids(updated, conflicts)
    return updated, issues


def _materialize_object_with_class(
    item: EvidenceBackedObject,
    *,
    evidence_class: EvidenceClass,
    transcript: TranscriptEnvelope,
    speaker_id: str,
    speaker_name: str,
    evidence_by_id: dict[str, EvidenceSpan],
    coreference_by_id: dict[str, CoreferenceLink],
    extractor_activity_id: str,
    sanitizer_activity_id: str,
) -> tuple[Any, list[EvidenceSpan], list[ClaimModelIssue]]:
    object_type, object_id = _object_identity(item)
    issues: list[ClaimModelIssue] = []
    valid_evidence_ids = [item_id for item_id in item.evidence_ids if item_id in evidence_by_id]
    invalid_evidence_ids = sorted(set(item.evidence_ids) - set(valid_evidence_ids))
    if invalid_evidence_ids:
        issues.append(
            ClaimModelIssue(
                object_type=object_type.value,
                object_id=object_id,
                stage="provenance",
                detail=f"unknown evidence IDs were removed: {invalid_evidence_ids}",
            )
        )

    generated: list[EvidenceSpan] = []
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

    final_class = _weakest_evidence_class(valid_evidence_ids, evidence_by_id, evidence_class)
    valid_coreference_ids = [
        item_id for item_id in item.coreference_link_ids if item_id in coreference_by_id
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
    return item.model_copy(update=update), generated, issues


def validate_extraction_contract_v2(
    transcript: TranscriptEnvelope,
    result: ExtractionResult,
) -> None:
    if result.schema_version != "extraction-v2":
        return

    segment_by_id = {segment.segment_id: segment for segment in transcript.segments}
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
        segment = segment_by_id[evidence.segment_id]
        if evidence.start_char is not None and evidence.end_char is not None:
            if segment.text[evidence.start_char : evidence.end_char] != evidence.text:
                raise ValueError(f"evidence {evidence.evidence_id} has invalid offsets")

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
        if provenance.speaker_id != result.speaker_id or provenance.speaker_name != result.speaker_name:
            raise ValueError(f"{object_type.value} {object_id} has wrong narrator provenance")
        if provenance.generated_by_activity_id not in activity_ids:
            raise ValueError(
                f"{object_type.value} {object_id} has unknown generation activity"
            )
        if set(provenance.validated_by_activity_ids) - activity_ids:
            raise ValueError(
                f"{object_type.value} {object_id} has unknown validation activity"
            )
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
            ref = ClaimReference(object_type=object_type, object_id=object_id)
            conflict_keys = {
                (candidate.object_type, candidate.object_id)
                for candidate in conflict_by_id[conflict_id].claim_refs
            }
            if (ref.object_type, ref.object_id) not in conflict_keys:
                raise ValueError(
                    f"{object_type.value} {object_id} is not included in conflict {conflict_id}"
                )

    for person in result.people_mentions:
        if not person.name_variants:
            raise ValueError(f"person mention {person.mention_id} has no name variants")
        primary = [
            variant
            for variant in person.name_variants
            if variant.variant_type is NameVariantType.PRIMARY
            and variant.normalized == normalize_evidence(person.name)
        ]
        if not primary:
            raise ValueError(f"person mention {person.mention_id} has no primary name variant")
        for variant in person.name_variants:
            if set(variant.evidence_ids) - set(evidence_by_id):
                raise ValueError(
                    f"name variant {variant.variant_id} references unknown evidence"
                )
