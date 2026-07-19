from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import Field
from sqlalchemy import DateTime, String, delete, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from mura.domain.models import (
    AssertionMode,
    EvidenceClass,
    FamilyEvent,
    NameVariantType,
    PersonDescription,
    PipelineResult,
    VerificationStatus,
)
from mura.storage.archive import (
    ArchiveClaimRow,
    ArchiveConflictRow,
    ArchivePersonRow,
    FamilyGraphEdgeRow,
    _json_payload,
    _mapped_people,
    _stable_id,
)
from mura.storage.conflict_resolution import (
    ArchiveConflictDecisionRow,
    ConflictAction,
    ConflictMutationResult,
    ConflictNotFoundError,
    ConflictResolutionError,
    ConflictResolutionService,
    ConflictReviewView,
)
from mura.storage.database import JSON_VALUE, Base, Database, RecordingRow, utcnow

ATTRIBUTE_OBJECT_TYPE = "attribute"

_AUTO_MATERIALIZABLE_CLASSES = {
    EvidenceClass.A_EXPLICIT.value,
    EvidenceClass.B_MORPHOLOGICALLY_EXPLICIT.value,
    EvidenceClass.C_SPEAKER_ANCHORED.value,
}

_PROFILE_ALIAS_VARIANTS = {
    NameVariantType.EXPLICIT_ALIAS,
    NameVariantType.NICKNAME,
    NameVariantType.DIMINUTIVE,
    NameVariantType.TRANSLITERATION,
    NameVariantType.SCRIPT_VARIANT,
}

_BIRTH_EVENT_TYPES = {
    "birth",
    "born",
    "birth event",
    "рождение",
    "родился",
    "родилась",
    "туған",
    "дүниеге келу",
}
_DEATH_EVENT_TYPES = {
    "death",
    "died",
    "deceased",
    "смерть",
    "умер",
    "умерла",
    "қайтыс болу",
    "қайтыс болды",
}
_PROFESSION_EVENT_TYPES = {
    "profession",
    "occupation",
    "career",
    "work",
    "job",
    "профессия",
    "работа",
    "карьера",
    "мамандық",
    "жұмыс",
}
_EDUCATION_EVENT_TYPES = {
    "education",
    "school",
    "university",
    "study",
    "учёба",
    "учеба",
    "школа",
    "университет",
    "білім",
    "мектеп",
    "университетте оқу",
}


class MaterializedPersonProfileRow(Base):
    __tablename__ = "materialized_person_profiles"

    person_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(128), index=True)
    canonical_name: Mapped[str] = mapped_column(String(256))
    profile_payload: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE)
    source_claim_ids: Mapped[list[str]] = mapped_column(JSON_VALUE, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )


class ProfileAttributeView(Field.__class__):
    pass


class MaterializedAttributeView(ConflictReviewView.__base__):
    attribute_type: str
    value: str
    normalized_value: str
    source_claim_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonProfileView(ConflictReviewView.__base__):
    person_id: str
    family_id: str
    canonical_name: str
    category: str
    birth_date: MaterializedAttributeView | None = None
    death_date: MaterializedAttributeView | None = None
    aliases: list[MaterializedAttributeView] = Field(default_factory=list)
    professions: list[MaterializedAttributeView] = Field(default_factory=list)
    locations: list[MaterializedAttributeView] = Field(default_factory=list)
    education: list[MaterializedAttributeView] = Field(default_factory=list)
    descriptions: list[MaterializedAttributeView] = Field(default_factory=list)
    events: list[MaterializedAttributeView] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    updated_at: datetime


class GenericProjectionReport(ConflictReviewView.__base__):
    projected_claims: int = Field(ge=0)
    open_conflicts: int = Field(ge=0)
    materialized_profiles: int = Field(ge=0)


class ProfileNotFoundError(LookupError):
    pass


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.replace("_", " ").split())


def _event_type(event: FamilyEvent) -> str:
    return _normalize(event.event_type)


def _assertion_mode(value: object) -> str | None:
    return value.value if isinstance(value, AssertionMode) else None


def _verification_status(value: object) -> str:
    if isinstance(value, VerificationStatus):
        return value.value
    return VerificationStatus.UNREVIEWED.value


def _source_claims_for_recording(
    session: Session,
    *,
    recording_id: str,
) -> dict[tuple[str, str], ArchiveClaimRow]:
    rows = list(
        session.scalars(
            select(ArchiveClaimRow).where(ArchiveClaimRow.recording_id == recording_id)
        )
    )
    return {(row.object_type, row.source_object_id): row for row in rows}


def _insert_attribute_claim(
    session: Session,
    *,
    recording: RecordingRow,
    source_claim: ArchiveClaimRow | None,
    source_object_id: str,
    subject_person_id: str | None,
    attribute_type: str,
    value: str,
    normalized_value: str,
    evidence_ids: list[str],
    evidence_class: str,
    verification_status: str,
    assertion_mode: str | None,
    metadata: dict[str, Any],
) -> int:
    payload = {
        "attribute_type": attribute_type,
        "value": value,
        "normalized_value": normalized_value,
        "source_object_id": source_object_id,
        "metadata": metadata,
    }
    claim_id = _stable_id(
        "claim",
        recording.recording_id,
        ATTRIBUTE_OBJECT_TYPE,
        source_object_id,
        subject_person_id or "unresolved",
        attribute_type,
        normalized_value,
        _json_payload(metadata),
    )
    if session.get(ArchiveClaimRow, claim_id) is not None:
        return 0
    session.add(
        ArchiveClaimRow(
            claim_id=claim_id,
            family_id=recording.family_id,
            recording_id=recording.recording_id,
            object_type=ATTRIBUTE_OBJECT_TYPE,
            source_object_id=source_object_id,
            predicate=attribute_type,
            subject_person_id=subject_person_id,
            object_person_id=None,
            payload=payload,
            evidence_ids=evidence_ids,
            evidence_class=evidence_class,
            verification_status=verification_status,
            assertion_mode=assertion_mode,
            status="active" if subject_person_id is not None else "unresolved",
            derived_from_claim_ids=[source_claim.claim_id] if source_claim is not None else [],
        )
    )
    return 1


def _project_aliases(
    session: Session,
    *,
    recording: RecordingRow,
    result: PipelineResult,
    mapped_people: dict[str, str],
    source_claims: dict[tuple[str, str], ArchiveClaimRow],
) -> int:
    count = 0
    for mention in result.extraction.people_mentions:
        person_id = mapped_people.get(mention.mention_id)
        source_claim = source_claims.get(("person_mention", mention.mention_id))
        for variant in mention.name_variants:
            if variant.variant_type not in _PROFILE_ALIAS_VARIANTS:
                continue
            count += _insert_attribute_claim(
                session,
                recording=recording,
                source_claim=source_claim,
                source_object_id=f"{mention.mention_id}:{variant.variant_id}",
                subject_person_id=person_id,
                attribute_type="alias",
                value=variant.surface,
                normalized_value=variant.normalized,
                evidence_ids=list(variant.evidence_ids or mention.evidence_ids),
                evidence_class=mention.evidence_class.value,
                verification_status=variant.verification_status.value,
                assertion_mode=mention.assertion_mode.value,
                metadata={
                    "variant_type": variant.variant_type.value,
                    "language": variant.language,
                    "script": variant.script,
                },
            )
    return count


def _project_descriptions(
    session: Session,
    *,
    recording: RecordingRow,
    result: PipelineResult,
    mapped_people: dict[str, str],
    source_claims: dict[tuple[str, str], ArchiveClaimRow],
) -> int:
    count = 0
    for description in result.extraction.descriptions:
        person_id = mapped_people.get(description.person_mention_id)
        source_claim = source_claims.get(("description", description.description_id))
        count += _insert_attribute_claim(
            session,
            recording=recording,
            source_claim=source_claim,
            source_object_id=description.description_id,
            subject_person_id=person_id,
            attribute_type="description",
            value=description.description,
            normalized_value=_normalize(description.description),
            evidence_ids=list(description.evidence_ids),
            evidence_class=description.evidence_class.value,
            verification_status=description.verification_status.value,
            assertion_mode=description.assertion_mode.value,
            metadata={"perspective": description.perspective},
        )
    return count


def _event_facets(event: FamilyEvent) -> list[tuple[str, str, dict[str, Any]]]:
    facets: list[tuple[str, str, dict[str, Any]]] = []
    normalized_type = _event_type(event)
    event_metadata = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "title": event.title,
        "date": event.date.model_dump(mode="json") if event.date is not None else None,
        "location": event.location,
    }
    facets.append(("event", event.description or event.title, event_metadata))

    if event.location:
        facets.append(
            (
                "location",
                event.location,
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "date": event.date.model_dump(mode="json") if event.date is not None else None,
                },
            )
        )

    if event.date is not None and event.date.value:
        if normalized_type in _BIRTH_EVENT_TYPES:
            facets.append(
                (
                    "birth_date",
                    event.date.value,
                    {
                        "precision": event.date.precision,
                        "original_expression": event.date.original_expression,
                        "event_id": event.event_id,
                    },
                )
            )
        elif normalized_type in _DEATH_EVENT_TYPES:
            facets.append(
                (
                    "death_date",
                    event.date.value,
                    {
                        "precision": event.date.precision,
                        "original_expression": event.date.original_expression,
                        "event_id": event.event_id,
                    },
                )
            )

    if normalized_type in _PROFESSION_EVENT_TYPES:
        facets.append(
            (
                "profession",
                event.description or event.title,
                {"event_id": event.event_id, "event_type": event.event_type},
            )
        )
    if normalized_type in _EDUCATION_EVENT_TYPES:
        facets.append(
            (
                "education",
                event.description or event.title,
                {"event_id": event.event_id, "event_type": event.event_type},
            )
        )
    return facets


def _project_events(
    session: Session,
    *,
    recording: RecordingRow,
    result: PipelineResult,
    mapped_people: dict[str, str],
    source_claims: dict[tuple[str, str], ArchiveClaimRow],
) -> int:
    count = 0
    for event in result.extraction.events:
        source_claim = source_claims.get(("event", event.event_id))
        for mention_id in event.participant_mention_ids:
            person_id = mapped_people.get(mention_id)
            for attribute_type, value, metadata in _event_facets(event):
                count += _insert_attribute_claim(
                    session,
                    recording=recording,
                    source_claim=source_claim,
                    source_object_id=f"{event.event_id}:{mention_id}:{attribute_type}",
                    subject_person_id=person_id,
                    attribute_type=attribute_type,
                    value=value,
                    normalized_value=_normalize(value),
                    evidence_ids=list(event.evidence_ids),
                    evidence_class=event.evidence_class.value,
                    verification_status=event.verification_status.value,
                    assertion_mode=event.assertion_mode.value,
                    metadata=metadata,
                )
    return count


def _claim_is_grounded(claim: ArchiveClaimRow) -> bool:
    return (
        claim.object_type == ATTRIBUTE_OBJECT_TYPE
        and claim.subject_person_id is not None
        and claim.evidence_class in _AUTO_MATERIALIZABLE_CLASSES
        and bool(claim.evidence_ids)
    )


def _claim_value(claim: ArchiveClaimRow) -> str:
    return str(claim.payload.get("normalized_value", ""))


def _generic_claims(session: Session, *, family_id: str) -> list[ArchiveClaimRow]:
    return list(
        session.scalars(
            select(ArchiveClaimRow).where(
                ArchiveClaimRow.family_id == family_id,
                ArchiveClaimRow.object_type == ATTRIBUTE_OBJECT_TYPE,
            )
        )
    )


def _graph_edge_count(session: Session, *, family_id: str) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(FamilyGraphEdgeRow)
            .where(FamilyGraphEdgeRow.family_id == family_id)
        )
        or 0
    )


class GenericProfileService:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def persist_pipeline_result(
        session: Session,
        *,
        recording: RecordingRow,
        result: PipelineResult,
    ) -> GenericProjectionReport:
        mapped_people = _mapped_people(result)
        source_claims = _source_claims_for_recording(
            session,
            recording_id=recording.recording_id,
        )
        projected = 0
        projected += _project_aliases(
            session,
            recording=recording,
            result=result,
            mapped_people=mapped_people,
            source_claims=source_claims,
        )
        projected += _project_descriptions(
            session,
            recording=recording,
            result=result,
            mapped_people=mapped_people,
            source_claims=source_claims,
        )
        projected += _project_events(
            session,
            recording=recording,
            result=result,
            mapped_people=mapped_people,
            source_claims=source_claims,
        )
        session.flush()
        open_conflicts = GenericProfileService._reconcile_conflicts(
            session,
            family_id=recording.family_id,
        )
        profiles = GenericProfileService._rebuild_profiles(
            session,
            family_id=recording.family_id,
        )
        return GenericProjectionReport(
            projected_claims=projected,
            open_conflicts=open_conflicts,
            materialized_profiles=profiles,
        )

    @staticmethod
    def _conflict_groups(
        claims: list[ArchiveClaimRow],
    ) -> dict[tuple[str, ...], tuple[str, list[ArchiveClaimRow], str]]:
        groups: dict[tuple[str, ...], tuple[str, list[ArchiveClaimRow], str]] = {}

        temporal: dict[tuple[str, str], list[ArchiveClaimRow]] = defaultdict(list)
        aliases: dict[str, list[ArchiveClaimRow]] = defaultdict(list)
        for claim in claims:
            if not _claim_is_grounded(claim):
                continue
            if claim.predicate in {"birth_date", "death_date"}:
                temporal[(claim.subject_person_id or "", claim.predicate)].append(claim)
            elif claim.predicate == "alias" and _claim_value(claim):
                aliases[_claim_value(claim)].append(claim)

        for (person_id, predicate), candidates in temporal.items():
            values = {_claim_value(candidate) for candidate in candidates}
            if len(values) <= 1:
                continue
            key = ("temporal", person_id, predicate)
            groups[key] = (
                "temporal",
                candidates,
                f"grounded {predicate} claims disagree for archive person {person_id}",
            )

        for normalized_alias, candidates in aliases.items():
            people = {candidate.subject_person_id for candidate in candidates}
            if len(people) <= 1:
                continue
            key = ("identity", "alias", normalized_alias)
            groups[key] = (
                "identity",
                candidates,
                f"the same grounded alias is assigned to multiple archive people: {normalized_alias}",
            )
        return groups

    @staticmethod
    def _upsert_conflict(
        session: Session,
        *,
        family_id: str,
        key: tuple[str, ...],
        conflict_type: str,
        candidates: list[ArchiveClaimRow],
        rationale: str,
    ) -> ArchiveConflictRow:
        conflict_id = _stable_id("conflict", family_id, *key)
        claim_ids = sorted(candidate.claim_id for candidate in candidates)
        conflict = session.get(ArchiveConflictRow, conflict_id)
        if conflict is None:
            conflict = ArchiveConflictRow(
                conflict_id=conflict_id,
                family_id=family_id,
                conflict_type=conflict_type,
                status="open",
                detected_by="deterministic",
                claim_ids=claim_ids,
                rationale=rationale,
            )
            session.add(conflict)
            session.flush()
        elif set(conflict.claim_ids) != set(claim_ids):
            previous_status = conflict.status
            previous_claim_ids = list(conflict.claim_ids)
            conflict.claim_ids = claim_ids
            conflict.status = "open"
            conflict.preferred_claim_id = None
            conflict.resolution_note = None
            conflict.updated_at = utcnow()
            ConflictResolutionService._append_decision(
                session,
                conflict=conflict,
                action=ConflictAction.AUTO_REOPEN,
                previous_status=previous_status,
                resulting_status="open",
                reviewer_reference="system:archive-ingestion",
                note="conflict reopened because its competing claim set changed",
                metadata_payload={
                    "previous_claim_ids": previous_claim_ids,
                    "current_claim_ids": claim_ids,
                },
            )
        elif conflict.status == "resolved" and conflict.preferred_claim_id not in claim_ids:
            previous_status = conflict.status
            conflict.status = "open"
            conflict.preferred_claim_id = None
            conflict.resolution_note = None
            conflict.updated_at = utcnow()
            ConflictResolutionService._append_decision(
                session,
                conflict=conflict,
                action=ConflictAction.AUTO_REOPEN,
                previous_status=previous_status,
                resulting_status="open",
                reviewer_reference="system:archive-ingestion",
                note="conflict reopened because the preferred claim is unavailable",
            )
        ConflictResolutionService._apply_conflict_state(session, conflict=conflict)
        return conflict

    @staticmethod
    def _reconcile_conflicts(session: Session, *, family_id: str) -> int:
        claims = _generic_claims(session, family_id=family_id)
        for claim in claims:
            claim.status = "active" if claim.subject_person_id is not None else "unresolved"

        conflicts = []
        for key, (conflict_type, candidates, rationale) in GenericProfileService._conflict_groups(
            claims
        ).items():
            conflicts.append(
                GenericProfileService._upsert_conflict(
                    session,
                    family_id=family_id,
                    key=key,
                    conflict_type=conflict_type,
                    candidates=candidates,
                    rationale=rationale,
                )
            )
        return sum(conflict.status == "open" for conflict in conflicts)

    @staticmethod
    def _rebuild_profiles(session: Session, *, family_id: str) -> int:
        session.execute(
            delete(MaterializedPersonProfileRow).where(
                MaterializedPersonProfileRow.family_id == family_id
            )
        )
        people = list(
            session.scalars(
                select(ArchivePersonRow)
                .where(ArchivePersonRow.family_id == family_id)
                .order_by(ArchivePersonRow.person_id)
            )
        )
        claims = [
            claim
            for claim in _generic_claims(session, family_id=family_id)
            if claim.status in {"active", "accepted"} and _claim_is_grounded(claim)
        ]
        by_person: dict[str, list[ArchiveClaimRow]] = defaultdict(list)
        for claim in claims:
            if claim.subject_person_id is not None:
                by_person[claim.subject_person_id].append(claim)

        for person in people:
            person_claims = by_person.get(person.person_id, [])
            grouped: dict[str, dict[str, list[ArchiveClaimRow]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for claim in person_claims:
                grouped[claim.predicate][_claim_value(claim)].append(claim)

            facets: dict[str, list[dict[str, Any]]] = {}
            for predicate, values in grouped.items():
                entries = []
                for normalized_value, sources in values.items():
                    first = sorted(sources, key=lambda item: item.created_at)[0]
                    entries.append(
                        {
                            "attribute_type": predicate,
                            "value": str(first.payload.get("value", "")),
                            "normalized_value": normalized_value,
                            "source_claim_ids": sorted(source.claim_id for source in sources),
                            "metadata": dict(first.payload.get("metadata", {})),
                        }
                    )
                facets[predicate] = sorted(
                    entries,
                    key=lambda item: (str(item["normalized_value"]), str(item["value"])),
                )

            source_claim_ids = sorted(claim.claim_id for claim in person_claims)
            payload = {
                "person_id": person.person_id,
                "family_id": family_id,
                "canonical_name": person.canonical_name,
                "category": person.category,
                "birth_date": (facets.get("birth_date") or [None])[0],
                "death_date": (facets.get("death_date") or [None])[0],
                "aliases": facets.get("alias", []),
                "professions": facets.get("profession", []),
                "locations": facets.get("location", []),
                "education": facets.get("education", []),
                "descriptions": facets.get("description", []),
                "events": facets.get("event", []),
                "source_claim_ids": source_claim_ids,
            }
            session.add(
                MaterializedPersonProfileRow(
                    person_id=person.person_id,
                    family_id=family_id,
                    canonical_name=person.canonical_name,
                    profile_payload=payload,
                    source_claim_ids=source_claim_ids,
                )
            )
        return len(people)

    def list_profiles(self, *, family_id: str) -> list[PersonProfileView]:
        with self.database.session_factory() as session:
            rows = list(
                session.scalars(
                    select(MaterializedPersonProfileRow)
                    .where(MaterializedPersonProfileRow.family_id == family_id)
                    .order_by(MaterializedPersonProfileRow.canonical_name)
                )
            )
            return [self._profile_view(row) for row in rows]

    def get_profile(self, *, family_id: str, person_id: str) -> PersonProfileView:
        with self.database.session_factory() as session:
            row = session.get(MaterializedPersonProfileRow, person_id)
            if row is None or row.family_id != family_id:
                raise ProfileNotFoundError("person profile not found in family archive")
            return self._profile_view(row)

    @staticmethod
    def _profile_view(row: MaterializedPersonProfileRow) -> PersonProfileView:
        payload = dict(row.profile_payload)
        payload["updated_at"] = row.updated_at
        return PersonProfileView.model_validate(payload)

    @staticmethod
    def _generic_conflict(
        session: Session,
        *,
        family_id: str,
        conflict_id: str,
    ) -> ArchiveConflictRow:
        conflict = session.get(ArchiveConflictRow, conflict_id)
        if conflict is None or conflict.family_id != family_id:
            raise ConflictNotFoundError("conflict not found in family archive")
        if conflict.conflict_type == "relationship":
            raise ConflictResolutionError("relationship conflict must use relationship resolver")
        return conflict

    @staticmethod
    def _selectable_claim(session: Session, claim_id: str) -> ArchiveClaimRow:
        claim = session.get(ArchiveClaimRow, claim_id)
        if claim is None or not _claim_is_grounded(claim):
            raise ConflictResolutionError("preferred claim is not eligible for profile materialization")
        return claim

    def resolve_generic(
        self,
        *,
        family_id: str,
        conflict_id: str,
        preferred_claim_id: str,
        reviewer_reference: str,
        note: str,
    ) -> ConflictMutationResult:
        with self.database.session_factory.begin() as session:
            conflict = self._generic_conflict(
                session,
                family_id=family_id,
                conflict_id=conflict_id,
            )
            if preferred_claim_id not in conflict.claim_ids:
                raise ConflictResolutionError("preferred claim must belong to the conflict")
            self._selectable_claim(session, preferred_claim_id)
            previous_status = conflict.status
            if (
                conflict.status == "resolved"
                and conflict.preferred_claim_id == preferred_claim_id
                and conflict.resolution_note == note
            ):
                self._rebuild_profiles(session, family_id=family_id)
                return ConflictMutationResult(
                    conflict=ConflictResolutionService._view(session, conflict),
                    graph_edges=_graph_edge_count(session, family_id=family_id),
                )
            conflict.status = "resolved"
            conflict.preferred_claim_id = preferred_claim_id
            conflict.resolution_note = note
            conflict.updated_at = utcnow()
            ConflictResolutionService._append_decision(
                session,
                conflict=conflict,
                action=ConflictAction.RESOLVE,
                previous_status=previous_status,
                resulting_status="resolved",
                reviewer_reference=reviewer_reference,
                note=note,
                preferred_claim_id=preferred_claim_id,
            )
            ConflictResolutionService._apply_conflict_state(session, conflict=conflict)
            self._rebuild_profiles(session, family_id=family_id)
            session.flush()
            return ConflictMutationResult(
                conflict=ConflictResolutionService._view(session, conflict),
                graph_edges=_graph_edge_count(session, family_id=family_id),
            )

    def dismiss_generic(
        self,
        *,
        family_id: str,
        conflict_id: str,
        reviewer_reference: str,
        note: str,
    ) -> ConflictMutationResult:
        with self.database.session_factory.begin() as session:
            conflict = self._generic_conflict(
                session,
                family_id=family_id,
                conflict_id=conflict_id,
            )
            previous_status = conflict.status
            if conflict.status == "dismissed" and conflict.resolution_note == note:
                self._rebuild_profiles(session, family_id=family_id)
                return ConflictMutationResult(
                    conflict=ConflictResolutionService._view(session, conflict),
                    graph_edges=_graph_edge_count(session, family_id=family_id),
                )
            conflict.status = "dismissed"
            conflict.preferred_claim_id = None
            conflict.resolution_note = note
            conflict.updated_at = utcnow()
            ConflictResolutionService._append_decision(
                session,
                conflict=conflict,
                action=ConflictAction.DISMISS,
                previous_status=previous_status,
                resulting_status="dismissed",
                reviewer_reference=reviewer_reference,
                note=note,
            )
            ConflictResolutionService._apply_conflict_state(session, conflict=conflict)
            self._rebuild_profiles(session, family_id=family_id)
            session.flush()
            return ConflictMutationResult(
                conflict=ConflictResolutionService._view(session, conflict),
                graph_edges=_graph_edge_count(session, family_id=family_id),
            )

    def reopen_generic(
        self,
        *,
        family_id: str,
        conflict_id: str,
        reviewer_reference: str,
        note: str,
    ) -> ConflictMutationResult:
        with self.database.session_factory.begin() as session:
            conflict = self._generic_conflict(
                session,
                family_id=family_id,
                conflict_id=conflict_id,
            )
            if conflict.status == "open":
                raise ConflictResolutionError("conflict is already open")
            previous_status = conflict.status
            conflict.status = "open"
            conflict.preferred_claim_id = None
            conflict.resolution_note = note
            conflict.updated_at = utcnow()
            ConflictResolutionService._append_decision(
                session,
                conflict=conflict,
                action=ConflictAction.REOPEN,
                previous_status=previous_status,
                resulting_status="open",
                reviewer_reference=reviewer_reference,
                note=note,
            )
            ConflictResolutionService._apply_conflict_state(session, conflict=conflict)
            self._rebuild_profiles(session, family_id=family_id)
            session.flush()
            return ConflictMutationResult(
                conflict=ConflictResolutionService._view(session, conflict),
                graph_edges=_graph_edge_count(session, family_id=family_id),
            )


class UnifiedConflictReviewService:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.relationships = ConflictResolutionService(database)
        self.generic = GenericProfileService(database)

    def _conflict_type(self, *, family_id: str, conflict_id: str) -> str:
        with self.database.session_factory() as session:
            conflict = session.get(ArchiveConflictRow, conflict_id)
            if conflict is None or conflict.family_id != family_id:
                raise ConflictNotFoundError("conflict not found in family archive")
            return conflict.conflict_type

    def list_conflicts(
        self,
        *,
        family_id: str,
        status: str | None = None,
    ) -> list[ConflictReviewView]:
        return self.relationships.list_conflicts(family_id=family_id, status=status)

    def get_conflict(self, *, family_id: str, conflict_id: str) -> ConflictReviewView:
        return self.relationships.get_conflict(family_id=family_id, conflict_id=conflict_id)

    def resolve(
        self,
        *,
        family_id: str,
        conflict_id: str,
        preferred_claim_id: str,
        reviewer_reference: str,
        note: str,
    ) -> ConflictMutationResult:
        if self._conflict_type(family_id=family_id, conflict_id=conflict_id) == "relationship":
            return self.relationships.resolve(
                family_id=family_id,
                conflict_id=conflict_id,
                preferred_claim_id=preferred_claim_id,
                reviewer_reference=reviewer_reference,
                note=note,
            )
        return self.generic.resolve_generic(
            family_id=family_id,
            conflict_id=conflict_id,
            preferred_claim_id=preferred_claim_id,
            reviewer_reference=reviewer_reference,
            note=note,
        )

    def dismiss(
        self,
        *,
        family_id: str,
        conflict_id: str,
        reviewer_reference: str,
        note: str,
    ) -> ConflictMutationResult:
        if self._conflict_type(family_id=family_id, conflict_id=conflict_id) == "relationship":
            return self.relationships.dismiss(
                family_id=family_id,
                conflict_id=conflict_id,
                reviewer_reference=reviewer_reference,
                note=note,
            )
        return self.generic.dismiss_generic(
            family_id=family_id,
            conflict_id=conflict_id,
            reviewer_reference=reviewer_reference,
            note=note,
        )

    def reopen(
        self,
        *,
        family_id: str,
        conflict_id: str,
        reviewer_reference: str,
        note: str,
    ) -> ConflictMutationResult:
        if self._conflict_type(family_id=family_id, conflict_id=conflict_id) == "relationship":
            return self.relationships.reopen(
                family_id=family_id,
                conflict_id=conflict_id,
                reviewer_reference=reviewer_reference,
                note=note,
            )
        return self.generic.reopen_generic(
            family_id=family_id,
            conflict_id=conflict_id,
            reviewer_reference=reviewer_reference,
            note=note,
        )
