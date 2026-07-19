from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from mura.domain.models import (
    KnownPerson,
    MentionResolution,
    PersonCategory,
    ResolutionStatus,
    StrictModel,
    VerificationStatus,
)


class ResolutionSignalKind(StrEnum):
    FAMILY_SCOPE = "family_scope"
    CANONICAL_NAME = "canonical_name"
    ESTABLISHED_ALIAS = "established_alias"
    STRUCTURED_ALIAS = "structured_alias"
    RELATION_TO_SPEAKER = "relation_to_speaker"
    GENERATION = "generation"
    CATEGORY = "category"
    GRAPH_NEIGHBOUR = "graph_neighbour"
    RELATION_CONFLICT = "relation_conflict"
    GENERATION_CONFLICT = "generation_conflict"
    CATEGORY_CONFLICT = "category_conflict"


class ResolutionSignal(StrictModel):
    rule_id: str = Field(min_length=1)
    kind: ResolutionSignalKind
    detail: str = Field(min_length=1)
    person_id: str | None = None
    related_mention_id: str | None = None
    related_person_id: str | None = None


class KnownPersonProfile(StrictModel):
    """Archive-scoped identity context loaded independently from extraction candidates."""

    family_id: str = Field(min_length=1)
    person: KnownPerson
    generation: int | None = None
    parent_person_ids: list[str] = Field(default_factory=list)
    child_person_ids: list[str] = Field(default_factory=list)
    spouse_person_ids: list[str] = Field(default_factory=list)
    sibling_person_ids: list[str] = Field(default_factory=list)
    source_recording_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profile_references(self) -> KnownPersonProfile:
        collections = (
            self.parent_person_ids,
            self.child_person_ids,
            self.spouse_person_ids,
            self.sibling_person_ids,
            self.source_recording_ids,
        )
        if any(len(values) != len(set(values)) for values in collections):
            raise ValueError("known-person profile references must be unique")
        if self.person.person_id in {
            *self.parent_person_ids,
            *self.child_person_ids,
            *self.spouse_person_ids,
            *self.sibling_person_ids,
        }:
            raise ValueError("known-person profile cannot relate a person to itself")
        return self


class EntityResolutionContext(StrictModel):
    schema_version: str = "entity-resolution-context-v1"
    family_id: str = Field(min_length=1)
    profiles: list[KnownPersonProfile] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> EntityResolutionContext:
        person_ids = [profile.person.person_id for profile in self.profiles]
        if len(person_ids) != len(set(person_ids)):
            raise ValueError("entity-resolution profile person IDs must be unique")
        foreign_profiles = [
            profile.person.person_id
            for profile in self.profiles
            if profile.family_id != self.family_id
        ]
        if foreign_profiles:
            raise ValueError("entity-resolution context cannot contain another family archive")
        return self


class ResolutionTrace(StrictModel):
    mention_id: str = Field(min_length=1)
    status: ResolutionStatus
    selected_person_id: str | None = None
    candidate_person_ids: list[str] = Field(default_factory=list)
    supporting_signals: list[ResolutionSignal] = Field(default_factory=list)
    conflicting_signals: list[ResolutionSignal] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    verification_status: VerificationStatus = VerificationStatus.UNREVIEWED

    @model_validator(mode="after")
    def validate_decision_state(self) -> ResolutionTrace:
        if len(self.candidate_person_ids) != len(set(self.candidate_person_ids)):
            raise ValueError("resolution candidate IDs must be unique")
        if len(self.rule_ids) != len(set(self.rule_ids)):
            raise ValueError("resolution rule IDs must be unique")
        if self.status is ResolutionStatus.RESOLVED:
            if self.selected_person_id is None:
                raise ValueError("resolved trace requires a selected person")
            if self.selected_person_id not in self.candidate_person_ids:
                raise ValueError("resolved person must be included in candidate IDs")
        elif self.selected_person_id is not None:
            raise ValueError("only resolved traces may select a person")
        if self.status is ResolutionStatus.NEW_PERSON and self.candidate_person_ids:
            raise ValueError("new-person trace cannot contain archive candidates")
        return self


class EntityResolutionMetrics(StrictModel):
    mentions: int = Field(ge=0)
    resolved: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    new_person: int = Field(ge=0)
    family_scope_violations: int = Field(default=0, ge=0)


class EntityResolutionRun(StrictModel):
    schema_version: str = "entity-resolution-run-v1"
    family_id: str = Field(min_length=1)
    resolutions: list[MentionResolution] = Field(default_factory=list)
    traces: list[ResolutionTrace] = Field(default_factory=list)
    metrics: EntityResolutionMetrics

    @model_validator(mode="after")
    def validate_alignment(self) -> EntityResolutionRun:
        resolution_ids = [item.mention_id for item in self.resolutions]
        trace_ids = [item.mention_id for item in self.traces]
        if resolution_ids != trace_ids:
            raise ValueError("resolution and trace mention ordering must match")
        if len(resolution_ids) != len(set(resolution_ids)):
            raise ValueError("entity-resolution mention IDs must be unique")
        if self.metrics.mentions != len(self.resolutions):
            raise ValueError("entity-resolution metrics must cover every mention")
        return self


def legacy_resolution_context(known_people: list[KnownPerson]) -> EntityResolutionContext:
    """Adapt the pre-persistence API without pretending account scope is a real family ID."""

    return EntityResolutionContext(
        family_id="legacy-family-scope",
        profiles=[
            KnownPersonProfile(family_id="legacy-family-scope", person=person)
            for person in known_people
        ],
    )


def categories_conflict(mention: PersonCategory, known: PersonCategory) -> bool:
    if mention is PersonCategory.UNKNOWN or known is PersonCategory.UNKNOWN:
        return False
    family = {PersonCategory.FAMILY_MEMBER}
    return (mention in family) != (known in family)
