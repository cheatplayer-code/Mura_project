from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Privacy(StrEnum):
    PRIVATE = "private"
    FAMILY = "family"
    PUBLIC = "public"


class VerificationStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class AssertionMode(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    UNCERTAIN = "uncertain"


class ResolutionStatus(StrEnum):
    NEW_PERSON = "new_person"
    RESOLVED = "resolved"
    NEEDS_REVIEW = "needs_review"


class CorrectionKind(StrEnum):
    ASR_NORMALIZATION = "asr_normalization"
    SPEAKER_SELF_CORRECTION = "speaker_self_correction"


class PersonCategory(StrEnum):
    FAMILY_MEMBER = "family_member"
    FRIEND = "friend"
    ROOMMATE = "roommate"
    ACQUAINTANCE = "acquaintance"
    OTHER_NON_FAMILY = "other_non_family"
    UNKNOWN = "unknown"


class RelationshipType(StrEnum):
    PARENT_CHILD = "parent_child"
    SPOUSE = "spouse"
    SIBLING = "sibling"


class RelationshipRole(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    SPOUSE = "spouse"
    OLDER_SIBLING = "older_sibling"
    YOUNGER_SIBLING = "younger_sibling"
    SIBLING = "sibling"


class RawSegment(StrictModel):
    segment_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)
    chunk_id: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> RawSegment:
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")
        return self


class TranscriptEnvelope(StrictModel):
    recording_id: str
    duration_seconds: float = Field(gt=0)
    language_hints: list[str] = Field(default_factory=list)
    full_text: str
    segments: list[RawSegment] = Field(min_length=1)
    asr_model: str
    asr_revision: str
    chunker_version: str
    processing_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_unique_segments(self) -> TranscriptEnvelope:
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment IDs must be unique")
        return self


class ReadableSegment(StrictModel):
    segment_id: str
    text: str = Field(min_length=1)


class DetectedCorrection(StrictModel):
    kind: CorrectionKind
    subject: str | None = None
    original_value: str = Field(min_length=1)
    corrected_value: str = Field(min_length=1)
    source_segment_ids: list[str] = Field(min_length=1)
    explanation: str
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_changed_value(self) -> DetectedCorrection:
        if self.original_value.casefold() == self.corrected_value.casefold():
            raise ValueError("a correction must change the value")
        return self


class UncertainFragment(StrictModel):
    source_segment_ids: list[str] = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    possible_interpretation: None = None
    reason: str


class CleanerResult(StrictModel):
    readable_segments: list[ReadableSegment] = Field(min_length=1)
    detected_corrections: list[DetectedCorrection] = Field(default_factory=list)
    uncertain_fragments: list[UncertainFragment] = Field(default_factory=list)
    full_readable_text: str = Field(min_length=1)


class PersonMention(StrictModel):
    mention_id: str
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    category: PersonCategory = PersonCategory.UNKNOWN
    relation_to_speaker: str | None = None
    source_segment_ids: list[str] = Field(min_length=1)
    assertion_mode: AssertionMode = AssertionMode.EXPLICIT
    verification_status: VerificationStatus = VerificationStatus.UNREVIEWED
    confidence: float = Field(ge=0, le=1)


class RelationshipClaim(StrictModel):
    relationship_id: str
    relationship_type: RelationshipType
    subject_mention_id: str
    subject_role: RelationshipRole
    object_mention_id: str
    object_role: RelationshipRole
    source_segment_ids: list[str] = Field(min_length=1)
    assertion_mode: AssertionMode = AssertionMode.EXPLICIT
    verification_status: VerificationStatus = VerificationStatus.UNREVIEWED
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_relationship_roles(self) -> RelationshipClaim:
        if self.subject_mention_id == self.object_mention_id:
            raise ValueError("a relationship must connect two different mentions")

        role_pair = (self.subject_role, self.object_role)
        allowed_pairs = {
            RelationshipType.PARENT_CHILD: {
                (RelationshipRole.PARENT, RelationshipRole.CHILD),
            },
            RelationshipType.SPOUSE: {
                (RelationshipRole.SPOUSE, RelationshipRole.SPOUSE),
            },
            RelationshipType.SIBLING: {
                (RelationshipRole.OLDER_SIBLING, RelationshipRole.YOUNGER_SIBLING),
                (RelationshipRole.SIBLING, RelationshipRole.SIBLING),
            },
        }
        if role_pair not in allowed_pairs[self.relationship_type]:
            raise ValueError(f"invalid role pair {role_pair!r} for {self.relationship_type.value}")
        return self


class EventDate(StrictModel):
    value: str | None = None
    precision: str = "unknown"
    original_expression: str | None = None


class FamilyEvent(StrictModel):
    event_id: str
    event_type: str
    title: str
    participant_mention_ids: list[str] = Field(default_factory=list)
    date: EventDate | None = None
    location: str | None = None
    description: str
    source_segment_ids: list[str] = Field(min_length=1)
    assertion_mode: AssertionMode = AssertionMode.EXPLICIT
    verification_status: VerificationStatus = VerificationStatus.UNREVIEWED
    confidence: float = Field(ge=0, le=1)


class PersonDescription(StrictModel):
    description_id: str
    person_mention_id: str
    description: str
    perspective: str
    source_segment_ids: list[str] = Field(min_length=1)
    assertion_mode: AssertionMode = AssertionMode.EXPLICIT
    verification_status: VerificationStatus = VerificationStatus.UNREVIEWED
    confidence: float = Field(ge=0, le=1)


class Story(StrictModel):
    story_id: str
    title: str
    summary: str
    person_mention_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    source_segment_ids: list[str] = Field(min_length=1)
    privacy: Privacy = Privacy.PRIVATE
    sensitivity: str = "normal"

    @model_validator(mode="after")
    def force_private_by_default(self) -> Story:
        self.privacy = Privacy.PRIVATE
        return self


class UnresolvedQuestion(StrictModel):
    question_id: str
    question: str
    reason: str
    related_mention_ids: list[str] = Field(default_factory=list)
    source_segment_ids: list[str] = Field(min_length=1)


class ExtractionResult(StrictModel):
    recording_id: str
    speaker_id: str
    speaker_name: str
    languages: list[str] = Field(default_factory=list)
    people_mentions: list[PersonMention] = Field(default_factory=list)
    relationship_claims: list[RelationshipClaim] = Field(default_factory=list)
    events: list[FamilyEvent] = Field(default_factory=list)
    descriptions: list[PersonDescription] = Field(default_factory=list)
    stories: list[Story] = Field(default_factory=list)
    unresolved_questions: list[UnresolvedQuestion] = Field(default_factory=list)


class KnownPerson(StrictModel):
    person_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    category: PersonCategory = PersonCategory.UNKNOWN
    relation_to_speaker: str | None = None


class MentionResolution(StrictModel):
    mention_id: str
    status: ResolutionStatus
    person_id: str | None = None
    candidate_person_ids: list[str] = Field(default_factory=list)
    reason: str


class PipelineRequest(StrictModel):
    transcript: TranscriptEnvelope
    speaker_id: str
    speaker_name: str
    known_people: list[KnownPerson] = Field(default_factory=list)


class PipelineResult(StrictModel):
    transcript: TranscriptEnvelope
    cleaned_transcript: CleanerResult
    extraction: ExtractionResult
    resolutions: list[MentionResolution] = Field(default_factory=list)
    processing: dict[str, Any] = Field(default_factory=dict)
