from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from mura.domain.models import (
    PersonCategory,
    RelationshipRole,
    RelationshipType,
    StrictModel,
    TranscriptEnvelope,
)


class LanguageBucket(StrEnum):
    KAZAKH = "kk"
    RUSSIAN = "ru"
    ENGLISH = "en"
    MIXED = "mixed"


class DatasetSplit(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


class GoldPerson(StrictModel):
    person_key: str = Field(min_length=1)
    accepted_surfaces: list[str] = Field(min_length=1)
    category: PersonCategory = PersonCategory.UNKNOWN


class GoldRelationship(StrictModel):
    relationship_key: str = Field(min_length=1)
    relationship_type: RelationshipType
    subject_person_key: str = Field(min_length=1)
    subject_role: RelationshipRole
    object_person_key: str = Field(min_length=1)
    object_role: RelationshipRole

    @model_validator(mode="after")
    def validate_roles(self) -> GoldRelationship:
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
        role_pair = (self.subject_role, self.object_role)
        if role_pair not in allowed_pairs[self.relationship_type]:
            raise ValueError(
                f"invalid role pair {role_pair!r} for {self.relationship_type.value}"
            )
        if self.subject_person_key == self.object_person_key:
            raise ValueError("gold relationship must connect two different people")
        return self


class BenchmarkGold(StrictModel):
    people: list[GoldPerson] = Field(default_factory=list)
    relationships: list[GoldRelationship] = Field(default_factory=list)
    quarantined_relationship_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> BenchmarkGold:
        person_keys = [person.person_key for person in self.people]
        relationship_keys = [item.relationship_key for item in self.relationships]
        if len(person_keys) != len(set(person_keys)):
            raise ValueError("gold person keys must be unique")
        if len(relationship_keys) != len(set(relationship_keys)):
            raise ValueError("gold relationship keys must be unique")
        if len(self.quarantined_relationship_ids) != len(
            set(self.quarantined_relationship_ids)
        ):
            raise ValueError("gold quarantined relationship IDs must be unique")
        return self


class BenchmarkCase(StrictModel):
    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: LanguageBucket
    construction_tags: list[str] = Field(default_factory=list)
    speaker_id: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    transcript: TranscriptEnvelope
    raw_extraction: dict[str, Any]
    gold: BenchmarkGold

    @model_validator(mode="after")
    def validate_recording_identity(self) -> BenchmarkCase:
        raw_recording_id = self.raw_extraction.get("recording_id")
        if raw_recording_id != self.transcript.recording_id:
            raise ValueError(
                "raw_extraction.recording_id must match transcript.recording_id"
            )
        return self


class BenchmarkDataset(StrictModel):
    dataset_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    description: str = ""
    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_cases(self) -> BenchmarkDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique within a dataset")
        return self


class ManifestDataset(StrictModel):
    dataset_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    split: DatasetSplit = DatasetSplit.VALIDATION
    enabled: bool = True


class BenchmarkManifest(StrictModel):
    schema_version: str = Field(min_length=1)
    datasets: list[ManifestDataset] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_datasets(self) -> BenchmarkManifest:
        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("manifest dataset IDs must be unique")
        return self


class PrecisionRecallF1(StrictModel):
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class RatioMetric(StrictModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float = Field(ge=0, le=1)


class CaseEvaluation(StrictModel):
    case_id: str
    dataset_id: str
    split: DatasetSplit
    language: LanguageBucket
    construction_tags: list[str]
    person_mentions: PrecisionRecallF1
    relationships: PrecisionRecallF1
    quarantined_relationships: PrecisionRecallF1
    relationship_direction_accuracy: RatioMetric
    provenance_completeness: RatioMetric
    unknown_segment_references: int = Field(ge=0)
    self_relationships: int = Field(ge=0)
    accepted_relationship_ids: list[str] = Field(default_factory=list)
    quarantined_relationship_ids: list[str] = Field(default_factory=list)
    extraction_issue_count: int = Field(ge=0)
    evidence_closure_relationships: int = Field(ge=0)


class BenchmarkSummary(StrictModel):
    case_count: int = Field(ge=0)
    person_mentions: PrecisionRecallF1
    relationships: PrecisionRecallF1
    quarantined_relationships: PrecisionRecallF1
    relationship_direction_accuracy: RatioMetric
    provenance_completeness: RatioMetric
    unknown_segment_references: int = Field(ge=0)
    self_relationships: int = Field(ge=0)


class BenchmarkReport(StrictModel):
    report_schema_version: str
    manifest_path: str
    pipeline_versions: dict[str, str]
    cases: list[CaseEvaluation]
    summary: BenchmarkSummary
