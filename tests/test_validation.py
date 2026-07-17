import pytest

from mura.domain.models import (
    CleanerResult,
    ExtractionResult,
    PersonMention,
    RawSegment,
    ReadableSegment,
    RelationshipClaim,
    TranscriptEnvelope,
)
from mura.validation import (
    ContractValidationError,
    validate_cleaner_result,
    validate_extraction_result,
)


def transcript() -> TranscriptEnvelope:
    return TranscriptEnvelope(
        recording_id="rec_1",
        duration_seconds=12,
        full_text="Әкемнің аты Сапар.",
        segments=[RawSegment(segment_id="seg_001", start=0, end=12, text="әкемнің аты сапар")],
        asr_model="gigaam",
        asr_revision="large_ctc",
        chunker_version="v1",
    )


def test_cleaner_requires_exact_segment_coverage() -> None:
    result = CleanerResult(
        readable_segments=[ReadableSegment(segment_id="seg_999", text="Wrong")],
        full_readable_text="Wrong",
    )
    with pytest.raises(ContractValidationError):
        validate_cleaner_result(transcript(), result)


def test_extraction_rejects_broken_relationship_reference() -> None:
    result = ExtractionResult(
        recording_id="rec_1",
        speaker_id="speaker_1",
        speaker_name="Күләш",
        people_mentions=[
            PersonMention(
                mention_id="mention_001",
                name="Сапар",
                source_segment_ids=["seg_001"],
                confidence=1,
            )
        ],
        relationship_claims=[
            RelationshipClaim(
                relationship_id="relationship_001",
                subject_mention_id="mention_001",
                relation="parent_of",
                object_mention_id="mention_missing",
                source_segment_ids=["seg_001"],
                confidence=1,
            )
        ],
    )
    with pytest.raises(ContractValidationError):
        validate_extraction_result(transcript(), result)
