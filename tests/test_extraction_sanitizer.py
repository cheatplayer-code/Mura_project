from __future__ import annotations

from mura.domain.models import RawSegment, TranscriptEnvelope
from mura.extraction_sanitizer import sanitize_extraction_output


def _transcript() -> TranscriptEnvelope:
    return TranscriptEnvelope(
        recording_id="rec_1",
        duration_seconds=30,
        full_text=(
            "Әкемнің аты Сапар. Сапардың інісі Нұрғали еді. Диас баскетбол ойнағанды жақсы көреді."
        ),
        segments=[
            RawSegment(
                segment_id="seg_001",
                start=0,
                end=10,
                text="әкемнің аты сапар",
            ),
            RawSegment(
                segment_id="seg_002",
                start=10,
                end=20,
                text="сапардың інісі нұрғали еді",
            ),
            RawSegment(
                segment_id="seg_003",
                start=20,
                end=30,
                text="диас баскетбол ойнағанды жақсы көреді",
            ),
        ],
        asr_model="gigaam",
        asr_revision="large_ctc",
        chunker_version="v1",
    )


def _base_raw() -> dict[str, object]:
    return {
        "recording_id": "rec_1",
        "speaker_id": "speaker_1",
        "speaker_name": "Күләш",
        "languages": ["kk"],
        "people_mentions": [
            {
                "mention_id": "mention_sapar",
                "name": "Сапар",
                "category": "family_member",
                "source_segment_ids": ["seg_001", "seg_002"],
                "confidence": 1.0,
            },
            {
                "mention_id": "mention_nurgali",
                "name": "Нұрғали",
                "category": "family_member",
                "source_segment_ids": ["seg_002"],
                "confidence": 1.0,
            },
            {
                "mention_id": "mention_dias",
                "name": "Диас",
                "category": "family_member",
                "source_segment_ids": ["seg_003"],
                "confidence": 1.0,
            },
        ],
        "relationship_claims": [
            {
                "relationship_id": "relationship_valid",
                "relationship_type": "sibling",
                "subject_mention_id": "mention_sapar",
                "subject_role": "older_sibling",
                "object_mention_id": "mention_nurgali",
                "object_role": "younger_sibling",
                "source_segment_ids": ["seg_002"],
                "confidence": 1.0,
            },
            {
                "relationship_id": "relationship_invalid",
                "relationship_type": "parent_child",
                "subject_mention_id": "mention_dias",
                "subject_role": "parent",
                "object_mention_id": "mention_dias",
                "object_role": "child",
                "source_segment_ids": ["seg_003"],
                "confidence": 1.0,
            },
        ],
        "events": [],
        "descriptions": [
            {
                "description_id": "description_valid",
                "person_mention_id": "mention_dias",
                "description": "баскетбол ойнағанды жақсы көреді",
                "perspective": "Күләш",
                "source_segment_ids": ["seg_003"],
                "confidence": 1.0,
            },
            {
                "description_id": "description_wrong_person",
                "person_mention_id": "mention_nurgali",
                "description": "баскетбол ойнағанды жақсы көреді",
                "perspective": "Күләш",
                "source_segment_ids": ["seg_003"],
                "confidence": 1.0,
            },
        ],
        "stories": [],
        "unresolved_questions": [],
    }


def test_sanitizer_keeps_valid_objects_and_quarantines_bad_ones() -> None:
    result, issues, closure_count = sanitize_extraction_output(
        raw=_base_raw(),
        transcript=_transcript(),
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )

    assert closure_count == 0
    assert [item.relationship_id for item in result.relationship_claims] == ["relationship_valid"]
    assert result.relationship_claims[0].source_segment_ids == ["seg_002"]
    assert [item.description_id for item in result.descriptions] == ["description_valid"]

    issue_by_id = {issue["object_id"]: issue for issue in issues}
    invalid_context = issue_by_id["relationship_invalid"]["context"]
    assert invalid_context["candidate"]["object_mention_id"] == "mention_dias"
    assert "description_wrong_person" in issue_by_id


def test_sanitizer_uses_authoritative_request_metadata() -> None:
    raw = _base_raw()
    raw["recording_id"] = "invented_recording"
    raw["speaker_name"] = "Invented Speaker"

    result, issues, _ = sanitize_extraction_output(
        raw=raw,
        transcript=_transcript(),
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )

    assert result.recording_id == "rec_1"
    assert result.speaker_name == "Күләш"
    metadata_issue_ids = {
        issue["object_id"] for issue in issues if issue["object_type"] == "metadata"
    }
    assert metadata_issue_ids == {"recording_id", "speaker_name"}


def _speaker_relationship_payload(
    *,
    speaker_sources: list[str],
) -> tuple[TranscriptEnvelope, dict[str, object]]:
    transcript = TranscriptEnvelope(
        recording_id="rec_speaker",
        duration_seconds=20,
        full_text="Менің атым Күләш. Біздің ұлымыз Ерлан.",
        segments=[
            RawSegment(segment_id="seg_001", start=0, end=10, text="менің атым күләш"),
            RawSegment(segment_id="seg_002", start=10, end=20, text="біздің ұлымыз ерлан"),
        ],
        asr_model="gigaam",
        asr_revision="large_ctc",
        chunker_version="v1",
    )
    raw: dict[str, object] = {
        "recording_id": "rec_speaker",
        "speaker_id": "speaker_1",
        "speaker_name": "Күләш",
        "languages": ["kk"],
        "people_mentions": [
            {
                "mention_id": "mention_kulash",
                "name": "Күләш",
                "category": "family_member",
                "source_segment_ids": speaker_sources,
                "confidence": 1.0,
            },
            {
                "mention_id": "mention_erlan",
                "name": "Ерлан",
                "category": "family_member",
                "source_segment_ids": ["seg_002"],
                "confidence": 1.0,
            },
        ],
        "relationship_claims": [
            {
                "relationship_id": "relationship_speaker_child",
                "relationship_type": "parent_child",
                "subject_mention_id": "mention_kulash",
                "subject_role": "parent",
                "object_mention_id": "mention_erlan",
                "object_role": "child",
                "source_segment_ids": ["seg_002"],
                "confidence": 1.0,
            }
        ],
        "events": [],
        "descriptions": [],
        "stories": [],
        "unresolved_questions": [],
    }
    return transcript, raw


def test_first_person_reference_supports_the_speaker_endpoint() -> None:
    transcript, raw = _speaker_relationship_payload(speaker_sources=["seg_001", "seg_002"])

    result, issues, closure_count = sanitize_extraction_output(
        raw=raw,
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )

    assert issues == []
    assert closure_count == 0
    assert [item.relationship_id for item in result.relationship_claims] == [
        "relationship_speaker_child"
    ]


def test_speaker_identity_closure_adds_only_one_identity_segment() -> None:
    transcript, raw = _speaker_relationship_payload(speaker_sources=["seg_001"])

    result, issues, closure_count = sanitize_extraction_output(
        raw=raw,
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )

    assert issues == []
    assert closure_count == 1
    assert result.relationship_claims[0].source_segment_ids == ["seg_001", "seg_002"]


def test_ambiguous_third_person_endpoint_is_quarantined_with_debug_context() -> None:
    transcript = TranscriptEnvelope(
        recording_id="rec_ambiguous",
        duration_seconds=20,
        full_text="Ерлан инженер. Оның ұлы Нұрлан.",
        segments=[
            RawSegment(segment_id="seg_001", start=0, end=10, text="ерлан инженер"),
            RawSegment(segment_id="seg_002", start=10, end=20, text="оның ұлы нұрлан"),
        ],
        asr_model="gigaam",
        asr_revision="large_ctc",
        chunker_version="v1",
    )
    raw: dict[str, object] = {
        "recording_id": "rec_ambiguous",
        "speaker_id": "speaker_1",
        "speaker_name": "Күләш",
        "languages": ["kk"],
        "people_mentions": [
            {
                "mention_id": "mention_erlan",
                "name": "Ерлан",
                "category": "family_member",
                "source_segment_ids": ["seg_001"],
                "confidence": 1.0,
            },
            {
                "mention_id": "mention_nurlan",
                "name": "Нұрлан",
                "category": "family_member",
                "source_segment_ids": ["seg_002"],
                "confidence": 1.0,
            },
        ],
        "relationship_claims": [
            {
                "relationship_id": "relationship_ambiguous",
                "relationship_type": "parent_child",
                "subject_mention_id": "mention_erlan",
                "subject_role": "parent",
                "object_mention_id": "mention_nurlan",
                "object_role": "child",
                "source_segment_ids": ["seg_002"],
                "confidence": 1.0,
            }
        ],
        "events": [],
        "descriptions": [],
        "stories": [],
        "unresolved_questions": [],
    }

    result, issues, closure_count = sanitize_extraction_output(
        raw=raw,
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )

    assert result.relationship_claims == []
    assert closure_count == 0
    relationship_issue = next(
        issue for issue in issues if issue["object_id"] == "relationship_ambiguous"
    )
    issue_context = relationship_issue["context"]
    analysis = issue_context["evidence_analysis"]
    assert issue_context["original_candidate"]["subject_mention_id"] == "mention_erlan"
    assert analysis["subject_name"] == "Ерлан"
    assert analysis["object_name"] == "Нұрлан"
    assert analysis["explicit_people"] == [{"mention_id": "mention_nurlan", "name": "Нұрлан"}]
    assert analysis["first_person_reference"] is False
    assert analysis["unsupported_endpoint_ids"] == ["mention_erlan"]
    assert analysis["source_text"] == "оның ұлы нұрлан"
