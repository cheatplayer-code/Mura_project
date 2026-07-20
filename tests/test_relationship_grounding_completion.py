from __future__ import annotations

from pathlib import Path
from typing import Any

from mura.deepseek.grounding_metrics import relationship_grounding_counters
from mura.domain.models import (
    CleanerResult,
    ConflictDetectionMethod,
    ConflictSet,
    ConflictStatus,
    ConflictType,
    EvidenceClass,
    EvidenceSpan,
    ExtractionResult,
    MentionResolution,
    PersonMention,
    PipelineResult,
    RawSegment,
    ReadableSegment,
    RelationshipClaim,
    RelationshipRole,
    RelationshipType,
    ResolutionStatus,
    TranscriptEnvelope,
)
from mura.evidence import complete_relationship_evidence
from mura.extraction_sanitizer import sanitize_extraction_output
from mura.relationship_grounding import select_relationship_grounding_contexts
from mura.storage.archive import ArchiveRepository
from mura.storage.database import Database, RecordingRepository, RecordingRow


def _transcript(*texts: str) -> TranscriptEnvelope:
    segments = [
        RawSegment(
            segment_id=f"seg_{index:03d}",
            start=float((index - 1) * 10),
            end=float(index * 10),
            text=text,
        )
        for index, text in enumerate(texts, start=1)
    ]
    return TranscriptEnvelope(
        recording_id="rec_completion",
        duration_seconds=float(len(segments) * 10),
        language_hints=["kk", "ru"],
        full_text=" ".join(texts),
        segments=segments,
        asr_model="fixture",
        asr_revision="v1",
        chunker_version="v1",
    )


def _raw(
    *,
    people: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    conflicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "recording_id": "rec_completion",
        "speaker_id": "speaker_1",
        "speaker_name": "Күләш",
        "languages": ["kk", "ru"],
        "people_mentions": people,
        "relationship_claims": relationships,
        "conflict_sets": conflicts or [],
        "events": [],
        "descriptions": [],
        "stories": [],
        "unresolved_questions": [],
    }


def _person(mention_id: str, name: str, *segments: str) -> dict[str, Any]:
    return {
        "mention_id": mention_id,
        "name": name,
        "category": "family_member",
        "source_segment_ids": list(segments),
        "confidence": 1.0,
    }


def _parent(
    relationship_id: str,
    parent_id: str,
    child_id: str,
    *segments: str,
) -> dict[str, Any]:
    return {
        "relationship_id": relationship_id,
        "relationship_type": "parent_child",
        "subject_mention_id": parent_id,
        "subject_role": "parent",
        "object_mention_id": child_id,
        "object_role": "child",
        "source_segment_ids": list(segments),
        "confidence": 1.0,
    }


def test_segment_boundary_preserves_bounded_coreference_and_provenance() -> None:
    transcript = _transcript(
        "әкемнің аты сапар",
        "оның інісі нұрғали еді",
    )
    extraction = ExtractionResult(
        recording_id=transcript.recording_id,
        speaker_id="speaker_1",
        speaker_name="Күләш",
        people_mentions=[
            PersonMention(
                mention_id="mention_sapar",
                name="Сапар",
                category="family_member",
                source_segment_ids=["seg_001"],
                confidence=1,
            ),
            PersonMention(
                mention_id="mention_nurgali",
                name="Нұрғали",
                category="family_member",
                source_segment_ids=["seg_002"],
                confidence=1,
            ),
        ],
        relationship_claims=[
            RelationshipClaim(
                relationship_id="relationship_siblings",
                relationship_type=RelationshipType.SIBLING,
                subject_mention_id="mention_sapar",
                subject_role=RelationshipRole.OLDER_SIBLING,
                object_mention_id="mention_nurgali",
                object_role=RelationshipRole.YOUNGER_SIBLING,
                source_segment_ids=["seg_002"],
                confidence=1,
            )
        ],
    )

    completed, closure_count = complete_relationship_evidence(extraction, transcript)
    relationship = completed.relationship_claims[0]
    contexts = select_relationship_grounding_contexts(
        relationship=relationship,
        transcript=transcript,
        people=completed.people_mentions,
        speaker_name=completed.speaker_name,
        resolved_antecedent_ids={"mention_sapar"},
    )

    assert closure_count == 1
    assert relationship.source_segment_ids == ["seg_001", "seg_002"]
    assert relationship.coreference_link_ids
    assert completed.coreference_links[0].antecedent_mention_ids == ["mention_sapar"]
    assert any("сапар" in item.text.casefold() for item in contexts)
    assert any("нұрғали" in item.text.casefold() for item in contexts)
    assert transcript.segments[0].text == "әкемнің аты сапар"
    assert transcript.segments[1].text == "оның інісі нұрғали еді"
    assert all(
        evidence.segment_id in {"seg_001", "seg_002"}
        for evidence in completed.evidence_spans
    )

    result, issues, _ = sanitize_extraction_output(
        raw=completed.model_dump(mode="json"),
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )
    assert issues == []
    assert [item.relationship_id for item in result.relationship_claims] == [
        "relationship_siblings"
    ]


def test_multiple_preceding_people_keep_third_person_antecedent_ambiguous() -> None:
    transcript = _transcript(
        "Сапар Болатпен сөйлесті",
        "оның ұлы Нұрлан",
    )
    raw = _raw(
        people=[
            _person("mention_sapar", "Сапар", "seg_001"),
            _person("mention_bolat", "Болат", "seg_001"),
            _person("mention_nurlan", "Нұрлан", "seg_002"),
        ],
        relationships=[
            _parent(
                "relationship_parent",
                "mention_sapar",
                "mention_nurlan",
                "seg_002",
            )
        ],
    )

    result, issues, _ = sanitize_extraction_output(
        raw=raw,
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )

    assert result.relationship_claims == []
    assert len(result.coreference_links) == 1
    assert result.coreference_links[0].status.value == "ambiguous"
    assert set(result.coreference_links[0].candidate_mention_ids) == {
        "mention_sapar",
        "mention_bolat",
    }
    assert any(item["object_id"] == "relationship_parent" for item in issues)


def test_third_sentence_and_oversized_context_are_not_used() -> None:
    people = [
        PersonMention(
            mention_id="mention_erlan",
            name="Ерлан",
            category="family_member",
            source_segment_ids=["seg_001"],
            confidence=1,
        ),
        PersonMention(
            mention_id="mention_nurlan",
            name="Нурлан",
            category="family_member",
            source_segment_ids=["seg_001"],
            confidence=1,
        ),
    ]
    relationship = RelationshipClaim(
        relationship_id="relationship_parent",
        relationship_type=RelationshipType.PARENT_CHILD,
        subject_mention_id="mention_erlan",
        subject_role=RelationshipRole.PARENT,
        object_mention_id="mention_nurlan",
        object_role=RelationshipRole.CHILD,
        source_segment_ids=["seg_001"],
        confidence=1,
    )
    distant = _transcript(
        "Ерлан инженер. Семья переехала. Его сын Нурлан учится."
    )
    oversized = _transcript(
        "Ерлан " + ("очень " * 90) + "его сын Нурлан"
    )

    assert select_relationship_grounding_contexts(
        relationship=relationship,
        transcript=distant,
        people=people,
        speaker_name="Күләш",
        resolved_antecedent_ids={"mention_erlan"},
    ) == []
    assert select_relationship_grounding_contexts(
        relationship=relationship,
        transcript=oversized,
        people=people,
        speaker_name="Күләш",
        resolved_antecedent_ids={"mention_erlan"},
    ) == []


def _open_conflict() -> list[dict[str, Any]]:
    return [
        {
            "conflict_id": "conflict_relationship",
            "conflict_type": "relationship",
            "claim_refs": [
                {"object_type": "relationship", "object_id": "relationship_spouse"},
                {"object_type": "relationship", "object_id": "relationship_parent"},
            ],
            "status": "open",
            "detected_by": "model",
            "rationale": "review required",
        }
    ]


def test_open_conflict_preserves_review_candidates_but_not_grounded_acceptance() -> None:
    transcript = _transcript("Ерлан Динара")
    raw = _raw(
        people=[
            _person("mention_erlan", "Ерлан", "seg_001"),
            _person("mention_dinara", "Динара", "seg_001"),
        ],
        relationships=[
            {
                "relationship_id": "relationship_spouse",
                "relationship_type": "spouse",
                "subject_mention_id": "mention_erlan",
                "subject_role": "spouse",
                "object_mention_id": "mention_dinara",
                "object_role": "spouse",
                "source_segment_ids": ["seg_001"],
                "confidence": 0.5,
            },
            _parent(
                "relationship_parent",
                "mention_erlan",
                "mention_dinara",
                "seg_001",
            ),
        ],
        conflicts=_open_conflict(),
    )

    result, issues, _ = sanitize_extraction_output(
        raw=raw,
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )
    counters = relationship_grounding_counters(
        result=result,
        transcript=transcript,
        extraction_issues=issues,
    )

    assert issues == []
    assert len(result.relationship_claims) == 2
    assert all(item.conflict_ids == ["conflict_relationship"] for item in result.relationship_claims)
    assert counters["conflict_linked_preserved"] == 2
    assert sum(
        counters[key]
        for key in counters
        if key.endswith("_accepted")
    ) == 0
    assert all(isinstance(value, int) for value in counters.values())


def test_closed_or_contradicted_conflict_does_not_bypass_grounding() -> None:
    transcript = _transcript("Амина — дочь Данияра")
    people = [
        _person("mention_daniyar", "Данияр", "seg_001"),
        _person("mention_amina", "Амина", "seg_001"),
    ]
    wrong = _parent(
        "relationship_wrong",
        "mention_amina",
        "mention_daniyar",
        "seg_001",
    )
    correct = _parent(
        "relationship_correct",
        "mention_daniyar",
        "mention_amina",
        "seg_001",
    )
    conflict = {
        "conflict_id": "conflict_direction",
        "conflict_type": "relationship",
        "claim_refs": [
            {"object_type": "relationship", "object_id": "relationship_wrong"},
            {"object_type": "relationship", "object_id": "relationship_correct"},
        ],
        "status": "open",
        "detected_by": "model",
        "rationale": "direction conflict",
    }

    result, issues, _ = sanitize_extraction_output(
        raw=_raw(
            people=people,
            relationships=[wrong, correct],
            conflicts=[conflict],
        ),
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )
    assert [item.relationship_id for item in result.relationship_claims] == [
        "relationship_correct"
    ]
    assert any(item["object_id"] == "relationship_wrong" for item in issues)

    closed = {**conflict, "status": "resolved", "preferred_claim": conflict["claim_refs"][1], "resolution_note": "reviewed"}
    result, issues, _ = sanitize_extraction_output(
        raw=_raw(
            people=people,
            relationships=[wrong, correct],
            conflicts=[closed],
        ),
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )
    assert [item.relationship_id for item in result.relationship_claims] == [
        "relationship_correct"
    ]
    assert any(item["object_id"] == "relationship_wrong" for item in issues)


def test_open_conflict_does_not_bypass_unknown_endpoint_or_segment() -> None:
    transcript = _transcript("Ерлан Динара")
    people = [
        _person("mention_erlan", "Ерлан", "seg_001"),
        _person("mention_dinara", "Динара", "seg_001"),
    ]
    invalid_endpoint = _parent(
        "relationship_parent",
        "mention_missing",
        "mention_dinara",
        "seg_001",
    )
    invalid_segment = {
        **{
            "relationship_id": "relationship_spouse",
            "relationship_type": "spouse",
            "subject_mention_id": "mention_erlan",
            "subject_role": "spouse",
            "object_mention_id": "mention_dinara",
            "object_role": "spouse",
            "confidence": 1.0,
        },
        "source_segment_ids": ["seg_missing"],
    }

    result, issues, _ = sanitize_extraction_output(
        raw=_raw(
            people=people,
            relationships=[invalid_endpoint, invalid_segment],
            conflicts=_open_conflict(),
        ),
        transcript=transcript,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )

    assert result.relationship_claims == []
    assert {item["object_id"] for item in issues if item["object_type"] == "relationship"} == {
        "relationship_parent",
        "relationship_spouse",
    }


def _pipeline_with_open_conflict() -> PipelineResult:
    transcript = _transcript("Ерлан Динара")
    evidence = EvidenceSpan(
        evidence_id="evidence_relationship",
        segment_id="seg_001",
        text="Ерлан Динара",
        evidence_class=EvidenceClass.A_EXPLICIT,
        mention_ids=["mention_erlan", "mention_dinara"],
    )
    people = [
        PersonMention(
            mention_id="mention_erlan",
            name="Ерлан",
            category="family_member",
            source_segment_ids=["seg_001"],
            evidence_ids=[evidence.evidence_id],
            evidence_class=EvidenceClass.A_EXPLICIT,
            confidence=1,
        ),
        PersonMention(
            mention_id="mention_dinara",
            name="Динара",
            category="family_member",
            source_segment_ids=["seg_001"],
            evidence_ids=[evidence.evidence_id],
            evidence_class=EvidenceClass.A_EXPLICIT,
            confidence=1,
        ),
    ]
    relationship = RelationshipClaim(
        relationship_id="relationship_spouse",
        relationship_type=RelationshipType.SPOUSE,
        subject_mention_id="mention_erlan",
        subject_role=RelationshipRole.SPOUSE,
        object_mention_id="mention_dinara",
        object_role=RelationshipRole.SPOUSE,
        source_segment_ids=["seg_001"],
        evidence_ids=[evidence.evidence_id],
        evidence_class=EvidenceClass.A_EXPLICIT,
        conflict_ids=["conflict_relationship"],
        confidence=1,
    )
    conflict = ConflictSet(
        conflict_id="conflict_relationship",
        conflict_type=ConflictType.RELATIONSHIP,
        claim_refs=[
            {"object_type": "relationship", "object_id": "relationship_spouse"},
            {"object_type": "relationship", "object_id": "relationship_parent_placeholder"},
        ],
        status=ConflictStatus.OPEN,
        detected_by=ConflictDetectionMethod.MODEL,
        rationale="review required",
    )
    return PipelineResult(
        transcript=transcript,
        cleaned_transcript=CleanerResult(
            readable_segments=[ReadableSegment(segment_id="seg_001", text="Ерлан Динара")],
            full_readable_text="Ерлан Динара",
        ),
        extraction=ExtractionResult(
            schema_version="extraction-v2",
            recording_id=transcript.recording_id,
            speaker_id="speaker_1",
            speaker_name="Күләш",
            evidence_spans=[evidence],
            people_mentions=people,
            relationship_claims=[relationship],
            conflict_sets=[conflict],
        ),
        resolutions=[
            MentionResolution(
                mention_id=person.mention_id,
                status=ResolutionStatus.NEW_PERSON,
                reason="fixture",
            )
            for person in people
        ],
    )


def test_conflict_linked_claim_is_persisted_but_not_materialized_as_edge(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'conflict_guard.db'}")
    database.create_schema()
    recording_repository = RecordingRepository(database)
    audio = tmp_path / "rec_completion.wav"
    audio.write_bytes(b"audio")
    recording_repository.create_recording_and_job(
        recording_id="rec_completion",
        job_id="job_completion",
        family_id="family_1",
        speaker_id="speaker_1",
        speaker_name="Күләш",
        original_filename=audio.name,
        content_type="audio/wav",
        audio_path=audio,
    )
    result = _pipeline_with_open_conflict()

    with database.session_factory.begin() as session:
        recording = session.get(RecordingRow, "rec_completion")
        assert recording is not None
        report = ArchiveRepository.persist_pipeline_result(
            session,
            recording=recording,
            result=result,
        )

    archive = ArchiveRepository(database)
    relationship_claims = [
        item
        for item in archive.list_claims("family_1")
        if item.object_type == "relationship"
    ]
    assert len(relationship_claims) == 1
    assert relationship_claims[0].status == "disputed"
    assert relationship_claims[0].payload["conflict_ids"] == [
        "conflict_relationship"
    ]
    assert report.graph_edges == 0
    assert archive.list_graph_edges("family_1") == []


def test_relationship_metrics_deduplicate_repeated_rejection_issues() -> None:
    transcript = _transcript("Ерлан Динара")
    result = ExtractionResult(
        recording_id=transcript.recording_id,
        speaker_id="speaker_1",
        speaker_name="Күләш",
    )
    issue = {
        "object_type": "relationship",
        "object_id": "relationship_rejected",
        "context": {
            "evidence_analysis": {
                "grounding_decision": "insufficient_deterministic_signal"
            }
        },
    }

    counters = relationship_grounding_counters(
        result=result,
        transcript=transcript,
        extraction_issues=[issue, issue],
    )

    assert counters["ambiguous_grounding_rejected"] == 1
    assert all(isinstance(value, int) for value in counters.values())
    assert "Ерлан" not in repr(counters)
