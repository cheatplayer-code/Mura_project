from __future__ import annotations

from typing import Any

from mura.deepseek.grounding_metrics import install_relationship_telemetry
from mura.domain.models import ExtractionResult, RawSegment, TranscriptEnvelope


class DummyService:
    calls = 0

    def extract(
        self,
        *,
        transcript: TranscriptEnvelope,
        cleaned: Any,
        speaker_id: str,
        speaker_name: str,
        known_people: Any = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        del cleaned, known_people
        type(self).calls += 1
        issue = {
            "object_type": "relationship",
            "object_id": "relationship_rejected",
            "context": {
                "evidence_analysis": {"grounding_decision": "insufficient_deterministic_signal"}
            },
        }
        return (
            ExtractionResult(
                recording_id=transcript.recording_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
            ),
            {
                "relationship_metrics": {
                    "candidates": 1,
                    "accepted": 0,
                    "quarantined": 2,
                    "acceptance_rate": 0.0,
                },
                "extraction_issues": [issue, issue],
            },
        )


def test_relationship_telemetry_is_idempotent_and_counts_unique_rejections() -> None:
    DummyService.calls = 0
    install_relationship_telemetry(DummyService)
    install_relationship_telemetry(DummyService)
    transcript = TranscriptEnvelope(
        recording_id="rec_metrics",
        duration_seconds=1,
        full_text="Alex",
        segments=[RawSegment(segment_id="seg_001", start=0, end=1, text="Alex")],
        asr_model="fixture",
        asr_revision="v1",
        chunker_version="v1",
    )

    _, usage = DummyService().extract(
        transcript=transcript,
        cleaned=None,
        speaker_id="speaker_1",
        speaker_name="Narrator",
    )

    assert DummyService.calls == 1
    assert usage["relationship_metrics"] == {
        "candidates": 1,
        "accepted": 0,
        "quarantined": 1,
        "acceptance_rate": 0.0,
    }
    counters = usage["relationship_grounding_metrics"]
    assert counters["ambiguous_grounding_rejected"] == 1
    assert all(isinstance(value, int) for value in counters.values())
    assert "Alex" not in repr(counters)
