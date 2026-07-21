from mura.domain.models import RawSegment, TranscriptEnvelope
from mura.long_form import LongFormExtractionPlanner, LongFormMode, LongFormPolicy


def _transcript(segment_count: int, *, words_per_segment: int = 12) -> TranscriptEnvelope:
    segments = [
        RawSegment(
            segment_id=f"seg_{index:02d}",
            start=float(index * 10),
            end=float(index * 10 + 8),
            text=" ".join(f"сөз{word}" for word in range(words_per_segment)),
        )
        for index in range(segment_count)
    ]
    return TranscriptEnvelope(
        recording_id="rec_long_form",
        duration_seconds=float(segment_count * 10),
        language_hints=["kk", "ru"],
        full_text=" ".join(item.text for item in segments),
        segments=segments,
        asr_model="fixture",
        asr_revision="fixture-v1",
        chunker_version="fixture-v1",
    )


def test_short_transcript_stays_on_existing_path() -> None:
    plan = LongFormExtractionPlanner().plan(_transcript(4))

    assert plan.mode is LongFormMode.SHORT
    assert len(plan.windows) == 1
    assert plan.windows[0].source_segment_ids == [f"seg_{index:02d}" for index in range(4)]


def test_eighteen_segments_create_stable_bounded_windows_with_full_coverage() -> None:
    planner = LongFormExtractionPlanner()
    transcript = _transcript(18)

    first = planner.plan(transcript)
    second = planner.plan(transcript)

    assert first.mode is LongFormMode.WINDOWED
    assert 3 <= len(first.windows) <= 6
    assert first == second
    assert [item.window_id for item in first.windows] == [item.window_id for item in second.windows]
    covered = {segment_id for window in first.windows for segment_id in window.source_segment_ids}
    assert covered == {item.segment_id for item in transcript.segments}
    assert all(len(window.overlap_segment_ids) <= 1 for window in first.windows)


def test_token_threshold_boundary_is_deterministic() -> None:
    policy = LongFormPolicy(
        segment_count_threshold=30,
        normalized_token_threshold=200,
        estimated_input_token_threshold=10_000,
    )
    planner = LongFormExtractionPlanner(policy)

    assert planner.plan(_transcript(2, words_per_segment=100)).mode is LongFormMode.SHORT
    assert planner.plan(_transcript(2, words_per_segment=101)).mode is LongFormMode.WINDOWED


def test_oversized_single_segment_is_split_without_changing_source_identity() -> None:
    policy = LongFormPolicy(
        segment_count_threshold=30,
        normalized_token_threshold=10_000,
        estimated_input_token_threshold=10_000,
        maximum_segment_tokens=64,
        target_window_tokens=128,
        maximum_window_tokens=256,
    )
    transcript = _transcript(1, words_per_segment=700)
    planner = LongFormExtractionPlanner(policy)

    plan = planner.plan(transcript)

    assert plan.mode is LongFormMode.WINDOWED
    assert len(plan.windows) > 1
    assert {segment_id for window in plan.windows for segment_id in window.source_segment_ids} == {
        "seg_00"
    }
    assert all(window.segment_slices for window in plan.windows)
    assert all(
        planner.materialize_window(transcript, window).segments[0].segment_id == "seg_00"
        for window in plan.windows
    )
