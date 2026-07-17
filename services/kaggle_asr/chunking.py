from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass(frozen=True)
class SpeechRegion:
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class ChunkRecord:
    index: int
    path: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TranscriptPart:
    chunk: ChunkRecord
    text: str


def build_smart_ranges(
    regions: list[SpeechRegion],
    *,
    sample_rate: int,
    max_chunk_seconds: float = 22.0,
    max_internal_gap_seconds: float = 1.2,
) -> list[SpeechRegion]:
    if not regions:
        return []

    ordered = sorted(regions, key=lambda item: item.start_sample)
    current_start = ordered[0].start_sample
    current_end = ordered[0].end_sample
    output: list[SpeechRegion] = []

    for region in ordered[1:]:
        proposed_duration = (region.end_sample - current_start) / sample_rate
        gap = (region.start_sample - current_end) / sample_rate
        if proposed_duration <= max_chunk_seconds and gap <= max_internal_gap_seconds:
            current_end = max(current_end, region.end_sample)
        else:
            output.append(SpeechRegion(current_start, current_end))
            current_start, current_end = region.start_sample, region.end_sample

    output.append(SpeechRegion(current_start, current_end))
    return output


def apply_edge_padding(
    ranges: list[SpeechRegion],
    *,
    sample_rate: int,
    total_samples: int,
    edge_padding_seconds: float = 0.45,
    max_final_seconds: float = 24.9,
) -> list[SpeechRegion]:
    padding = int(edge_padding_seconds * sample_rate)
    padded: list[SpeechRegion] = []

    for region in ranges:
        start = max(0, region.start_sample - padding)
        end = min(total_samples, region.end_sample + padding)
        if (end - start) / sample_rate > max_final_seconds:
            overflow = (end - start) - int(max_final_seconds * sample_rate)
            trim_left = overflow // 2
            trim_right = overflow - trim_left
            start += trim_left
            end -= trim_right
        padded.append(SpeechRegion(start, end))

    return padded


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w]+", "", word.casefold(), flags=re.UNICODE)


def merge_transcript_parts(
    parts: list[TranscriptPart],
    *,
    max_overlap_words: int = 18,
    similarity_threshold: float = 0.88,
) -> str:
    """Merge chunks; remove a duplicate only when the audio ranges also overlap."""
    if not parts:
        return ""

    ordered = sorted(parts, key=lambda part: part.chunk.start)
    merged = ordered[0].text.strip()
    previous = ordered[0]

    for current in ordered[1:]:
        new_text = current.text.strip()
        if not new_text:
            previous = current
            continue
        if not merged:
            merged = new_text
            previous = current
            continue

        audio_overlaps = current.chunk.start < previous.chunk.end
        if audio_overlaps:
            old_words = merged.split()
            new_words = new_text.split()
            maximum = min(max_overlap_words, len(old_words), len(new_words))
            removed = False
            for size in range(maximum, 1, -1):
                old_tail = " ".join(_normalize_word(word) for word in old_words[-size:])
                new_head = " ".join(_normalize_word(word) for word in new_words[:size])
                if SequenceMatcher(None, old_tail, new_head).ratio() >= similarity_threshold:
                    merged = f"{merged} {' '.join(new_words[size:])}".strip()
                    removed = True
                    break
            if not removed:
                merged = f"{merged} {new_text}".strip()
        else:
            merged = f"{merged} {new_text}".strip()
        previous = current

    return merged
