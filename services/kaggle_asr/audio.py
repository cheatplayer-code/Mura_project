from __future__ import annotations

import json
import subprocess
from pathlib import Path


class AudioProcessingError(RuntimeError):
    pass


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16_000) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise AudioProcessingError(f"ffmpeg failed: {process.stderr[-2000:]}")


def audio_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise AudioProcessingError(f"ffprobe failed: {process.stderr[-1000:]}")
    try:
        return float(json.loads(process.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioProcessingError("ffprobe returned no valid duration") from exc
