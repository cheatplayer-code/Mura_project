from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from mura.orchestration import AudioStorageError, LocalAudioStorage


def test_local_audio_storage_uses_generated_recording_name(tmp_path: Path) -> None:
    storage = LocalAudioStorage(tmp_path, max_upload_bytes=10)
    result = storage.save(
        recording_id="rec_safe",
        original_filename="../../family story.m4a",
        source=BytesIO(b"12345"),
    )

    assert result == tmp_path / "rec_safe.m4a"
    assert result.read_bytes() == b"12345"


def test_local_audio_storage_rejects_bad_extension_and_oversize(tmp_path: Path) -> None:
    storage = LocalAudioStorage(tmp_path, max_upload_bytes=4)

    with pytest.raises(AudioStorageError, match="unsupported audio extension"):
        storage.save(
            recording_id="rec_bad",
            original_filename="payload.exe",
            source=BytesIO(b"123"),
        )

    with pytest.raises(AudioStorageError, match="maximum upload size"):
        storage.save(
            recording_id="rec_large",
            original_filename="story.wav",
            source=BytesIO(b"12345"),
        )
    assert not (tmp_path / "rec_large.wav").exists()
