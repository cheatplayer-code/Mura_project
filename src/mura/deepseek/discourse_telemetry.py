from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mura.domain.models import (
    CoreferenceMethod,
    CoreferenceStatus,
    ExtractionResult,
    GrammaticalNumber,
    TranscriptEnvelope,
)

_INSTALL_MARKER = "_mura_discourse_telemetry_installed"


def discourse_link_counters(result: ExtractionResult) -> dict[str, int]:
    counters = {
        "singular_coreference_resolved": 0,
        "plural_coreference_resolved": 0,
        "ambiguous_coreference_rejected": 0,
        "unresolved_coreference_rejected": 0,
    }
    links = {
        link.coreference_id: link
        for link in result.coreference_links
        if link.method is CoreferenceMethod.DETERMINISTIC_DISCOURSE
    }
    for link in links.values():
        if link.status is CoreferenceStatus.RESOLVED:
            key = (
                "plural_coreference_resolved"
                if link.grammatical_number is GrammaticalNumber.PLURAL
                else "singular_coreference_resolved"
            )
            counters[key] += 1
        elif link.status is CoreferenceStatus.AMBIGUOUS:
            counters["ambiguous_coreference_rejected"] += 1
        else:
            counters["unresolved_coreference_rejected"] += 1
    return counters


def install_discourse_telemetry(service_type: type[Any]) -> None:
    if getattr(service_type, _INSTALL_MARKER, False):
        return

    original: Callable[..., tuple[ExtractionResult, dict[str, Any]]] = service_type.extract

    def extract(
        self: Any,
        *,
        transcript: TranscriptEnvelope,
        cleaned: Any,
        speaker_id: str,
        speaker_name: str,
        known_people: Any = None,
    ) -> tuple[ExtractionResult, dict[str, Any]]:
        result, payload = original(
            self,
            transcript=transcript,
            cleaned=cleaned,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            known_people=known_people,
        )
        payload["coreference_metrics"] = discourse_link_counters(result)
        return result, payload

    service_type.extract = extract
    setattr(service_type, _INSTALL_MARKER, True)
