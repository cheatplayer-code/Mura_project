from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mura.evaluation.models import (
    BenchmarkDataset,
    BenchmarkManifest,
    BenchmarkReport,
    BenchmarkSlice,
    CaseEvaluation,
    DatasetCoverage,
)
from mura.evaluation.scoring import aggregate_case_metrics, score_case
from mura.extraction_sanitizer import sanitize_extraction_output
from mura.versioning import get_pipeline_versions


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_manifest(path: str | Path) -> BenchmarkManifest:
    return BenchmarkManifest.model_validate(_read_json(Path(path)))


def load_dataset(path: str | Path) -> BenchmarkDataset:
    return BenchmarkDataset.model_validate(_read_json(Path(path)))


def _resolve_dataset_path(manifest_path: Path, dataset_path: str) -> Path:
    candidate = Path(dataset_path)
    if candidate.is_absolute():
        return candidate
    return manifest_path.parent / candidate


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_slices(evaluations: list[CaseEvaluation]) -> list[BenchmarkSlice]:
    buckets: dict[tuple[str, str], list[CaseEvaluation]] = defaultdict(list)
    for case in evaluations:
        buckets[("language", case.language.value)].append(case)
        buckets[("layer", case.dataset_layer.value)].append(case)
        buckets[("dataset", case.dataset_id)].append(case)
        for tag in case.construction_tags:
            buckets[("construction", tag)].append(case)
    return [
        BenchmarkSlice(
            dimension=dimension,
            key=key,
            summary=aggregate_case_metrics(cases),
        )
        for (dimension, key), cases in sorted(buckets.items())
    ]


def run_benchmark(manifest_path: str | Path) -> BenchmarkReport:
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(resolved_manifest_path)
    evaluations: list[CaseEvaluation] = []
    coverage: list[DatasetCoverage] = []

    for entry in manifest.datasets:
        if not entry.enabled:
            coverage.append(
                DatasetCoverage(
                    dataset_id=entry.dataset_id,
                    split=entry.split,
                    layer=entry.layer,
                    enabled=False,
                    loaded=False,
                    case_count=0,
                    approved_anonymized=entry.approved_anonymized,
                    narrator_count=entry.narrator_count,
                    required_for_production=entry.required_for_production,
                )
            )
            continue

        dataset_path = _resolve_dataset_path(resolved_manifest_path, entry.path)
        dataset = load_dataset(dataset_path)
        if dataset.dataset_id != entry.dataset_id:
            raise ValueError(
                f"manifest dataset_id={entry.dataset_id!r} does not match "
                f"dataset_id={dataset.dataset_id!r} in {dataset_path}"
            )

        coverage.append(
            DatasetCoverage(
                dataset_id=entry.dataset_id,
                split=entry.split,
                layer=entry.layer,
                enabled=True,
                loaded=True,
                case_count=len(dataset.cases),
                approved_anonymized=entry.approved_anonymized,
                narrator_count=entry.narrator_count,
                required_for_production=entry.required_for_production,
            )
        )
        for case in dataset.cases:
            extraction, issues, closure_count = sanitize_extraction_output(
                raw=case.raw_extraction,
                transcript=case.transcript,
                speaker_id=case.speaker_id,
                speaker_name=case.speaker_name,
            )
            evaluations.append(
                score_case(
                    case=case,
                    dataset_id=dataset.dataset_id,
                    split=entry.split,
                    dataset_layer=entry.layer,
                    extraction=extraction,
                    issues=issues,
                    evidence_closure_relationships=closure_count,
                )
            )

    if not evaluations:
        raise ValueError("benchmark manifest contains no enabled cases")

    return BenchmarkReport(
        report_schema_version="evaluation-report-v2",
        manifest_path=_display_path(resolved_manifest_path),
        pipeline_versions=get_pipeline_versions().model_dump(mode="json"),
        cases=evaluations,
        summary=aggregate_case_metrics(evaluations),
        slices=_build_slices(evaluations),
        dataset_coverage=coverage,
    )
