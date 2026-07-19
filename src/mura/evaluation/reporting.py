from __future__ import annotations

import json
from pathlib import Path

from mura.evaluation.models import BenchmarkReport, BenchmarkSummary, PrecisionRecallF1, RatioMetric


def write_json_report(report: BenchmarkReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_prf(metric: PrecisionRecallF1) -> str:
    return (
        f"P={metric.precision:.3f}, R={metric.recall:.3f}, F1={metric.f1:.3f} "
        f"(TP={metric.true_positive}, FP={metric.false_positive}, "
        f"FN={metric.false_negative})"
    )


def _format_ratio(metric: RatioMetric) -> str:
    return f"{metric.value:.3f} ({metric.numerator}/{metric.denominator})"


def _summary_row(label: str, summary: BenchmarkSummary) -> str:
    return (
        f"| {label} | {summary.case_count} | {summary.relationships.precision:.3f} | "
        f"{summary.relationships.recall:.3f} | {summary.relationships.f1:.3f} | "
        f"{summary.provenance_completeness.value:.3f} | "
        f"{summary.unsupported_relationship_acceptance.value:.3f} | "
        f"{summary.critical_graph_violations} |"
    )


def render_markdown_report(report: BenchmarkReport) -> str:
    summary = report.summary
    lines = [
        "# Mura ML Core Baseline",
        "",
        f"Manifest: `{report.manifest_path}`",
        "",
        "## Pipeline versions",
        "",
        "| Component | Version |",
        "|---|---|",
    ]
    lines.extend(
        f"| {component} | `{version}` |"
        for component, version in sorted(report.pipeline_versions.items())
    )
    lines.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            f"- Cases: **{summary.case_count}**",
            f"- Person mentions: {_format_prf(summary.person_mentions)}",
            f"- Relationships: {_format_prf(summary.relationships)}",
            f"- Expected quarantine: {_format_prf(summary.quarantined_relationships)}",
            (
                "- Relationship direction accuracy: "
                f"{_format_ratio(summary.relationship_direction_accuracy)}"
            ),
            f"- Provenance completeness: {_format_ratio(summary.provenance_completeness)}",
            (
                "- Unsupported relationship acceptance: "
                f"{_format_ratio(summary.unsupported_relationship_acceptance)}"
            ),
            f"- Unknown segment references: **{summary.unknown_segment_references}**",
            f"- Self relationships: **{summary.self_relationships}**",
            (
                "- Accepted claims without evidence: "
                f"**{summary.accepted_claims_without_evidence}**"
            ),
            f"- Critical graph violations: **{summary.critical_graph_violations}**",
            "",
            "## Dataset coverage",
            "",
            (
                "| Dataset | Layer | Split | Enabled | Loaded | Cases | "
                "Approved anonymized | Narrators |"
            ),
            "|---|---|---|---|---|---:|---|---:|",
        ]
    )
    for item in report.dataset_coverage:
        lines.append(
            f"| {item.dataset_id} | {item.layer.value} | {item.split.value} | "
            f"{'yes' if item.enabled else 'no'} | {'yes' if item.loaded else 'no'} | "
            f"{item.case_count} | {'yes' if item.approved_anonymized else 'no'} | "
            f"{item.narrator_count} |"
        )

    lines.extend(
        [
            "",
            "## Language breakdown",
            "",
            (
                "| Bucket | Cases | Precision | Recall | F1 | Provenance | "
                "Unsupported rate | Violations |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report.slices:
        if item.dimension == "language":
            lines.append(_summary_row(item.key, item.summary))

    lines.extend(
        [
            "",
            "## Layer breakdown",
            "",
            (
                "| Layer | Cases | Precision | Recall | F1 | Provenance | "
                "Unsupported rate | Violations |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report.slices:
        if item.dimension == "layer":
            lines.append(_summary_row(item.key, item.summary))

    lines.extend(
        [
            "",
            "## Cases",
            "",
            (
                "| Case | Layer | Language | Relationships | Quarantine | "
                "Unsupported | Accepted | Quarantined |"
            ),
            "|---|---|---|---|---|---:|---|---|",
        ]
    )
    for case in report.cases:
        accepted = ", ".join(case.accepted_relationship_ids) or "—"
        quarantined = ", ".join(case.quarantined_relationship_ids) or "—"
        lines.append(
            "| "
            f"{case.case_id} | {case.dataset_layer.value} | {case.language.value} | "
            f"{case.relationships.f1:.3f} | "
            f"{case.quarantined_relationships.f1:.3f} | "
            f"{case.unsupported_relationship_acceptance.value:.3f} | "
            f"{accepted} | {quarantined} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report measures the deterministic validation layer against fixed extraction "
            "candidates. It does not measure live DeepSeek candidate generation or ASR quality. "
            "Disabled private datasets are reported as coverage gaps rather than silently treated "
            "as passing production evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: BenchmarkReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_report(report), encoding="utf-8")
