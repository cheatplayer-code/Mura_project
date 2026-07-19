from __future__ import annotations

import argparse
from pathlib import Path

from mura.evaluation.entity_resolution import run_entity_resolution_benchmark
from mura.evaluation.release_gates import (
    GateProfile,
    evaluate_release_gates,
    load_benchmark_report,
    load_release_gate_config,
    render_release_gate_markdown,
    write_release_gate_result,
)
from mura.evaluation.reporting import (
    render_markdown_report,
    write_json_report,
    write_markdown_report,
)
from mura.evaluation.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Mura's deterministic ML core on versioned benchmark fixtures."
    )
    parser.add_argument(
        "--manifest",
        default="benchmarks/manifest.json",
        help="Path to the benchmark manifest.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional path for a Markdown report.",
    )
    parser.add_argument(
        "--gate-config",
        default=None,
        help="Optional release-gate configuration. Supplying it makes gate failure exit non-zero.",
    )
    parser.add_argument(
        "--gate-profile",
        choices=[item.value for item in GateProfile],
        default=GateProfile.PULL_REQUEST.value,
        help="Release-gate profile to evaluate.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional approved benchmark report used for common-case regression checks.",
    )
    parser.add_argument(
        "--entity-resolution-dataset",
        default="benchmarks/entity_resolution_v1.json",
        help="Entity-resolution benchmark used for false-merge and review-routing gates.",
    )
    parser.add_argument(
        "--gate-output",
        default=None,
        help="Optional path for the machine-readable release-gate result.",
    )
    parser.add_argument(
        "--gate-markdown-output",
        default=None,
        help="Optional path for the human-readable release-gate result.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(Path(args.manifest))

    if args.json_output:
        write_json_report(report, args.json_output)
    if args.markdown_output:
        write_markdown_report(report, args.markdown_output)

    print(render_markdown_report(report))
    if not args.gate_config:
        return 0

    config = load_release_gate_config(args.gate_config)
    entity_report = run_entity_resolution_benchmark(Path(args.entity_resolution_dataset))
    baseline = load_benchmark_report(args.baseline) if args.baseline else None
    gate_result = evaluate_release_gates(
        report=report,
        entity_report=entity_report,
        config=config,
        profile=GateProfile(args.gate_profile),
        baseline=baseline,
    )
    gate_markdown = render_release_gate_markdown(gate_result)
    if args.gate_output:
        write_release_gate_result(gate_result, args.gate_output)
    if args.gate_markdown_output:
        output = Path(args.gate_markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(gate_markdown, encoding="utf-8")
    print(gate_markdown)
    return 0 if gate_result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
