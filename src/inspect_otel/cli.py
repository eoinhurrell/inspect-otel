"""Batch CLI: export .eval files to OTLP spans."""

from __future__ import annotations

from pathlib import Path

import click
from inspect_ai.log import read_eval_log
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from inspect_otel.config import capture_content as get_capture_content
from inspect_otel.config import endpoint as get_endpoint
from inspect_otel.config import headers as get_headers
from inspect_otel.config import sample_span_limit as get_sample_span_limit
from inspect_otel.config import service_name
from inspect_otel.translate import translate_eval


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--endpoint", "endpoint_url", help="OTLP endpoint URL")
@click.option("--headers", "headers_str", help="OTLP headers (key=val,key=val)")
@click.option("--dry-run", is_flag=True, help="Print spans to stdout instead of sending")
@click.option("--latest-only", is_flag=True, help="Export only the latest log per task")
def export(path: str, endpoint_url: str | None, headers_str: str | None, dry_run: bool, latest_only: bool) -> None:
    eval_files = _find_eval_files(path)
    if not eval_files:
        click.echo("No .eval files found.")
        return

    if latest_only:
        eval_files = _filter_latest_only(eval_files)

    provider = _setup_provider(endpoint_url, headers_str, dry_run)
    tracer = provider.get_tracer("inspect-otel")

    cc = get_capture_content()
    sl = get_sample_span_limit()

    for f in eval_files:
        click.echo(f"Exporting {f}...", err=True)
        log = read_eval_log(f)
        translate_eval(log, tracer, capture_content=cc, span_limit=sl)

    provider.force_flush()
    provider.shutdown()

    click.echo(f"Exported {len(eval_files)} file(s).")


def _find_eval_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file() and p.suffix == ".eval":
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.eval"))
    return []


def _filter_latest_only(files: list[Path]) -> list[Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for f in files:
        log = read_eval_log(str(f), header_only=True)
        task = log.eval.task
        started = log.stats.started_at if log.stats.started_at else ""
        existing = latest.get(task)
        if existing is None or started > existing[0]:
            latest[task] = (started, f)
    return [v[1] for v in latest.values()]


def _setup_provider(endpoint_url: str | None, headers_str: str | None, dry_run: bool) -> TracerProvider:
    resource = Resource.create({"service.name": service_name()})
    provider = TracerProvider(resource=resource)

    if dry_run:
        exporter = ConsoleSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        _export_with_otlp(provider, endpoint_url, headers_str)

    return provider


def _export_with_otlp(provider: TracerProvider, endpoint_url: str | None, headers_str: str | None) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    url = endpoint_url or get_endpoint()
    hdrs = _parse_headers(headers_str) if headers_str else _parse_headers(get_headers())

    exporter = OTLPSpanExporter(endpoint=url, headers=hdrs)
    provider.add_span_processor(BatchSpanProcessor(exporter))


def _parse_headers(headers_str: str | None) -> dict[str, str]:
    if not headers_str:
        return {}
    result = {}
    for pair in headers_str.split(","):
        if "=" in pair:
            key, val = pair.split("=", 1)
            result[key.strip()] = val.strip()
    return result


def main() -> None:
    cli()
