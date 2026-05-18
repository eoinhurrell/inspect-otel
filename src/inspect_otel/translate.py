"""Core translator: EvalLog -> OpenTelemetry spans."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from inspect_ai.event._error import ErrorEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._sandbox import SandboxEvent
from inspect_ai.event._score import ScoreEvent
from inspect_ai.event._span import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._subtask import SubtaskEvent
from inspect_ai.event._tool import ToolEvent
from opentelemetry.trace import SpanKind, StatusCode, set_span_in_context

if TYPE_CHECKING:
    from inspect_ai.log._log import EvalLog, EvalSample
    from opentelemetry.trace import Span, Tracer


def _to_ns(value: datetime | str | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1_000_000_000)


def translate_eval(
    log: EvalLog,
    tracer: Tracer,
    *,
    capture_content: bool = False,
    span_limit: int | None = None,
) -> None:
    start = _to_ns(log.stats.started_at)
    end = _to_ns(log.stats.completed_at)
    eval_span = tracer.start_span(
        "inspect.eval",
        kind=SpanKind.INTERNAL,
        start_time=start,
    )
    _set_eval_attributes(eval_span, log)
    _set_eval_metrics(eval_span, log)
    if log.samples:
        for sample in log.samples:
            translate_sample(sample, tracer, eval_span, capture_content=capture_content, span_limit=span_limit)
    eval_span.end(end_time=end)


def translate_sample(
    sample: EvalSample,
    tracer: Tracer,
    parent_span: Span,
    *,
    capture_content: bool = False,
    span_limit: int | None = None,
) -> None:
    ctx = set_span_in_context(parent_span)
    start = _to_ns(sample.started_at)
    end = _to_ns(sample.completed_at)
    sample_span = tracer.start_span(
        "inspect.sample",
        kind=SpanKind.INTERNAL,
        context=ctx,
        start_time=start,
    )
    _set_sample_attributes(sample_span, sample, capture_content=capture_content)
    _translate_events(
        sample.events,
        tracer,
        sample_span,
        capture_content=capture_content,
        span_limit=span_limit,
        limit_span=sample_span,
    )
    sample_span.end(end_time=end)


def _translate_events(
    events: list,
    tracer: Tracer,
    parent_span: Span,
    *,
    capture_content: bool = False,
    span_limit: int | None = None,
    limit_span: Span | None = None,
) -> None:
    open_spans: dict[str, Span] = {}
    span_count = 0

    for event in events:
        if isinstance(event, SpanBeginEvent):
            if span_limit is not None and span_count >= span_limit:
                if limit_span is not None:
                    limit_span.set_attribute("inspect.sample.span_limit_reached", True)
                continue
            ctx = set_span_in_context(parent_span)
            span = tracer.start_span(
                "inspect.solver",
                kind=SpanKind.INTERNAL,
                context=ctx,
                attributes={"inspect.solver.name": event.name},
                start_time=_to_ns(event.timestamp),
            )
            open_spans[event.id] = span
            span_count += 1
        elif isinstance(event, SpanEndEvent):
            span = open_spans.pop(event.id, None)
            if span:
                span.end(end_time=_to_ns(event.timestamp))
        elif isinstance(event, ModelEvent):
            if span_limit is not None and span_count >= span_limit:
                if limit_span is not None:
                    limit_span.set_attribute("inspect.sample.span_limit_reached", True)
                continue
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_model_event(event, tracer, parent, capture_content=capture_content)
            span_count += 1
        elif isinstance(event, ToolEvent):
            if span_limit is not None and span_count >= span_limit:
                if limit_span is not None:
                    limit_span.set_attribute("inspect.sample.span_limit_reached", True)
                continue
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_tool_event(event, tracer, parent, capture_content=capture_content)
            span_count += 1
        elif isinstance(event, ScoreEvent):
            if span_limit is not None and span_count >= span_limit:
                if limit_span is not None:
                    limit_span.set_attribute("inspect.sample.span_limit_reached", True)
                continue
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_score_event(event, tracer, parent, capture_content=capture_content)
            span_count += 1
        elif isinstance(event, SubtaskEvent):
            if span_limit is not None and span_count >= span_limit:
                if limit_span is not None:
                    limit_span.set_attribute("inspect.sample.span_limit_reached", True)
                continue
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_subtask_event(event, tracer, parent)
            span_count += 1
        elif isinstance(event, SandboxEvent):
            if span_limit is not None and span_count >= span_limit:
                if limit_span is not None:
                    limit_span.set_attribute("inspect.sample.span_limit_reached", True)
                continue
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_sandbox_event(event, tracer, parent)
            span_count += 1
        elif isinstance(event, ErrorEvent):
            if parent_span and parent_span.is_recording():
                parent_span.set_status(StatusCode.ERROR, event.error.message)
                parent_span.set_attribute("error.type", event.error.message)

    for span in open_spans.values():
        span.end()


def _derive_provider(model: str) -> tuple[str, str]:
    if "/" in model:
        provider = model.split("/", 1)[0]
        return provider, provider
    return "unknown", "unknown"


def _emit_model_event(event: ModelEvent, tracer: Tracer, parent_span: Span, *, capture_content: bool = False) -> None:
    ctx = set_span_in_context(parent_span)
    start = _to_ns(event.timestamp)
    end = _to_ns(event.completed)
    span = tracer.start_span("gen_ai.chat", kind=SpanKind.CLIENT, context=ctx, start_time=start)
    span.set_attribute("gen_ai.request.model", event.model)

    system, provider = _derive_provider(event.model)
    span.set_attribute("gen_ai.system", system)
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.operation.name", "chat")

    config = event.config
    if config.max_tokens is not None:
        span.set_attribute("gen_ai.request.max_tokens", config.max_tokens)
    if config.temperature is not None:
        span.set_attribute("gen_ai.request.temperature", config.temperature)
    if config.top_p is not None:
        span.set_attribute("gen_ai.request.top_p", config.top_p)
    if config.stop_seqs is not None:
        span.set_attribute("gen_ai.request.stop_sequences", json.dumps(config.stop_seqs))

    if event.output:
        if event.output.model:
            span.set_attribute("gen_ai.response.model", event.output.model)
        if event.output.usage:
            span.set_attribute("gen_ai.usage.input_tokens", event.output.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", event.output.usage.output_tokens)
        if event.output.choices:
            reasons = [c.stop_reason for c in event.output.choices if c.stop_reason]
            if reasons:
                span.set_attribute("gen_ai.response.finish_reasons", ",".join(reasons))
        if event.output.metadata and "id" in event.output.metadata:
            span.set_attribute("gen_ai.response.id", str(event.output.metadata["id"]))

    if event.error:
        span.set_status(StatusCode.ERROR, event.error)

    if capture_content:
        prompt_data = json.dumps([_serialize_msg(m) for m in event.input])
        span.add_event("gen_ai.content.prompt", attributes={"content": prompt_data})
        if event.output and event.output.completion:
            span.add_event("gen_ai.content.completion", attributes={"content": event.output.completion})

    span.end(end_time=end)


def _serialize_msg(msg: object) -> dict:
    dump = getattr(msg, "model_dump", None)
    if callable(dump):
        result = dump(exclude_none=True)
        return result if isinstance(result, dict) else {"data": result}
    if isinstance(msg, dict):
        return msg
    return {"text": str(msg)}


def _emit_tool_event(event: ToolEvent, tracer: Tracer, parent_span: Span, *, capture_content: bool = False) -> None:
    ctx = set_span_in_context(parent_span)
    start = _to_ns(event.timestamp)
    end = _to_ns(event.completed)
    span = tracer.start_span("execute_tool", kind=SpanKind.INTERNAL, context=ctx, start_time=start)
    span.set_attribute("gen_ai.tool.name", event.function)
    span.set_attribute("gen_ai.tool.call.id", event.id)
    if event.error:
        span.set_status(StatusCode.ERROR, str(event.error.message))
        span.set_attribute("inspect.tool.error", str(event.error.message))
    if capture_content:
        span.set_attribute("gen_ai.tool.arguments", json.dumps(event.arguments))
        if event.result is not None:
            span.set_attribute("gen_ai.tool.result", str(event.result))
    span.end(end_time=end)


def _emit_score_event(event: ScoreEvent, tracer: Tracer, parent_span: Span, *, capture_content: bool = False) -> None:
    ctx = set_span_in_context(parent_span)
    start = _to_ns(event.timestamp)
    end = _to_ns(getattr(event, "completed", None))
    span = tracer.start_span("inspect.score", kind=SpanKind.INTERNAL, context=ctx, start_time=start)
    if event.scorer:
        span.set_attribute("inspect.score.scorer", event.scorer)
    span.set_attribute("inspect.score.value", _score_value(event.score.value))
    if event.score.answer is not None:
        span.set_attribute("inspect.score.answer", event.score.answer)
    if event.score.explanation is not None:
        span.set_attribute("inspect.score.explanation", event.score.explanation)
    if event.score.metadata is not None:
        span.set_attribute("inspect.score.metadata", json.dumps(event.score.metadata))
    span.end(end_time=end)


def _emit_subtask_event(event: SubtaskEvent, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    start = _to_ns(event.timestamp)
    end = _to_ns(event.completed)
    span = tracer.start_span("gen_ai.agent", kind=SpanKind.INTERNAL, context=ctx, start_time=start)
    span.set_attribute("gen_ai.agent.name", event.name)
    span.set_attribute("inspect.subtask.name", event.name)
    if event.uuid:
        span.set_attribute("gen_ai.agent.id", event.uuid)
    span.end(end_time=end)


def _emit_sandbox_event(event: SandboxEvent, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    start = _to_ns(event.timestamp)
    span = tracer.start_span("execute_tool", kind=SpanKind.INTERNAL, context=ctx, start_time=start)
    span.set_attribute("gen_ai.tool.name", f"sandbox.{event.action}")
    span.set_attribute("gen_ai.tool.type", "sandbox")
    span.end()


def _set_eval_attributes(span: Span, log: EvalLog) -> None:
    span.set_attribute("inspect.eval.task", log.eval.task)
    span.set_attribute("inspect.eval.run_id", log.eval.run_id)
    span.set_attribute("inspect.eval.task_id", log.eval.task_id)
    span.set_attribute("inspect.eval.status", log.status)
    span.set_attribute("gen_ai.request.model", log.eval.model)
    if log.eval.dataset.name:
        span.set_attribute("inspect.dataset.name", log.eval.dataset.name)
    if log.eval.task_args:
        span.set_attribute("inspect.eval.task_args", json.dumps(log.eval.task_args))
    if log.eval.solver:
        span.set_attribute("inspect.eval.solver", log.eval.solver)


def _set_eval_metrics(span: Span, log: EvalLog) -> None:
    if not log.results or not log.results.scores:
        return
    for eval_score in log.results.scores:
        for metric_name, metric in eval_score.metrics.items():
            span.add_event(
                f"inspect.metric.{metric_name}",
                attributes={
                    "inspect.metric.value": metric.value,
                    "inspect.metric.scorer": eval_score.scorer,
                },
            )


def _set_sample_attributes(span: Span, sample: EvalSample, *, capture_content: bool = False) -> None:
    span.set_attribute("inspect.sample.id", sample.id)
    span.set_attribute("inspect.sample.epoch", sample.epoch)
    if sample.uuid:
        span.set_attribute("inspect.sample.uuid", sample.uuid)
    if sample.metadata:
        span.set_attribute("inspect.sample.metadata", json.dumps(sample.metadata))
    if sample.scores:
        for scorer_name, score in sample.scores.items():
            span.set_attribute(f"inspect.sample.score.{scorer_name}", _score_value(score.value))
    if capture_content:
        if sample.input is not None:
            inp = sample.input if isinstance(sample.input, str) else json.dumps(sample.input)
            span.set_attribute("inspect.sample.input", inp)
        if sample.target is not None:
            target = sample.target if isinstance(sample.target, str) else json.dumps(sample.target)
            span.set_attribute("inspect.sample.target", target)


def _score_value(value: object) -> str | int | float:
    if isinstance(value, (int, float)):
        return value
    return str(value)
