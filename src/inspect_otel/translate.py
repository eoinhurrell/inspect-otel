"""Core translator: EvalLog -> OpenTelemetry spans."""

from __future__ import annotations

import json
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


def translate_eval(log: EvalLog, tracer: Tracer) -> None:
    with tracer.start_as_current_span(
        "inspect.eval",
        kind=SpanKind.INTERNAL,
    ) as eval_span:
        _set_eval_attributes(eval_span, log)
        if log.samples:
            for sample in log.samples:
                translate_sample(sample, tracer, eval_span)


def translate_sample(sample: EvalSample, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    with tracer.start_as_current_span(
        "inspect.sample",
        kind=SpanKind.INTERNAL,
        context=ctx,
    ) as sample_span:
        _set_sample_attributes(sample_span, sample)
        _translate_events(sample.events, tracer, sample_span)


def _translate_events(events: list, tracer: Tracer, parent_span: Span) -> None:
    open_spans: dict[str, Span] = {}

    for event in events:
        if isinstance(event, SpanBeginEvent):
            ctx = set_span_in_context(parent_span)
            span = tracer.start_span(
                "inspect.solver",
                kind=SpanKind.INTERNAL,
                context=ctx,
                attributes={"inspect.solver.name": event.name},
            )
            open_spans[event.id] = span
        elif isinstance(event, SpanEndEvent):
            span = open_spans.pop(event.id, None)
            if span:
                span.end()
        elif isinstance(event, ModelEvent):
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_model_event(event, tracer, parent)
        elif isinstance(event, ToolEvent):
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_tool_event(event, tracer, parent)
        elif isinstance(event, ScoreEvent):
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_score_event(event, tracer, parent)
        elif isinstance(event, SubtaskEvent):
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_subtask_event(event, tracer, parent)
        elif isinstance(event, SandboxEvent):
            parent = open_spans.get(event.span_id, parent_span) if event.span_id else parent_span
            _emit_sandbox_event(event, tracer, parent)
        elif isinstance(event, ErrorEvent):
            from opentelemetry.trace import get_current_span

            current = get_current_span()
            if current and current.is_recording():
                current.set_status(StatusCode.ERROR, event.error.message)
                current.set_attribute("error.type", event.error.message)

    for span in open_spans.values():
        span.end()


def _emit_model_event(event: ModelEvent, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    with tracer.start_as_current_span("gen_ai.chat", kind=SpanKind.CLIENT, context=ctx) as span:
        span.set_attribute("gen_ai.request.model", event.model)
        if event.output and event.output.model:
            span.set_attribute("gen_ai.response.model", event.output.model)
        if event.output and event.output.usage:
            span.set_attribute("gen_ai.usage.input_tokens", event.output.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", event.output.usage.output_tokens)
        if event.error:
            span.set_status(StatusCode.ERROR, event.error)


def _emit_tool_event(event: ToolEvent, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    with tracer.start_as_current_span("execute_tool", kind=SpanKind.INTERNAL, context=ctx) as span:
        span.set_attribute("gen_ai.tool.name", event.function)
        span.set_attribute("gen_ai.tool.call.id", event.id)
        if event.error:
            span.set_status(StatusCode.ERROR, str(event.error.message))
            span.set_attribute("inspect.tool.error", str(event.error.message))


def _emit_score_event(event: ScoreEvent, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    with tracer.start_as_current_span("inspect.score", kind=SpanKind.INTERNAL, context=ctx) as span:
        if event.scorer:
            span.set_attribute("inspect.score.scorer", event.scorer)
        span.set_attribute("inspect.score.value", _score_value(event.score.value))
        if event.score.answer is not None:
            span.set_attribute("inspect.score.answer", event.score.answer)
        if event.score.explanation is not None:
            span.set_attribute("inspect.score.explanation", event.score.explanation)


def _emit_subtask_event(event: SubtaskEvent, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    with tracer.start_as_current_span("gen_ai.agent", kind=SpanKind.INTERNAL, context=ctx) as span:
        span.set_attribute("gen_ai.agent.name", event.name)
        span.set_attribute("inspect.subtask.name", event.name)


def _emit_sandbox_event(event: SandboxEvent, tracer: Tracer, parent_span: Span) -> None:
    ctx = set_span_in_context(parent_span)
    with tracer.start_as_current_span("execute_tool", kind=SpanKind.INTERNAL, context=ctx) as span:
        span.set_attribute("gen_ai.tool.name", f"sandbox.{event.action}")
        span.set_attribute("gen_ai.tool.type", "sandbox")


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


def _set_sample_attributes(span: Span, sample: EvalSample) -> None:
    span.set_attribute("inspect.sample.id", sample.id)
    span.set_attribute("inspect.sample.epoch", sample.epoch)
    if sample.uuid:
        span.set_attribute("inspect.sample.uuid", sample.uuid)
    if sample.metadata:
        span.set_attribute("inspect.sample.metadata", json.dumps(sample.metadata))
    if sample.scores:
        for scorer_name, score in sample.scores.items():
            span.set_attribute(f"inspect.sample.score.{scorer_name}", _score_value(score.value))


def _score_value(value: object) -> str | int | float:
    if isinstance(value, (int, float)):
        return value
    return str(value)
