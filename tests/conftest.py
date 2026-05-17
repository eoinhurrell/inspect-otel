"""Shared test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from inspect_ai._util.error import EvalError
from inspect_ai.event._error import ErrorEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._sandbox import SandboxEvent
from inspect_ai.event._score import ScoreEvent
from inspect_ai.event._span import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._subtask import SubtaskEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, EvalStats
from inspect_ai.model._model_output import ModelOutput, ModelUsage
from inspect_ai.scorer._metric import Score
from inspect_ai.tool._tool_call import ToolCallError
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer  # noqa: TC002


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def make_eval_log(
    *,
    samples: list[EvalSample] | None = None,
    task: str = "test_task",
    model: str = "test/model",
    run_id: str = "run-001",
    eval_id: str = "eval-001",
    status: str = "success",
) -> EvalLog:
    return EvalLog(
        status=status,
        eval=EvalSpec(
            task=task,
            model=model,
            run_id=run_id,
            eval_id=eval_id,
            created=_utc_now(),
            dataset=EvalDataset(),
            config=EvalConfig(),
        ),
        stats=EvalStats(),
        samples=samples,
    )


def make_sample(
    *,
    id: int | str = 1,
    epoch: int = 1,
    events: list | None = None,
    scores: dict[str, Score] | None = None,
    target: str = "answer",
    uuid: str | None = "sample-uuid-1",
) -> EvalSample:
    return EvalSample(
        id=id,
        epoch=epoch,
        input="test input",
        target=target,
        events=events or [],
        scores=scores,
        uuid=uuid,
    )


def make_model_event(
    *,
    model: str = "test/model",
    input_tokens: int = 100,
    output_tokens: int = 50,
    output_model: str | None = None,
    error: str | None = None,
    span_id: str | None = None,
) -> ModelEvent:
    return ModelEvent(
        model=model,
        input=[],
        tools=[],
        tool_choice="auto",
        config={},
        output=ModelOutput(
            model=output_model or model,
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        ),
        error=error,
        span_id=span_id,
    )


def make_tool_event(
    *,
    function: str = "search",
    tool_id: str = "call-001",
    error: str | None = None,
) -> ToolEvent:
    return ToolEvent(
        id=tool_id,
        function=function,
        arguments={"query": "test"},
        error=ToolCallError(type="unknown", message=error) if error else None,
    )


def make_score_event(
    *,
    scorer: str = "match",
    value: str | int | float = 1.0,
    answer: str | None = None,
    explanation: str | None = None,
) -> ScoreEvent:
    return ScoreEvent(
        score=Score(value=value, answer=answer, explanation=explanation),
        scorer=scorer,
        target="answer",
        intermediate=False,
    )


def make_subtask_event(
    *,
    name: str = "research_agent",
) -> SubtaskEvent:
    return SubtaskEvent(
        name=name,
        input={},
    )


def make_span_begin(
    *,
    span_id: str = "span-001",
    name: str = "solver_step",
) -> SpanBeginEvent:
    return SpanBeginEvent(id=span_id, name=name)


def make_span_end(
    *,
    span_id: str = "span-001",
) -> SpanEndEvent:
    return SpanEndEvent(id=span_id)


def make_sandbox_event(
    *,
    action: str = "exec",
    cmd: str | None = None,
) -> SandboxEvent:
    return SandboxEvent(action=action, cmd=cmd)


def make_error_event(
    *,
    message: str = "Something went wrong",
) -> ErrorEvent:
    return ErrorEvent(error=EvalError(message=message, traceback="traceback...", traceback_ansi="traceback..."))


def get_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return exporter.get_finished_spans()
