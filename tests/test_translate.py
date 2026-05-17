"""Tests for the core translator."""

from __future__ import annotations

from opentelemetry.trace import SpanKind, StatusCode

from inspect_otel.translate import translate_eval
from tests.conftest import (
    get_spans,
    make_error_event,
    make_eval_log,
    make_model_event,
    make_sample,
    make_sandbox_event,
    make_score_event,
    make_span_begin,
    make_span_end,
    make_subtask_event,
    make_tool_event,
)


def span_by_name(spans, name: str):
    return next(s for s in spans if s.name == name)


def spans_by_name(spans, name: str):
    return [s for s in spans if s.name == name]


class TestEvalSpan:
    def test_eval_span_root(self, tracer, exporter):
        log = make_eval_log(task="my_task", model="anthropic/claude-4", run_id="run-42")
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        assert eval_span.kind == SpanKind.INTERNAL
        attrs = eval_span.attributes
        assert attrs["inspect.eval.task"] == "my_task"
        assert attrs["inspect.eval.run_id"] == "run-42"
        assert attrs["gen_ai.request.model"] == "anthropic/claude-4"
        assert attrs["inspect.eval.status"] == "success"

    def test_eval_span_error_status(self, tracer, exporter):
        log = make_eval_log(status="error")
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        assert eval_span.attributes["inspect.eval.status"] == "error"

    def test_empty_eval(self, tracer, exporter):
        log = make_eval_log(samples=None)
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        assert len(spans) == 1
        assert spans[0].name == "inspect.eval"

    def test_eval_span_includes_task_args(self, tracer, exporter):
        log = make_eval_log()
        log.eval.task_args = {"key": "value"}
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        assert "inspect.eval.task_args" in eval_span.attributes


class TestSampleSpan:
    def test_sample_spans_are_children_of_eval(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(id=1, epoch=1),
                make_sample(id=2, epoch=1),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        sample_spans = spans_by_name(spans, "inspect.sample")
        assert len(sample_spans) == 2
        for s in sample_spans:
            assert s.parent.span_id == eval_span.context.span_id
            assert s.kind == SpanKind.INTERNAL

    def test_sample_attributes(self, tracer, exporter):
        log = make_eval_log(samples=[make_sample(id=42, epoch=3, uuid="uuid-abc")])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        sample_span = span_by_name(spans, "inspect.sample")
        assert sample_span.attributes["inspect.sample.id"] == 42
        assert sample_span.attributes["inspect.sample.epoch"] == 3
        assert sample_span.attributes["inspect.sample.uuid"] == "uuid-abc"

    def test_sample_with_scores(self, tracer, exporter):
        from inspect_ai.scorer._metric import Score

        log = make_eval_log(
            samples=[
                make_sample(scores={"match": Score(value=1.0, answer="yes")}),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        sample_span = span_by_name(spans, "inspect.sample")
        assert "inspect.sample.score.match" in sample_span.attributes
        assert sample_span.attributes["inspect.sample.score.match"] == 1.0


class TestModelEvent:
    def test_model_event_span(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_model_event(
                            model="anthropic/claude-4",
                            input_tokens=100,
                            output_tokens=50,
                        )
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.kind == SpanKind.CLIENT
        attrs = model_span.attributes
        assert attrs["gen_ai.request.model"] == "anthropic/claude-4"
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50

    def test_model_event_parent_is_sample(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_model_event()]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        sample_span = span_by_name(spans, "inspect.sample")
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.parent.span_id == sample_span.context.span_id

    def test_model_event_with_error(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_model_event(error="rate limited")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.status.status_code == StatusCode.ERROR

    def test_model_event_response_model(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_model_event(output_model="gpt-4o-2024-08-06")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.response.model"] == "gpt-4o-2024-08-06"


class TestToolEvent:
    def test_tool_event_span(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_tool_event(function="search", tool_id="call-123")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        tool_span = span_by_name(spans, "execute_tool")
        assert tool_span.kind == SpanKind.INTERNAL
        assert tool_span.attributes["gen_ai.tool.name"] == "search"
        assert tool_span.attributes["gen_ai.tool.call.id"] == "call-123"

    def test_tool_event_with_error(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_tool_event(error="file not found")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        tool_span = span_by_name(spans, "execute_tool")
        assert tool_span.status.status_code == StatusCode.ERROR
        assert tool_span.attributes["inspect.tool.error"] == "file not found"


class TestScoreEvent:
    def test_score_event_span(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_score_event(
                            scorer="match",
                            value=1.0,
                            answer="yes",
                            explanation="correct",
                        )
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        score_span = span_by_name(spans, "inspect.score")
        assert score_span.kind == SpanKind.INTERNAL
        attrs = score_span.attributes
        assert attrs["inspect.score.scorer"] == "match"
        assert attrs["inspect.score.value"] == 1.0
        assert attrs["inspect.score.answer"] == "yes"
        assert attrs["inspect.score.explanation"] == "correct"


class TestSubtaskEvent:
    def test_subtask_event_span(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_subtask_event(name="research_agent")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        subtask_span = span_by_name(spans, "gen_ai.agent")
        assert subtask_span.kind == SpanKind.INTERNAL
        assert subtask_span.attributes["gen_ai.agent.name"] == "research_agent"
        assert subtask_span.attributes["inspect.subtask.name"] == "research_agent"


class TestSolverSpans:
    def test_solver_spans_from_span_begin_end(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_span_begin(span_id="s1", name="my_solver"),
                        make_span_end(span_id="s1"),
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        solver_span = span_by_name(spans, "inspect.solver")
        assert solver_span.kind == SpanKind.INTERNAL
        assert solver_span.attributes["inspect.solver.name"] == "my_solver"

    def test_events_between_span_begin_end_parent_under_solver(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_span_begin(span_id="s1", name="solver_step"),
                        make_model_event(span_id="s1"),
                        make_span_end(span_id="s1"),
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        solver_span = span_by_name(spans, "inspect.solver")
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.parent.span_id == solver_span.context.span_id

    def test_unmatched_span_begin_still_creates_span(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_span_begin(span_id="orphan", name="orphan_span"),
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        solver_span = span_by_name(spans, "inspect.solver")
        assert solver_span.attributes["inspect.solver.name"] == "orphan_span"


class TestSandboxEvent:
    def test_sandbox_event_span(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_sandbox_event(action="exec", cmd="ls -la")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        sandbox_span = span_by_name(spans, "execute_tool")
        assert sandbox_span.kind == SpanKind.INTERNAL
        assert sandbox_span.attributes["gen_ai.tool.name"] == "sandbox.exec"
        assert sandbox_span.attributes["gen_ai.tool.type"] == "sandbox"

    def test_sandbox_read_file(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_sandbox_event(action="read_file")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        sandbox_span = span_by_name(spans, "execute_tool")
        assert sandbox_span.attributes["gen_ai.tool.name"] == "sandbox.read_file"


class TestErrorEvent:
    def test_error_event_sets_current_span_status(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_model_event(),
                        make_error_event(message="timeout exceeded"),
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        sample_span = span_by_name(spans, "inspect.sample")
        assert sample_span.status.status_code == StatusCode.ERROR
        assert sample_span.attributes.get("error.type") == "timeout exceeded"
