"""Tests for Phase 2: Translator Completeness."""

from __future__ import annotations

import json

from inspect_ai.log._log import EvalMetric, EvalResults, EvalScore
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage

from inspect_otel.translate import translate_eval
from tests.conftest import (
    get_spans,
    make_eval_log,
    make_model_event,
    make_sample,
    make_score_event,
    make_subtask_event,
    make_tool_event,
)


def span_by_name(spans, name: str):
    return next(s for s in spans if s.name == name)


def spans_by_name(spans, name: str):
    return [s for s in spans if s.name == name]


class TestDeriveProvider:
    def test_provider_from_slash_model(self, tracer, exporter):
        log = make_eval_log(
            model="anthropic/claude-4",
            samples=[make_sample(events=[make_model_event(model="anthropic/claude-4")])],
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.system"] == "anthropic"
        assert model_span.attributes["gen_ai.provider.name"] == "anthropic"

    def test_provider_from_openai_model(self, tracer, exporter):
        log = make_eval_log(
            model="openai/gpt-4o",
            samples=[make_sample(events=[make_model_event(model="openai/gpt-4o")])],
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.system"] == "openai"

    def test_provider_bare_model_falls_back_to_unknown(self, tracer, exporter):
        log = make_eval_log(
            model="gpt-4o",
            samples=[make_sample(events=[make_model_event(model="gpt-4o")])],
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.system"] == "unknown"
        assert model_span.attributes["gen_ai.provider.name"] == "unknown"


class TestOperationName:
    def test_operation_name_is_chat(self, tracer, exporter):
        log = make_eval_log(samples=[make_sample(events=[make_model_event()])])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.operation.name"] == "chat"


class TestRequestConfig:
    def test_request_config_attributes_set(self, tracer, exporter):
        event = make_model_event(model="test/model")
        event.config = GenerateConfig(max_tokens=1024, temperature=0.7)
        log = make_eval_log(samples=[make_sample(events=[event])])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.request.max_tokens"] == 1024
        assert model_span.attributes["gen_ai.request.temperature"] == 0.7

    def test_request_config_with_top_p_and_stop_seqs(self, tracer, exporter):
        event = make_model_event(model="test/model")
        event.config = GenerateConfig(top_p=0.9, stop_seqs=["stop1", "stop2"])
        log = make_eval_log(samples=[make_sample(events=[event])])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.request.top_p"] == 0.9
        assert model_span.attributes["gen_ai.request.stop_sequences"] == json.dumps(["stop1", "stop2"])

    def test_request_config_default_no_extra_attributes(self, tracer, exporter):
        log = make_eval_log(samples=[make_sample(events=[make_model_event()])])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert "gen_ai.request.max_tokens" not in model_span.attributes
        assert "gen_ai.request.temperature" not in model_span.attributes


class TestResponseFinishReasons:
    def test_finish_reasons_from_choices(self, tracer, exporter):
        from inspect_ai.model._chat_message import ChatMessageAssistant

        event = make_model_event(model="test/model")
        event.output = ModelOutput(
            model="test/model",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content="hi"),
                    stop_reason="stop",
                )
            ],
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        log = make_eval_log(samples=[make_sample(events=[event])])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.response.finish_reasons"] == "stop"

    def test_response_id_from_metadata(self, tracer, exporter):
        event = make_model_event(model="test/model")
        event.output = ModelOutput(
            model="test/model",
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            metadata={"id": "resp-abc123"},
        )
        log = make_eval_log(samples=[make_sample(events=[event])])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert model_span.attributes["gen_ai.response.id"] == "resp-abc123"


class TestEvalSolver:
    def test_eval_solver_attribute_present(self, tracer, exporter):
        log = make_eval_log()
        log.eval.solver = "my_solver"
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        assert eval_span.attributes["inspect.eval.solver"] == "my_solver"

    def test_eval_solver_absent_when_none(self, tracer, exporter):
        log = make_eval_log()
        log.eval.solver = None
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        assert "inspect.eval.solver" not in eval_span.attributes


class TestScoreMetadata:
    def test_score_metadata_json(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_score_event(
                            scorer="match",
                            value=1.0,
                            metadata={"key": "val"},
                        )
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        score_span = span_by_name(spans, "inspect.score")
        assert score_span.attributes["inspect.score.metadata"] == json.dumps({"key": "val"})

    def test_score_metadata_absent_when_none(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_score_event(
                            scorer="match",
                            value=1.0,
                            metadata=None,
                        )
                    ]
                ),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        score_span = span_by_name(spans, "inspect.score")
        assert "inspect.score.metadata" not in score_span.attributes


class TestSubtaskAgentId:
    def test_subtask_agent_id(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_subtask_event(name="agent1", uuid="uuid-123")]),
            ]
        )
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        subtask_span = span_by_name(spans, "gen_ai.agent")
        assert subtask_span.attributes["gen_ai.agent.id"] == "uuid-123"


class TestContentCapture:
    def test_content_capture_model_prompt(self, tracer, exporter):
        log = make_eval_log(samples=[make_sample(events=[make_model_event()])])
        translate_eval(log, tracer, capture_content=True)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        events = model_span.events
        prompt_events = [e for e in events if e.name == "gen_ai.content.prompt"]
        assert len(prompt_events) == 1

    def test_content_capture_model_completion(self, tracer, exporter):
        event = make_model_event(model="test/model")
        event.output = ModelOutput(
            model="test/model",
            completion="Hello world",
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        log = make_eval_log(samples=[make_sample(events=[event])])
        translate_eval(log, tracer, capture_content=True)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        events = model_span.events
        completion_events = [e for e in events if e.name == "gen_ai.content.completion"]
        assert len(completion_events) == 1

    def test_content_capture_tool_args(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_tool_event(function="search", tool_id="call-001")]),
            ]
        )
        translate_eval(log, tracer, capture_content=True)
        spans = get_spans(exporter)
        tool_span = span_by_name(spans, "execute_tool")
        assert "gen_ai.tool.arguments" in tool_span.attributes

    def test_content_capture_tool_result(self, tracer, exporter):
        event = make_tool_event(function="search", tool_id="call-001")
        event.result = "search result data"
        log = make_eval_log(samples=[make_sample(events=[event])])
        translate_eval(log, tracer, capture_content=True)
        spans = get_spans(exporter)
        tool_span = span_by_name(spans, "execute_tool")
        assert tool_span.attributes["gen_ai.tool.result"] == "search result data"

    def test_content_capture_disabled(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[make_model_event(), make_tool_event()]),
            ]
        )
        translate_eval(log, tracer, capture_content=False)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        assert len(model_span.events) == 0
        tool_span = span_by_name(spans, "execute_tool")
        assert "gen_ai.tool.arguments" not in tool_span.attributes

    def test_content_capture_sample_input(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(events=[]),
            ]
        )
        translate_eval(log, tracer, capture_content=True)
        spans = get_spans(exporter)
        sample_span = span_by_name(spans, "inspect.sample")
        assert "inspect.sample.input" in sample_span.attributes
        assert sample_span.attributes["inspect.sample.input"] == "test input"

    def test_content_capture_sample_target(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(target="expected answer", events=[]),
            ]
        )
        translate_eval(log, tracer, capture_content=True)
        spans = get_spans(exporter)
        sample_span = span_by_name(spans, "inspect.sample")
        assert "inspect.sample.target" in sample_span.attributes
        assert sample_span.attributes["inspect.sample.target"] == "expected answer"


class TestAggregatedMetrics:
    def test_metrics_as_span_events(self, tracer, exporter):
        results = EvalResults(
            scores=[
                EvalScore(
                    name="accuracy",
                    scorer="match",
                    reducer=None,
                    scored_samples=10,
                    unscored_samples=0,
                    params={},
                    metrics={
                        "accuracy": EvalMetric(name="accuracy", value=0.85, params={}, metadata=None),
                        "stderr": EvalMetric(name="stderr", value=0.05, params={}, metadata=None),
                    },
                    metadata=None,
                )
            ]
        )
        log = make_eval_log(results=results)
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        metric_events = [e for e in eval_span.events if e.name.startswith("inspect.metric.")]
        assert len(metric_events) == 2
        event_names = {e.name for e in metric_events}
        assert "inspect.metric.accuracy" in event_names
        assert "inspect.metric.stderr" in event_names

    def test_no_metrics_when_results_empty(self, tracer, exporter):
        log = make_eval_log(results=None)
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        metric_events = [e for e in eval_span.events if e.name.startswith("inspect.metric.")]
        assert len(metric_events) == 0


class TestSpanLimit:
    def test_span_limit_enforced(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_model_event(),
                        make_tool_event(),
                        make_score_event(),
                    ]
                ),
            ]
        )
        translate_eval(log, tracer, span_limit=2)
        spans = get_spans(exporter)
        child_spans = [s for s in spans if s.name not in ("inspect.eval", "inspect.sample")]
        assert len(child_spans) == 2
        sample_span = span_by_name(spans, "inspect.sample")
        assert sample_span.attributes.get("inspect.sample.span_limit_reached") is True

    def test_no_limit_all_spans_emitted(self, tracer, exporter):
        log = make_eval_log(
            samples=[
                make_sample(
                    events=[
                        make_model_event(),
                        make_tool_event(),
                        make_score_event(),
                    ]
                ),
            ]
        )
        translate_eval(log, tracer, span_limit=None)
        spans = get_spans(exporter)
        child_spans = [s for s in spans if s.name not in ("inspect.eval", "inspect.sample")]
        assert len(child_spans) == 3
        sample_span = span_by_name(spans, "inspect.sample")
        assert "inspect.sample.span_limit_reached" not in sample_span.attributes


class TestBackdatedTimestamps:
    def test_eval_span_uses_log_timestamps(self, tracer, exporter):
        log = make_eval_log(samples=[make_sample(events=[])])
        log.stats.started_at = "2024-01-01T10:00:00+00:00"
        log.stats.completed_at = "2024-01-01T10:05:00+00:00"
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        eval_span = span_by_name(spans, "inspect.eval")
        from datetime import datetime

        expected_start = int(datetime.fromisoformat("2024-01-01T10:00:00+00:00").timestamp() * 1_000_000_000)
        expected_end = int(datetime.fromisoformat("2024-01-01T10:05:00+00:00").timestamp() * 1_000_000_000)
        assert eval_span.start_time == expected_start
        assert eval_span.end_time == expected_end

    def test_sample_span_uses_sample_timestamps(self, tracer, exporter):
        sample = make_sample(events=[])
        sample.started_at = "2024-01-01T10:01:00+00:00"
        sample.completed_at = "2024-01-01T10:02:00+00:00"
        log = make_eval_log(samples=[sample])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        sample_span = span_by_name(spans, "inspect.sample")
        from datetime import datetime

        expected_start = int(datetime.fromisoformat("2024-01-01T10:01:00+00:00").timestamp() * 1_000_000_000)
        expected_end = int(datetime.fromisoformat("2024-01-01T10:02:00+00:00").timestamp() * 1_000_000_000)
        assert sample_span.start_time == expected_start
        assert sample_span.end_time == expected_end

    def test_model_event_uses_event_timestamps(self, tracer, exporter):
        from datetime import datetime

        event = make_model_event()
        event.timestamp = datetime.fromisoformat("2024-06-15T12:00:00+00:00")
        event.completed = datetime.fromisoformat("2024-06-15T12:00:05+00:00")
        log = make_eval_log(samples=[make_sample(events=[event])])
        translate_eval(log, tracer)
        spans = get_spans(exporter)
        model_span = span_by_name(spans, "gen_ai.chat")
        expected_start = int(datetime.fromisoformat("2024-06-15T12:00:00+00:00").timestamp() * 1_000_000_000)
        expected_end = int(datetime.fromisoformat("2024-06-15T12:00:05+00:00").timestamp() * 1_000_000_000)
        assert model_span.start_time == expected_start
        assert model_span.end_time == expected_end
