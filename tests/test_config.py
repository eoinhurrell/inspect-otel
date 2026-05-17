from inspect_otel.config import capture_content, enabled, sample_span_limit, service_name


def test_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_ENABLED", raising=False)
    assert enabled() is True


def test_enabled_false(monkeypatch):
    monkeypatch.setenv("INSPECT_OTEL_ENABLED", "false")
    assert enabled() is False


def test_capture_content_defaults_false(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_CAPTURE_CONTENT", raising=False)
    assert capture_content() is False


def test_service_name_default(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_SERVICE_NAME", raising=False)
    assert service_name() == "inspect"


def test_sample_span_limit_default(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_SAMPLE_SPAN_LIMIT", raising=False)
    assert sample_span_limit() is None


def test_sample_span_limit_set(monkeypatch):
    monkeypatch.setenv("INSPECT_OTEL_SAMPLE_SPAN_LIMIT", "500")
    assert sample_span_limit() == 500
