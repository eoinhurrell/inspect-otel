from inspect_otel.config import capture_content, enabled, endpoint, headers, sample_span_limit, service_name


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


def test_endpoint_inspect_env_takes_priority(monkeypatch):
    monkeypatch.setenv("INSPECT_OTEL_ENDPOINT", "https://custom.endpoint")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert endpoint() == "https://custom.endpoint"


def test_endpoint_fallback_to_otel_env(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel.endpoint")
    assert endpoint() == "https://otel.endpoint"


def test_endpoint_default_none(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert endpoint() is None


def test_headers_inspect_env_takes_priority(monkeypatch):
    monkeypatch.setenv("INSPECT_OTEL_HEADERS", "key1=val1,key2=val2")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    assert headers() == "key1=val1,key2=val2"


def test_headers_fallback_to_otel_env(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_HEADERS", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "auth=token123")
    assert headers() == "auth=token123"


def test_headers_default_none(monkeypatch):
    monkeypatch.delenv("INSPECT_OTEL_HEADERS", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    assert headers() is None
