import importlib.util

from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer

import lg_tracing.provider as provider_module
from lg_tracing.provider import build_exporter, build_tracer_provider, instrument


def test_allowlisted_tracers_record_and_others_are_noop():
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("svc", exporter, batch=False)
    assert isinstance(provider.get_tracer("fastapi"), NoOpTracer)
    assert isinstance(provider.get_tracer("mcp.client"), NoOpTracer)
    with provider.get_tracer("lg_tracing").start_as_current_span("kept"):
        with provider.get_tracer("httpx").start_as_current_span("dropped"):
            with provider.get_tracer("lg_tracing").start_as_current_span("child"):
                pass
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert set(by_name) == {"kept", "child"}
    kept, child = by_name["kept"], by_name["child"]
    # "child" is parented to "kept": the NoOp span in between passes context through
    assert child.parent.span_id == kept.context.span_id


def test_resource_has_project_and_service(monkeypatch):
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "proj-x")
    provider = build_tracer_provider("research-agent", InMemorySpanExporter())
    assert provider.resource.attributes["openinference.project.name"] == "proj-x"
    assert provider.resource.attributes["service.name"] == "research-agent"


def test_console_exporter_without_arize_keys(monkeypatch):
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    assert isinstance(build_exporter(), ConsoleSpanExporter)


def test_grpc_exporter_with_arize_keys(monkeypatch):
    monkeypatch.setenv("ARIZE_API_KEY", "k")
    monkeypatch.setenv("ARIZE_SPACE_ID", "s")
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    assert isinstance(build_exporter(), OTLPSpanExporter)


class SpyInstrumentor:
    calls = 0

    def instrument(self, **kwargs):
        SpyInstrumentor.calls += 1


def test_instrument_skipped_without_langchain_core(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "langchain_core" else real_find_spec(name, *a),
    )
    monkeypatch.setattr(provider_module, "LangChainInstrumentor", SpyInstrumentor)
    SpyInstrumentor.calls = 0
    instrument(build_tracer_provider("websearch-mcp", InMemorySpanExporter()))
    assert SpyInstrumentor.calls == 0


def test_instrument_runs_with_langchain_core(monkeypatch):
    monkeypatch.setattr(provider_module, "LangChainInstrumentor", SpyInstrumentor)
    SpyInstrumentor.calls = 0
    instrument(build_tracer_provider("research-agent", InMemorySpanExporter()))
    assert SpyInstrumentor.calls == 1
