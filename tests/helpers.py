from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def spans_named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def only_span(
    exporter: InMemorySpanExporter, name: str, *, scope: str | None = None
) -> ReadableSpan:
    spans = [
        s
        for s in spans_named(exporter, name)
        if scope is None or s.instrumentation_scope.name == scope
    ]
    assert len(spans) == 1, f"expected one span named {name!r}, got {len(spans)}"
    return spans[0]
