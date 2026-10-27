"""Tracer name and trace-id helpers."""

from opentelemetry import trace

TRACER_NAME = "lg_tracing"


def current_trace_id() -> str:
    """32-char hex trace id of the current span, or "" when there is none."""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""
