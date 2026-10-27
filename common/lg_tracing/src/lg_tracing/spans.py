"""Tracer name and trace-id helpers."""

from openinference.instrumentation.langchain import get_current_span
from opentelemetry import context as otel_context
from opentelemetry import trace

TRACER_NAME = "lg_tracing"


def _active_context() -> otel_context.Context:
    """Current context, with the running LangChain node/tool span as the current span.

    The LangChain instrumentor never attaches its spans as current, so look the span up by run.
    """
    ctx = otel_context.get_current()
    try:
        span = get_current_span()
    except ImportError:  # get_current_span imports langchain-core, absent in non-LangChain services
        span = None
    return trace.set_span_in_context(span, ctx) if span is not None else ctx


def current_trace_id() -> str:
    """32-char hex trace id of the current span, or "" when there is none."""
    span_ctx = trace.get_current_span(_active_context()).get_span_context()
    return format(span_ctx.trace_id, "032x") if span_ctx.is_valid else ""
