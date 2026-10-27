"""Tracer name, trace-id helper and the TOOL span used by non-LangChain services."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.langchain import get_current_span
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

TRACER_NAME = "lg_tracing"


def _active_context() -> otel_context.Context:
    """Current context, with the running LangChain node/tool span as the current span.

    The LangChain instrumentor never attaches its spans as current, so look the span up by run.
    """
    ctx = otel_context.get_current()
    try:
        span = get_current_span()
    except (ImportError, AttributeError):  # langchain-core absent or its runnables not imported
        span = None
    return trace.set_span_in_context(span, ctx) if span is not None else ctx


def current_trace_id() -> str:
    """32-char hex trace id of the current span, or "" when there is none."""
    span_ctx = trace.get_current_span(_active_context()).get_span_context()
    return format(span_ctx.trace_id, "032x") if span_ctx.is_valid else ""


def _serialize(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return value, OpenInferenceMimeTypeValues.TEXT.value
    return json.dumps(value, default=str), OpenInferenceMimeTypeValues.JSON.value


def set_output(span: Span, value: Any) -> None:
    text, mime = _serialize(value)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, text)
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime)


@contextmanager
def tool_span(name: str, arguments: dict) -> Iterator[Span]:
    """TOOL span parented on the current context; OK on exit, ERROR (re-raised) on exception."""
    with trace.get_tracer(TRACER_NAME).start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
        # Plain OTel spans don't get using_attributes context (session, user, ...) automatically.
        span.set_attributes(dict(get_attributes_from_context()))
        span.set_attribute(SpanAttributes.TOOL_NAME, name)
        text, mime = _serialize(arguments)
        span.set_attribute(SpanAttributes.INPUT_VALUE, text)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, mime)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_status(Status(StatusCode.OK))
