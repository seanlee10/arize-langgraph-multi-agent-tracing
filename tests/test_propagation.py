import json
import sys
import uuid
from contextlib import contextmanager

import pytest
from openinference.instrumentation import get_attributes_from_context
from opentelemetry import baggage, trace

from lg_tracing.propagation import (
    SESSION_KEY,
    TAGS,
    continue_request_context,
    inject_carrier,
    request_context,
)
import lg_tracing.spans as lg_spans
from tests.helpers import only_span

tracer = trace.get_tracer("lg_tracing")


@contextmanager
def stamped_span(name):
    """A plain span stamped with the context attributes, as Task 3's span helpers will do."""
    with tracer.start_as_current_span(name) as s:
        s.set_attributes(dict(get_attributes_from_context()))
        yield s


def test_round_trip_preserves_trace_parent_and_baggage(exporter):
    with request_context(session_id="s1", user_id="u1", request_id="r1", ticker="NVDA"):
        with tracer.start_as_current_span("caller") as caller:
            carrier = inject_carrier()
    assert set(carrier) == {"traceparent", "baggage"}
    with continue_request_context(carrier):
        with stamped_span("callee"):
            assert baggage.get_baggage(SESSION_KEY) == "s1"
    callee_span = only_span(exporter, "callee")
    assert callee_span.context.trace_id == caller.get_span_context().trace_id
    assert callee_span.parent.span_id == caller.get_span_context().span_id
    assert callee_span.attributes["session.id"] == "s1"
    assert callee_span.attributes["user.id"] == "u1"
    assert json.loads(callee_span.attributes["metadata"])["ticker"] == "NVDA"
    assert list(callee_span.attributes["tag.tags"]) == TAGS


@pytest.mark.parametrize("value", ["analyst@example.com", "a b", "x,y;z=1", "세션"])
def test_baggage_values_with_reserved_characters_round_trip(exporter, value):
    with request_context(session_id=value, user_id=value, request_id="r", ticker="NVDA"):
        carrier = inject_carrier()
    with continue_request_context(carrier):
        with stamped_span("callee"):
            pass
    span = only_span(exporter, "callee")
    assert span.attributes["session.id"] == value
    assert span.attributes["user.id"] == value


def test_missing_carrier_starts_new_root_with_defaults(exporter):
    with continue_request_context({}, ticker="NVDA"):
        with stamped_span("root"):
            pass
    span = only_span(exporter, "root")
    assert span.parent is None
    assert span.attributes["user.id"] == "anonymous"
    uuid.UUID(span.attributes["session.id"])


def test_header_keys_are_case_insensitive(exporter):
    with request_context(session_id="s1", user_id="u1", request_id="r1", ticker="NVDA"):
        with tracer.start_as_current_span("caller") as caller:
            carrier = inject_carrier()
    headers = {"Traceparent": carrier["traceparent"], "Baggage": carrier["baggage"]}
    with continue_request_context(headers):
        with stamped_span("callee"):
            pass
    span = only_span(exporter, "callee")
    assert span.parent.span_id == caller.get_span_context().span_id
    assert span.attributes["session.id"] == "s1"


def test_inject_without_langchain_core_uses_plain_current_span(exporter, monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_core.runnables.config", None)
    with tracer.start_as_current_span("plain") as span:
        carrier = inject_carrier()
    assert carrier["traceparent"].split("-")[2] == format(span.get_span_context().span_id, "016x")


def test_span_lookup_failure_is_not_swallowed(monkeypatch):
    def broken():
        raise AttributeError("openinference API changed")

    monkeypatch.setattr(lg_spans, "get_current_span", broken)
    with pytest.raises(AttributeError):
        inject_carrier()
