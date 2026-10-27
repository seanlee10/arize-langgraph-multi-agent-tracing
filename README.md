# One Arize trace across LangGraph services

Three services turn a ticker into a buy / hold / sell decision. Two are LangGraph agents, one is an MCP tool server. Each exports its own spans straight to Arize AX, with no shared collector. Arize joins them into **one trace** because every hop passes W3C `traceparent` and `baggage`. Graph, node, LLM and tool spans come from OpenInference's `LangChainInstrumentor`; the only code we add is the context hand-off at each remote call.

```
client ─POST /invoke─► research-agent :8001 ──MCP (_meta)──► websearch-mcp :8000/mcp ─► Tavily
                       (LangGraph)      │
                                        └──POST /invoke (headers)──► decision-agent :8002
                                                                      (LangGraph + tools)
```

This repo is the LangGraph version of [arize-custom-multi-agent-tracing](https://github.com/seanlee10/arize-custom-multi-agent-tracing), which hand-writes every span without a framework.

## Run it

Needs Docker, [uv](https://docs.astral.sh/uv/), an Anthropic key and a Tavily key. Arize keys are optional.

```bash
cp .env.example .env     # fill in ANTHROPIC_API_KEY, TAVILY_API_KEY, ARIZE_SPACE_ID, ARIZE_API_KEY
docker compose up --build
uv run scripts/analyze.py NVDA
```

The script prints the decision and a `trace_id`. If the Arize keys are blank, spans print to the container logs instead. Pass `--session my-session` to set the Arize session id.

## Trace screenshot

![One trace from three services](images/arize-trace.png)
<!-- TODO: screenshot -->

## Expected trace shape

The research graph is flat: its nodes are siblings under the root, and the tool span holds the MCP server's span. Names come from the instrumentor; scopes are `openinference.instrumentation.langchain` unless noted.

```
research_graph                                   [research-agent]
├─ research
│  ├─ ChatAnthropic
│  └─ after_research
├─ tools
│  └─ web_search
│     └─ web_search  (lg_tracing)                [websearch-mcp]
├─ research  (second pass)
│  └─ ChatAnthropic
├─ summarize
│  └─ ChatAnthropic
└─ call_decision
   └─ decision_graph                             [decision-agent]
      ├─ agent → ChatAnthropic
      ├─ tools → force_index
      ├─ agent → ChatAnthropic
      └─ tools → submit_decision
```

Every span in all three services carries the same `session.id`, `user.id`, `metadata` and `tag.tags`.

## Apply this to your app

Three additions, about 12 lines.

1. At startup in every service:

```python
from lg_tracing import setup_tracing
setup_tracing("my-service")   # provider, OTLP export, LangChainInstrumentor
```

2. Inside the node or tool that makes the remote call:

```python
from lg_tracing import inject_carrier
httpx.post(url, json=payload, headers=inject_carrier())        # HTTP
await session.call_tool(name, args, meta=inject_carrier())     # MCP
```

3. In the handler of the receiving service, around the graph:

```python
from lg_tracing import continue_request_context
with continue_request_context(request.headers):                # MCP: ctx.request_context.meta
    result = await graph.ainvoke(state)
```

The rule: **inject where your node or tool makes the outbound call; extract in the handler before you invoke the graph.** The instrumentor handles everything inside a process. The entry service opens `request_context(...)` once, instead of `continue_request_context`.

One catch: the instrumentor creates node and tool spans but never makes them the current OpenTelemetry span. So `inject_carrier()` finds the running span with `openinference.instrumentation.langchain.get_current_span()`. If you copy the helper, keep that call; plain `trace.get_current_span()` returns the caller's span and the remote spans attach one level too high. `tests/test_instrumentor_contract.py` pins this behaviour per version.

The code lives in `common/lg_tracing/src/lg_tracing/`. The working call sites are `services/research_agent/src/research_agent/graph.py` (HTTP), `web_search.py` (MCP) and `services/decision_agent/src/decision_agent/app.py` (handler).

## Framework spans

Once a global tracer provider exists, FastAPI, httpx and the MCP SDK emit their own spans, and an HTTP span can become the trace root. By default this repo records only OpenInference spans: `lg_tracing/provider.py` gives every other tracer a `NoOpTracer`, which records nothing but passes the parent context through, so parent links stay intact.

To keep a library's spans, add its instrumentation scope name to `ALLOWED_TRACERS` in `lg_tracing/provider.py`, for example `"opentelemetry.instrumentation.fastapi"`. Then expect the root to become the HTTP span (`POST /invoke`), with the LangGraph root below it. The traces table shows the root's input and output, and an HTTP span has none.

## Tested versions

From `uv.lock`. Re-run `uv run pytest` after upgrading any of them.

| Package | Version |
|---|---|
| langgraph | 1.2.18 |
| langchain-core | 1.7.0 |
| openinference-instrumentation-langchain | 0.1.80 |
| opentelemetry-sdk | 1.45.1 |

## Known limitations

- The instrumentor does not set node or tool spans as the current span (see above). A newer version may change this; the contract tests will fail if it does, which is the signal to revisit `_active_context` in `lg_tracing/spans.py`.
- The MCP server masks raw tool exception text. The client gets a generic error, not the original message; the detail is on the server-side span.
- Demo scope: no auth between services, no retries, one exporter per service.

## Tests

```bash
uv run pytest -q
```
