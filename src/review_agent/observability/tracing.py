"""OpenTelemetry tracing. One trace per request, with child spans for guardrails, each LLM call
and each tool call. Span attributes follow the OpenTelemetry GenAI conventions (gen_ai.*), so the
traces open in Jaeger, Grafana Tempo, Honeycomb or Datadog without custom parsing.

RIA_OTEL_EXPORTER=none     tracing calls are no-ops (default)
RIA_OTEL_EXPORTER=console  spans printed to stdout (local debugging)
RIA_OTEL_EXPORTER=otlp     spans sent to an OpenTelemetry collector at RIA_OTEL_ENDPOINT
LangSmith works alongside: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY and LangChain sends
its own traces of every model call.
"""
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_tracer = trace.get_tracer("review_agent")
_configured = False


def setup_tracing(service_name: str, env: str, exporter: str, endpoint: str) -> None:
    global _configured
    if exporter == "none" or _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name,
                                                        "deployment.environment": env}))
    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


def _attr(value):
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return str(value)


@contextmanager
def span(name: str, **attributes):
    with _tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            if value is not None:
                s.set_attribute(key, _attr(value))
        yield s


def current_trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""
