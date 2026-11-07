"""Structured JSON logs. Every line carries request_id, session_id and trace_id, so one search in
the log tool shows everything that happened in a request. Message text is never logged raw:
only lengths, flags and IDs, because user messages can contain personal data.
"""
import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_CONTEXT = ("request_id", "session_id", "trace_id")
_base_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs) -> logging.LogRecord:
    """Stamp request context onto each record when it is created, so it survives queued handlers."""
    from .tracing import current_trace_id

    record = _base_factory(*args, **kwargs)
    record.request_id = request_id_var.get()
    record.session_id = session_id_var.get()
    record.trace_id = current_trace_id()
    return record


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        line |= {k: getattr(record, k, "") for k in _CONTEXT}
        line.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED and k not in _CONTEXT})
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure_logging(level: str = "INFO") -> None:
    logging.setLogRecordFactory(_record_factory)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
