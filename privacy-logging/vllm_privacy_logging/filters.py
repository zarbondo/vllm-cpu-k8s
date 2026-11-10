"""Second line of defence: a logging.Filter on every handler of the server.

It does not depend on the plugin being loaded (``VLLM_PLUGINS`` could disable
it, or a new vLLM release could move RequestLogger). Wired in logging.json:

* records of loggers that write request/response content are dropped unless
  they were produced by the plugin (``privacy_safe`` attribute);
* DEBUG records of the modules that handle raw request/response text
  (entrypoints, tool and reasoning parsers) are dropped.
"""

from __future__ import annotations

import logging

from .plugin import AUDIT_LOGGER_NAME, SAFE_RECORD_ATTR

# Loggers whose records contain prompts or generated text.
CONTENT_LOGGERS = (
    "vllm.entrypoints.serve.utils.request_logger",  # RequestLogger, vLLM >= 0.2x
    "vllm.entrypoints.logger",  # RequestLogger, older releases
    "vllm.entrypoints.serve.middleware.log_response",  # VLLM_DEBUG_LOG_API_SERVER_RESPONSE
)

# These packages see raw request/response text; their DEBUG output may echo it.
DEBUG_SENSITIVE_PREFIXES = (
    "vllm.entrypoints",
    "vllm.tool_parsers",
    "vllm.reasoning",
    "vllm.parser",
    "vllm.renderers",
)


# Error handlers of the API server. With --log-error-stack they log the traceback,
# and its text can quote the request (pydantic validation errors include input values).
ERROR_HANDLER_LOGGERS = ("vllm.entrypoints.serve.exception_handling",)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


class PrivacyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # The format in logging.json uses vLLM's process name; vLLM adds it via a
        # record factory, this fallback keeps formatting safe without it.
        if not hasattr(record, "vllm_process_name"):
            record.vllm_process_name = record.processName
        if getattr(record, SAFE_RECORD_ATTR, False) or record.name == AUDIT_LOGGER_NAME:
            return True
        if _matches(record.name, CONTENT_LOGGERS):
            return False
        if record.levelno < logging.INFO and _matches(record.name, DEBUG_SENSITIVE_PREFIXES):
            return False
        if record.exc_info and _matches(record.name, ERROR_HANDLER_LOGGERS):
            # keep the error line (it carries the request id), drop the traceback
            record.exc_info = None
            record.exc_text = None
            record.msg = f"{record.msg} [traceback hidden by privacy filter]"
        return True
