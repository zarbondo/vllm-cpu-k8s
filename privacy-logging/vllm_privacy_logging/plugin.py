"""vLLM general plugin: request logging without prompts and completions.

In vLLM (checked against 0.27 - 0.33) user content reaches the log from
``RequestLogger``:

* ``log_inputs``       - prompt text and prompt token ids (DEBUG level);
* ``log_request_body`` - the whole JSON body of the request;
* ``log_outputs``      - the generated text (INFO level, ``--enable-log-outputs``).

``register()`` replaces these methods with versions that only write metadata:
request id, token counts, a few numeric sampling parameters, finish reason and
duration. The text never becomes part of a log record, so it cannot leak
through any handler, formatter or log level.

The fact of every request is still logged (``request_received`` /
``request_finished`` lines), plus the uvicorn access log line.

A second, independent layer is ``filters.PrivacyFilter`` (see logging.json).
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any

AUDIT_LOGGER_NAME = "vllm.privacy.audit"
# Records created by this module carry this attribute; PrivacyFilter lets them through.
SAFE_RECORD_ATTR = "privacy_safe"

# RequestLogger moved between releases, both known locations are supported.
REQUEST_LOGGER_MODULES = (
    "vllm.entrypoints.serve.utils.request_logger",
    "vllm.entrypoints.logger",
)

# Only numeric sampling parameters are logged. Stop strings, logit_bias,
# structured-output schemas and similar fields are user-provided text.
SAFE_PARAM_FIELDS = ("max_tokens", "temperature", "top_p", "top_k", "n", "seed")

_PATCH_MARKER = "_privacy_logging_patched"
_SAFE = {SAFE_RECORD_ATTR: True}

audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
_log = logging.getLogger(__name__)


class RequestTracker:
    """Start time and streamed token count per request id.

    Size is bounded: entries of requests that never finish (client went away)
    are evicted oldest-first instead of piling up.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_size = max_size

    def start(self, request_id: str) -> None:
        self._items[request_id] = {"t0": time.monotonic(), "tokens": 0, "finish": None}
        self._items.move_to_end(request_id)
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)

    def _find(self, request_id: str) -> str | None:
        # Completions log inputs as "<id>-<i>" and outputs as "<id>",
        # chat with several prompts uses "<id>_<i>".
        for key in (request_id, f"{request_id}-0", f"{request_id}_0"):
            if key in self._items:
                return key
        return None

    def add_tokens(self, request_id: str, count: int, finish_reason: str | None) -> None:
        key = self._find(request_id)
        if key is None:
            return
        item = self._items[key]
        item["tokens"] += count
        if finish_reason:
            item["finish"] = finish_reason

    def finish(self, request_id: str) -> dict[str, Any] | None:
        key = self._find(request_id)
        return self._items.pop(key) if key is not None else None

    def __len__(self) -> int:
        return len(self._items)


tracker = RequestTracker()


def _params_summary(params: Any) -> str:
    if params is None:
        return ""
    parts = []
    for field in SAFE_PARAM_FIELDS:
        value = getattr(params, field, None)
        if isinstance(value, (bool, int, float)):
            parts.append(f"{field}={value}")
    return " ".join(parts)


def _prompt_len(prompt: Any, prompt_token_ids: Any, prompt_embeds: Any) -> str:
    if prompt_token_ids is not None:
        return f"prompt_tokens={len(prompt_token_ids)}"
    if prompt_embeds is not None and getattr(prompt_embeds, "shape", None):
        return f"prompt_tokens={prompt_embeds.shape[0]}"
    if prompt is not None:
        return f"prompt_chars={len(prompt)}"
    return "prompt_tokens=unknown"


def log_inputs(
    self: Any,
    request_id: str,
    prompt: Any = None,
    prompt_token_ids: Any = None,
    prompt_embeds: Any = None,
    params: Any = None,
    lora_request: Any = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    tracker.start(request_id)
    lora = getattr(lora_request, "lora_name", None)
    audit_logger.info(
        "request_received id=%s %s %s%s",
        request_id,
        _prompt_len(prompt, prompt_token_ids, prompt_embeds),
        _params_summary(params),
        f" lora={lora}" if lora else "",
        extra=_SAFE,
    )


def log_outputs(
    self: Any,
    request_id: str,
    outputs: Any = None,
    output_token_ids: Any = None,
    finish_reason: str | None = None,
    is_streaming: bool = False,
    delta: bool = False,
    *args: Any,
    **kwargs: Any,
) -> None:
    if is_streaming and delta:
        # Called for every streamed chunk: only count tokens, write nothing.
        count = len(output_token_ids) if output_token_ids is not None else 0
        tracker.add_tokens(request_id, count, finish_reason)
        return

    state = tracker.finish(request_id) or {}
    if output_token_ids is not None:
        tokens: Any = len(output_token_ids)
    elif state:
        tokens = state["tokens"]
    else:
        tokens = "unknown"
    if finish_reason == "streaming_complete":
        finish_reason = state.get("finish") or finish_reason
    duration = (
        f" duration_ms={(time.monotonic() - state['t0']) * 1000:.0f}" if state else ""
    )
    audit_logger.info(
        "request_finished id=%s output_tokens=%s finish_reason=%s stream=%s%s",
        request_id,
        tokens,
        finish_reason,
        str(bool(is_streaming)).lower(),
        duration,
        extra=_SAFE,
    )


def log_request_body(self: Any, request: Any, *args: Any, **kwargs: Any) -> None:
    # The body holds the messages, so it is never written. The request itself
    # is recorded by log_inputs.
    return None


def _patch_request_logger() -> list[str]:
    patched = []
    for module_name in REQUEST_LOGGER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, "RequestLogger", None)
        if cls is None:
            continue
        if not getattr(cls, _PATCH_MARKER, False):
            cls.log_inputs = log_inputs
            cls.log_outputs = log_outputs
            if hasattr(cls, "log_request_body"):
                cls.log_request_body = log_request_body
            setattr(cls, _PATCH_MARKER, True)
        patched.append(f"{module_name}.RequestLogger")
    return patched


def register() -> None:
    """Entry point of the ``vllm.general_plugins`` group.

    vLLM calls it in every process and possibly several times per process,
    so it has to be idempotent.
    """
    # Debug middleware that dumps response bodies: switched off for good.
    if os.environ.get("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "").lower() in ("1", "true"):
        _log.warning(
            "VLLM_DEBUG_LOG_API_SERVER_RESPONSE is ignored: privacy logging is enabled"
        )
    os.environ["VLLM_DEBUG_LOG_API_SERVER_RESPONSE"] = "false"

    try:
        patched = _patch_request_logger()
    except Exception:  # noqa: BLE001 - must never break server start-up
        _log.exception("privacy logging: failed to patch RequestLogger")
        return
    if patched:
        _log.debug("privacy logging: patched %s", ", ".join(patched))
    else:
        # PrivacyFilter still drops records of the request logger.
        _log.error("privacy logging: RequestLogger not found, relying on PrivacyFilter")
