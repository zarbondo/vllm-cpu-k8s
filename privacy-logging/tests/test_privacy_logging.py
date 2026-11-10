"""Checks that prompts and completions never reach stdout/stderr.

Every scenario runs in a fresh interpreter: vLLM configures logging and loads
plugins once per process, exactly as the server does. Needs vllm and this
package installed (inside the image: `python -m pytest /opt/privacy-logging/tests`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

LOGGING_JSON = Path(__file__).resolve().parents[1] / "logging.json"
SECRET = "SECRET"

SCENARIO = textwrap.dedent(
    """
    import logging, sys
    from vllm.config.logging import LoggingConfig
    from vllm.logger import configure_logging
    # same call the API server and every engine/worker process make at start-up;
    # picks up VLLM_LOGGING_CONFIG_PATH and VLLM_LOGGING_LEVEL
    configure_logging(LoggingConfig())
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    from vllm.entrypoints.serve.utils.request_logger import RequestLogger
    from vllm.sampling_params import SamplingParams

    rl = RequestLogger(max_log_len=None)
    params = SamplingParams(max_tokens=16, temperature=0.5, stop=["SECRET_STOP"])

    class Body:
        request_id = "chatcmpl-1"
        def model_dump_json(self, **kw):
            return '{"messages": [{"role": "user", "content": "SECRET_BODY"}]}'

    # streaming chat request
    rl.log_request_body(Body())
    rl.log_inputs("chatcmpl-1", "SECRET_PROMPT", [1, 2, 3], None,
                  params=params, lora_request=None)
    rl.log_outputs("chatcmpl-1", "SECRET_DELTA_1", [7], None, is_streaming=True, delta=True)
    rl.log_outputs("chatcmpl-1", "SECRET_DELTA_2", [8, 9], "stop", is_streaming=True, delta=True)
    rl.log_outputs("chatcmpl-1", "SECRET_FULL", None, "streaming_complete",
                   is_streaming=True, delta=False)

    # non-streaming completion request (inputs are logged as <id>-<i>)
    rl.log_inputs("cmpl-2-0", "SECRET_PROMPT_2", list(range(10)), None,
                  params=params, lora_request=None)
    rl.log_outputs("cmpl-2", "SECRET_OUTPUT_2", [5, 5, 5, 5], "length",
                   is_streaming=False, delta=False)

    if "--direct" in sys.argv:
        # content written straight to loggers, bypassing RequestLogger
        logging.getLogger("vllm.entrypoints.serve.middleware.log_response").info(
            "response_body={%s}", "SECRET_RESPONSE")
        logging.getLogger("vllm.tool_parsers.hermes_tool_parser").debug(
            "tool call text: %s", "SECRET_TOOL_CALL")
        # --log-error-stack: a validation error quoting the request body
        try:
            raise ValueError("1 validation error: input_value='SECRET_IN_TRACEBACK'")
        except ValueError:
            logging.getLogger(
                "vllm.entrypoints.serve.exception_handling.handlers.validation"
            ).exception("RequestValidationError caught. Request id: %s", "chatcmpl-9")
    logging.getLogger("vllm.engine.dummy").info("ordinary record %d", 42)
    """
)


def run(tmp_path: Path, *, config: str | None, level: str, plugins: str | None,
        direct: bool = False) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env["VLLM_LOGGING_LEVEL"] = level
    if config == "privacy":
        cfg = json.loads(LOGGING_JSON.read_text())
        for section in (cfg["handlers"]["console"], cfg["loggers"]["vllm"]):
            section["level"] = level
        path = tmp_path / "logging.json"
        path.write_text(json.dumps(cfg))
        env["VLLM_LOGGING_CONFIG_PATH"] = str(path)
    if plugins is not None:
        env["VLLM_PLUGINS"] = plugins
    args = [sys.executable, "-c", SCENARIO] + (["--direct"] if direct else [])
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout + proc.stderr


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_plugin_and_filter(tmp_path, level):
    out = run(tmp_path, config="privacy", level=level, plugins=None, direct=True)
    assert SECRET not in out, out
    assert "request_received id=chatcmpl-1 prompt_tokens=3 max_tokens=16" in out
    assert "request_finished id=chatcmpl-1 output_tokens=3 finish_reason=stop stream=true" in out
    assert "request_received id=cmpl-2-0 prompt_tokens=10" in out
    assert "request_finished id=cmpl-2 output_tokens=4 finish_reason=length stream=false" in out
    assert "duration_ms=" in out
    assert "RequestValidationError caught. Request id: chatcmpl-9 [traceback hidden" in out
    assert "ordinary record 42" in out  # unrelated logs are untouched


def test_plugin_alone_with_default_vllm_logging(tmp_path):
    # no custom logging config at all: the patched RequestLogger is enough
    out = run(tmp_path, config=None, level="DEBUG", plugins=None)
    assert SECRET not in out, out
    assert "request_received id=chatcmpl-1" in out


def test_filter_alone_when_plugin_disabled(tmp_path):
    # VLLM_PLUGINS="" loads no general plugins: PrivacyFilter must still hold
    out = run(tmp_path, config="privacy", level="DEBUG", plugins="", direct=True)
    assert SECRET not in out, out
    assert "ordinary record 42" in out


UVICORN_SCENARIO = textwrap.dedent(
    """
    import json, sys, threading, time, urllib.request
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/v1/chat/completions")
    def chat():
        return {"ok": True}

    # vLLM hands the same dictConfig file to uvicorn (get_uvicorn_log_config)
    config = uvicorn.Config(app, host="127.0.0.1", port=int(sys.argv[2]),
                            log_config=json.load(open(sys.argv[1])))
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    req = urllib.request.Request(
        f"http://127.0.0.1:{sys.argv[2]}/v1/chat/completions",
        data=b'{"messages": [{"role": "user", "content": "SECRET_PROMPT"}]}',
        headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req).read()
    server.should_exit = True
    time.sleep(0.5)
    """
)


def test_uvicorn_access_log_keeps_fact_of_request(tmp_path):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.run(
        [sys.executable, "-c", UVICORN_SCENARIO, str(LOGGING_JSON), str(port)],
        capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert '"POST /v1/chat/completions HTTP/1.1" 200' in out, out
    assert SECRET not in out, out


def test_baseline_vllm_leaks(tmp_path):
    # sanity check of the test itself: stock vLLM logging does print content
    out = run(tmp_path, config=None, level="DEBUG", plugins="")
    assert "SECRET_PROMPT" in out and "SECRET_OUTPUT_2" in out
