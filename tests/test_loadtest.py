"""Runs loadtest/loadtest.py against a fake vLLM server with known timings."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

from aiohttp import web

SPEC = importlib.util.spec_from_file_location(
    "loadtest", Path(__file__).resolve().parents[1] / "loadtest" / "loadtest.py"
)
lt = importlib.util.module_from_spec(SPEC)
sys.modules["loadtest"] = lt  # dataclasses look the module up there
SPEC.loader.exec_module(lt)

KEY = "test-key"
PREFILL_S_PER_TOKEN = 0.0005  # 1000 tokens -> 0.5 s to first token
DECODE_S = 0.01  # per output token


def fake_vllm() -> web.Application:
    app = web.Application()

    async def models(_):
        return web.json_response({"data": [{"id": "fake-model"}]})

    async def version(_):
        return web.json_response({"version": "0.32.0"})

    async def chat(request: web.Request):
        if request.headers.get("Authorization") != f"Bearer {KEY}":
            return web.json_response({"error": "Unauthorized"}, status=401)
        body = await request.json()
        assert body["stream"] and body["ignore_eos"]
        prompt_tokens = len(body["messages"][0]["content"].split()) + 10  # "chat template"
        n = body["max_tokens"]
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        async def send(obj):
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        await send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
        await asyncio.sleep(prompt_tokens * PREFILL_S_PER_TOKEN)
        for i in range(n):
            if i:
                await asyncio.sleep(DECODE_S)
            await send({"choices": [{"index": 0, "delta": {"content": " tok"}}]})
        await send({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n}})
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app.add_routes([web.get("/v1/models", models), web.get("/version", version), web.post("/v1/chat/completions", chat)])
    return app


async def run_against_fake(tmp_path: Path, api_key: str) -> tuple[int, dict]:
    runner = web.AppRunner(fake_vllm())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    args = argparse.Namespace(
        base_url=f"http://127.0.0.1:{port}", api_key=api_key, model=None,
        scenario=["short:50:20:1,4", "long:1000:20:2"], rounds=2, timeout=60, seed=1,
        output_dir=str(tmp_path), update_readme=str(tmp_path / "README.md"),
    )
    (tmp_path / "README.md").write_text(f"# x\n{lt.README_BEGIN}\nold\n{lt.README_END}\ntail\n")
    try:
        code = await lt.main_async(args)
    finally:
        await runner.cleanup()
    (report,) = tmp_path.glob("loadtest-*.json")
    return code, json.loads(report.read_text())


def test_metrics_match_known_timings(tmp_path):
    started = time.perf_counter()
    code, report = asyncio.run(run_against_fake(tmp_path, KEY))
    assert code == 0
    rows = {(r["scenario"], r["concurrency"]): r for r in report["rows"]}
    assert set(rows) == {("short", 1), ("short", 4), ("long", 2)}

    long = rows[("long", 2)]
    # calibration makes prompts ~1000 tokens long; TTFT follows the fake prefill time
    assert 900 <= long["prompt_tokens_mean"] <= 1100
    assert 0.4 <= long["ttft"]["p50"] <= 0.75
    assert long["output_tokens_mean"] == 20
    # TPOT ~= decode delay, latency = TTFT + 19 * TPOT
    assert 8 <= long["tpot_ms_mean"] <= 20
    assert abs(long["latency"]["mean"] - (long["ttft"]["mean"] + 19 * long["tpot_ms_mean"] / 1000)) < 0.05

    short1, short4 = rows[("short", 1)], rows[("short", 4)]
    assert short1["ttft"]["p50"] < long["ttft"]["p50"]
    # 4 parallel clients against a server that does not slow down -> ~4x throughput
    assert short4["output_tok_s"] > 2.5 * short1["output_tok_s"]
    assert all(r["failed"] == 0 for r in rows.values())
    assert list(tmp_path.glob("loadtest-*.md"))
    readme = (tmp_path / "README.md").read_text()
    assert "old" not in readme and "| long | 2 |" in readme and readme.endswith("tail\n")
    assert time.perf_counter() - started < 60


def test_wrong_key_is_reported_as_failures(tmp_path):
    code, report = asyncio.run(run_against_fake(tmp_path, "wrong"))
    assert code == 1
    assert all(r["failed"] == r["requests"] for r in report["rows"])
