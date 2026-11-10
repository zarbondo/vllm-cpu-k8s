#!/usr/bin/env python3
"""Load test of an OpenAI-compatible vLLM endpoint: TTFT, latency, throughput.

Sends streaming /v1/chat/completions requests with a fixed number of
concurrent clients, for prompts of different length, and prints a markdown
table (also saved as .md and .json in --output-dir).

  python loadtest/loadtest.py --base-url http://127.0.0.1:8000 --api-key "$VLLM_API_KEY"

Scenarios are "name:input_tokens:output_tokens:concurrency[,concurrency...]".
Defaults: short:64:128:1,4,8 and long:3000:128:1,4.

Every prompt starts with a unique id, so the prefix cache cannot serve one
request from another; ignore_eos makes every answer exactly output_tokens long.

Metrics (per scenario and concurrency level):
  TTFT        time from sending the request to the first generated token
  latency     time from sending the request to the end of the stream
  TPOT        (latency - TTFT) / (output_tokens - 1), time per output token
  out tok/s   generated tokens of all requests / wall time of the level
  total tok/s (prompt + generated tokens) / wall time
  req/s       completed requests / wall time
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp

WORDS = (
    "time person year way day thing man world life hand part child eye woman place work week "
    "case point government company number group problem fact system program question night "
    "water room mother area money story month lot right study book job word business issue "
    "side kind head house service friend father power hour game line end member law car city "
    "community name president team minute idea kid body information back parent face others "
    "level office door health art war history party result change morning reason research girl "
    "guy moment air teacher force education river model network cloud server memory signal "
    "engine market price energy planet forest garden window street bridge island ocean music"
).split()

DEFAULT_SCENARIOS = ["short:64:128:1,4,8", "long:3000:128:1,4"]


@dataclass
class Scenario:
    name: str
    input_tokens: int
    output_tokens: int
    concurrency: list[int]

    @classmethod
    def parse(cls, spec: str) -> "Scenario":
        name, inp, out, conc = spec.split(":")
        return cls(name, int(inp), int(out), [int(c) for c in conc.split(",")])


@dataclass
class Result:
    ok: bool
    ttft: float = 0.0
    latency: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


@dataclass
class LevelStats:
    scenario: str
    concurrency: int
    requests: int
    failed: int
    prompt_tokens_mean: float
    output_tokens_mean: float
    ttft: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    tpot_ms_mean: float = 0.0
    output_tok_s: float = 0.0
    total_tok_s: float = 0.0
    req_s: float = 0.0
    wall_s: float = 0.0


class Client:
    def __init__(self, base_url: str, api_key: str, model: str | None, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "Client":
        connector = aiohttp.TCPConnector(limit=0)
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout, connector=connector)
        if not self.model:
            async with self.session.get(f"{self.base_url}/v1/models") as r:
                r.raise_for_status()
                self.model = (await r.json())["data"][0]["id"]
        return self

    async def __aexit__(self, *exc) -> None:
        await self.session.close()

    async def get_json(self, path: str) -> dict:
        try:
            async with self.session.get(f"{self.base_url}{path}") as r:
                return await r.json() if r.status == 200 else {}
        except (aiohttp.ClientError, ValueError):
            return {}

    async def chat(self, prompt: str, max_tokens: int) -> Result:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            # vLLM extensions: fixed answer length makes runs comparable
            "ignore_eos": True,
            "min_tokens": max_tokens,
        }
        start = time.perf_counter()
        ttft = None
        usage = {}
        try:
            async with self.session.post(f"{self.base_url}/v1/chat/completions", json=body) as r:
                if r.status != 200:
                    return Result(ok=False, error=f"HTTP {r.status}: {(await r.text())[:200]}")
                async for raw in r.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                            ttft = time.perf_counter() - start
        except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
            return Result(ok=False, error=f"{type(exc).__name__}: {exc}")
        latency = time.perf_counter() - start
        return Result(
            ok=ttft is not None,
            ttft=ttft or 0.0,
            latency=latency,
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            error="" if ttft is not None else "no tokens in the stream",
        )


class PromptFactory:
    """Builds unique prompts of a target token length.

    tokens_per_word is calibrated on the server (usage.prompt_tokens) before the run.
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.tokens_per_word = 1.0
        self.overhead = 0

    def text(self, words: int) -> str:
        return " ".join(self.rng.choice(WORDS) for _ in range(words))

    def make(self, input_tokens: int) -> str:
        head = f"Request {uuid.uuid4().hex}. Continue the story using these words:"
        n = max(1, round((input_tokens - self.overhead) / self.tokens_per_word))
        return f"{head} {self.text(n)}"

    async def calibrate(self, client: Client) -> None:
        small, big = 20, 520
        r1 = await client.chat(f"Request {uuid.uuid4().hex}. Continue the story using these words: {self.text(small)}", 1)
        r2 = await client.chat(f"Request {uuid.uuid4().hex}. Continue the story using these words: {self.text(big)}", 1)
        if r1.ok and r2.ok and r2.prompt_tokens > r1.prompt_tokens:
            self.tokens_per_word = (r2.prompt_tokens - r1.prompt_tokens) / (big - small)
            self.overhead = round(r1.prompt_tokens - small * self.tokens_per_word)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": pct(values, 0.50),
        "p90": pct(values, 0.90),
        "p99": pct(values, 0.99),
    }


async def run_level(client: Client, prompts: PromptFactory, sc: Scenario, concurrency: int, rounds: int) -> LevelStats:
    total = max(4, concurrency * rounds)
    queue: asyncio.Queue[str] = asyncio.Queue()
    for _ in range(total):
        queue.put_nowait(prompts.make(sc.input_tokens))
    results: list[Result] = []

    async def worker() -> None:
        while True:
            try:
                prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await client.chat(prompt, sc.output_tokens))

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - started

    ok = [r for r in results if r.ok]
    for r in results:
        if not r.ok:
            print(f"  request failed: {r.error}", file=sys.stderr)
    out_tokens = sum(r.output_tokens for r in ok)
    in_tokens = sum(r.prompt_tokens for r in ok)
    tpot = [(r.latency - r.ttft) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    return LevelStats(
        scenario=sc.name,
        concurrency=concurrency,
        requests=len(results),
        failed=len(results) - len(ok),
        prompt_tokens_mean=in_tokens / len(ok) if ok else 0.0,
        output_tokens_mean=out_tokens / len(ok) if ok else 0.0,
        ttft=summary([r.ttft for r in ok]),
        latency=summary([r.latency for r in ok]),
        tpot_ms_mean=statistics.fmean(tpot) * 1000 if tpot else 0.0,
        output_tok_s=out_tokens / wall,
        total_tok_s=(in_tokens + out_tokens) / wall,
        req_s=len(ok) / wall,
        wall_s=wall,
    )


def to_markdown(rows: list[LevelStats], meta: dict) -> str:
    head = (
        "| scenario | conc | req (fail) | in / out tok | TTFT mean / p50 / p90, s "
        "| latency mean / p50 / p90, s | TPOT, ms | out tok/s | total tok/s | req/s |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| {r.scenario} | {r.concurrency} | {r.requests} ({r.failed}) "
        f"| {r.prompt_tokens_mean:.0f} / {r.output_tokens_mean:.0f} "
        f"| {r.ttft['mean']:.2f} / {r.ttft['p50']:.2f} / {r.ttft['p90']:.2f} "
        f"| {r.latency['mean']:.2f} / {r.latency['p50']:.2f} / {r.latency['p90']:.2f} "
        f"| {r.tpot_ms_mean:.0f} | {r.output_tok_s:.1f} | {r.total_tok_s:.1f} | {r.req_s:.3f} |"
        for r in rows
    ]
    info = ", ".join(f"{k}: {v}" for k, v in meta.items())
    return f"_{info}_\n\n{head}" + "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> int:
    scenarios = [Scenario.parse(s) for s in (args.scenario or DEFAULT_SCENARIOS)]
    prompts = PromptFactory(args.seed)
    async with Client(args.base_url, args.api_key, args.model, args.timeout) as client:
        version = (await client.get_json("/version")).get("version", "unknown")
        print(f"model {client.model}, vLLM {version}; warm-up and prompt-length calibration...")
        await prompts.calibrate(client)
        await client.chat(prompts.make(64), 16)
        rows = []
        for sc in scenarios:
            for c in sc.concurrency:
                print(f"- {sc.name}: ~{sc.input_tokens} in / {sc.output_tokens} out tokens, concurrency {c}")
                row = await run_level(client, prompts, sc, c, args.rounds)
                print(
                    f"  TTFT p50 {row.ttft['p50']:.2f}s, latency p50 {row.latency['p50']:.2f}s, "
                    f"{row.output_tok_s:.1f} out tok/s, failed {row.failed}"
                )
                rows.append(row)

    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": client.model,
        "vllm": version,
        "server": args.base_url,
        "client": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
    }
    md = to_markdown(rows, meta)
    print("\n" + md)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"loadtest-{stamp}.md").write_text(md)
    (out / f"loadtest-{stamp}.json").write_text(json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2))
    print(f"saved to {out}/loadtest-{stamp}.{{md,json}}")
    if args.update_readme:
        update_readme(Path(args.update_readme), md)
    return 1 if any(r.failed for r in rows) else 0


README_BEGIN, README_END = "<!-- loadtest:begin -->", "<!-- loadtest:end -->"


def update_readme(path: Path, md: str) -> None:
    """Puts the table between the loadtest markers of README.md."""
    text = path.read_text(encoding="utf-8")
    start, end = text.index(README_BEGIN) + len(README_BEGIN), text.index(README_END)
    path.write_text(text[:start] + "\n" + md + text[end:], encoding="utf-8")
    print(f"table written to {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:8000"))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    p.add_argument("--model", default=None, help="served model name (default: first from /v1/models)")
    p.add_argument("--scenario", action="append", help="name:input_tokens:output_tokens:c1,c2 (repeatable)")
    p.add_argument("--rounds", type=int, default=4, help="requests per level = concurrency * rounds (min 4)")
    p.add_argument("--timeout", type=float, default=1800, help="per-request timeout, s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    p.add_argument("--update-readme", metavar="README", help="also put the table into this README.md")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
