"""
Load generator. Point it at Level 1 or Level 2 and compare the numbers.

    uvicorn level_1_fastapi:app --port 8001      # terminal 1
    python bench.py --port 8001 -c 32 -n 200

    uvicorn level_2_batching:app --port 8002     # terminal 2
    python bench.py --port 8002 -c 32 -n 200

Same model, same hardware, same images. The only difference is Level 2's
batcher. Read throughput first, then p99, then the server's own /metrics --
the story is in inference_batch_size_avg.
"""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

IMAGE = Path(__file__).parent / "assets" / "dog.jpg"


async def one(client: httpx.AsyncClient, url: str, raw: bytes, field: str) -> float | None:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, files={field: ("img.jpg", raw, "image/jpeg")})
        # A 503 is Level 2 shedding load on purpose (backpressure) -- it is a
        # successful rejection, not a crash, so it is excluded from latency.
        return (time.perf_counter() - t0) * 1000 if r.status_code == 200 else None
    except Exception:
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("-c", "--concurrency", type=int, default=32)
    ap.add_argument("-n", "--requests", type=int, default=200)
    # Levels 0-2 name the upload field "file"; the BentoML service in Level 4
    # derives it from the parameter name, which is "images".
    # Levels 0-2 name the upload field "file"; the BentoML service derives it
    # from its parameter name, which is "images".
    ap.add_argument("--field", default="file", help="multipart field name")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/predict"
    raw = IMAGE.read_bytes()
    sem = asyncio.Semaphore(args.concurrency)

    limits = httpx.Limits(max_connections=args.concurrency + 8)
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
        await one(client, url, raw, args.field)                         # warm-up

        async def guarded() -> float | None:
            async with sem:
                return await one(client, url, raw, args.field)

        print(f"{args.requests} requests, {args.concurrency} concurrent -> {url}")
        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded() for _ in range(args.requests)))
        wall = time.perf_counter() - t0

        oks = [r for r in results if r is not None]
        if not oks:
            print("all requests failed -- is the server running?")
            return

        oks.sort()
        p = lambda q: oks[min(int(len(oks) * q), len(oks) - 1)]   # noqa: E731
        print(f"""
{"-" * 46}
  ok / failed        {len(oks)} / {len(results) - len(oks)}
  wall clock         {wall:.2f} s
  throughput         {len(oks) / wall:7.1f} req/s     <-- the number that matters
  latency p50        {statistics.median(oks):7.1f} ms
  latency p95        {p(0.95):7.1f} ms
  latency p99        {p(0.99):7.1f} ms
{"-" * 46}
now: curl -s http://{args.host}:{args.port}/metrics""")


if __name__ == "__main__":
    asyncio.run(main())
