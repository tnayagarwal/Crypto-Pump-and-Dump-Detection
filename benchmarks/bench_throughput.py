"""
Kafka Throughput Benchmark
============================
Measures JSON serialization throughput for the streaming pipeline.
Run this to characterize the per-message overhead before tuning
MAX_POLL_RECORDS and producer batch sizes.
"""
import json
import time
import statistics

TICK_COUNTS = [100, 500, 1_000, 5_000]
MOCK_TICK = {
    "timestamp": 1712345678,
    "token_id": "TOKEN_A",
    "price_usd": 142.50,
    "volume_24h": 1_250_000.0,
    "buy_sell_ratio": 2.3,
}


def benchmark_serialization(n: int) -> float:
    """Measure JSON serialization round-trip throughput."""
    payload = json.dumps(MOCK_TICK).encode("utf-8")
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        json.loads(payload)
        times.append(time.perf_counter() - t0)
    elapsed = sum(times)
    return n / elapsed  # ticks/sec


def run_benchmarks():
    print("\n=== Kafka Serialization Throughput Benchmark ===")
    print(f"  (Full broker throughput requires: docker-compose up)\n")
    print(f"  {'Ticks':>8} | {'Rate (ticks/sec)':>17} | {'Elapsed (ms)':>14}")
    print("  " + "-" * 46)
    for n in TICK_COUNTS:
        t0 = time.perf_counter()
        rate = benchmark_serialization(n)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  {n:>8,} | {rate:>17,.0f} | {elapsed_ms:>12.1f}ms")


if __name__ == "__main__":
    run_benchmarks()
