# Benchmark Results

> Last run: May 2026 | CPU: 8-core x86-64 | Python 3.11 | PyTorch 2.1.0

## 1. Kafka Serialization Throughput

| Ticks | Rate (ticks/sec) | Elapsed |
|-------|-----------------|---------|
| 100   | ~280,000        | 0.4ms   |
| 500   | ~310,000        | 1.6ms   |
| 1,000 | ~315,000        | 3.2ms   |
| 5,000 | ~318,000        | 15.7ms  |

> Real broker limit: ~1M msgs/sec (single partition). Bottleneck is network I/O.

## 2. GNN + LSTM Forward Pass (CPU)

| Nodes | Mean (ms) | P99 (ms) |
|-------|-----------|----------|
| 10    | 2.1       | 2.6      |
| 50    | 4.7       | 5.4      |
| 100   | 9.3       | 11.1     |
| 200   | 18.3      | 20.2     |
| 500   | 44.8      | 49.6     |

> Sub 50ms at 500 nodes on CPU. GPU target: <5ms at 500 nodes.

*Reproduce: `python benchmarks/bench_throughput.py` and `python benchmarks/bench_model.py`*
