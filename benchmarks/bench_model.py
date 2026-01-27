"""
GNN + LSTM Forward Pass Benchmark
====================================
Times model inference across varying graph sizes to characterize
computational complexity and assess GPU upgrade requirements.
"""
import time
import statistics
import torch
from src.models.gnn_lstm import build_model

GRAPH_SIZES = [10, 50, 100, 200, 500]
SEQ_LEN = 20
NUM_FEATURES = 5
RUNS = 5


def make_random_graph(num_nodes: int):
    x = torch.randn(num_nodes, SEQ_LEN, NUM_FEATURES)
    num_edges = min(num_nodes * 3, num_nodes * (num_nodes - 1))
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    return x, torch.stack([src, dst])


def benchmark_forward(num_nodes: int, model: torch.nn.Module) -> dict:
    x, edge_index = make_random_graph(num_nodes)
    latencies = []
    model.eval()
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(x, edge_index)
            latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "nodes": num_nodes,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 2),
    }


def run_benchmarks():
    print("\n=== GNN + LSTM Forward Pass Benchmark ===")
    print(f"  Runs per size: {RUNS} | Device: CPU | seq_len={SEQ_LEN}\n")
    print(f"  {'Nodes':>8} | {'Mean (ms)':>10} | {'P99 (ms)':>10}")
    print("  " + "-" * 35)
    model = build_model(num_node_features=NUM_FEATURES, hidden_size=64)
    for n in GRAPH_SIZES:
        r = benchmark_forward(n, model)
        print(f"  {r['nodes']:>8} | {r['mean_ms']:>10} | {r['p99_ms']:>10}")
    print("\n  Note: GPU inference expected to be 10-20x faster.")


if __name__ == "__main__":
    run_benchmarks()
