"""Unit tests for the AnomalyDetector model."""

import torch
import pytest
from src.models.gnn_lstm import AnomalyDetector, build_model


def test_model_output_shape():
    """Model output should have shape [num_nodes, num_classes]."""
    model = build_model(num_node_features=5, hidden_size=32)
    num_nodes = 4
    x = torch.randn(num_nodes, 10, 5)  # [nodes, seq_len, features]
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (num_nodes, 2), f"Expected (4, 2) but got {out.shape}"


def test_model_probabilities_sum_to_one():
    """Softmax output probabilities must sum to 1 per node."""
    model = build_model(num_node_features=5, hidden_size=32)
    x = torch.randn(3, 10, 5)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    out = model(x, edge_index)
    row_sums = out.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)


def test_null_graph_edge_case():
    """Model must not crash on a disconnected graph (no edges)."""
    model = build_model(num_node_features=5, hidden_size=32)
    x = torch.randn(2, 10, 5)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape[0] == 2
