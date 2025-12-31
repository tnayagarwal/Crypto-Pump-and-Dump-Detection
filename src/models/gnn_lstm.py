"""
Anomaly Detector: Dual-Framework GNN + LSTM Model
==================================================
Combines temporal LSTM encoding of price sequences with graph-level
structural analysis via Graph Convolutional Networks (GCN) to classify
coordinated pump-and-dump market manipulation in real time.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")


class AnomalyDetector(nn.Module):
    """
    Dual-framework model: LSTM (temporal) + GCN (structural) for anomaly detection.

    Architecture:
        1. LSTM encodes key-sequence time-series features per node.
        2. Two GCN layers propagate messages across the token interaction graph.
        3. Linear classifier outputs a binary anomaly probability.

    Args:
        num_node_features: Number of input features per node (e.g., OHLCV = 5).
        hidden_size: Dimensionality of LSTM hidden state.
        num_classes: Output classes (2 = normal / pump-and-dump).
        dropout: Dropout rate for regularization.
    """

    def __init__(
        self,
        num_node_features: int = 5,
        hidden_size: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Temporal encoder: models each token's price-action over time
        self.lstm = nn.LSTM(
            input_size=num_node_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Structural encoder: propagates manipulation signals across the graph
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size // 2)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)

        # Binary classifier
        self.classifier = nn.Linear(hidden_size // 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node feature tensor [num_nodes, seq_len, num_node_features]
            edge_index: Graph edge connectivity [2, num_edges]

        Returns:
            Class probabilities [num_nodes, num_classes]
        """
        # 1. Temporal encoding via LSTM
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]  # Take final hidden state

        # 2. Graph-level message passing
        features = F.relu(self.conv1(features, edge_index))
        features = self.dropout(features)
        features = F.relu(self.conv2(features, edge_index))
        features = self.batch_norm(features)

        # 3. Classification
        logits = self.classifier(features)
        return F.softmax(logits, dim=1)


def build_model(num_node_features: int = 5, hidden_size: int = 64) -> AnomalyDetector:
    """Factory function to instantiate a fresh AnomalyDetector."""
    return AnomalyDetector(
        num_node_features=num_node_features,
        hidden_size=hidden_size,
    )
