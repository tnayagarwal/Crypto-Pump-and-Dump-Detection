import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class AnomalyDetector(nn.Module):
    """
    Dual-framework model merging Temporal LSTM networks with Graph Convolutional Networks (GCN)
    to detect real-time coordinated pump and dump market manipulation.
    """
    def __init__(self, num_node_features: int, hidden_size: int, num_classes: int = 2):
        super(AnomalyDetector, self).__init__()
        
        # Temporal analysis for individual token price action
        self.lstm = nn.LSTM(input_size=num_node_features, hidden_size=hidden_size, batch_first=True)
        
        # Structural manipulation detection across crypto ecosystems
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size // 2)
        
        # Binary Classifier (0 = Normal, 1 = Pump & Dump Anomaly)
        self.classifier = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x, edge_index):
        # 1. Temporal encoding
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :] # Take the latest temporal state
        
        # 2. Graph Message Passing
        features = self.conv1(features, edge_index)
        features = torch.relu(features)
        
        features = self.conv2(features, edge_index)
        features = torch.relu(features)
        
        # 3. Anomaly Classification
        out = self.classifier(features)
        return torch.softmax(out, dim=1)
