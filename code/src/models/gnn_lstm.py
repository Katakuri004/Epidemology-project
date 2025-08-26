from typing import Tuple

import torch
import torch.nn as nn


class SimpleGNNLayer(nn.Module):
    """
    Minimal GCN-style layer using precomputed normalized adjacency matrix.
    Expects inputs with shape (batch, nodes, features).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F), norm_adj: (N, N)
        support = self.linear(x)  # (B, N, out_features)
        # Convert (N, N) adjacency to batched (B, N, N) and perform bmm
        if norm_adj.dim() == 2:
            norm_adj_batched = norm_adj.unsqueeze(0).expand(support.size(0), -1, -1)
        else:
            norm_adj_batched = norm_adj
        propagated = torch.bmm(norm_adj_batched, support)  # (B, N, out_features)
        return propagated


class GNNLSTM(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_features: int,
        hidden_gnn: int,
        hidden_lstm: int,
        forecast_horizon: int,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.forecast_horizon = forecast_horizon

        self.gnn1 = SimpleGNNLayer(input_features, hidden_gnn)
        self.gnn2 = SimpleGNNLayer(hidden_gnn, hidden_gnn)
        # We will run an LSTM per node by flattening (B, N) into the batch dimension
        self.lstm = nn.LSTM(input_size=hidden_gnn, hidden_size=hidden_lstm, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_lstm, forecast_horizon)

    def forward(self, x_seq: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (batch, lookback, nodes, features)
        norm_adj: (nodes, nodes)
        Returns: (batch, horizon, nodes)
        """
        b, t, n, f = x_seq.shape
        assert n == self.num_nodes, "num_nodes mismatch"
        # Apply GNN at each time independently
        x_g = x_seq.reshape(b * t, n, f)
        x_g = torch.relu(self.gnn1(x_g, norm_adj))
        x_g = torch.relu(self.gnn2(x_g, norm_adj))
        x_g = x_g.reshape(b, t, n, -1)  # (b, t, n, hidden_gnn)

        # Keep node dimension; run LSTM per node via batch flattening
        x_flat = x_g.permute(0, 2, 1, 3).contiguous().reshape(b * n, t, -1)  # (b*n, t, hidden_gnn)
        lstm_out, _ = self.lstm(x_flat)  # (b*n, t, hidden_lstm)
        last = lstm_out[:, -1, :]  # (b*n, hidden_lstm)
        out = self.proj(last)  # (b*n, horizon)
        out = out.reshape(b, n, self.forecast_horizon)  # (b, n, horizon)
        return out


def dummy_batch(num_nodes: int, lookback: int, features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(2, lookback, num_nodes, features)
    a = torch.eye(num_nodes)
    return x, a


