import numpy as np
import torch

from src.data.build_graph import add_self_loops, build_normalized_graph, normalize_symmetric
from src.models.gnn_lstm import GNNLSTM


def test_graph_normalization_identity_on_isolated_nodes():
    adj = np.zeros((3, 3), dtype=np.float32)
    a_hat = add_self_loops(adj)
    norm = normalize_symmetric(a_hat)
    assert np.allclose(np.eye(3), norm, atol=1e-6)


def test_gnn_lstm_forward_shapes():
    num_nodes, lookback, features, horizon = 5, 4, 2, 3
    x = torch.randn(2, lookback, num_nodes, features)
    _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
    norm_t = torch.from_numpy(norm.astype(np.float32))
    model = GNNLSTM(num_nodes, features, hidden_gnn=8, hidden_lstm=16, forecast_horizon=horizon)
    y = model(x, norm_t)
    assert y.shape == (2, num_nodes, horizon)


