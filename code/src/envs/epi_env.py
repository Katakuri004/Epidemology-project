from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch

from src.models.gnn_lstm import GNNLSTM
from src.data.build_graph import build_normalized_graph, build_norm_from_coords
from src.data.dataset import TimeSeriesWindowDataset


@dataclass
class EpiEnvConfig:
    num_nodes: int
    state_dim: int
    num_actions: int
    seed: int = 42


class SimpleEpiEnv:
    """
    Minimal placeholder environment for RL training. Dynamics are deterministic given seed.
    """

    def __init__(self, cfg: EpiEnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.state = self.rng.normal(size=(cfg.state_dim,)).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.state = self.rng.normal(size=(self.cfg.state_dim,)).astype(np.float32)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Simple linear dynamics with action effect
        noise = self.rng.normal(scale=0.01, size=self.cfg.state_dim).astype(np.float32)
        self.state = 0.99 * self.state + 0.01 * action + noise
        reward = -float(np.linalg.norm(self.state))
        done = False
        return self.state.copy(), reward, done, {}


@dataclass
class PredictorEnvConfig:
    series_path: str
    weights_path: str
    lookback: int
    horizon: int
    state_dim: int
    num_actions: int
    seed: int = 42
    alpha_cases: float = 1.0
    beta_npi: float = 0.1
    action_costs: Optional[list[float]] = None
    graph_mode: str = "distance_threshold"
    radius_km: float = 800.0
    knn_k: int = 8
    hidden_gnn: int = 32
    hidden_lstm: int = 64
    add_log1p: bool = True
    add_roll_7: bool = True
    add_roll_14: bool = False


class PredictorEpiEnv:
    """
    Model-based simulator: frozen GNN-LSTM predicts next cases given last lookback window and action.
    Reward = -alpha * mean(pred_cases) - beta * action_cost[action].
    State = last 'state_dim' features from aggregated window stats.
    """

    def __init__(self, cfg: PredictorEnvConfig):
        import os
        import pandas as pd

        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        series_np = np.load(cfg.series_path, mmap_mode="r")  # (T, N, F_raw)
        # Use the same augmentation as training via dataset helper to ensure feature dim matches
        # Build a small dataset instance to get the augmented series and scalers
        # We can use train split just to initialize
        ds = TimeSeriesWindowDataset(
            cfg.series_path,
            cfg.lookback,
            cfg.horizon,
            split="train",
            train_ratio=0.7,
            val_ratio=0.15,
            fit_scaler_on_train=True,
            add_log1p=cfg.add_log1p,
            add_roll_7=cfg.add_roll_7,
            add_roll_14=cfg.add_roll_14,
        )
        self.series = ds.series.astype(np.float32)  # augmented (T, N, F_aug)
        self.T, self.N, self.F = self.series.shape
        nodes_csv = os.path.join(os.path.dirname(cfg.series_path), "nodes.csv")
        if os.path.exists(nodes_csv):
            nodes_df = pd.read_csv(nodes_csv)
            if {"lat", "lon"}.issubset(nodes_df.columns):
                coords = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
                _, norm = build_norm_from_coords(coords, mode=self.cfg.graph_mode, radius_km=self.cfg.radius_km, k=self.cfg.knn_k, seed=self.cfg.seed)
            else:
                _, norm = build_normalized_graph(np.eye(self.N, dtype=np.float32))
        else:
            _, norm = build_normalized_graph(np.eye(self.N, dtype=np.float32))
        self.norm_t = torch.from_numpy(norm.astype(np.float32))

        # Load frozen predictor
        features = self.F
        hidden_gnn = int(getattr(cfg, "hidden_gnn", 32)) if hasattr(cfg, "hidden_gnn") else 32
        hidden_lstm = int(getattr(cfg, "hidden_lstm", 64)) if hasattr(cfg, "hidden_lstm") else 64
        self.model = GNNLSTM(
            num_nodes=self.N,
            input_features=features,
            hidden_gnn=hidden_gnn,
            hidden_lstm=hidden_lstm,
            forecast_horizon=cfg.horizon,
        )
        sd = torch.load(cfg.weights_path, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.alpha = float(cfg.alpha_cases)
        self.beta = float(cfg.beta_npi)
        self.action_costs = np.array(cfg.action_costs if cfg.action_costs is not None else [0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        # Rolling window indices
        self.t = 0
        self.window: np.ndarray = np.zeros((cfg.lookback, self.N, self.F), dtype=np.float32)

        self.state_dim = cfg.state_dim
        self.num_actions = cfg.num_actions

    def _encode_state(self) -> np.ndarray:
        K = min(self.window.shape[0], self.state_dim)
        if K <= 0:
            return np.zeros((self.state_dim,), dtype=np.float32)
        vals = self.window[-K:, :, 0].mean(axis=1)
        flat = vals.astype(np.float32).ravel()
        if flat.size >= self.state_dim:
            return flat[: self.state_dim].copy()
        pad = np.zeros((self.state_dim - flat.size,), dtype=np.float32)
        return np.concatenate([flat, pad], axis=0)

    def reset(self) -> np.ndarray:
        max_start = max(1, self.T - (self.cfg.lookback + self.cfg.horizon + 1))
        self.t = int(self.rng.integers(low=self.cfg.lookback, high=max_start))
        self.window = self.series[self.t - self.cfg.lookback : self.t].copy()
        return self._encode_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        with torch.no_grad():
            x = torch.from_numpy(self.window[None, ...])  # (1, L, N, F)
            pred = self.model(x, self.norm_t)[0]  # (N, H)
            next_cases = pred[:, 0].cpu().numpy().astype(np.float32)

        a = int(action)
        cost = self.action_costs[a] if 0 <= a < len(self.action_costs) else float(a)
        damp = 1.0 / (1.0 + 0.1 * cost)
        next_cases_eff = next_cases * damp

        next_step = self.window[-1].copy()
        next_step[:, 0] = next_cases_eff
        self.window = np.concatenate([self.window[1:], next_step[None, ...]], axis=0)
        self.t += 1
        done = False

        reward = -self.alpha * float(next_cases_eff.mean()) - self.beta * float(cost)
        state = self._encode_state()
        info = {"mean_pred_cases": float(next_cases.mean()), "mean_cases_eff": float(next_cases_eff.mean()), "cost": float(cost), "action": a}
        return state, reward, done, info


