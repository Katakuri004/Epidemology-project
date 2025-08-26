import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceConfig:
    lookback: int
    forecast_horizon: int


def _make_time_indices(T: int, lookback: int, horizon: int) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    # End index exclusive for input window; start >= 0 and target end < T
    for end in range(lookback, T - horizon + 1):
        start = end - lookback
        indices.append((start, end))
    return indices


class TimeSeriesWindowDataset(Dataset):
    """
    Loads processed series from numpy file of shape (T, N, F) and returns
    input windows (lookback, N, F) and the next-step target per node (N, 1).
    For simplicity, we predict only the first step of the horizon during training.
    Evaluation can roll out the full horizon if desired.
    """

    def __init__(
        self,
        series_path: str,
        lookback: int,
        forecast_horizon: int,
        split: str,
        train_ratio: float,
        val_ratio: float,
        fit_scaler_on_train: bool = True,
        add_log1p: bool = True,
        add_roll_7: bool = True,
        add_roll_14: bool = False,
    ) -> None:
        super().__init__()
        if not os.path.exists(series_path):
            raise FileNotFoundError(f"Series file not found: {series_path}")
        series = np.load(series_path)  # (T, N, F)
        # Optionally augment features: log1p(new_cases), rolling means
        aug = [series]
        # Assume original features [..., 0]=new_cases, [..., 1]=new_deaths, [..., 2]=cum_cases
        if add_log1p:
            aug.append(np.log1p(series[..., 0:1]))
        if add_roll_7:
            # Rolling mean over window with same length T by padding the first (win-1) with the first valid value
            c = np.cumsum(series[..., 0], axis=0)
            pad = np.concatenate([np.zeros((1, c.shape[1]), dtype=np.float32), c], axis=0)
            win = 7
            r_valid = (pad[win:] - pad[:-win]) / float(win)  # shape (T, N) when including leading pad below
            # Prepend the first valid value (broadcast) to keep length T
            lead = np.repeat(r_valid[:1], repeats=(series.shape[0] - r_valid.shape[0]), axis=0) if r_valid.shape[0] < series.shape[0] else np.empty((0, r_valid.shape[1]), dtype=np.float32)
            r = np.concatenate([lead, r_valid], axis=0)
            if r.shape[0] > series.shape[0]:
                r = r[-series.shape[0]:]
            aug.append(r[..., None])
        if add_roll_14:
            c = np.cumsum(series[..., 0], axis=0)
            pad = np.concatenate([np.zeros((1, c.shape[1]), dtype=np.float32), c], axis=0)
            win = 14
            r_valid = (pad[win:] - pad[:-win]) / float(win)
            lead = np.repeat(r_valid[:1], repeats=(series.shape[0] - r_valid.shape[0]), axis=0) if r_valid.shape[0] < series.shape[0] else np.empty((0, r_valid.shape[1]), dtype=np.float32)
            r = np.concatenate([lead, r_valid], axis=0)
            if r.shape[0] > series.shape[0]:
                r = r[-series.shape[0]:]
            aug.append(r[..., None])
        self.series = np.concatenate(aug, axis=-1).astype(np.float32)

        T = series.shape[0]
        horizon = forecast_horizon
        all_idx = _make_time_indices(T, lookback, horizon)

        # Time-based contiguous splits
        train_end = int(len(all_idx) * train_ratio)
        val_end = train_end + int(len(all_idx) * val_ratio)

        if split == "train":
            self.indices = all_idx[:train_end]
        elif split == "val":
            self.indices = all_idx[train_end:val_end]
        elif split == "test":
            self.indices = all_idx[val_end:]
        else:
            raise ValueError("split must be one of {'train','val','test'}")

        self.lookback = lookback
        self.horizon = forecast_horizon

        # Train-only scaling: compute scale on train split and store for all splits
        self._scaler_min: Optional[np.ndarray] = None
        self._scaler_max: Optional[np.ndarray] = None
        self._target_min: Optional[float] = None
        self._target_max: Optional[float] = None
        if fit_scaler_on_train:
            # Compute on full training window coverage (values that appear in train inputs and targets)
            train_idx = all_idx[:train_end]
            if len(train_idx) > 0:
                t0 = train_idx[0][0]
                t1 = train_idx[-1][1]  # exclusive end for inputs
                # Include the immediate next step as well
                t1 = min(series.shape[0], t1 + forecast_horizon)
                window = self.series[t0:t1]  # (T_train, N, F)
                self._scaler_min = window.min(axis=(0, 1), keepdims=True)
                self._scaler_max = window.max(axis=(0, 1), keepdims=True)
                # Target is feature index 0 (new_cases); store scalar min/max for it
                self._target_min = float(self._scaler_min[..., 0])
                self._target_max = float(self._scaler_max[..., 0])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        x = self.series[start:end]  # (lookback, N, F)
        # Predict the immediate next step (first horizon step)
        y_next = self.series[end]  # (N, F)
        # Target = next-step new_cases (feature index 0)
        y = y_next[:, 0].copy()  # (N,)

        # Apply scaling if available (broadcast over time and nodes)
        if self._scaler_min is not None and self._scaler_max is not None:
            denom = np.maximum(self._scaler_max - self._scaler_min, 1e-8)
            x = (x - self._scaler_min) / denom
            if self._target_min is not None and self._target_max is not None:
                t_denom = max(self._target_max - self._target_min, 1e-8)
                y = (y - self._target_min) / t_denom

        # Ensure numerical safety
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

        return torch.from_numpy(x), torch.from_numpy(y)


