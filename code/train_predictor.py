import argparse
import json
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.build_graph import build_normalized_graph, build_norm_from_coords
from src.models.gnn_lstm import GNNLSTM
from src.data.dataset import TimeSeriesWindowDataset
from src.utils.config import load_config, resolve_path, set_global_seed


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def synthetic_data(num_nodes: int, lookback: int, features: int, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.randn(steps, lookback, num_nodes, features).astype(np.float32)
    y = np.random.randn(steps, num_nodes, 1).astype(np.float32)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN-LSTM predictor")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--out", type=str, default="models/gnn_lstm_model.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.training.get("seed", 42)
    set_global_seed(seed)

    lookback = cfg.model.get("lookback", 14)
    # Paths and flags
    series_path = cfg.dataset.get("series_file", "data/processed/series.npy")
    add_log1p = cfg.model.get("add_log1p", True)
    add_roll_7 = cfg.model.get("add_roll_7", True)
    add_roll_14 = cfg.model.get("add_roll_14", False)
    hidden_gnn = cfg.model.get("hidden_gnn", 32)
    hidden_lstm = cfg.model.get("hidden_lstm", 64)
    horizon = cfg.model.get("forecast_horizon", 7)

    # Prefer real dataset if available
    # series_path already defined above
    bs = cfg.training.get("batch_size", 32)
    num_workers = cfg.training.get("num_workers", 0)
    train_ratio = cfg.dataset.get("train_ratio", 0.7)
    val_ratio = cfg.dataset.get("val_ratio", 0.15)

    if os.path.exists(series_path):
        # Infer num_nodes from data to avoid mismatch
        tmp = np.load(series_path, mmap_mode="r")
        inferred_nodes = tmp.shape[1]
        num_nodes = inferred_nodes
        # Try to load lat/lon for geo graph
        nodes_csv = os.path.join(os.path.dirname(series_path), "nodes.csv")
        if os.path.exists(nodes_csv):
            try:
                import pandas as pd  # local import to keep deps minimal
                nodes_df = pd.read_csv(nodes_csv)
                if {"lat", "lon"}.issubset(nodes_df.columns):
                    coords = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
                    _, norm = build_norm_from_coords(coords, mode="distance_threshold", radius_km=800.0)
                else:
                    _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
            except Exception:
                _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
        else:
            _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
        norm_t = torch.from_numpy(norm.astype(np.float32))

        train_ds = TimeSeriesWindowDataset(series_path, lookback, horizon, split="train", train_ratio=train_ratio, val_ratio=val_ratio, fit_scaler_on_train=True, add_log1p=add_log1p, add_roll_7=add_roll_7, add_roll_14=add_roll_14)
        val_ds = TimeSeriesWindowDataset(series_path, lookback, horizon, split="val", train_ratio=train_ratio, val_ratio=val_ratio, fit_scaler_on_train=True, add_log1p=add_log1p, add_roll_7=add_roll_7, add_roll_14=add_roll_14)
        features_aug = train_ds.series.shape[2]
        model = GNNLSTM(
            num_nodes=num_nodes,
            input_features=features_aug,
            hidden_gnn=hidden_gnn,
            hidden_lstm=hidden_lstm,
            forecast_horizon=horizon,
        )
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.training.get("lr", 1e-3))
        loss_fn = nn.MSELoss()

        epochs = cfg.training.get("epochs", 5)
        patience = cfg.training.get("early_stopping_patience", 3)
        best_val = float("inf")
        stale = 0
        returns = []
        for _ in range(epochs):
            model.train()
            running = 0.0
            count = 0
            for xb, yb in train_dl:
                # xb: (B, T, N, F), yb: (N, 1) per example → make batch (B, N)
                xb = torch.nan_to_num(xb, nan=0.0, posinf=1.0, neginf=0.0)
                pred = model(xb, norm_t)  # (B, N, horizon)
                pred_first = pred[:, :, 0]
                target = yb  # (B, N) after dataset fix
                loss = loss_fn(pred_first, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                count += xb.size(0)
            train_loss = running / max(1, count)

            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = torch.nan_to_num(xb, nan=0.0, posinf=1.0, neginf=0.0)
                    pred = model(xb, norm_t)  # (B, N, horizon)
                    pred_first = pred[:, :, 0]
                    target = yb
                    loss = loss_fn(pred_first, target)
                    val_running += loss.item() * xb.size(0)
                    val_count += xb.size(0)
            val_loss = val_running / max(1, val_count)
            # Log to console
            print({"train_mse": round(train_loss, 6), "val_mse": round(val_loss, 6)})
            # Append to CSV under fixed repo path: <repo>/code/code/logs
            logs_dir = os.path.join(os.path.dirname(__file__), "code", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            with open(os.path.join(logs_dir, "predictor_metrics.csv"), "a", encoding="utf-8") as f:
                f.write(f"{train_loss},{val_loss}\n")
            returns.append(val_loss)
            # Early stopping
            if val_loss + 1e-9 < best_val:
                best_val = val_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    else:
        num_nodes = cfg.model.get("num_nodes", 10)
        adj = np.eye(num_nodes, dtype=np.float32)
        _, norm = build_normalized_graph(adj)
        norm_t = torch.from_numpy(norm.astype(np.float32))

        model = GNNLSTM(
            num_nodes=num_nodes,
            input_features=features,
            hidden_gnn=hidden_gnn,
            hidden_lstm=hidden_lstm,
            forecast_horizon=horizon,
        )
        # Fallback to synthetic data
        x, y = synthetic_data(num_nodes, lookback, features, steps=64)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y).squeeze(-1)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.training.get("lr", 1e-3))
        loss_fn = nn.MSELoss()
        epochs = cfg.training.get("epochs", 2)
        for _ in range(epochs):
            pred = model(x_t, norm_t)
            target = y_t
            pred_first = pred[:, :, 0]
            loss = loss_fn(pred_first, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved predictor to {args.out}")

    # Write metrics under runs/
    runs_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(runs_dir, exist_ok=True)
    metrics = {
        "seed": int(seed),
        "git_sha": _git_sha(),
        "epochs": int(cfg.training.get("epochs", 5)),
        "best_val_mse": float(best_val) if 'best_val' in locals() else None,
    }
    with open(os.path.join(runs_dir, "predictor_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()


