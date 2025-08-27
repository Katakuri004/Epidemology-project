import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.build_graph import build_normalized_graph, build_norm_from_coords
from src.models.gnn_lstm import GNNLSTM, LSTMOnly, GNNOnly
from src.data.dataset import TimeSeriesWindowDataset
from src.utils.config import load_config, set_global_seed


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_predictor(model_type: str, num_nodes: int, input_features: int, hidden_gnn: int, hidden_lstm: int, horizon: int):
    mt = (model_type or "gnn_lstm").lower()
    if mt == "lstm_only":
        return LSTMOnly(num_nodes=num_nodes, input_features=input_features, hidden_lstm=hidden_lstm, forecast_horizon=horizon)
    if mt == "gnn_only":
        return GNNOnly(num_nodes=num_nodes, input_features=input_features, hidden_gnn=hidden_gnn, forecast_horizon=horizon)
    return GNNLSTM(num_nodes=num_nodes, input_features=input_features, hidden_gnn=hidden_gnn, hidden_lstm=hidden_lstm, forecast_horizon=horizon)


def _infer_dims_from_checkpoint(state_dict: dict) -> dict:
    dims = {}
    try:
        if "gnn1.linear.weight" in state_dict:
            w = state_dict["gnn1.linear.weight"]
            dims["hidden_gnn"] = int(w.shape[0])
            dims["features"] = int(w.shape[1])
        if "lstm.weight_ih_l0" in state_dict:
            wih = state_dict["lstm.weight_ih_l0"]
            dims["hidden_lstm"] = int(wih.shape[0] // 4)
        if "proj.weight" in state_dict:
            pw = state_dict["proj.weight"]
            dims["horizon"] = int(pw.shape[0])
    except Exception:
        pass
    return dims

def _choose_aug_flags_for_features(base_features: int, required_features: int) -> tuple[bool, bool, bool]:
    extra = max(0, required_features - base_features)
    add_log1p = extra >= 1
    add_roll_7 = extra >= 2
    add_roll_14 = extra >= 3
    return add_log1p, add_roll_7, add_roll_14


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictor and report simple metrics")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--weights", type=str, default="models/gnn_lstm_model.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.training.get("seed", 42)
    set_global_seed(seed)

    num_nodes = cfg.model.get("num_nodes", 10)
    lookback = cfg.model.get("lookback", 14)
    # Infer feature dimension from dataset object (after augmentation) to avoid mismatch
    series_path = cfg.dataset.get("series_file", "data/processed/series.npy")
    add_log1p = cfg.model.get("add_log1p", True)
    add_roll_7 = cfg.model.get("add_roll_7", True)
    add_roll_14 = cfg.model.get("add_roll_14", False)
    hidden_gnn = cfg.model.get("hidden_gnn", 32)
    hidden_lstm = cfg.model.get("hidden_lstm", 64)
    horizon = cfg.model.get("forecast_horizon", 7)
    model_type = cfg.model.get("type", "gnn_lstm")

    adj = np.eye(num_nodes, dtype=np.float32)
    _, norm = build_normalized_graph(adj)
    norm_t = torch.from_numpy(norm.astype(np.float32))

    # Will rebuild model after creating dataset to know augmented feature size

    # If processed data exists, compute MAE/RMSE for next-step prediction on test split
    # series_path defined above
    train_ratio = cfg.dataset.get("train_ratio", 0.7)
    val_ratio = cfg.dataset.get("val_ratio", 0.15)
    if os.path.exists(series_path):
        # Infer nodes from data and rebuild adjacency/model
        tmp = np.load(series_path, mmap_mode="r")
        num_nodes = tmp.shape[1]
        # Build graph: prefer geo graph if lat/lon present
        nodes_csv = os.path.join(os.path.dirname(series_path), "nodes.csv")
        if os.path.exists(nodes_csv):
            import pandas as pd
            nodes_df = pd.read_csv(nodes_csv)
            if {"lat", "lon"}.issubset(nodes_df.columns):
                coords = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
                _, norm = build_norm_from_coords(coords, mode="distance_threshold", radius_km=800.0)
            else:
                _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
        else:
            _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
        norm_t = torch.from_numpy(norm.astype(np.float32))
        # Load checkpoint and infer dims; ensure dataset augmentation matches checkpoint features
        state_dict = torch.load(args.weights, map_location="cpu")
        inferred = _infer_dims_from_checkpoint(state_dict)
        required_features = int(inferred.get("features", 0))
        if required_features and required_features != (3 + int(add_log1p) + int(add_roll_7) + int(add_roll_14)):
            base_features = 3
            add_log1p, add_roll_7, add_roll_14 = _choose_aug_flags_for_features(base_features, required_features)
        test_ds = TimeSeriesWindowDataset(series_path, lookback, horizon, split="test", train_ratio=train_ratio, val_ratio=val_ratio, add_log1p=add_log1p, add_roll_7=add_roll_7, add_roll_14=add_roll_14)
        features = test_ds.series.shape[2]
        # Use inferred hidden sizes/horizon if available to avoid size mismatch
        hidden_gnn = int(inferred.get("hidden_gnn", hidden_gnn))
        hidden_lstm = int(inferred.get("hidden_lstm", hidden_lstm))
        horizon = int(inferred.get("horizon", horizon))
        model = _build_predictor(model_type, num_nodes, features, hidden_gnn, hidden_lstm, horizon)
        model.load_state_dict(state_dict)
        model.eval()
        dl = DataLoader(test_ds, batch_size=64, shuffle=False)
        mae_sum = 0.0
        rmse_sum = 0.0
        count = 0
        logs_dir = os.path.join(os.path.dirname(__file__), "code", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        # Aggregated per-country metrics over full test
        per_country_abs_err_sum = np.zeros((num_nodes,), dtype=np.float64)
        per_country_sq_err_sum = np.zeros((num_nodes,), dtype=np.float64)
        per_country_count = np.zeros((num_nodes,), dtype=np.float64)
        with torch.no_grad():
            for xb, yb in dl:
                pred = model(xb, norm_t)  # (B, N, H)
                pred_first = pred[:, :, 0]
                target = yb
                err = pred_first - target
                mae_sum += err.abs().sum().item()
                rmse_sum += (err.pow(2)).sum().item()
                count += err.numel()
                # Accumulate per-country
                per_country_abs_err_sum += err.abs().sum(dim=0).cpu().numpy()
                per_country_sq_err_sum += (err.pow(2)).sum(dim=0).cpu().numpy()
                per_country_count += np.array([err.shape[0]] * num_nodes, dtype=np.float64)
        mae = mae_sum / max(1, count)
        rmse = (rmse_sum / max(1, count)) ** 0.5
        per_country_mae = per_country_abs_err_sum / np.maximum(per_country_count, 1.0)
        per_country_rmse = np.sqrt(per_country_sq_err_sum / np.maximum(per_country_count, 1.0))
        # mean±sd per-node MAE
        mae_mean = float(np.mean(per_country_mae))
        mae_std = float(np.std(per_country_mae))
        # Save with header that aligns to nodes order
        nodes_csv = os.path.join(os.path.dirname(series_path), "nodes.csv")
        header = None
        if os.path.exists(nodes_csv):
            import pandas as pd
            nodes_df = pd.read_csv(nodes_csv)
            names = nodes_df.get("name", nodes_df.get("Province/State", pd.Series(range(num_nodes)))).tolist()
            header = ",".join([str(n) for n in names])
        np.savetxt(os.path.join(logs_dir, "per_country_mae.csv"), per_country_mae.reshape(1, -1), delimiter=",", fmt="%.6f", header=header or "", comments="")
        np.savetxt(os.path.join(logs_dir, "per_country_rmse.csv"), per_country_rmse.reshape(1, -1), delimiter=",", fmt="%.6f", header=header or "", comments="")
        result = {"pred_shape": (len(test_ds), num_nodes, horizon), "horizon": horizon, "nodes": num_nodes, "mae": round(mae, 6), "rmse": round(rmse, 6), "mae_per_node_mean": round(mae_mean, 6), "mae_per_node_std": round(mae_std, 6), "model_type": model_type}
        print(result)
        # Write aggregate to runs/
        runs_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(runs_dir, exist_ok=True)
        metrics = {"seed": int(seed), "git_sha": _git_sha(), **result}
        with open(os.path.join(runs_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    else:
        features = cfg.model.get("features", 3)
        model = _build_predictor(model_type, num_nodes, features, hidden_gnn, hidden_lstm, horizon)
        x = torch.randn(8, lookback, num_nodes, features)
        with torch.no_grad():
            pred = model(x, norm_t)
        print({"pred_shape": tuple(pred.shape), "horizon": horizon, "nodes": num_nodes, "model_type": model_type})


if __name__ == "__main__":
    main()


