import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/process datasets and persist scalers")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg.dataset.get("raw_dir", "data/raw"))
    proc_dir = resolve_path(cfg.dataset.get("processed_dir", "data/processed"))
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    # Minimal placeholder: create synthetic time series and a nodes list.
    nodes = pd.DataFrame({"node_id": np.arange(cfg.model.get("num_nodes", 10))})
    nodes.to_csv(Path(proc_dir) / "nodes.csv", index=False)

    T = int(cfg.dataset.get("timesteps", 100))
    features = cfg.model.get("features", 3)
    series = np.abs(np.random.randn(T, len(nodes), features)).astype(np.float32)
    scaler = MinMaxScaler()
    flat = series.reshape(T * len(nodes), features)
    scaler.fit(flat)
    series_scaled = scaler.transform(flat).reshape(T, len(nodes), features)
    np.save(Path(proc_dir) / "series.npy", series_scaled)
    joblib.dump(scaler, Path(proc_dir) / "scaler.joblib")
    print(f"Prepared data at {proc_dir}")


if __name__ == "__main__":
    main()


