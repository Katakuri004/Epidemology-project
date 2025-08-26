import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def run_prepare_with_cfg(tmp_path, timesteps: int, num_nodes: int, features: int = 3):
    cfg = {
        "dataset": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "nodes_file": str(tmp_path / "processed" / "nodes.csv"),
            "series_file": str(tmp_path / "processed" / "series.npy"),
            "timesteps": timesteps,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
        "model": {
            "num_nodes": num_nodes,
            "features": features,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Import module and invoke main with argv
    import importlib

    mod = importlib.import_module("prepare_data")
    argv_backup = sys.argv[:]
    try:
        sys.argv = ["prepare_data.py", "--config", str(cfg_path)]
        mod.main()
    finally:
        sys.argv = argv_backup

    return cfg


def test_prepare_data_outputs_exist_and_shapes(tmp_path):
    cfg = run_prepare_with_cfg(tmp_path, timesteps=30, num_nodes=5, features=3)
    proc_dir = Path(cfg["dataset"]["processed_dir"]) 
    nodes_path = proc_dir / "nodes.csv"
    series_path = proc_dir / "series.npy"
    scaler_path = proc_dir / "scaler.joblib"

    assert nodes_path.exists()
    assert series_path.exists()
    assert scaler_path.exists()

    nodes_df = pd.read_csv(nodes_path)
    assert "node_id" in nodes_df.columns
    assert len(nodes_df) == 5

    series = np.load(series_path)
    assert series.shape == (30, 5, 3)
    assert np.isfinite(series).all()
    # Scaled to [0,1]
    assert series.min() >= 0.0 - 1e-6
    assert series.max() <= 1.0 + 1e-6


def test_prepare_data_small_timesteps(tmp_path):
    cfg = run_prepare_with_cfg(tmp_path, timesteps=1, num_nodes=3, features=2)
    proc_dir = Path(cfg["dataset"]["processed_dir"]) 
    series = np.load(proc_dir / "series.npy")
    assert series.shape == (1, 3, 2)
    assert np.isfinite(series).all()
    assert series.min() >= 0.0 - 1e-6
    assert series.max() <= 1.0 + 1e-6


