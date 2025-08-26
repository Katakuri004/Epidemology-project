#!/usr/bin/env python3
"""
Demonstration script for model explainability features.
Shows how to explain GNN-LSTM predictions and RL agent decisions.
"""

import argparse
import os
import numpy as np
import torch
import pandas as pd

from src.models.gnn_lstm import GNNLSTM
from src.models.agent import DQNAgent, DQNConfig
from src.data.build_graph import build_normalized_graph, build_norm_from_coords
from src.data.dataset import TimeSeriesWindowDataset
from src.utils.config import load_config, set_global_seed
from src.utils.explainability import (
    explain_gnn_attention, explain_rl_action, 
    plot_gnn_explanation, plot_rl_explanation,
    generate_explanation_report
)


def load_models_and_data(cfg_path: str, predictor_weights: str, rl_weights: str):
    """Load trained models and prepare sample data for explanation."""
    cfg = load_config(cfg_path)
    set_global_seed(cfg.training.get("seed", 42))
    
    # Load predictor model
    series_path = cfg.dataset.get("series_file", "data/processed/series.npy")
    if not os.path.exists(series_path):
        raise FileNotFoundError(f"Series file not found: {series_path}")
    
    # Create dataset to get augmented features
    train_ratio = cfg.dataset.get("train_ratio", 0.7)
    val_ratio = cfg.dataset.get("val_ratio", 0.15)
    lookback = cfg.model.get("lookback", 14)
    horizon = cfg.model.get("forecast_horizon", 7)
    
    ds = TimeSeriesWindowDataset(
        series_path, lookback, horizon, split="train",
        train_ratio=train_ratio, val_ratio=val_ratio,
        fit_scaler_on_train=True,
        add_log1p=cfg.model.get("add_log1p", True),
        add_roll_7=cfg.model.get("add_roll_7", True),
        add_roll_14=cfg.model.get("add_roll_14", False)
    )
    
    features = ds.series.shape[2]
    num_nodes = ds.series.shape[1]
    
    # Build predictor model
    model = GNNLSTM(
        num_nodes=num_nodes,
        input_features=features,
        hidden_gnn=cfg.model.get("hidden_gnn", 32),
        hidden_lstm=cfg.model.get("hidden_lstm", 64),
        forecast_horizon=horizon
    )
    model.load_state_dict(torch.load(predictor_weights, map_location="cpu"))
    model.eval()
    
    # Build graph
    nodes_csv = os.path.join(os.path.dirname(series_path), "nodes.csv")
    if os.path.exists(nodes_csv):
        nodes_df = pd.read_csv(nodes_csv)
        if {"lat", "lon"}.issubset(nodes_df.columns):
            coords = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
            _, norm = build_norm_from_coords(
                coords, 
                mode=cfg.graph.get("mode", "distance_threshold"),
                radius_km=float(cfg.graph.get("radius_km", 800.0)),
                k=int(cfg.graph.get("knn_k", 8)),
                seed=cfg.training.get("seed", 42)
            )
        else:
            _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
    else:
        _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
    
    norm_t = torch.from_numpy(norm.astype(np.float32))
    
    # Load RL agent
    state_dim = cfg.rl.get("state_dim", 16)
    num_actions = cfg.rl.get("num_actions", 4)
    agent = DQNAgent(DQNConfig(
        state_dim=state_dim,
        num_actions=num_actions,
        lr=cfg.rl.get("lr", 1e-3)
    ))
    agent.q.load_state_dict(torch.load(rl_weights, map_location="cpu"))
    agent.q.eval()
    
    # Get sample data
    sample_x, sample_y = ds[0]  # Get first sample
    sample_x = sample_x.unsqueeze(0)  # Add batch dimension
    
    # Create sample state for RL (simulate from predictor env)
    sample_state = np.random.normal(size=(state_dim,)).astype(np.float32)
    
    return model, agent, sample_x, norm_t, sample_state, nodes_df if os.path.exists(nodes_csv) else None


def main():
    parser = argparse.ArgumentParser(description="Demonstrate model explainability")
    parser.add_argument("--config", type=str, default="configs/toy.yaml")
    parser.add_argument("--predictor_weights", type=str, default="models/gnn_lstm_model.pth")
    parser.add_argument("--rl_weights", type=str, default="models/rl_agent_q_network.pth")
    parser.add_argument("--target_node", type=int, default=0, help="Node to explain predictions for")
    parser.add_argument("--action", type=int, default=0, help="Action to explain")
    parser.add_argument("--save_dir", type=str, default="explanations")
    args = parser.parse_args()
    
    print("Loading models and data...")
    model, agent, sample_x, norm_t, sample_state, nodes_df = load_models_and_data(
        args.config, args.predictor_weights, args.rl_weights
    )
    
    print(f"Model loaded: {args.predictor_weights}")
    print(f"RL agent loaded: {args.rl_weights}")
    print(f"Sample input shape: {sample_x.shape}")
    print(f"Sample state shape: {sample_state.shape}")
    
    # Get node names if available
    node_names = None
    if nodes_df is not None:
        if "name" in nodes_df.columns:
            node_names = nodes_df["name"].tolist()
        elif "Province/State" in nodes_df.columns:
            node_names = nodes_df["Province/State"].tolist()
    
    print(f"Node names: {node_names}")
    
    # Generate explanations
    print("\nGenerating GNN-LSTM explanation...")
    gnn_explanation = explain_gnn_attention(model, sample_x, norm_t, args.target_node)
    
    print("Generating RL agent explanation...")
    rl_explanation = explain_rl_action(agent, sample_state, args.action)
    
    # Print summary
    print(f"\n=== GNN-LSTM Explanation Summary ===")
    print(f"Target node: {node_names[args.target_node] if node_names else args.target_node}")
    print(f"Prediction horizon: {gnn_explanation['target_prediction']}")
    print(f"Most important feature: {np.argmax(gnn_explanation['feature_importance'])}")
    print(f"Most influential neighbor: {np.argmax(gnn_explanation['neighbor_influence'])}")
    
    print(f"\n=== RL Agent Explanation Summary ===")
    print(f"Chosen action: {args.action}")
    print(f"Q-value for chosen action: {rl_explanation['chosen_q']:.3f}")
    print(f"Action confidence: {rl_explanation['action_confidence']:.3f}")
    print(f"Most important state dimension: {np.argmax(rl_explanation['state_importance'])}")
    
    # Generate plots
    print(f"\nGenerating explanation plots...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    gnn_plot_path = os.path.join(args.save_dir, "gnn_explanation.png")
    rl_plot_path = os.path.join(args.save_dir, "rl_explanation.png")
    
    plot_gnn_explanation(gnn_explanation, node_names, gnn_plot_path)
    plot_rl_explanation(rl_explanation, save_path=rl_plot_path)
    
    print(f"Plots saved to:")
    print(f"  GNN explanation: {gnn_plot_path}")
    print(f"  RL explanation: {rl_plot_path}")
    
    # Save explanation data as JSON
    import json
    explanation_data = {
        'gnn_explanation': {
            'target_prediction': gnn_explanation['target_prediction'].tolist(),
            'feature_importance': gnn_explanation['feature_importance'].tolist(),
            'neighbor_influence': gnn_explanation['neighbor_influence'].tolist(),
            'node_names': node_names
        },
        'rl_explanation': {
            'q_values': rl_explanation['q_values'].tolist(),
            'chosen_q': rl_explanation['chosen_q'],
            'action_confidence': rl_explanation['action_confidence'],
            'state_importance': rl_explanation['state_importance'].tolist()
        }
    }
    
    json_path = os.path.join(args.save_dir, "explanation_data.json")
    with open(json_path, 'w') as f:
        json.dump(explanation_data, f, indent=2)
    print(f"Explanation data saved to: {json_path}")


if __name__ == "__main__":
    main()
