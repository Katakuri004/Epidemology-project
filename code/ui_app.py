import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import streamlit as st

from src.models.gnn_lstm import GNNLSTM
from src.data.build_graph import build_normalized_graph, build_norm_from_coords
from src.utils.config import load_config


@st.cache_data
def load_nodes(nodes_path: str) -> pd.DataFrame:
    return pd.read_csv(nodes_path)


@st.cache_resource
def load_model(weights_path: str, num_nodes: int, features: int, hidden_gnn: int, hidden_lstm: int, horizon: int) -> GNNLSTM:
    model = GNNLSTM(num_nodes=num_nodes, input_features=features, hidden_gnn=hidden_gnn, hidden_lstm=hidden_lstm, forecast_horizon=horizon)
    sd = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


def build_graph(series_path: str, cfg) -> torch.Tensor:
    nodes_csv = os.path.join(os.path.dirname(series_path), "nodes.csv")
    series = np.load(series_path, mmap_mode="r")
    num_nodes = series.shape[1]
    if os.path.exists(nodes_csv):
        nodes_df = pd.read_csv(nodes_csv)
        if {"lat", "lon"}.issubset(nodes_df.columns):
            coords = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
            mode = cfg.graph.get("mode", "distance_threshold")
            if mode == "distance_threshold":
                _, norm = build_norm_from_coords(coords, mode=mode, radius_km=float(cfg.graph.get("radius_km", 800.0)))
            else:
                _, norm = build_norm_from_coords(coords, mode="knn", k=int(cfg.graph.get("knn_k", 8)))
            return torch.from_numpy(norm.astype(np.float32))
    _, norm = build_normalized_graph(np.eye(num_nodes, dtype=np.float32))
    return torch.from_numpy(norm.astype(np.float32))


def main():
    st.set_page_config(page_title="Epi Forecast UI", layout="wide")
    st.title("Spatiotemporal Epidemiology Forecasts")

    cfg_path = st.text_input("Config path", value="configs/base.yaml")
    cfg = load_config(cfg_path)
    series_path = cfg.dataset.get("series_file", "data/processed/series.npy")
    weights_path = st.text_input("Model weights", value="models/gnn_lstm_model.pth")

    if not os.path.exists(series_path) or not os.path.exists(weights_path):
        st.warning("Prepare data and train the model first.")
        st.stop()

    series = np.load(series_path)
    nodes_df = load_nodes(os.path.join(os.path.dirname(series_path), "nodes.csv"))
    num_nodes = series.shape[1]
    lookback = int(cfg.model.get("lookback", 14))
    features = int(series.shape[2])
    hidden_gnn = int(cfg.model.get("hidden_gnn", 32))
    hidden_lstm = int(cfg.model.get("hidden_lstm", 64))
    horizon = int(cfg.model.get("forecast_horizon", 7))

    model = load_model(weights_path, num_nodes, features, hidden_gnn, hidden_lstm, horizon)
    norm_t = build_graph(series_path, cfg)

    st.sidebar.header("Controls")
    node_name = st.sidebar.selectbox("Region", options=nodes_df.get("name", nodes_df.get("Province/State", pd.Series(range(num_nodes)))))
    node_idx = int(nodes_df[nodes_df.get("name", nodes_df.get("Province/State")) == node_name]["node_id"].iloc[0]) if "node_id" in nodes_df.columns else int(nodes_df[nodes_df.get("name", nodes_df.get("Province/State")) == node_name].index[0])
    t_end = st.sidebar.slider("End time index (T)", min_value=lookback, max_value=int(series.shape[0]-1), value=int(series.shape[0]-1))

    # Build one batch window ending at t_end
    x = series[t_end - lookback:t_end]  # (lookback, N, F)
    x = np.expand_dims(x, axis=0)  # (1, lookback, N, F)
    x_t = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        pred = model(x_t, norm_t)  # (1, N, H)
    pred_np = pred.squeeze(0).numpy()  # (N, H)

    # Show plots
    import plotly.express as px
    import plotly.graph_objects as go

    # History and forecast for selected node
    hist = series[t_end - lookback:t_end, node_idx, 0]
    forecast = pred_np[node_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist, name="history (new_cases)", mode="lines"))
    fig.add_trace(go.Scatter(x=list(range(lookback, lookback + horizon)), y=forecast, name="forecast", mode="lines+markers"))
    fig.update_layout(title=f"{node_name} | next {horizon} steps", xaxis_title="time index", yaxis_title="scaled value")
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap over nodes for first forecast step
    step = st.slider("Forecast step to map", 1, horizon, 1)
    z = pred_np[:, step - 1]
    heat_df = pd.DataFrame({"node": nodes_df.get("name", nodes_df.get("Province/State")), "pred": z})
    st.dataframe(heat_df.sort_values("pred", ascending=False).head(20), use_container_width=True)


if __name__ == "__main__":
    main()


