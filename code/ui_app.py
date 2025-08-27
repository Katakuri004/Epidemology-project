#      python -m streamlit run ui_app.py
# 
import os
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import streamlit as st

from src.models.gnn_lstm import GNNLSTM
from src.data.build_graph import build_normalized_graph, build_norm_from_coords
from src.envs.epi_env import PredictorEnvConfig, PredictorEpiEnv
from src.models.agent import DQNAgent, DQNConfig
from src.utils.config import load_config


@st.cache_data
def load_nodes(nodes_path: str) -> pd.DataFrame:
    return pd.read_csv(nodes_path)


def _infer_dims_from_state(sd: dict, fallback: dict) -> dict:
    dims = dict(fallback)
    try:
        if "gnn1.linear.weight" in sd:
            w = sd["gnn1.linear.weight"]
            dims["hidden_gnn"] = int(w.shape[0])
            dims["features"] = int(w.shape[1])
        if "lstm.weight_ih_l0" in sd:
            wih = sd["lstm.weight_ih_l0"]
            # PyTorch LSTM: weight_ih shape (4*hidden_lstm, input_size)
            dims["hidden_lstm"] = int(wih.shape[0] // 4)
        if "proj.weight" in sd:
            pw = sd["proj.weight"]
            dims["horizon"] = int(pw.shape[0])
    except Exception:
        pass
    return dims


@st.cache_resource
def load_model(weights_path: str, num_nodes: int, features: int, hidden_gnn: int, hidden_lstm: int, horizon: int):
    sd = torch.load(weights_path, map_location="cpu")
    # Guard: if this looks like an RL agent checkpoint, not a predictor, raise a clear error
    sd_keys = list(sd.keys())
    has_predictor_keys = any(k.startswith("gnn1.linear") for k in sd_keys) or ("proj.weight" in sd)
    has_rl_keys = any(k.startswith("feature.") or k.startswith("value.") or k.startswith("advantage.") for k in sd_keys)
    if has_rl_keys and not has_predictor_keys:
        raise ValueError("Selected weights appear to be an RL agent checkpoint, not a predictor model. Please select a predictor .pth (e.g., models/gnn_lstm_model.pth).")
    dims = {"features": features, "hidden_gnn": hidden_gnn, "hidden_lstm": hidden_lstm, "horizon": horizon}
    # First try cfg dims
    try:
        model = GNNLSTM(num_nodes=num_nodes, input_features=dims["features"], hidden_gnn=dims["hidden_gnn"], hidden_lstm=dims["hidden_lstm"], forecast_horizon=dims["horizon"])
        model.load_state_dict(sd)
    except Exception:
        # Fallback: infer from checkpoint
        inferred = _infer_dims_from_state(sd, dims)
        model = GNNLSTM(num_nodes=num_nodes, input_features=inferred.get("features", features), hidden_gnn=inferred.get("hidden_gnn", hidden_gnn), hidden_lstm=inferred.get("hidden_lstm", hidden_lstm), forecast_horizon=inferred.get("horizon", horizon))
        model.load_state_dict(sd)
    model.eval()
    # Return inferred dims so caller can align inputs
    final_dims = {
        "features": model.gnn1.linear.in_features,
        "hidden_gnn": model.gnn1.linear.out_features,
        "hidden_lstm": model.lstm.hidden_size,
        "horizon": model.proj.out_features,
    }
    return model, final_dims


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


def augment_series(series: np.ndarray, add_log1p: bool, add_roll_7: bool, add_roll_14: bool) -> np.ndarray:
    aug = [series]
    # log1p of first channel
    if add_log1p:
        aug.append(np.log1p(series[..., 0:1]))
    # rolling means on first channel
    def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
        c = np.cumsum(x, axis=0)
        pad = np.concatenate([np.zeros((1, c.shape[1]), dtype=np.float32), c], axis=0)
        r_valid = (pad[win:] - pad[:-win]) / float(win)
        lead = np.repeat(r_valid[:1], repeats=(x.shape[0] - r_valid.shape[0]), axis=0) if r_valid.shape[0] < x.shape[0] else np.empty((0, r_valid.shape[1]), dtype=np.float32)
        r = np.concatenate([lead, r_valid], axis=0)
        if r.shape[0] > x.shape[0]:
            r = r[-x.shape[0]:]
        return r
    if add_roll_7:
        r7 = _rolling_mean(series[..., 0], 7)
        aug.append(r7[..., None])
    if add_roll_14:
        r14 = _rolling_mean(series[..., 0], 14)
        aug.append(r14[..., None])
    return np.concatenate(aug, axis=-1).astype(np.float32)


def choose_aug_flags_for_features(base_features: int, required_features: int) -> tuple[bool, bool, bool]:
    # Base features are always present. We add in order: log1p, roll_7, roll_14
    extra = required_features - base_features
    add_log1p = extra >= 1
    add_roll_7 = extra >= 2
    add_roll_14 = extra >= 3
    return add_log1p, add_roll_7, add_roll_14


def list_files_by_ext(directory: str, extensions: List[str]) -> List[str]:
    try:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)]
        return sorted(files)
    except Exception:
        return []


def load_latest_run_metrics(runs_dir: str = "runs") -> Dict[str, Optional[str]]:
    try:
        if not os.path.exists(runs_dir):
            return {}
        subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not subdirs:
            return {}
        latest = sorted(subdirs)[-1]
        eval_path = os.path.join(latest, "eval_metrics.json")
        rl_path = os.path.join(latest, "rl_metrics.json")
        out = {}
        if os.path.exists(eval_path):
            out["eval_metrics_path"] = eval_path
        if os.path.exists(rl_path):
            out["rl_metrics_path"] = rl_path
        return out
    except Exception:
        return {}


def main():
    st.set_page_config(page_title="Epi Forecast UI", layout="wide")
    st.title("Spatiotemporal Epidemiology Forecasts")
    st.caption("Data source: JHU CSSE COVID-19 time series [link](https://github.com/CSSEGISandData/COVID-19/tree/master)")

    # Sidebar: config and weights selection
    st.sidebar.header("Setup")
    available_cfgs = list_files_by_ext("configs", [".yaml", ".yml"]) or ["configs/base.yaml"]
    cfg_path = st.sidebar.selectbox("Config", options=available_cfgs, index=available_cfgs.index("configs/base.yaml") if "configs/base.yaml" in available_cfgs else 0)
    cfg = load_config(cfg_path)
    series_path = cfg.dataset.get("series_file", "data/processed/series.npy")

    available_weights = list_files_by_ext("models", [".pth"]) or ["models/gnn_lstm_model.pth"]
    weights_path = st.sidebar.selectbox("Predictor weights (.pth)", options=available_weights, index=available_weights.index("models/gnn_lstm_model.pth") if "models/gnn_lstm_model.pth" in available_weights else 0)

    refresh = st.sidebar.button("Reload")

    if not os.path.exists(series_path) or not os.path.exists(weights_path):
        st.warning("Prepare data and train the model first.")
        st.stop()

    # Load data and model
    series_raw = np.load(series_path)
    # Optional: load dates if available for user-friendly time labels
    dates_csv = os.path.join(os.path.dirname(series_path), "dates.csv")
    dates: Optional[List[str]] = None
    if os.path.exists(dates_csv):
        try:
            dates_df = pd.read_csv(dates_csv)
            # Expect a column named 'date' as ISO string
            if 'date' in dates_df.columns:
                dates = dates_df['date'].astype(str).tolist()
        except Exception:
            dates = None
    nodes_df = load_nodes(os.path.join(os.path.dirname(series_path), "nodes.csv"))
    num_nodes = series_raw.shape[1]
    lookback = int(cfg.model.get("lookback", 14))
    # Use same augmentation flags as training
    add_log1p = bool(cfg.model.get("add_log1p", True))
    add_roll_7 = bool(cfg.model.get("add_roll_7", True))
    add_roll_14 = bool(cfg.model.get("add_roll_14", False))
    series = augment_series(series_raw, add_log1p, add_roll_7, add_roll_14)
    features = int(series.shape[2])
    hidden_gnn = int(cfg.model.get("hidden_gnn", 32))
    hidden_lstm = int(cfg.model.get("hidden_lstm", 64))
    horizon = int(cfg.model.get("forecast_horizon", 7))

    model, ckpt_dims = load_model(weights_path, num_nodes, features, hidden_gnn, hidden_lstm, horizon)
    norm_t = build_graph(series_path, cfg)

    # Sidebar: forecasting controls
    st.sidebar.header("Controls")
    # Robust name column resolution
    if "name" in nodes_df.columns:
        name_col = "name"
    elif "Country/Region" in nodes_df.columns:
        name_col = "Country/Region"
    elif "Province/State" in nodes_df.columns:
        name_col = "Province/State"
    else:
        # Fallback create a name column from index
        nodes_df = nodes_df.copy()
        nodes_df["_tmp_name"] = nodes_df.index.astype(str)
        name_col = "_tmp_name"
    options = nodes_df[name_col].tolist()
    node_name = st.sidebar.selectbox("Region", options=options)
    if "node_id" in nodes_df.columns:
        node_idx = int(nodes_df.loc[nodes_df[name_col] == node_name, "node_id"].iloc[0])
    else:
        node_idx = int(nodes_df.loc[nodes_df[name_col] == node_name].index[0])
    t_end = st.sidebar.slider("End time", min_value=lookback, max_value=int(series.shape[0]-1), value=int(series.shape[0]-1))
    if dates and len(dates) == series.shape[0]:
        st.sidebar.caption(f"Selected date: {dates[t_end]}")

    # Forecast for selected node
    # Ensure input feature dim matches checkpoint's expected input
    if ckpt_dims["features"] != features:
        # Derive augmentation flags to reach required features from base
        base_features = int(np.load(series_path, mmap_mode="r").shape[2])
        add_log1p_auto, add_roll_7_auto, add_roll_14_auto = choose_aug_flags_for_features(base_features, ckpt_dims["features"])
        series = augment_series(series_raw, add_log1p_auto, add_roll_7_auto, add_roll_14_auto)
        features = series.shape[2]
    x = series[t_end - lookback:t_end]  # (lookback, N, F)
    x = np.expand_dims(x, axis=0)  # (1, lookback, N, F)
    x_t = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        pred = model(x_t, norm_t)  # (1, N, H)
    pred_np = pred.squeeze(0).numpy()  # (N, H)
    pred_h = int(pred_np.shape[1])

    # Show plots
    import plotly.graph_objects as go

    col1, col2 = st.columns(2)
    with col1:
        hist = series[t_end - lookback:t_end, node_idx, 0]
        forecast = pred_np[node_idx]
        fig = go.Figure()
        if dates and len(dates) == series.shape[0]:
            x_hist = dates[t_end - lookback:t_end]
            x_fc = [dates[t_end-1]] + [f"t+{k}" for k in range(1, pred_h)]
            fig.add_trace(go.Scatter(x=x_hist, y=hist, name="history (new_cases, scaled)", mode="lines"))
            fig.add_trace(go.Scatter(x=list(range(lookback, lookback + pred_h)), y=forecast, name="forecast (scaled)", mode="lines+markers"))
            fig.update_xaxes(tickangle=45)
        else:
            fig.add_trace(go.Scatter(y=hist, name="history (new_cases, scaled)", mode="lines"))
            fig.add_trace(go.Scatter(x=list(range(lookback, lookback + pred_h)), y=forecast, name="forecast (scaled)", mode="lines+markers"))
        fig.update_layout(title=f"{node_name} | next {pred_h} days", xaxis_title="time", yaxis_title="scaled value (0-1)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        step = st.slider("Forecast step to rank (1..H)", 1, pred_h, min(1, pred_h))
        z = pred_np[:, min(step - 1, pred_h - 1)]
        # Use the resolved name column for display to avoid None labels
        heat_df = pd.DataFrame({"region": nodes_df[name_col].tolist(), "pred_scaled": z})
        st.dataframe(heat_df.sort_values("pred_scaled", ascending=False).head(20), use_container_width=True)
        st.caption("Values are normalized to [0,1] by training-time scaler; higher means more cases relative to training range.")

    # RL rollout (optional)
    st.subheader("Intervention Policy (RL) – optional")
    rl_cols = st.columns(3)
    with rl_cols[0]:
        rl_weights = st.selectbox("RL agent weights (.pth)", options=["(none)"] + available_weights, index=0)
    with rl_cols[1]:
        steps_ep = st.number_input("Steps per episode", min_value=10, max_value=200, value=int(cfg.rl.get("steps_per_episode", 40)))
    with rl_cols[2]:
        epsilon = st.slider("Epsilon (exploration)", 0.0, 1.0, float(cfg.rl.get("epsilon", 0.1)), 0.01)

    if rl_weights != "(none)":
        try:
            # Build predictor-driven environment using the selected predictor weights
            # Align environment feature augmentation to checkpoint input features
            base_features_env = int(np.load(series_path, mmap_mode="r").shape[2])
            add_log1p_env, add_roll_7_env, add_roll_14_env = choose_aug_flags_for_features(base_features_env, int(ckpt_dims.get("features", base_features_env)))

            penv = PredictorEpiEnv(PredictorEnvConfig(
                series_path=series_path,
                weights_path=weights_path,
                lookback=lookback,
                horizon=ckpt_dims.get("horizon", horizon),
                state_dim=int(cfg.rl.get("state_dim", 16)),
                num_actions=int(cfg.rl.get("num_actions", 4)),
                seed=int(cfg.training.get("seed", 42)),
                alpha_cases=float(cfg.rl.get("alpha_cases", 1.0)),
                beta_npi=float(cfg.rl.get("beta_npi", 0.1)),
                action_costs=cfg.rl.get("action_costs", [0.0, 1.0, 2.0, 3.0]),
                graph_mode=cfg.graph.get("mode", "distance_threshold"),
                radius_km=float(cfg.graph.get("radius_km", 800.0)),
                knn_k=int(cfg.graph.get("knn_k", 8)),
                hidden_gnn=int(ckpt_dims.get("hidden_gnn", cfg.model.get("hidden_gnn", 32))),
                hidden_lstm=int(ckpt_dims.get("hidden_lstm", cfg.model.get("hidden_lstm", 64))),
                add_log1p=bool(add_log1p_env),
                add_roll_7=bool(add_roll_7_env),
                add_roll_14=bool(add_roll_14_env),
            ))
            agent = DQNAgent(DQNConfig(
                state_dim=int(cfg.rl.get("state_dim", 16)),
                num_actions=int(cfg.rl.get("num_actions", 4)),
                lr=float(cfg.rl.get("lr", 1e-3)),
            ))
            agent.q.load_state_dict(torch.load(rl_weights, map_location="cpu"))
            agent.q.eval()

            # Run one episode
            import numpy as _np
            s = penv.reset()
            ep_ret = 0.0
            action_counts: Dict[int, int] = {}
            last_a = None
            switches = 0
            for _ in range(int(steps_ep)):
                a = agent.act(s, epsilon=epsilon)
                ns, r, done, info = penv.step(a)
                ep_ret += float(r)
                action_counts[a] = action_counts.get(a, 0) + 1
                if last_a is not None and a != last_a:
                    switches += 1
                last_a = a
                s = ns
                if done:
                    break
            m1, m2 = st.columns(2)
            with m1:
                st.write({"episode_return": round(ep_ret, 3), "switches": int(switches), "alpha": penv.alpha, "beta": penv.beta})
            with m2:
                if action_counts:
                    ah_df = pd.DataFrame({"action": list(action_counts.keys()), "count": list(action_counts.values())})
                    st.bar_chart(ah_df.set_index("action"))
            st.caption("Higher (less negative) return is better. Switches reflect policy stability. Actions are NPI levels (0=none → 3=strong).")
        except Exception as e:
            st.warning(f"RL rollout failed: {e}")

    # Context box to guide non-technical users
    with st.expander("What am I looking at?", expanded=False):
        st.markdown(
            "- The left plot shows recent history of daily new cases (scaled) for the selected region and the forecast for the next days.\n"
            "- The table ranks regions by predicted case level at a chosen forecast step.\n"
            "- The RL panel simulates an intervention policy (e.g., NPIs) learned with Deep RL against the frozen predictor; it reports expected return, policy stability, and action usage.\n"
            "- Scaled values lie in [0,1] due to normalization; compare shapes and relative magnitudes, not raw counts.\n"
        )

    # Metrics pane
    st.subheader("Metrics")
    latest = load_latest_run_metrics("runs")
    k1, k2, k3 = st.columns(3)
    # Predictor metrics
    if latest.get("eval_metrics_path"):
        try:
            import json as _json
            with open(latest["eval_metrics_path"], "r", encoding="utf-8") as f:
                em = _json.load(f)
            mae_mean = float(em.get("mae_per_node_mean", np.nan))
            mae_std = float(em.get("mae_per_node_std", np.nan))
            rmse = float(em.get("rmse", np.nan))
            horizon_k = int(em.get("horizon", 0))
            with k1:
                st.metric("MAE (per-node mean)", f"{mae_mean:.4f}", help="Lower is better")
            with k2:
                st.metric("MAE std", f"{mae_std:.4f}")
            with k3:
                st.metric("RMSE", f"{rmse:.4f}")
            st.caption(f"Forecast horizon: {horizon_k} days")
            # Optional per-region bar charts if present
            logs_dir = os.path.join(os.path.dirname(__file__), "code", "logs")
            mae_csv = os.path.join(logs_dir, "per_country_mae.csv")
            rmse_csv = os.path.join(logs_dir, "per_country_rmse.csv")
            if os.path.exists(mae_csv):
                try:
                    # Read header as names row
                    with open(mae_csv, "r", encoding="utf-8") as f:
                        header = f.readline().strip().lstrip('#').split(',') if f.readline else []
                    arr = np.loadtxt(mae_csv, delimiter=",", skiprows=1)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    vals = arr[0]
                    names = header if header and len(header) == len(vals) else [f"n{i}" for i in range(len(vals))]
                    df = pd.DataFrame({"region": names, "MAE": vals})
                    st.bar_chart(df.sort_values("MAE").head(20).set_index("region"))
                except Exception:
                    pass
        except Exception as e:
            st.caption(f"No predictor metrics available: {e}")
    # RL metrics
    if latest.get("rl_metrics_path"):
        try:
            import json as _json
            with open(latest["rl_metrics_path"], "r", encoding="utf-8") as f:
                rl = _json.load(f)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Mean episodic return", f"{rl.get('mean_return', 0.0):.2f}")
            with c2:
                st.metric("Std of return", f"{rl.get('std_return', 0.0):.2f}")
            with c3:
                st.metric("Switches/episode", f"{rl.get('mean_switches_per_episode', 0.0):.2f}")
            ah = rl.get("action_histogram", {})
            if isinstance(ah, dict) and len(ah) > 0:
                ah_df = pd.DataFrame({"action": list(ah.keys()), "count": list(ah.values())})
                st.bar_chart(ah_df.set_index("action"))
        except Exception as e:
            st.caption(f"No RL metrics available: {e}")


if __name__ == "__main__":
    main()


