import argparse
import json
import os
import time
from collections import deque, Counter
from typing import Deque, Tuple, Dict, Any

import numpy as np
import torch

from src.envs.epi_env import EpiEnvConfig, SimpleEpiEnv, PredictorEnvConfig, PredictorEpiEnv
from src.models.agent import DQNAgent, DQNConfig
from src.utils.config import load_config, set_global_seed


def sample_batch(buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]], batch_size: int):
    idx = np.random.choice(len(buffer), size=min(batch_size, len(buffer)), replace=False)
    s, a, r, ns, d = zip(*(buffer[i] for i in idx))
    return (
        torch.from_numpy(np.stack(s).astype(np.float32)),
        torch.from_numpy(np.array(a).astype(np.int64)),
        torch.from_numpy(np.array(r).astype(np.float32)),
        torch.from_numpy(np.stack(ns).astype(np.float32)),
        torch.from_numpy(np.array(d).astype(np.float32)),
    )


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--out", type=str, default="models/rl_agent_q_network.pth")
    parser.add_argument("--use_predictor_env", action="store_true", help="Use frozen predictor as environment")
    parser.add_argument("--predictor_weights", type=str, default=None, help="Path to predictor weights when using predictor env")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.training.get("seed", 42)
    set_global_seed(seed)

    state_dim = cfg.rl.get("state_dim", 16)
    num_actions = cfg.rl.get("num_actions", 4)

    if args.use_predictor_env:
        series_path = cfg.dataset.get("series_file", "data/processed/series.npy")
        weights_path = args.predictor_weights or cfg.training.get("predictor_weights", "models/gnn_lstm_model.pth")
        env = PredictorEpiEnv(PredictorEnvConfig(
            series_path=series_path,
            weights_path=weights_path,
            lookback=cfg.model.get("lookback", 14),
            horizon=cfg.model.get("forecast_horizon", 7),
            state_dim=state_dim,
            num_actions=num_actions,
            seed=seed,
            alpha_cases=cfg.rl.get("alpha_cases", 1.0),
            beta_npi=cfg.rl.get("beta_npi", 0.1),
            action_costs=cfg.rl.get("action_costs", [0.0, 1.0, 2.0, 3.0]),
            graph_mode=cfg.graph.get("mode", "distance_threshold"),
            radius_km=float(cfg.graph.get("radius_km", 800.0)),
            knn_k=int(cfg.graph.get("knn_k", 8)),
            hidden_gnn=int(cfg.model.get("hidden_gnn", 32)),
            hidden_lstm=int(cfg.model.get("hidden_lstm", 64)),
            add_log1p=bool(cfg.model.get("add_log1p", True)),
            add_roll_7=bool(cfg.model.get("add_roll_7", True)),
            add_roll_14=bool(cfg.model.get("add_roll_14", False)),
        ))
    else:
        env = SimpleEpiEnv(EpiEnvConfig(num_nodes=cfg.model.get("num_nodes", 10), state_dim=state_dim, num_actions=num_actions, seed=seed))

    agent = DQNAgent(DQNConfig(state_dim=state_dim, num_actions=num_actions, lr=cfg.rl.get("lr", 1e-3)))

    buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=10000)
    episodes = cfg.rl.get("episodes", 5)
    steps_per_ep = cfg.rl.get("steps_per_episode", 50)
    batch_size = cfg.rl.get("batch_size", 32)
    epsilon = cfg.rl.get("epsilon", 0.1)

    returns = []
    action_counts: Counter = Counter()
    comp_infect_means = []
    comp_cost_means = []
    switches_per_ep = []
    start_time = time.time()
    for _ in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        last_a = None
        ep_switches = 0
        infect_comp_accum = []
        cost_comp_accum = []
        for _ in range(steps_per_ep):
            a = agent.act(s, epsilon=epsilon)
            ns, r, done, info = env.step(a)
            buffer.append((s, a, r, ns, float(done)))
            s = ns
            ep_ret += float(r)
            action_counts[a] += 1
            if last_a is not None and a != last_a:
                ep_switches += 1
            last_a = a
            # component logs if provided by env
            if isinstance(info, dict):
                if "mean_cases_eff" in info:
                    infect_comp_accum.append(float(info["mean_cases_eff"]))
                if "cost" in info:
                    cost_comp_accum.append(float(info["cost"]))
            if len(buffer) >= 8:
                batch = sample_batch(buffer, batch_size)
                agent.update(batch)
            if done:
                break
        returns.append(ep_ret)
        switches_per_ep.append(ep_switches)
        if infect_comp_accum:
            comp_infect_means.append(float(np.mean(infect_comp_accum)))
        if cost_comp_accum:
            comp_cost_means.append(float(np.mean(cost_comp_accum)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(agent.q.state_dict(), args.out)
    print(f"Saved RL agent to {args.out}")

    # Write metrics under runs/
    runs_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(runs_dir, exist_ok=True)
    metrics: Dict[str, Any] = {
        "seed": int(seed),
        "git_sha": _git_sha(),
        "episodes": int(episodes),
        "steps_per_episode": int(steps_per_ep),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "action_histogram": {int(k): int(v) for k, v in sorted(action_counts.items())},
        "mean_switches_per_episode": float(np.mean(switches_per_ep)) if switches_per_ep else 0.0,
        "comp_mean_cases_eff_mean": float(np.mean(comp_infect_means)) if comp_infect_means else None,
        "comp_cost_mean": float(np.mean(comp_cost_means)) if comp_cost_means else None,
        "alpha_cases": float(cfg.rl.get("alpha_cases", 1.0)),
        "beta_npi": float(cfg.rl.get("beta_npi", 0.1)),
    }
    with open(os.path.join(runs_dir, "rl_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()


