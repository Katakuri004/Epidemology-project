import argparse
import os
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch

from src.envs.epi_env import EpiEnvConfig, SimpleEpiEnv
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--out", type=str, default="models/rl_agent_q_network.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.training.get("seed", 42)
    set_global_seed(seed)

    state_dim = cfg.rl.get("state_dim", 16)
    num_actions = cfg.rl.get("num_actions", 4)

    env = SimpleEpiEnv(EpiEnvConfig(num_nodes=cfg.model.get("num_nodes", 10), state_dim=state_dim, num_actions=num_actions, seed=seed))
    agent = DQNAgent(DQNConfig(state_dim=state_dim, num_actions=num_actions, lr=cfg.rl.get("lr", 1e-3)))

    buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=10000)
    episodes = cfg.rl.get("episodes", 5)
    steps_per_ep = cfg.rl.get("steps_per_episode", 50)
    batch_size = cfg.rl.get("batch_size", 32)
    epsilon = cfg.rl.get("epsilon", 0.1)

    for _ in range(episodes):
        s = env.reset()
        for _ in range(steps_per_ep):
            a = agent.act(s, epsilon=epsilon)
            ns, r, done, _ = env.step(a)
            buffer.append((s, a, r, ns, float(done)))
            s = ns
            if len(buffer) >= 8:
                batch = sample_batch(buffer, batch_size)
                agent.update(batch)
            if done:
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(agent.q.state_dict(), args.out)
    print(f"Saved RL agent to {args.out}")


if __name__ == "__main__":
    main()


