from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_actions))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        f = self.feature(state)
        v = self.value(f)
        a = self.advantage(f)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


@dataclass
class DQNConfig:
    state_dim: int
    num_actions: int
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.01


class DQNAgent:
    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.q = DuelingQNetwork(cfg.state_dim, cfg.num_actions)
        self.target = DuelingQNetwork(cfg.state_dim, cfg.num_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.cfg.num_actions)
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        q = self.q(s)
        return int(q.argmax(dim=1).item())

    def update(self, batch: Tuple[torch.Tensor, ...]) -> float:
        states, actions, rewards, next_states, dones = batch
        q_values = self.q(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            target = rewards + self.cfg.gamma * next_q * (1.0 - dones)
        loss = torch.nn.functional.mse_loss(q_values, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)
        self.opt.step()
        # soft update
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
        return float(loss.item())


