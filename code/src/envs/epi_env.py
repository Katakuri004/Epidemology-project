from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EpiEnvConfig:
    num_nodes: int
    state_dim: int
    num_actions: int
    seed: int = 42


class SimpleEpiEnv:
    """
    Minimal placeholder environment for RL training. Dynamics are deterministic given seed.
    """

    def __init__(self, cfg: EpiEnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.state = self.rng.normal(size=(cfg.state_dim,)).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.state = self.rng.normal(size=(self.cfg.state_dim,)).astype(np.float32)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Simple linear dynamics with action effect
        noise = self.rng.normal(scale=0.01, size=self.cfg.state_dim).astype(np.float32)
        self.state = 0.99 * self.state + 0.01 * action + noise
        reward = -float(np.linalg.norm(self.state))
        done = False
        return self.state.copy(), reward, done, {}


