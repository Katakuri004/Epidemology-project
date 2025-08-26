import numpy as np
import torch

from src.envs.epi_env import EpiEnvConfig, SimpleEpiEnv
from src.models.agent import DQNAgent, DQNConfig
from src.utils.config import set_global_seed


def test_dqn_determinism_fixed_seed():
    seed = 123
    set_global_seed(seed)
    env1 = SimpleEpiEnv(EpiEnvConfig(num_nodes=1, state_dim=8, num_actions=3, seed=seed))
    env2 = SimpleEpiEnv(EpiEnvConfig(num_nodes=1, state_dim=8, num_actions=3, seed=seed))
    agent1 = DQNAgent(DQNConfig(state_dim=8, num_actions=3))
    agent2 = DQNAgent(DQNConfig(state_dim=8, num_actions=3))
    s1, s2 = env1.reset(), env2.reset()
    a1, a2 = agent1.act(s1, epsilon=0.0), agent2.act(s2, epsilon=0.0)
    assert a1 == a2


