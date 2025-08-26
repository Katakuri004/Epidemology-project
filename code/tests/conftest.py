import os
import sys
import numpy as np
import torch


def pytest_runtest_setup(item):
    torch.manual_seed(0)
    np.random.seed(0)


# Ensure project root is on sys.path so `import src` works when running from repo root
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)



