import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw.get("dataset", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def graph(self) -> Dict[str, Any]:
        return self.raw.get("graph", {})

    @property
    def rl(self) -> Dict[str, Any]:
        return self.raw.get("rl", {})


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data or {})


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    anchor = base_dir or os.getcwd()
    return os.path.normpath(os.path.join(anchor, path))


