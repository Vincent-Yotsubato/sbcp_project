# utils.py
import json
import os
import time
import random
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def save_json(path: str, data: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)


def save_npz(path: str, **kwargs) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez(path, **kwargs)


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start