"""Shared helpers: seeding, config loading, JSON I/O, timing and environment capture."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


class Timer:
    """Accumulates wall-clock time per named stage (ingest / retrieve / generate)."""

    def __init__(self):
        self.records: dict[str, list[float]] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self.records.items():
            a = np.asarray(v)
            out[k] = {
                "median_ms": float(np.median(a)),
                "p95_ms": float(np.percentile(a, 95)),
                "iqr_ms": float(np.percentile(a, 75) - np.percentile(a, 25)),
                "n": int(len(a)),
            }
        return out


def environment_info() -> dict:
    """Records software/hardware details for the report (Task 3 requirement)."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    pkgs = [
        "torch", "transformers", "sentence_transformers", "peft", "bitsandbytes",
        "accelerate", "faiss", "numpy", "scikit-learn", "datasets", "bs4",
    ]
    for p in pkgs:
        mod = "sklearn" if p == "scikit-learn" else p
        try:
            m = __import__(mod)
            info[p] = getattr(m, "__version__", "installed")
        except Exception:
            info[p] = "not installed"
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        pass
    return info
