"""
System hardware monitoring, RAM usage verification, memory safety guards,
and auto-detection of available CPU cores and GPU acceleration.
Directly inspired by memory management patterns in multi-horizon-ofi and minor-project.
"""

from __future__ import annotations

import gc
import os
import logging
from typing import Dict, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("system_memory")


def get_available_ram_gb() -> float:
    """Return available system RAM in Gigabytes (GB)."""
    if psutil is None:
        return 8.0  # Safe fallback estimate if psutil is unavailable
    return psutil.virtual_memory().available / 1e9


def get_total_ram_gb() -> float:
    """Return total system RAM in Gigabytes (GB)."""
    if psutil is None:
        return 16.0
    return psutil.virtual_memory().total / 1e9


def get_ram_usage_percent() -> float:
    """Return system RAM usage percentage (0.0 to 100.0)."""
    if psutil is None:
        return 50.0
    return psutil.virtual_memory().percent


def get_cpu_count() -> int:
    """Return optimal number of CPU worker threads."""
    count = os.cpu_count() or 2
    return max(1, count)


def get_gpu_memory_info() -> Dict[str, Any]:
    """Return GPU memory allocation and reservation metrics in GB."""
    if torch is None or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "device_count": 0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_vram_gb": 0.0,
        }

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(dev),
        "device_count": torch.cuda.device_count(),
        "allocated_gb": torch.cuda.memory_allocated(dev) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(dev) / 1e9,
        "total_vram_gb": props.total_memory / 1e9,
    }


def deep_cleanup_memory() -> None:
    """
    Perform deep memory garbage collection and release cached CUDA memory.
    Safely handles both CPU RAM and GPU VRAM cleanup.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def check_memory_pressure(critical_ram_gb: float = 1.5, max_usage_pct: float = 88.0, auto_clean: bool = True) -> Dict[str, Any]:
    """
    Check if the system is under dangerous RAM memory pressure.
    Returns status dictionary and triggers aggressive garbage collection if needed.
    """
    available_gb = get_available_ram_gb()
    usage_pct = get_ram_usage_percent()

    should_throttle = (available_gb < critical_ram_gb) or (usage_pct > max_usage_pct)

    if should_throttle and auto_clean:
        deep_cleanup_memory()
        available_gb = get_available_ram_gb()
        usage_pct = get_ram_usage_percent()
        should_throttle = (available_gb < critical_ram_gb) or (usage_pct > max_usage_pct)

    return {
        "should_throttle": should_throttle,
        "available_gb": available_gb,
        "usage_pct": usage_pct,
        "critical_ram_gb": critical_ram_gb,
        "max_usage_pct": max_usage_pct,
    }


def format_memory_summary() -> str:
    """Returns a single formatted string summarizing current RAM and GPU memory usage."""
    ram_avail = get_available_ram_gb()
    ram_tot = get_total_ram_gb()
    ram_pct = get_ram_usage_percent()
    gpu_info = get_gpu_memory_info()

    if gpu_info["cuda_available"]:
        vram_used = max(gpu_info["allocated_gb"], gpu_info["reserved_gb"])
        return (
            f"RAM: {ram_pct:.1f}% ({ram_avail:.2f}/{ram_tot:.2f} GB free) | "
            f"GPU: {gpu_info['device_name']} "
            f"({vram_used:.2f}/{gpu_info['total_vram_gb']:.2f} GB VRAM)"
        )
    return f"RAM: {ram_pct:.1f}% ({ram_avail:.2f}/{ram_tot:.2f} GB free) | GPU: None (CPU mode)"
