"""
Atomic Checkpointing and Resumption Engine.
Provides crash-resilient persistence for pipeline stages, chunked inference,
and trained models, preventing data loss during Google Colab disconnections or timeouts.
Directly inspired by checkpointing patterns in multi-horizon-ofi and minor-project.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import datetime

logger = logging.getLogger("checkpoint")


def atomic_save_json(data: Any, filepath: Union[Path, str], indent: int = 2) -> None:
    """
    Atomically persist a JSON-serializable dictionary to disk.
    Writes first to a `.tmp` file and replaces the destination file atomically,
    ensuring no half-written or corrupted files occur during unexpected crashes.
    """
    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, target_path)


def load_json(filepath: Union[Path, str]) -> Optional[Dict[str, Any]]:
    """Safely load JSON file if it exists, otherwise return None."""
    target_path = Path(filepath).resolve()
    if not target_path.is_file():
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read checkpoint at {target_path}: {e}")
        return None


def get_checkpoint_dir(checkpoint_root: Union[Path, str]) -> Path:
    """Return the checkpoint directory, ensuring it exists."""
    ckpt_dir = Path(checkpoint_root).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def is_stage_completed(checkpoint_root: Union[Path, str], stage_name: str) -> bool:
    """Check if a specific pipeline stage has already completed successfully."""
    ckpt_dir = get_checkpoint_dir(checkpoint_root)
    marker = ckpt_dir / f"{stage_name}.done"
    return marker.is_file()


def mark_stage_completed(checkpoint_root: Union[Path, str], stage_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Mark a pipeline stage as completed, atomically saving metadata."""
    ckpt_dir = get_checkpoint_dir(checkpoint_root)
    marker = ckpt_dir / f"{stage_name}.done"
    payload = {
        "stage": stage_name,
        "completed_at": datetime.datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    atomic_save_json(payload, marker)


def save_chunk_progress(checkpoint_root: Union[Path, str], chunk_idx: int, total_chunks: int, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save progress for chunked processing so interrupted runs can resume from exact chunk."""
    ckpt_dir = get_checkpoint_dir(checkpoint_root)
    progress_file = ckpt_dir / "chunk_progress.json"
    data = {
        "last_completed_chunk": chunk_idx,
        "total_chunks": total_chunks,
        "updated_at": datetime.datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    atomic_save_json(data, progress_file)


def load_chunk_progress(checkpoint_root: Union[Path, str]) -> Optional[Dict[str, Any]]:
    """Load chunk progress to resume processing."""
    ckpt_dir = get_checkpoint_dir(checkpoint_root)
    progress_file = ckpt_dir / "chunk_progress.json"
    return load_json(progress_file)


def clear_checkpoint_stage(checkpoint_root: Union[Path, str], stage_name: str) -> None:
    """Remove a completion marker if re-running a stage."""
    ckpt_dir = get_checkpoint_dir(checkpoint_root)
    marker = ckpt_dir / f"{stage_name}.done"
    if marker.is_file():
        marker.unlink()
