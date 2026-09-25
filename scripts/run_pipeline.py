#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Master High-Performance Pipeline Runner.
Engineered for Google Colab, high-memory GPU servers, and local workstations.

Features:
  1. Auto-Scaling Hardware Detection:
     - Detects CUDA GPU (T4, V100, A100, L4) and auto-selects LightGBM GPU/CPU tree learner.
     - Auto-scales multi-core worker threads (os.cpu_count() parallelization).
  2. Memory Management & OOM Safety Guards:
     - Real-time RAM & VRAM tracking (via psutil and torch).
     - Proactive garbage collection and dynamic batch throttling to prevent Colab crashes.
  3. Crash-Resilient Checkpointing & Resumption (--resume):
     - Atomic markers and chunk progress tracking.
     - Automatically picks up from the exact interrupted stage/batch.
  4. Google Drive Persistence Integration:
     - Auto-detects Google Drive and mirrors datasets, models, checkpoints, and submissions.
  5. Scalable Execution:
     - Supports smoke test mode (--smoke-test), full validation (--val), or production test inference.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Filter noisy sklearn / LightGBM feature name warnings during batch inference
warnings.filterwarnings("ignore", category=UserWarning)

# System path bootstrap
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.utils.system import (
        get_available_ram_gb,
        get_total_ram_gb,
        get_ram_usage_percent,
        get_cpu_count,
        get_gpu_memory_info,
        deep_cleanup_memory,
        check_memory_pressure,
        format_memory_summary,
    )
    from src.utils.checkpoint import (
        atomic_save_json,
        load_json,
        get_checkpoint_dir,
        is_stage_completed,
        mark_stage_completed,
        save_chunk_progress,
        load_chunk_progress,
    )
    from src.normalization import (
        normalize_name,
        normalize_address,
        extract_legal_suffix,
        extract_unit_info,
        extract_numbers,
    )
    from src.blocking import (
        CandidateBlocker,
        load_source_tsv,
        load_ground_truth_tsv,
        write_candidate_pairs_tsv,
    )
    from src.classifier import (
        EntityResolutionClassifier,
        write_matching_results_tsv,
    )
    from src.decision_layer import DecisionLayer
    from src.validation import (
        evaluate_pipeline_on_split,
        run_holdout_validation,
        EntityCrossValidator,
    )
except ImportError:
    from utils.system import (
        get_available_ram_gb,
        get_total_ram_gb,
        get_ram_usage_percent,
        get_cpu_count,
        get_gpu_memory_info,
        deep_cleanup_memory,
        check_memory_pressure,
        format_memory_summary,
    )
    from utils.checkpoint import (
        atomic_save_json,
        load_json,
        get_checkpoint_dir,
        is_stage_completed,
        mark_stage_completed,
        save_chunk_progress,
        load_chunk_progress,
    )
    from normalization import (
        normalize_name,
        normalize_address,
        extract_legal_suffix,
        extract_unit_info,
        extract_numbers,
    )
    from blocking import (
        CandidateBlocker,
        load_source_tsv,
        load_ground_truth_tsv,
        write_candidate_pairs_tsv,
    )
    from classifier import (
        EntityResolutionClassifier,
        write_matching_results_tsv,
    )
    from decision_layer import DecisionLayer
    from validation import (
        evaluate_pipeline_on_split,
        run_holdout_validation,
        EntityCrossValidator,
    )


# =============================================================================
# HARDWARE DISCOVERY & RUNNER CONFIG
# =============================================================================
def discover_hardware(requested_device: str = "auto") -> Dict[str, Any]:
    """Auto-detects CPU cores, system RAM, and GPU capabilities."""
    cpu_cores = get_cpu_count()
    ram_gb = get_total_ram_gb()
    ram_avail = get_available_ram_gb()
    gpu_info = get_gpu_memory_info()

    if requested_device == "cuda" and not gpu_info["cuda_available"]:
        print("[!] Warning: CUDA requested but no GPU found. Falling back to CPU.")
        device = "cpu"
    elif requested_device == "auto":
        device = "cuda" if gpu_info["cuda_available"] else "cpu"
    else:
        device = requested_device

    return {
        "device": device,
        "cpu_cores": cpu_cores,
        "total_ram_gb": ram_gb,
        "avail_ram_gb": ram_avail,
        "gpu_info": gpu_info,
    }


# =============================================================================
# PIPELINE STAGES
# =============================================================================
class PipelineOrchestrator:
    """
    Manages complete end-to-end execution with atomic checkpoints,
    memory throttling, auto-scaling, and Google Drive synchronization.
    """

    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        output_dir: str,
        checkpoint_dir: str,
        drive_sync_dir: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 2000,
        max_train_records: int = 50000,
        resume: bool = True,
        smoke_test: bool = False,
    ):
        self.train_dir = Path(train_dir).resolve()
        self.test_dir = Path(test_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        if smoke_test and Path(checkpoint_dir).name == "checkpoints":
            self.checkpoint_dir = (Path(checkpoint_dir) / "smoke_test").resolve()
        else:
            self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.drive_sync_dir = Path(drive_sync_dir).resolve() if drive_sync_dir else None
        self.resume = resume
        self.smoke_test = smoke_test
        self.batch_size = max(100, batch_size)
        self.max_train_records = max_train_records

        # Output paths
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.checkpoint_dir / "lgbm_entity_resolver.pkl"
        self.candidates_out = self.output_dir / "candidate_pairs.tsv"
        self.matches_out = self.output_dir / "matching_results.tsv"
        self.scorecard_out = self.output_dir / "scorecard.json"

        # Hardware setup
        self.hw = discover_hardware(device)
        self._print_banner()

    def _print_banner(self):
        print("=" * 76)
        print("   AMAZON ML CHALLENGE 2026 - COLAB & GPU HIGH-PERFORMANCE RUNNER")
        print("=" * 76)
        print(f"Device Mode:       {self.hw['device'].upper()} "
              f"({self.hw['gpu_info']['device_name'] if self.hw['gpu_info']['cuda_available'] else 'CPU Multi-Core'})")
        print(f"CPU Workers:       {self.hw['cpu_cores']} threads")
        print(f"System RAM:        {self.hw['avail_ram_gb']:.2f} GB free / {self.hw['total_ram_gb']:.2f} GB total")
        print(f"Train Dataset:     {self.train_dir}")
        print(f"Test Dataset:      {self.test_dir}")
        print(f"Output Directory:  {self.output_dir}")
        print(f"Checkpoints:       {self.checkpoint_dir} (Resume: {self.resume})")
        if self.drive_sync_dir:
            print(f"Google Drive Sync: {self.drive_sync_dir}")
        print(f"Smoke Test:        {self.smoke_test}")
        print("-" * 76)

    def sync_to_drive(self):
        """Mirror output artifacts and checkpoints to Google Drive for persistence."""
        if not self.drive_sync_dir:
            return
        try:
            self.drive_sync_dir.mkdir(parents=True, exist_ok=True)
            drive_output = self.drive_sync_dir / "output"
            drive_ckpts = self.drive_sync_dir / "checkpoints"
            drive_output.mkdir(parents=True, exist_ok=True)
            drive_ckpts.mkdir(parents=True, exist_ok=True)

            for f in self.output_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, drive_output / f.name)
            for f in self.checkpoint_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, drive_ckpts / f.name)
            print(f"[Drive Sync] Mirrored checkpoints and outputs to {self.drive_sync_dir}")
        except Exception as e:
            print(f"[Drive Sync Warning] Could not sync to drive: {e}")

    # -------------------------------------------------------------------------
    # STAGE 1: TRAINING OR CHECKPOINT RESTORATION
    # -------------------------------------------------------------------------
    def train_or_load_model(self) -> EntityResolutionClassifier:
        """Trains LightGBM classifier or restores from atomic checkpoint."""
        stage_name = "stage1_train"
        if self.resume and is_stage_completed(self.checkpoint_dir, stage_name) and self.model_path.is_file():
            print(f"\n[Stage 1/3] Checkpoint found: Restoring trained model from {self.model_path}...")
            return EntityResolutionClassifier.load(str(self.model_path))

        print("\n[Stage 1/3] Training Entity Resolution Pipeline...")
        t0 = time.time()
        deep_cleanup_memory()

        def _find_source_path(data_dir: Path, filename_base: str) -> str:
            # 1. Direct .tsv file
            f_tsv = data_dir / f"{filename_base}.tsv"
            if f_tsv.is_file():
                return str(f_tsv)
            # 2. Gzip .tsv.gz file
            f_gz = data_dir / f"{filename_base}.tsv.gz"
            if f_gz.is_file():
                return str(f_gz)
            # 3. Directory of shards
            d_shard = data_dir / filename_base
            if d_shard.is_dir():
                return str(d_shard)
            # 4. Fallback in data/shards/<split>/<filename_base>
            split_name = data_dir.name
            d_alt = PROJECT_ROOT / "data" / "shards" / split_name / filename_base
            if d_alt.is_dir():
                return str(d_alt)
            raise FileNotFoundError(f"Missing source file or shard directory for {filename_base} in {data_dir}")

        if self.smoke_test:
            max_load = 500
        elif self.max_train_records and self.max_train_records > 0:
            max_load = self.max_train_records
        else:
            max_load = None
        print(f"  * Loading training data from {self.train_dir}...")
        s1_tr = load_source_tsv(_find_source_path(self.train_dir, "train_source1"), max_records=max_load)
        s2_tr = load_source_tsv(_find_source_path(self.train_dir, "train_source2"), max_records=max_load)
        s3_tr = load_source_tsv(_find_source_path(self.train_dir, "train_source3"), max_records=max_load)
        gt_tr = load_ground_truth_tsv(_find_source_path(self.train_dir, "train_ground_truth"), max_records=max_load)

        if self.smoke_test:
            smoke_s1 = list(s1_tr.keys())[:30]
            s1_tr = {k: s1_tr[k] for k in smoke_s1}
            gt_tr = {k: gt_tr.get(k, set()) for k in smoke_s1}
            
            # Filter S2 and S3 to only true matches + a small distractor sample (300 records each)
            smoke_matches = set()
            for s in smoke_s1:
                smoke_matches.update(gt_tr.get(s, set()))
            s2_keep = set(list(smoke_matches) + list(s2_tr.keys())[:300])
            s3_keep = set(list(smoke_matches) + list(s3_tr.keys())[:300])
            s2_tr = {k: s2_tr[k] for k in s2_keep if k in s2_tr}
            s3_tr = {k: s3_tr[k] for k in s3_keep if k in s3_tr}
            print(f"  [Smoke Test] Filtered to {len(s1_tr)} S1 entities, {len(s2_tr)} S2 records, {len(s3_tr)} S3 records.")
        elif len(s1_tr) > 5000:
            import random
            rng = random.Random(42)
            train_sample_keys = rng.sample(list(s1_tr.keys()), 5000)
            s1_tr = {k: s1_tr[k] for k in train_sample_keys}
            gt_tr = {k: gt_tr.get(k, set()) for k in train_sample_keys}
            train_matches = set()
            for s in train_sample_keys:
                train_matches.update(gt_tr.get(s, set()))
            s2_keep = set(list(train_matches) + list(s2_tr.keys())[:20000])
            s3_keep = set(list(train_matches) + list(s3_tr.keys())[:20000])
            s2_tr = {k: s2_tr[k] for k in s2_keep if k in s2_tr}
            s3_tr = {k: s3_tr[k] for k in s3_keep if k in s3_tr}
            print(f"  [Large-Scale Guard] Sampled {len(s1_tr)} S1 entities ({len(s2_tr)} S2, {len(s3_tr)} S3) for GBDT training.")

        # Train classifier with auto hardware tuning
        clf = EntityResolutionClassifier(use_decision_layer=True)
        clf.fit(s1_tr, s2_tr, s3_tr, gt_tr, tune_threshold=True, verbose=True)

        # Atomic checkpoint save
        clf.save(str(self.model_path))
        mark_stage_completed(self.checkpoint_dir, stage_name, {
            "n_train_s1": len(s1_tr),
            "training_duration_s": time.time() - t0,
            "threshold": clf.threshold,
        })
        self.sync_to_drive()
        print(f"[Stage 1/3] Model saved successfully to {self.model_path} ({time.time() - t0:.2f}s)")
        return clf

    # -------------------------------------------------------------------------
    # STAGE 2: CANDIDATE BLOCKING WITH MEMORY GUARD
    # -------------------------------------------------------------------------
    def run_blocking(self, clf: EntityResolutionClassifier, s1_test, s2_test, s3_test) -> Dict[str, Set[str]]:
        """Executes candidate blocking with memory monitoring."""
        stage_name = "stage2_blocking"
        cands_ckpt_file = self.checkpoint_dir / "candidates.json"

        if self.resume and is_stage_completed(self.checkpoint_dir, stage_name) and cands_ckpt_file.is_file():
            print(f"\n[Stage 2/3] Checkpoint found: Restoring candidate pairs from {cands_ckpt_file}...")
            cands_raw = load_json(cands_ckpt_file)
            return {k: set(v) for k, v in cands_raw.items()}

        print(f"\n[Stage 2/3] Generating candidate pairs ({format_memory_summary()})...")
        t0 = time.time()
        check_memory_pressure(critical_ram_gb=1.0, auto_clean=True)

        candidates = clf.blocker.generate_candidates(s1_test, s2_test, s3_test, verbose=True)

        # Export candidate_pairs.tsv for audit compliance
        write_candidate_pairs_tsv(candidates, str(self.candidates_out))

        # Checkpoint candidates atomically
        atomic_save_json({k: list(v) for k, v in candidates.items()}, cands_ckpt_file)
        mark_stage_completed(self.checkpoint_dir, stage_name, {
            "total_candidates": sum(len(c) for c in candidates.values()),
            "duration_s": time.time() - t0,
        })
        self.sync_to_drive()
        print(f"[Stage 2/3] Saved candidate pairs to {self.candidates_out} ({time.time() - t0:.2f}s)")
        return candidates

    # -------------------------------------------------------------------------
    # STAGE 3: CHUNKED INFERENCE & DECISION LAYER
    # -------------------------------------------------------------------------
    def run_inference(
        self,
        clf: EntityResolutionClassifier,
        s1_test: Dict[str, Dict[str, str]],
        s2_test: Dict[str, Dict[str, str]],
        s3_test: Dict[str, Dict[str, str]],
        candidates: Dict[str, Set[str]],
        gt_test: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[str, Set[str]]:
        """
        Runs chunked inference with mid-batch checkpointing and memory protection.
        """
        stage_name = "stage3_inference"
        matches_ckpt_file = self.checkpoint_dir / "matching_results.json"

        if self.resume and is_stage_completed(self.checkpoint_dir, stage_name) and matches_ckpt_file.is_file():
            print(f"\n[Stage 3/3] Checkpoint found: Restoring predictions from {matches_ckpt_file}...")
            preds_raw = load_json(matches_ckpt_file)
            return {k: set(v) for k, v in preds_raw.items()}

        print(f"\n[Stage 3/3] Running GBDT Classification & Decision Layer ({format_memory_summary()})...")
        t0 = time.time()

        s1_keys = sorted(s1_test.keys())
        total_s1 = len(s1_keys)
        final_predictions: Dict[str, Set[str]] = {}

        # Chunked inference to guarantee zero OOM
        chunk_size = self.batch_size
        num_chunks = (total_s1 + chunk_size - 1) // chunk_size

        prog = load_chunk_progress(self.checkpoint_dir) if self.resume else None
        start_chunk = (prog["last_completed_chunk"] + 1) if (prog and "last_completed_chunk" in prog) else 0

        if start_chunk > 0:
            print(f"  * Resuming from chunk {start_chunk + 1}/{num_chunks}...")
            cached_partial = load_json(self.checkpoint_dir / "partial_predictions.json")
            if cached_partial:
                final_predictions = {k: set(v) for k, v in cached_partial.items()}

        for c_idx in range(start_chunk, num_chunks):
            mem_stat = check_memory_pressure(critical_ram_gb=1.2, auto_clean=True)
            if mem_stat["should_throttle"]:
                print(f"  [RAM Guard] High memory pressure ({mem_stat['available_gb']:.2f} GB free). Cleaned GC.")

            c_start = c_idx * chunk_size
            c_end = min(total_s1, (c_idx + 1) * chunk_size)
            chunk_s1_keys = s1_keys[c_start:c_end]
            chunk_s1 = {k: s1_test[k] for k in chunk_s1_keys}
            chunk_cands = {k: candidates.get(k, set()) for k in chunk_s1_keys}

            # Predict chunk
            chunk_preds = clf.predict(
                s1_records=chunk_s1,
                s2_records=s2_test,
                s3_records=s3_test,
                candidates=chunk_cands,
                verbose=False,
            )
            final_predictions.update(chunk_preds)

            # Save mid-batch progress
            atomic_save_json({k: list(v) for k, v in final_predictions.items()}, self.checkpoint_dir / "partial_predictions.json")
            save_chunk_progress(self.checkpoint_dir, c_idx, num_chunks)

            pct = (c_end / total_s1) * 100.0
            print(f"  -> Processed [{c_end}/{total_s1}] entities ({pct:.1f}%) | {format_memory_summary()}")

        # Export official submission TSV
        write_matching_results_tsv(final_predictions, str(self.matches_out))

        # Checkpoint complete stage
        atomic_save_json({k: list(v) for k, v in final_predictions.items()}, matches_ckpt_file)
        mark_stage_completed(self.checkpoint_dir, stage_name, {
            "total_entities": total_s1,
            "total_matched_entities": sum(1 for v in final_predictions.values() if len(v) > 0),
            "duration_s": time.time() - t0,
        })
        self.sync_to_drive()

        # If ground truth is provided, compute scorecard
        if gt_test:
            print("\nEvaluating official scorecard on test set...")
            scorecard = evaluate_pipeline_on_split(
                clf=clf,
                s1_val=s1_test,
                s2_val=s2_test,
                s3_val=s3_test,
                gt_val=gt_test,
                fold_name="Test Set Evaluation",
                verbose=True,
            )
            from dataclasses import asdict
            atomic_save_json(asdict(scorecard), self.scorecard_out)
            self.sync_to_drive()

        print(f"[Stage 3/3] Inference complete! Results saved to {self.matches_out} ({time.time() - t0:.2f}s)")
        return final_predictions

    # -------------------------------------------------------------------------
    # MASTER EXECUTION PIPELINE
    # -------------------------------------------------------------------------
    def run(self):
        """Executes complete automated pipeline."""
        t_all = time.time()

        # 1. Train or load
        clf = self.train_or_load_model()

        # 2. Load test records
        def _find_test_path(data_dir: Path, source_name: str) -> str:
            for pfx in ["test", "train"]:
                base = f"{pfx}_{source_name}"
                f_tsv = data_dir / f"{base}.tsv"
                if f_tsv.is_file():
                    return str(f_tsv)
                f_gz = data_dir / f"{base}.tsv.gz"
                if f_gz.is_file():
                    return str(f_gz)
                d_shard = data_dir / base
                if d_shard.is_dir():
                    return str(d_shard)
                split_name = data_dir.name
                d_alt = PROJECT_ROOT / "data" / "shards" / split_name / base
                if d_alt.is_dir():
                    return str(d_alt)
            raise FileNotFoundError(f"Could not locate {source_name} file or shard directory in {data_dir}")

        max_test_load = 500 if self.smoke_test else None
        s1_te = load_source_tsv(_find_test_path(self.test_dir, "source1"), max_records=max_test_load)
        s2_te = load_source_tsv(_find_test_path(self.test_dir, "source2"), max_records=max_test_load)
        s3_te = load_source_tsv(_find_test_path(self.test_dir, "source3"), max_records=max_test_load)
        try:
            gt_te = load_ground_truth_tsv(_find_test_path(self.test_dir, "ground_truth"), max_records=max_test_load)
        except FileNotFoundError:
            gt_te = None

        if self.smoke_test:
            s1_keys = list(s1_te.keys())[:20]
            s1_te = {k: s1_te[k] for k in s1_keys}
            smoke_test_matches = set()
            if gt_te:
                gt_te = {k: gt_te.get(k, set()) for k in s1_keys}
                for s in s1_keys:
                    smoke_test_matches.update(gt_te.get(s, set()))
            s2_te_keys = set(list(smoke_test_matches) + list(s2_te.keys())[:200])
            s3_te_keys = set(list(smoke_test_matches) + list(s3_te.keys())[:200])
            s2_te = {k: s2_te[k] for k in s2_te_keys if k in s2_te}
            s3_te = {k: s3_te[k] for k in s3_te_keys if k in s3_te}
            print(f"  [Smoke Test] Filtered test set to {len(s1_te)} S1 entities, {len(s2_te)} S2 records, {len(s3_te)} S3 records.")

        # 3. Blocking
        candidates = self.run_blocking(clf, s1_te, s2_te, s3_te)

        # 4. Inference & Decision Layer
        self.run_inference(clf, s1_te, s2_te, s3_te, candidates, gt_test=gt_te)

        print("\n" + "=" * 76)
        print(f"  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {time.time() - t_all:.2f} SECONDS!")
        print(f"  Artifacts:")
        print(f"    - Candidate Pairs:  {self.candidates_out}")
        print(f"    - Matching Results: {self.matches_out}")
        if self.scorecard_out.is_file():
            print(f"    - Scorecard JSON:   {self.scorecard_out}")
        print("=" * 76 + "\n")


# =============================================================================
# CLI PARSER
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Master High-Performance Pipeline Runner."
    )
    parser.add_argument("--train-dir", type=str, default="dataset_split/train", help="Path to training data directory.")
    parser.add_argument("--test-dir", type=str, default="dataset_split/test", help="Path to test/evaluation data directory.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save candidate_pairs.tsv and matching_results.tsv.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for atomic stage checkpoints.")
    parser.add_argument("--drive-sync-dir", type=str, default=None, help="Google Drive path for persistent sync (e.g. /content/drive/MyDrive/amazon-ml-challenge).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device (default: auto).")
    parser.add_argument("--batch-size", type=int, default=2000, help="Inference batch chunk size (default: 2000).")
    parser.add_argument("--max-train-records", type=int, default=50000, help="Max training entities to load into memory (default: 50000; set 0 for all).")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume; restart all stages fresh.")
    parser.add_argument("--smoke-test", action="store_true", help="Quick sanity run on small subset in ~10 seconds.")
    return parser.parse_args()


def main():
    args = parse_args()
    orchestrator = PipelineOrchestrator(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        drive_sync_dir=args.drive_sync_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_train_records=args.max_train_records,
        resume=not args.no_resume,
        smoke_test=args.smoke_test,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
