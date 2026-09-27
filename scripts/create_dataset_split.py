#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Stratified Train/Test Dataset Split Generator

Creates a representative, fully independent Train and Test split dataset:
  - Both Train and Test maintain the exact population distributions:
      1. Country Distribution: 60% US / 40% India
      2. Match Combinations: Both S2+S3 (~80%), S3-only (~7.5%), S2-only (~6.5%), Singletons (~5.6%)
      3. Address Null Rates: ~3.3% in S2 and S3
      4. Realistic Distractors: ~26% unmatched noise in S2 and S3
  - Strict Independence: No entity overlap between Train and Test across all sources.
  - Generates:
      <output_dir>/train/
          train_source1.tsv
          train_source2.tsv
          train_source3.tsv
          train_ground_truth.tsv
      <output_dir>/test/
          test_source1.tsv
          test_source2.tsv
          test_source3.tsv
          test_ground_truth.tsv (ground truth for scoring/evaluation)
  - Automatically runs official validate_submission.py check on the test split.
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict


def find_default_data_dir():
    candidates = [
        "6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        "student_resource/dataset/train",
        "dataset/train",
        "../dataset/train",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "train_source1.tsv")):
            return c
    return "dataset/train"


def find_validator_script():
    candidates = [
        "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
        "student_resource/utils/validate_submission.py",
        "utils/validate_submission.py",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a stratified Train/Test split dataset for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=20,
        help="Number of Source 1 companies in the test split (default: 20)."
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=80,
        help="Number of Source 1 companies in the train split (default: 80, for 100 total companies with 20 test)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="dataset_split",
        help="Directory to save the split dataset (default: dataset_split)."
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default=None,
        help="Path to training data directory (auto-detected if omitted)."
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--no-distractors",
        action="store_true",
        help="Exclude unmatched distractor records from S2 and S3."
    )
    parser.add_argument(
        "--distractor-ratio",
        type=float,
        default=1340997 / 2206821,
        help="Ratio of unmatched S2/S3 records per S1 entity (default: ~0.608)."
    )
    return parser.parse_args()


def get_stratified_quotas(n_target, pop_counts):
    """Largest Remainder (Hare-Niemeyer) quota allocation."""
    total_pop = sum(pop_counts.values())
    quotas = {}
    remainders = []
    floor_sum = 0
    for stratum, count in pop_counts.items():
        exact = n_target * count / total_pop
        fl = int(exact)
        quotas[stratum] = fl
        floor_sum += fl
        remainders.append((exact - fl, stratum))

    remainders.sort(key=lambda x: x[0], reverse=True)
    leftover = n_target - floor_sum
    for i in range(leftover):
        stratum = remainders[i][1]
        quotas[stratum] += 1
    return quotas


def main():
    args = parse_args()
    t_start = time.time()

    data_dir = args.data_dir if args.data_dir else find_default_data_dir()
    output_dir = args.output_dir
    n_train = args.n_train
    n_test = args.n_test
    seed = args.seed
    include_distractors = not args.no_distractors
    distractor_ratio = args.distractor_ratio

    train_out_dir = os.path.join(output_dir, "train")
    test_out_dir = os.path.join(output_dir, "test")
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)

    random.seed(seed)

    print("=" * 68)
    print("  Amazon ML Challenge 2026 - Stratified Train/Test Dataset Split")
    print("=" * 68)
    print(f"Config:")
    print(f"  Input Directory:   {data_dir}")
    print(f"  Output Directory:  {output_dir}")
    print(f"  Train Companies:   {n_train}")
    print(f"  Test Companies:    {n_test} (Test Split = {n_test / (n_train + n_test) * 100:.1f}%)")
    print(f"  Total Companies:   {n_train + n_test}")
    print(f"  Random Seed:       {seed}")
    print(f"  Distractors:       {include_distractors} (ratio: {distractor_ratio:.4f})")
    print("-" * 68)

    # -------------------------------------------------------------
    # Step 1: Scan S1 Metadata
    # -------------------------------------------------------------
    t0 = time.time()
    print("[1/5] Scanning Source 1 entities and metadata...")
    s1_country = {}
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        header_s1 = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            s1_country[p[0]] = p[3].strip() if len(p) > 3 else ""

    pop_s1_total = len(s1_country)
    print(f"      Mapped {pop_s1_total:,} S1 entities in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------
    # Step 2: Stratified Reservoir Sampling over Ground Truth
    # -------------------------------------------------------------
    t1 = time.time()
    print("[2/5] Stratified reservoir sampling across Ground Truth...")
    M_RES = 150
    coarse_counts = Counter()
    coarse_reservoirs = defaultdict(list)

    with open(os.path.join(data_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        header_gt = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            s1_id = p[0].strip()
            cntry = s1_country.get(s1_id, "US")
            m_str = p[1].strip() if len(p) > 1 else ""

            if not m_str:
                n_s2, n_s3 = 0, 0
                matches = []
                cat = "singleton"
            else:
                matches = [x.strip() for x in m_str.split(",") if x.strip()]
                n_s2 = sum(1 for x in matches if x.startswith("S2-"))
                n_s3 = sum(1 for x in matches if x.startswith("S3-"))
                if n_s2 > 0 and n_s3 > 0:
                    cat = "both"
                elif n_s2 > 0:
                    cat = "s2_only"
                else:
                    cat = "s3_only"

            coarse_stratum = (cntry, cat)
            coarse_counts[coarse_stratum] += 1
            n_seen = coarse_counts[coarse_stratum]

            res = coarse_reservoirs[coarse_stratum]
            item = (s1_id, m_str, matches, n_s2, n_s3, cntry, cat)
            if len(res) < M_RES:
                res.append(item)
            else:
                r = random.randint(0, n_seen - 1)
                if r < M_RES:
                    res[r] = item

    print(f"      Profiled Ground Truth in {time.time() - t1:.2f}s")

    # -------------------------------------------------------------
    # Step 3: Quotas and Split Allocation
    # -------------------------------------------------------------
    quotas_test = get_stratified_quotas(n_test, coarse_counts)
    quotas_train = get_stratified_quotas(n_train, coarse_counts)

    # Allocate test items first, then train items (strictly disjoint)
    test_items = []
    train_items = []
    used_s1_ids = set()

    for stratum, q in quotas_test.items():
        if q > 0:
            pool = [x for x in coarse_reservoirs[stratum] if x[0] not in used_s1_ids]
            random.shuffle(pool)
            picked = pool[:q]
            test_items.extend(picked)
            for it in picked:
                used_s1_ids.add(it[0])

    for stratum, q in quotas_train.items():
        if q > 0:
            pool = [x for x in coarse_reservoirs[stratum] if x[0] not in used_s1_ids]
            random.shuffle(pool)
            picked = pool[:q]
            train_items.extend(picked)
            for it in picked:
                used_s1_ids.add(it[0])

    random.shuffle(test_items)
    random.shuffle(train_items)

    assert len(used_s1_ids) == len(test_items) + len(train_items), "Overlap between Train and Test!"
    print(f"      Allocated {len(train_items)} Train companies and {len(test_items)} Test companies.")

    # Collect matched IDs
    train_s1_ids = {x[0] for x in train_items}
    test_s1_ids = {x[0] for x in test_items}

    train_s2_match_ids = {m for x in train_items for m in x[2] if m.startswith("S2-")}
    train_s3_match_ids = {m for x in train_items for m in x[2] if m.startswith("S3-")}

    test_s2_match_ids = {m for x in test_items for m in x[2] if m.startswith("S2-")}
    test_s3_match_ids = {m for x in test_items for m in x[2] if m.startswith("S3-")}

    assert train_s2_match_ids.isdisjoint(test_s2_match_ids), "S2 matches overlap between train and test!"
    assert train_s3_match_ids.isdisjoint(test_s3_match_ids), "S3 matches overlap between train and test!"

    # -------------------------------------------------------------
    # Step 4: Write Source 1 and Ground Truth Files
    # -------------------------------------------------------------
    print("[3/5] Writing Source 1 and Ground Truth files...")
    t2 = time.time()

    # Train GT
    with open(os.path.join(train_out_dir, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
        f.write(header_gt)
        for x in train_items:
            f.write(f"{x[0]}\t{x[1]}\n")

    # Test GT (reference for scoring)
    with open(os.path.join(test_out_dir, "test_ground_truth.tsv"), "w", encoding="utf-8") as f:
        f.write(header_gt)
        for x in test_items:
            f.write(f"{x[0]}\t{x[1]}\n")

    # Stream S1 to write train_source1.tsv and test_source1.tsv
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f_in, \
         open(os.path.join(train_out_dir, "train_source1.tsv"), "w", encoding="utf-8") as f_tr, \
         open(os.path.join(test_out_dir, "test_source1.tsv"), "w", encoding="utf-8") as f_te:
        f_tr.write(header_s1)
        f_te.write(header_s1)
        tr_count = te_count = 0
        for line in f_in:
            eid = line.split("\t", 1)[0]
            if eid in train_s1_ids:
                f_tr.write(line)
                tr_count += 1
            elif eid in test_s1_ids:
                f_te.write(line)
                te_count += 1
            if tr_count == len(train_s1_ids) and te_count == len(test_s1_ids):
                break

    print(f"      Saved S1 and GT files in {time.time() - t2:.2f}s")

    # -------------------------------------------------------------
    # Step 5: Extract Source 2 (Train & Test Disjointly)
    # -------------------------------------------------------------
    print("[4/5] Extracting Source 2 records (Train and Test)...")
    t3 = time.time()
    num_s2_dist_train = round(n_train * distractor_ratio) if include_distractors else 0
    num_s2_dist_test = round(n_test * distractor_ratio) if include_distractors else 0

    train_s2_matches = []
    test_s2_matches = []
    s2_distractor_pool = {"US_null": [], "US_valid": [], "India_null": [], "India_valid": []}
    M_DIST = max(100, (num_s2_dist_train + num_s2_dist_test) * 2)

    with open(os.path.join(data_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
        header_s2 = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            eid = p[0]
            if eid in train_s2_match_ids:
                train_s2_matches.append(line)
            elif eid in test_s2_match_ids:
                test_s2_matches.append(line)
            elif (num_s2_dist_train + num_s2_dist_test) > 0:
                cntry = p[3].strip() if len(p) > 3 else ""
                is_null = (len(p) < 3 or p[2].strip() == "")
                key = f"{cntry}_{'null' if is_null else 'valid'}"
                if key in s2_distractor_pool:
                    pool = s2_distractor_pool[key]
                    if len(pool) < M_DIST:
                        pool.append(line)

    def sample_distractors(n_dist, match_lines, target_null_rate, pool_dict):
        if n_dist <= 0:
            return []
        n_us = round(n_dist * 0.60)
        n_in = n_dist - n_us
        tot = len(match_lines) + n_dist
        target_null = round(tot * target_null_rate)
        match_nulls = sum(1 for l in match_lines if l.split("\t")[2].strip() == "")
        needed_null = max(0, min(n_dist, target_null - match_nulls))

        null_us = min(len(pool_dict["US_null"]), round(needed_null * 0.6))
        null_in = min(len(pool_dict["India_null"]), needed_null - null_us)
        valid_us = min(len(pool_dict["US_valid"]), n_us - null_us)
        valid_in = min(len(pool_dict["India_valid"]), n_in - null_in)

        chosen = []
        for k, count in [("US_null", null_us), ("US_valid", valid_us), ("India_null", null_in), ("India_valid", valid_in)]:
            picked = random.sample(pool_dict[k], count)
            chosen.extend(picked)
            # Remove chosen from pool to prevent reuse
            pool_dict[k] = [x for x in pool_dict[k] if x not in set(picked)]
        return chosen

    s2_dist_test = sample_distractors(num_s2_dist_test, test_s2_matches, 0.03356, s2_distractor_pool)
    s2_dist_train = sample_distractors(num_s2_dist_train, train_s2_matches, 0.03356, s2_distractor_pool)

    all_s2_train = train_s2_matches + s2_dist_train
    all_s2_test = test_s2_matches + s2_dist_test
    random.shuffle(all_s2_train)
    random.shuffle(all_s2_test)

    with open(os.path.join(train_out_dir, "train_source2.tsv"), "w", encoding="utf-8") as f:
        f.write(header_s2)
        for l in all_s2_train:
            f.write(l)

    with open(os.path.join(test_out_dir, "test_source2.tsv"), "w", encoding="utf-8") as f:
        f.write(header_s2)
        for l in all_s2_test:
            f.write(l)

    print(f"      Saved Source 2 files in {time.time() - t3:.2f}s (Train: {len(all_s2_train)}, Test: {len(all_s2_test)})")

    # -------------------------------------------------------------
    # Step 6: Extract Source 3 (Train & Test Disjointly)
    # -------------------------------------------------------------
    print("[5/5] Extracting Source 3 records (Train and Test)...")
    t4 = time.time()
    num_s3_dist_train = round(n_train * distractor_ratio) if include_distractors else 0
    num_s3_dist_test = round(n_test * distractor_ratio) if include_distractors else 0

    train_s3_matches = []
    test_s3_matches = []
    s3_distractor_pool = {"US_null": [], "US_valid": [], "India_null": [], "India_valid": []}

    with open(os.path.join(data_dir, "train_source3.tsv"), "r", encoding="utf-8") as f:
        header_s3 = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            eid = p[0]
            if eid in train_s3_match_ids:
                train_s3_matches.append(line)
            elif eid in test_s3_match_ids:
                test_s3_matches.append(line)
            elif (num_s3_dist_train + num_s3_dist_test) > 0:
                cntry = p[3].strip() if len(p) > 3 else ""
                is_null = (len(p) < 3 or p[2].strip() == "")
                key = f"{cntry}_{'null' if is_null else 'valid'}"
                if key in s3_distractor_pool:
                    pool = s3_distractor_pool[key]
                    if len(pool) < M_DIST:
                        pool.append(line)

    s3_dist_test = sample_distractors(num_s3_dist_test, test_s3_matches, 0.03328, s3_distractor_pool)
    s3_dist_train = sample_distractors(num_s3_dist_train, train_s3_matches, 0.03328, s3_distractor_pool)

    all_s3_train = train_s3_matches + s3_dist_train
    all_s3_test = test_s3_matches + s3_dist_test
    random.shuffle(all_s3_train)
    random.shuffle(all_s3_test)

    with open(os.path.join(train_out_dir, "train_source3.tsv"), "w", encoding="utf-8") as f:
        f.write(header_s3)
        for l in all_s3_train:
            f.write(l)

    with open(os.path.join(test_out_dir, "test_source3.tsv"), "w", encoding="utf-8") as f:
        f.write(header_s3)
        for l in all_s3_test:
            f.write(l)

    print(f"      Saved Source 3 files in {time.time() - t4:.2f}s (Train: {len(all_s3_train)}, Test: {len(all_s3_test)})")
    total_duration = time.time() - t_start

    # -------------------------------------------------------------
    # Step 7: Statistical Verification & Summary
    # -------------------------------------------------------------
    def compute_stats(items, s2_lines, s3_lines, n_total):
        cntry = Counter(x[5] for x in items)
        combos = Counter(x[6] for x in items)
        tot_s2_m = sum(x[3] for x in items)
        tot_s3_m = sum(x[4] for x in items)
        s2_null = sum(1 for l in s2_lines if l.split("\t")[2].strip() == "")
        s3_null = sum(1 for l in s3_lines if l.split("\t")[2].strip() == "")
        s2_n = len(s2_lines)
        s3_n = len(s3_lines)
        return {
            "n_s1": n_total,
            "us_pct": cntry["US"] / n_total * 100,
            "in_pct": cntry["India"] / n_total * 100,
            "both_pct": combos["both"] / n_total * 100,
            "s3_pct": combos["s3_only"] / n_total * 100,
            "s2_pct": combos["s2_only"] / n_total * 100,
            "single_pct": combos["singleton"] / n_total * 100,
            "avg_s2": tot_s2_m / n_total,
            "avg_s3": tot_s3_m / n_total,
            "s2_tot": s2_n,
            "s3_tot": s3_n,
            "s2_null_pct": s2_null / s2_n * 100 if s2_n > 0 else 0,
            "s3_null_pct": s3_null / s3_n * 100 if s3_n > 0 else 0,
            "s2_null_count": s2_null,
            "s3_null_count": s3_null,
        }

    tr_st = compute_stats(train_items, all_s2_train, all_s3_train, n_train)
    te_st = compute_stats(test_items, all_s2_test, all_s3_test, n_test)

    print("\n" + "=" * 76)
    print("           STRATIFIED TRAIN / TEST SPLIT FIDELITY REPORT")
    print("=" * 76)
    print(f"{'Metric':<28} | {'Original (2.2M)':<15} | {'Train Split':<13} | {'Test Split':<13}")
    print("-" * 76)
    print(f"{'Source 1 Companies':<28} | {'2,206,821':<15} | {tr_st['n_s1']:<13} | {te_st['n_s1']:<13}")
    print(f"{'Source 2 Total Records':<28} | {'5,034,616':<15} | {tr_st['s2_tot']:<13} | {te_st['s2_tot']:<13}")
    print(f"{'Source 3 Total Records':<28} | {'5,285,603':<15} | {tr_st['s3_tot']:<13} | {te_st['s3_tot']:<13}")
    print("-" * 76)
    print(f"{'Country: US':<28} | {'60.0%':<15} | {tr_st['us_pct']:.1f}%{'':<8} | {te_st['us_pct']:.1f}%{'':<8}")
    print(f"{'Country: India':<28} | {'40.0%':<15} | {tr_st['in_pct']:.1f}%{'':<8} | {te_st['in_pct']:.1f}%{'':<8}")
    print("-" * 76)
    print(f"{'Match: Both S2 & S3':<28} | {'80.48%':<15} | {tr_st['both_pct']:.1f}%{'':<8} | {te_st['both_pct']:.1f}%{'':<8}")
    print(f"{'Match: S3 Only':<28} | {'7.45%':<15} | {tr_st['s3_pct']:.1f}%{'':<8} | {te_st['s3_pct']:.1f}%{'':<8}")
    print(f"{'Match: S2 Only':<28} | {'6.48%':<15} | {tr_st['s2_pct']:.1f}%{'':<8} | {te_st['s2_pct']:.1f}%{'':<8}")
    print(f"{'Match: Singletons':<28} | {'5.58%':<15} | {tr_st['single_pct']:.1f}%{'':<8} | {te_st['single_pct']:.1f}%{'':<8}")
    print("-" * 76)
    print(f"{'Avg Matched S2 per S1':<28} | {'1.674':<15} | {tr_st['avg_s2']:.3f}{'':<8} | {te_st['avg_s2']:.3f}{'':<8}")
    print(f"{'Avg Matched S3 per S1':<28} | {'1.787':<15} | {tr_st['avg_s3']:.3f}{'':<8} | {te_st['avg_s3']:.3f}{'':<8}")
    print("-" * 76)
    print(f"{'S2 Null Address Rate':<28} | {'3.36%':<15} | {tr_st['s2_null_pct']:.2f}%{'':<8} | {te_st['s2_null_pct']:.2f}%{'':<8}")
    print(f"{'S3 Null Address Rate':<28} | {'3.33%':<15} | {tr_st['s3_null_pct']:.2f}%{'':<8} | {te_st['s3_null_pct']:.2f}%{'':<8}")
    print("=" * 76)

    # -------------------------------------------------------------
    # Step 8: Automatic Submission Validation Check
    # -------------------------------------------------------------
    val_script = find_validator_script()
    if val_script:
        print("\nRunning official submission validator on test split...")
        cmd = [
            sys.executable, val_script,
            "--matching", os.path.join(test_out_dir, "test_ground_truth.tsv"),
            "--test-dir", test_out_dir,
            "--check-ids"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(">> Validation Check on Test Split: PASS (100% compliant with competition rules!)")
        else:
            print(f">> Validation warning/output:\n{res.stdout}\n{res.stderr}")

    print(f"\nCompleted in {total_duration:.2f} seconds.")
    print(f"Files created in '{output_dir}/':")
    for s_dir in [train_out_dir, test_out_dir]:
        for fn in os.listdir(s_dir):
            p = os.path.join(s_dir, fn)
            print(f"  - {p} ({os.path.getsize(p) / 1024:.1f} KB)")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
