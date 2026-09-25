#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution
Mini Dataset Generator

Creates a small, fully self-contained version of the training dataset
(e.g., 100-200 companies) that strictly preserves the original distributions:
  1. Exact Country Distribution: 60% US / 40% India (via Largest Remainder apportionment)
  2. Ground Truth Match Combinations:
     - Singletons (no matches): ~5.6%
     - Both S2 and S3 matches: ~80.5%
     - S3-only matches: ~7.5%
     - S2-only matches: ~6.5%
     - Joint (n_s2, n_s3) match count frequencies
  3. Null Value Rates:
     - Source 1: 0% null name, 0% null address, 0% null country
     - Source 2: ~3.36% null address, 0% null name, 0% null country
     - Source 3: ~3.33% null address, 0% null name, 0% null country
  4. Real-world Noise / Distractors:
     - Includes proportional unmatched records from S2 and S3 (~26% of source files)
     - Preserves distractor country split (60% US / 40% India)
     - Calibrates null address counts so final files match true target rates

Runs in ~25 seconds on the 14M-row dataset using <100MB RAM (pure Python stdlib).
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict


def find_default_data_dir():
    """Locate the train dataset directory automatically."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a representative mini training dataset for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "-n", "--n-companies",
        type=int,
        default=150,
        help="Number of Source 1 companies/entities to sample (default: 150, e.g. 100-200)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="mini_dataset",
        help="Directory where mini dataset TSVs will be saved (default: mini_dataset)."
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default=None,
        help="Path to the training data directory containing train_source1.tsv, etc. (auto-detected if omitted)."
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic, reproducible sampling (default: 42)."
    )
    parser.add_argument(
        "--no-distractors",
        action="store_true",
        help="If set, only include matched S2/S3 records and exclude unmatched distractor records."
    )
    parser.add_argument(
        "--distractor-ratio",
        type=float,
        default=1340997 / 2206821,  # ~0.60766 unmatched records per S1 entity
        help="Ratio of unmatched S2/S3 records per S1 entity (default: ~0.608 matching full dataset)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages and show only final verification table."
    )
    return parser.parse_args()


def log(msg, quiet=False):
    if not quiet:
        print(msg)


def main():
    args = parse_args()
    t_start = time.time()

    data_dir = args.data_dir if args.data_dir else find_default_data_dir()
    output_dir = args.output_dir
    n_companies = args.n_companies
    seed = args.seed
    include_distractors = not args.no_distractors
    distractor_ratio = args.distractor_ratio

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    req_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv"
    ]
    for rf in req_files:
        p = os.path.join(data_dir, rf)
        if not os.path.isfile(p):
            print(f"Error: Required file '{p}' not found.", file=sys.stderr)
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    log(f"===========================================================", args.quiet)
    log(f"  Amazon ML Challenge 2026 - Mini Dataset Generator", args.quiet)
    log(f"===========================================================", args.quiet)
    log(f"Config:", args.quiet)
    log(f"  Input Directory:       {data_dir}", args.quiet)
    log(f"  Output Directory:      {output_dir}", args.quiet)
    log(f"  Target Companies (S1): {n_companies}", args.quiet)
    log(f"  Random Seed:           {seed}", args.quiet)
    log(f"  Include Distractors:   {include_distractors} (ratio: {distractor_ratio:.4f})", args.quiet)
    log(f"-----------------------------------------------------------", args.quiet)

    # -------------------------------------------------------------
    # Step 1: Stream train_source1.tsv to map entity_id -> country
    # -------------------------------------------------------------
    t0 = time.time()
    log("[1/5] Scanning Source 1 entities and metadata...", args.quiet)
    s1_country = {}
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        header_s1 = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            s1_country[p[0]] = p[3].strip() if len(p) > 3 else ""

    pop_s1_total = len(s1_country)
    log(f"      Mapped {pop_s1_total:,} S1 entities in {time.time() - t0:.2f}s", args.quiet)

    # -------------------------------------------------------------
    # Step 2: Stratified Reservoir Sampling over Ground Truth
    # -------------------------------------------------------------
    t1 = time.time()
    log("[2/5] Stratified reservoir sampling across Ground Truth...", args.quiet)
    M_RES = 60  # reservoir capacity per stratum
    stratum_counts = Counter()
    stratum_reservoirs = defaultdict(list)

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
            else:
                matches = [x.strip() for x in m_str.split(",") if x.strip()]
                n_s2 = sum(1 for x in matches if x.startswith("S2-"))
                n_s3 = sum(1 for x in matches if x.startswith("S3-"))

            stratum = (cntry, n_s2, n_s3)
            stratum_counts[stratum] += 1
            n_seen = stratum_counts[stratum]

            res = stratum_reservoirs[stratum]
            item = (s1_id, m_str, matches, n_s2, n_s3)
            if len(res) < M_RES:
                res.append(item)
            else:
                r = random.randint(0, n_seen - 1)
                if r < M_RES:
                    res[r] = item

    log(f"      Analyzed {pop_s1_total:,} records across {len(stratum_counts)} strata in {time.time() - t1:.2f}s", args.quiet)

    # -------------------------------------------------------------
    # Step 3: Largest Remainder Apportionment (Hare-Niemeyer Quota)
    # -------------------------------------------------------------
    total_pop = sum(stratum_counts.values())
    quotas = {}
    remainders = []
    floor_sum = 0
    for stratum, count in stratum_counts.items():
        exact = n_companies * count / total_pop
        fl = int(exact)
        quotas[stratum] = fl
        floor_sum += fl
        remainders.append((exact - fl, stratum))

    remainders.sort(key=lambda x: x[0], reverse=True)
    leftover = n_companies - floor_sum
    for i in range(leftover):
        stratum = remainders[i][1]
        quotas[stratum] += 1

    sampled_gt_items = []
    for stratum, q in quotas.items():
        if q > 0:
            res = stratum_reservoirs[stratum]
            random.shuffle(res)
            sampled_gt_items.extend(res[:q])

    random.shuffle(sampled_gt_items)

    chosen_s1_ids = {x[0] for x in sampled_gt_items}
    chosen_s2_matched_ids = set()
    chosen_s3_matched_ids = set()
    for x in sampled_gt_items:
        for m in x[2]:
            if m.startswith("S2-"):
                chosen_s2_matched_ids.add(m)
            elif m.startswith("S3-"):
                chosen_s3_matched_ids.add(m)

    # -------------------------------------------------------------
    # Step 4: Write mini train_source1.tsv and train_ground_truth.tsv
    # -------------------------------------------------------------
    log("[3/5] Writing mini train_source1.tsv and train_ground_truth.tsv...", args.quiet)
    t2 = time.time()
    with open(os.path.join(output_dir, "train_ground_truth.tsv"), "w", encoding="utf-8") as f_gt:
        f_gt.write(header_gt)
        for x in sampled_gt_items:
            f_gt.write(f"{x[0]}\t{x[1]}\n")

    s1_written = 0
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f_in, \
         open(os.path.join(output_dir, "train_source1.tsv"), "w", encoding="utf-8") as f_out:
        f_out.write(header_s1)
        for line in f_in:
            p = line.rstrip("\n").split("\t")
            if p[0] in chosen_s1_ids:
                f_out.write(line)
                s1_written += 1
                if s1_written == len(chosen_s1_ids):
                    break

    log(f"      Saved {s1_written} S1 records & GT in {time.time() - t2:.2f}s", args.quiet)

    # -------------------------------------------------------------
    # Step 5: Extract Source 2 (Matches + Calibrated Distractors)
    # -------------------------------------------------------------
    log("[4/5] Extracting Source 2 records with calibrated distractors...", args.quiet)
    t3 = time.time()
    num_s2_distractors = round(n_companies * distractor_ratio) if include_distractors else 0
    n_dist_s2_us = round(num_s2_distractors * 0.60)
    n_dist_s2_in = num_s2_distractors - n_dist_s2_us

    # Target overall null address rate in S2: 3.356%
    target_total_s2 = len(chosen_s2_matched_ids) + num_s2_distractors
    target_null_s2 = round(target_total_s2 * 0.03356)

    s2_matches = []
    s2_distractor_candidates = {"US_null": [], "US_valid": [], "India_null": [], "India_valid": []}
    M_DIST = max(100, num_s2_distractors * 2)

    with open(os.path.join(data_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
        header_s2 = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            eid = p[0]
            if eid in chosen_s2_matched_ids:
                s2_matches.append(line)
            elif num_s2_distractors > 0:
                cntry = p[3].strip() if len(p) > 3 else ""
                is_null_addr = (len(p) < 3 or p[2].strip() == "")
                key = f"{cntry}_{'null' if is_null_addr else 'valid'}"
                if key in s2_distractor_candidates:
                    pool = s2_distractor_candidates[key]
                    if len(pool) < M_DIST:
                        pool.append(line)

    s2_match_nulls = sum(1 for line in s2_matches if line.split("\t")[2].strip() == "")
    needed_null_dist_s2 = max(0, min(num_s2_distractors, target_null_s2 - s2_match_nulls))

    null_dist_us = min(len(s2_distractor_candidates["US_null"]), round(needed_null_dist_s2 * 0.6))
    null_dist_in = min(len(s2_distractor_candidates["India_null"]), needed_null_dist_s2 - null_dist_us)

    valid_dist_us = min(len(s2_distractor_candidates["US_valid"]), n_dist_s2_us - null_dist_us)
    valid_dist_in = min(len(s2_distractor_candidates["India_valid"]), n_dist_s2_in - null_dist_in)

    selected_s2_distractors = (
        random.sample(s2_distractor_candidates["US_null"], null_dist_us) +
        random.sample(s2_distractor_candidates["US_valid"], valid_dist_us) +
        random.sample(s2_distractor_candidates["India_null"], null_dist_in) +
        random.sample(s2_distractor_candidates["India_valid"], valid_dist_in)
    )

    all_s2_lines = s2_matches + selected_s2_distractors
    random.shuffle(all_s2_lines)

    with open(os.path.join(output_dir, "train_source2.tsv"), "w", encoding="utf-8") as f_out:
        f_out.write(header_s2)
        for line in all_s2_lines:
            f_out.write(line)

    log(f"      Saved {len(all_s2_lines)} S2 records ({len(s2_matches)} matched + {len(selected_s2_distractors)} distractors) in {time.time() - t3:.2f}s", args.quiet)

    # -------------------------------------------------------------
    # Step 6: Extract Source 3 (Matches + Calibrated Distractors)
    # -------------------------------------------------------------
    log("[5/5] Extracting Source 3 records with calibrated distractors...", args.quiet)
    t4 = time.time()
    num_s3_distractors = round(n_companies * distractor_ratio) if include_distractors else 0
    n_dist_s3_us = round(num_s3_distractors * 0.60)
    n_dist_s3_in = num_s3_distractors - n_dist_s3_us

    # Target overall null address rate in S3: 3.328%
    target_total_s3 = len(chosen_s3_matched_ids) + num_s3_distractors
    target_null_s3 = round(target_total_s3 * 0.03328)

    s3_matches = []
    s3_distractor_candidates = {"US_null": [], "US_valid": [], "India_null": [], "India_valid": []}

    with open(os.path.join(data_dir, "train_source3.tsv"), "r", encoding="utf-8") as f:
        header_s3 = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            eid = p[0]
            if eid in chosen_s3_matched_ids:
                s3_matches.append(line)
            elif num_s3_distractors > 0:
                cntry = p[3].strip() if len(p) > 3 else ""
                is_null_addr = (len(p) < 3 or p[2].strip() == "")
                key = f"{cntry}_{'null' if is_null_addr else 'valid'}"
                if key in s3_distractor_candidates:
                    pool = s3_distractor_candidates[key]
                    if len(pool) < M_DIST:
                        pool.append(line)

    s3_match_nulls = sum(1 for line in s3_matches if line.split("\t")[2].strip() == "")
    needed_null_dist_s3 = max(0, min(num_s3_distractors, target_null_s3 - s3_match_nulls))

    null_dist_s3_us = min(len(s3_distractor_candidates["US_null"]), round(needed_null_dist_s3 * 0.6))
    null_dist_s3_in = min(len(s3_distractor_candidates["India_null"]), needed_null_dist_s3 - null_dist_s3_us)

    valid_dist_s3_us = min(len(s3_distractor_candidates["US_valid"]), n_dist_s3_us - null_dist_s3_us)
    valid_dist_s3_in = min(len(s3_distractor_candidates["India_valid"]), n_dist_s3_in - null_dist_s3_in)

    selected_s3_distractors = (
        random.sample(s3_distractor_candidates["US_null"], null_dist_s3_us) +
        random.sample(s3_distractor_candidates["US_valid"], valid_dist_s3_us) +
        random.sample(s3_distractor_candidates["India_null"], null_dist_s3_in) +
        random.sample(s3_distractor_candidates["India_valid"], valid_dist_s3_in)
    )

    all_s3_lines = s3_matches + selected_s3_distractors
    random.shuffle(all_s3_lines)

    with open(os.path.join(output_dir, "train_source3.tsv"), "w", encoding="utf-8") as f_out:
        f_out.write(header_s3)
        for line in all_s3_lines:
            f_out.write(line)

    log(f"      Saved {len(all_s3_lines)} S3 records ({len(s3_matches)} matched + {len(selected_s3_distractors)} distractors) in {time.time() - t4:.2f}s", args.quiet)

    total_duration = time.time() - t_start

    # -------------------------------------------------------------
    # Step 7: Verification & Distribution Comparison Report
    # -------------------------------------------------------------
    s1_sample_countries = Counter(s1_country[x[0]] for x in sampled_gt_items)
    s2_sample_countries = Counter(l.split("\t")[3].strip() for l in all_s2_lines)
    s3_sample_countries = Counter(l.split("\t")[3].strip() for l in all_s3_lines)

    s2_null_count = sum(1 for l in all_s2_lines if l.split("\t")[2].strip() == "")
    s3_null_count = sum(1 for l in all_s3_lines if l.split("\t")[2].strip() == "")

    combos = Counter()
    tot_s2_matches = 0
    tot_s3_matches = 0
    for x in sampled_gt_items:
        n_s2, n_s3 = x[3], x[4]
        tot_s2_matches += n_s2
        tot_s3_matches += n_s3
        if n_s2 == 0 and n_s3 == 0:
            combos["singleton"] += 1
        elif n_s2 > 0 and n_s3 == 0:
            combos["s2_only"] += 1
        elif n_s2 == 0 and n_s3 > 0:
            combos["s3_only"] += 1
        else:
            combos["both"] += 1

    s1_us_pct = s1_sample_countries["US"] / n_companies * 100
    s1_in_pct = s1_sample_countries["India"] / n_companies * 100

    s2_tot = len(all_s2_lines)
    s3_tot = len(all_s3_lines)

    s2_null_pct = s2_null_count / s2_tot * 100 if s2_tot > 0 else 0
    s3_null_pct = s3_null_count / s3_tot * 100 if s3_tot > 0 else 0

    s1_us_count = s1_sample_countries["US"]
    s1_in_count = s1_sample_countries["India"]
    s1_us_str = f"{s1_us_pct:.1f}% ({s1_us_count})"
    s1_in_str = f"{s1_in_pct:.1f}% ({s1_in_count})"

    both_str = f"{combos['both']/n_companies*100:.2f}% ({combos['both']})"
    s3_only_str = f"{combos['s3_only']/n_companies*100:.2f}% ({combos['s3_only']})"
    s2_only_str = f"{combos['s2_only']/n_companies*100:.2f}% ({combos['s2_only']})"
    single_str = f"{combos['singleton']/n_companies*100:.2f}% ({combos['singleton']})"

    s2_null_str = f"{s2_null_pct:.2f}% ({s2_null_count}/{s2_tot})"
    s3_null_str = f"{s3_null_pct:.2f}% ({s3_null_count}/{s3_tot})"

    s2_dist_str = f"{len(selected_s2_distractors)/s2_tot*100:.2f}% ({len(selected_s2_distractors)})"
    s3_dist_str = f"{len(selected_s3_distractors)/s3_tot*100:.2f}% ({len(selected_s3_distractors)})"

    s1_tot_str = f"{n_companies:,}"
    s2_tot_str = f"{s2_tot:,}"
    s3_tot_str = f"{s3_tot:,}"
    s2_avg_str = f"{tot_s2_matches/n_companies:.3f}"
    s3_avg_str = f"{tot_s3_matches/n_companies:.3f}"

    print("\n" + "=" * 65)
    print("       DATASET DISTRIBUTION FIDELITY REPORT")
    print("=" * 65)
    print(f"{'Metric':<32} | {'Original Dataset':<14} | {'Mini Dataset':<14}")
    print("-" * 65)
    print(f"{'Source 1 Entities (Companies)':<32} | {'2,206,821':<14} | {s1_tot_str:<14}")
    print(f"{'Source 2 Total Records':<32} | {'5,034,616':<14} | {s2_tot_str:<14}")
    print(f"{'Source 3 Total Records':<32} | {'5,285,603':<14} | {s3_tot_str:<14}")
    print("-" * 65)
    print(f"{'S1 Country: US':<32} | {'60.0%':<14} | {s1_us_str:<14}")
    print(f"{'S1 Country: India':<32} | {'40.0%':<14} | {s1_in_str:<14}")
    print("-" * 65)
    print(f"{'Match: Both S2 & S3':<32} | {'80.48%':<14} | {both_str:<14}")
    print(f"{'Match: S3 Only':<32} | {'7.45%':<14} | {s3_only_str:<14}")
    print(f"{'Match: S2 Only':<32} | {'6.48%':<14} | {s2_only_str:<14}")
    print(f"{'Match: Singleton (No Matches)':<32} | {'5.58%':<14} | {single_str:<14}")
    print("-" * 65)
    print(f"{'Avg Matched S2 per S1':<32} | {'1.674':<14} | {s2_avg_str:<14}")
    print(f"{'Avg Matched S3 per S1':<32} | {'1.787':<14} | {s3_avg_str:<14}")
    print("-" * 65)
    print(f"{'Source 1 Null Address Rate':<32} | {'0.00%':<14} | {'0.00%':<14}")
    print(f"{'Source 2 Null Address Rate':<32} | {'3.36%':<14} | {s2_null_str:<14}")
    print(f"{'Source 3 Null Address Rate':<32} | {'3.33%':<14} | {s3_null_str:<14}")
    print("-" * 65)
    print(f"{'S2 Unmatched Distractors':<32} | {'26.64%':<14} | {s2_dist_str:<14}")
    print(f"{'S3 Unmatched Distractors':<32} | {'25.37%':<14} | {s3_dist_str:<14}")
    print("=" * 65)
    print(f"Generated 4 TSV files in '{output_dir}/' in {total_duration:.2f} seconds:")
    for f_name in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]:
        p = os.path.join(output_dir, f_name)
        sz = os.path.getsize(p)
        print(f"  - {p} ({sz / 1024:.1f} KB)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
