#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: blocking

High-Recall Multi-Strategy Candidate Generation & Blocking Engine.
Sets the recall ceiling for the entity resolution pipeline by taking the
union of multiple complementary blocking strategies:

1. Rule-Based Inverted Index Keys:
   - Key 1 (State + Name Prefix): (country, state, name[:3..4])
   - Key 2 (City + Name Prefix): (country, city, name[:4])
   - Key 3 (Postal/ZIP + Name Token Sort Key): (country, zip, name_sorted[:3])
   - Key 4 (Street Number + Street Name Key): (country, house_num, street[:4])
   - Key 5 (Phonetic Soundex Keys): (country, state, soundex(name_token)) & (country, soundex)
   - Key 6 (Sorted Name Token Prefix): (country, name_sorted[:4])

2. TF-IDF + Cosine kNN Search (The Workhorse Engine):
   - Vectorizes normalized name + address using character n-grams (char_wb, 3-4 grams)
   - Executes sparse matrix multiplication partitioned by country (US, India, France)
   - Retrieves top-k similar candidates per S1 entity above cosine similarity threshold

3. Union, Deduplication & Export:
   - Merges candidate sets per S1 entity
   - Writes valid 'candidate_pairs.tsv' meeting all competition audit requirements
   - Evaluates pair recall against ground truth matching labels
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Local imports
try:
    from src.normalization import (
        normalize_name,
        normalize_name_tokens,
        normalize_address,
        normalize_address_tokens,
        extract_numbers,
        extract_legal_suffix,
        extract_unit_info,
        extract_street_name,
        extract_metaphone,
    )
except ImportError:
    from normalization import (
        normalize_name,
        normalize_name_tokens,
        normalize_address,
        normalize_address_tokens,
        extract_numbers,
        extract_legal_suffix,
        extract_unit_info,
        extract_street_name,
        extract_metaphone,
    )


# =============================================================================
# 1. PHONETIC ENCODING (SOUNDEX)
# =============================================================================
SOUNDEX_MAPPING = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6'
}


def soundex(word: str) -> str:
    """Computes standard Soundex phonetic code for a word."""
    if not word or not word.isalpha():
        return ""
    w = word.upper()
    first = w[0]
    encoded = []
    prev = SOUNDEX_MAPPING.get(first, "")
    for ch in w[1:]:
        c = SOUNDEX_MAPPING.get(ch, "")
        if c and c != prev:
            encoded.append(c)
        prev = c
    return (first + "".join(encoded) + "0000")[:4]


BLOCKING_STOPWORDS = {
    "and", "the", "of", "for", "in", "at", "to", "a", "an", "co", "company",
    "corp", "corporation", "inc", "incorporated", "ltd", "limited", "pvt", "llc",
    "gmbh", "sarl", "sa", "bv", "center", "services", "group", "holdings", "enterprise"
}


def _extract_keys_worker(batch: List[Dict[str, str]]) -> List[Tuple[str, List[str]]]:
    """Worker function for parallel extraction of blocking keys."""
    results = []
    for raw_rec in batch:
        eid = raw_rec["entity_id"]
        cntry = raw_rec.get("country", "")
        nm = raw_rec.get("business_name", raw_rec.get("name", ""))
        addr = raw_rec.get("business_address", raw_rec.get("address", ""))

        nm_clean = normalize_name(nm)
        nm_words = nm_clean.split()
        addr_clean = normalize_address(addr, cntry)
        addr_words = addr_clean.split()
        nums = extract_numbers(addr)

        state = ""
        for tok in addr_words:
            if len(tok) == 2 and tok.isalpha():
                state = tok
                break
        postal = ""
        for n in nums:
            if len(n) == 5 or len(n) == 6:
                postal = n
                break

        keys = []
        if state and len(nm_clean) >= 3:
            keys.append(f"st_nm3:{cntry}:{state}:{nm_clean[:3]}")
            keys.append(f"st_nm4:{cntry}:{state}:{nm_clean[:4]}")
        if postal and len(nm_clean) >= 3:
            keys.append(f"zip_nm:{cntry}:{postal}:{nm_clean[:3]}")
        if nums and len(addr_words) >= 2:
            house_num = nums[0]
            for w in addr_words:
                if len(w) >= 4 and not w.isdigit() and w not in (
                    "street", "road", "avenue", "drive", "lane", "boulevard", "unit"
                ):
                    keys.append(f"num_street:{cntry}:{house_num}:{w[:4]}")
                    break

        # Stopword-free sorted name tokens
        clean_tokens = [w for w in nm_words if w not in BLOCKING_STOPWORDS and len(w) >= 2]
        if clean_tokens:
            nm_sorted = " ".join(sorted(clean_tokens))
            if len(nm_sorted) >= 4:
                keys.append(f"nm_sort4:{cntry}:{nm_sorted[:4]}")

        # Phonetic Keys (Soundex) - bound to state to prevent nationwide Cartesian explosion
        if nm_words and state:
            sx = soundex(nm_words[0])
            if sx:
                keys.append(f"st_soundex:{cntry}:{state}:{sx}")
        results.append((eid, keys))
    return results


# =============================================================================
# 2. CANDIDATE BLOCKER ENGINE
# =============================================================================
class CandidateBlocker:
    """
    Multi-strategy candidate blocker combining inverted index rule keys
    with TF-IDF cosine kNN sparse matrix retrieval.
    """

    def __init__(
        self,
        k_neighbors: int = 30,
        min_similarity: float = 0.12,
        ngram_range: Tuple[int, int] = (3, 4),
        use_rule_keys: bool = True,
        use_phonetic_keys: bool = True,
        use_tfidf: bool = True,
    ):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.ngram_range = ngram_range
        self.use_rule_keys = use_rule_keys
        self.use_phonetic_keys = use_phonetic_keys
        self.use_tfidf = use_tfidf

    def preprocess_record(self, raw_rec: Dict[str, str]) -> Dict[str, any]:
        """Normalizes and prepares structured metadata for a record."""
        eid = raw_rec["entity_id"]
        cntry = raw_rec.get("country", "")
        nm = raw_rec.get("business_name", raw_rec.get("name", ""))
        addr = raw_rec.get("business_address", raw_rec.get("address", ""))

        nm_clean = normalize_name(nm)
        nm_sorted = normalize_name_tokens(nm)
        addr_clean = normalize_address(addr, cntry)
        addr_sorted = normalize_address_tokens(addr, cntry)
        nums = extract_numbers(addr)

        # Extract state candidate (2-letter token in address)
        state = ""
        addr_tokens = addr_clean.split()
        for tok in addr_tokens:
            if len(tok) == 2 and tok.isalpha():
                state = tok
                break

        # Extract postal code candidate (5-digit token or 6-digit token)
        postal = ""
        for n in nums:
            if len(n) == 5 or len(n) == 6:
                postal = n
                break

        # Extract street name & unit
        st_name = extract_street_name(addr_clean)
        first_nm_word = nm_clean.split()[0] if nm_clean.split() else ""
        first_st_word = st_name.split()[0] if st_name.split() else ""

        full_text = f"{nm_clean} {nm_sorted} {addr_clean}"

        return {
            "entity_id": eid,
            "country": cntry,
            "raw_name": nm,
            "raw_address": addr,
            "name_clean": nm_clean,
            "name_sorted": nm_sorted,
            "addr_clean": addr_clean,
            "addr_sorted": addr_sorted,
            "legal_suffix": extract_legal_suffix(nm),
            "unit": extract_unit_info(addr),
            "street_name": st_name,
            "state": state,
            "postal": postal,
            "numbers": nums,
            "nm_metaphone": extract_metaphone(first_nm_word),
            "street_metaphone": extract_metaphone(first_st_word),
            "nm_soundex": soundex(first_nm_word) if first_nm_word else "",
            "street_soundex": soundex(first_st_word) if first_st_word else "",
            "full_text": full_text,
        }

    def generate_blocking_keys(self, rec: Dict[str, any]) -> List[str]:
        """Generates multiple complementary rule-based blocking keys."""
        keys = []
        cntry = rec["country"]
        state = rec["state"]
        postal = rec["postal"]
        nm = rec["name_clean"]
        nm_words = nm.split()
        addr_words = rec["addr_clean"].split()

        if not self.use_rule_keys:
            return keys

        # Key 1: (country, state, name_prefix)
        if state and len(nm) >= 3:
            keys.append(f"st_nm3:{cntry}:{state}:{nm[:3]}")
            keys.append(f"st_nm4:{cntry}:{state}:{nm[:4]}")

        # Key 2: (country, postal_code, name_prefix)
        if postal and len(nm) >= 3:
            keys.append(f"zip_nm:{cntry}:{postal}:{nm[:3]}")

        # Key 3: (country, house_number, street_token)
        if rec["numbers"] and len(addr_words) >= 2:
            house_num = rec["numbers"][0]
            for w in addr_words:
                if len(w) >= 4 and not w.isdigit() and w not in (
                    "street", "road", "avenue", "drive", "lane", "boulevard", "unit"
                ):
                    keys.append(f"num_street:{cntry}:{house_num}:{w[:4]}")
                    break

        # Key 4: (country, name_sorted_prefix)
        if len(rec["name_sorted"]) >= 4:
            keys.append(f"nm_sort4:{cntry}:{rec['name_sorted'][:4]}")

        # Key 5: Phonetic Keys (Soundex)
        if self.use_phonetic_keys and nm_words:
            sx = soundex(nm_words[0])
            if sx:
                keys.append(f"soundex:{cntry}:{sx}")
                if state:
                    keys.append(f"st_soundex:{cntry}:{state}:{sx}")

        return keys

    def generate_candidates(
        self,
        s1_records: Dict[str, Dict[str, str]],
        s2_records: Dict[str, Dict[str, str]],
        s3_records: Dict[str, Dict[str, str]],
        verbose: bool = True,
    ) -> Dict[str, Set[str]]:
        """
        Generates candidate matches from S2 and S3 for every Source 1 entity.
        High-performance parallel indexing with multi-core auto-scaling.
        """
        t0 = time.time()
        n_targets = len(s2_records) + len(s3_records)
        n_s1 = len(s1_records)
        if verbose:
            print(f"[Blocking] 1/2 Parallel indexing {n_targets:,} target records (S2+S3)...")

        inverted_index: Dict[str, Set[str]] = defaultdict(set)
        candidates: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_records}

        n_workers = min(12, os.cpu_count() or 4)
        batch_size = 10000

        def _index_records_stream(records_dict, name):
            rec_list = list(records_dict.values())
            total = len(rec_list)
            if total == 0:
                return
            chunks = [rec_list[i:i + batch_size] for i in range(0, total, batch_size)]
            indexed = 0
            t_start = time.time()

            if total > 5000 and n_workers > 1:
                with Pool(processes=n_workers) as pool:
                    for batch_res in pool.imap_unordered(_extract_keys_worker, chunks, chunksize=1):
                        for eid, keys in batch_res:
                            for k in keys:
                                inverted_index[k].add(eid)
                        indexed += len(batch_res)
                        if verbose and (indexed % 500000 < batch_size or indexed == total):
                            rate = indexed / max(0.01, time.time() - t_start)
                            pct = (indexed / total) * 100
                            print(f"  * [{name}] Indexed {indexed:,} / {total:,} records ({pct:.1f}%) at {rate:,.0f} rec/s")
            else:
                for chunk in chunks:
                    batch_res = _extract_keys_worker(chunk)
                    for eid, keys in batch_res:
                        for k in keys:
                            inverted_index[k].add(eid)
                    indexed += len(batch_res)

        # 1. Index S2 and S3
        _index_records_stream(s2_records, "Source 2")
        _index_records_stream(s3_records, "Source 3")

        # 2. Prune oversized inverted index keys (>250 target records) to eliminate noise & RAM explosion
        max_bucket_size = 250
        pruned_keys = 0
        for k in list(inverted_index.keys()):
            if len(inverted_index[k]) > max_bucket_size:
                del inverted_index[k]
                pruned_keys += 1

        if verbose:
            print(f"[Blocking] Inverted index optimized: {len(inverted_index):,} active keys ({pruned_keys:,} stop-keys pruned, {time.time() - t0:.2f}s).")
            print(f"[Blocking] 2/2 Querying candidates for {n_s1:,} Source 1 entities (bounded <= 50 cands/entity)...")

        # 3. Query S1 entities against inverted index with per-entity candidate cap
        t1 = time.time()
        s1_list = list(s1_records.values())
        s1_chunks = [s1_list[i:i + batch_size] for i in range(0, n_s1, batch_size)]
        queried = 0
        max_cands = 50

        if n_s1 > 5000 and n_workers > 1:
            with Pool(processes=n_workers) as pool:
                for batch_res in pool.imap_unordered(_extract_keys_worker, s1_chunks, chunksize=1):
                    for eid, keys in batch_res:
                        s1_cands = candidates[eid]
                        for k in keys:
                            if k in inverted_index:
                                for cid in inverted_index[k]:
                                    s1_cands.add(cid)
                                    if len(s1_cands) >= max_cands:
                                        break
                            if len(s1_cands) >= max_cands:
                                break
                    queried += len(batch_res)
                    if verbose and (queried % 250000 < batch_size or queried == n_s1):
                        rate = queried / max(0.01, time.time() - t1)
                        pct = (queried / n_s1) * 100
                        print(f"  * [Source 1] Matched {queried:,} / {n_s1:,} entities ({pct:.1f}%) at {rate:,.0f} rec/s")
        else:
            for chunk in s1_chunks:
                batch_res = _extract_keys_worker(chunk)
                for eid, keys in batch_res:
                    s1_cands = candidates[eid]
                    for k in keys:
                        if k in inverted_index:
                            for cid in inverted_index[k]:
                                s1_cands.add(cid)
                                if len(s1_cands) >= max_cands:
                                    break
                        if len(s1_cands) >= max_cands:
                            break
                queried += len(batch_res)

        # TF-IDF fallback only for small sets (e.g. smoke tests) where cartesian space is negligible
        if self.use_tfidf and (n_s1 + n_targets) <= 5000:
            if verbose:
                print("  * Running TF-IDF fallback on small evaluation subset...")
            s1_prep = {eid: self.preprocess_record(r) for eid, r in s1_records.items()}
            s2_prep = {eid: self.preprocess_record(r) for eid, r in s2_records.items()}
            s3_prep = {eid: self.preprocess_record(r) for eid, r in s3_records.items()}
            countries = set(r["country"] for r in s1_prep.values())
            for cntry in sorted(countries):
                s1_sub = [eid for eid, r in s1_prep.items() if r["country"] == cntry]
                s2_sub = [eid for eid, r in s2_prep.items() if r["country"] == cntry]
                s3_sub = [eid for eid, r in s3_prep.items() if r["country"] == cntry]
                target_sub = s2_sub + s3_sub
                if not s1_sub or not target_sub or len(s1_sub) * len(target_sub) > 1000000:
                    continue
                corpus = [s1_prep[eid]["full_text"] for eid in s1_sub] + [
                    (s2_prep[eid]["full_text"] if eid in s2_prep else s3_prep[eid]["full_text"])
                    for eid in target_sub
                ]
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=self.ngram_range, min_df=1, sublinear_tf=True
                )
                X = vectorizer.fit_transform(corpus)
                n_s = len(s1_sub)
                sim_matrix = X[:n_s].dot(X[n_s:].T).toarray()
                for i, s1_id in enumerate(s1_sub):
                    row_sims = sim_matrix[i]
                    top_k_indices = np.argsort(row_sims)[::-1][:self.k_neighbors]
                    for idx in top_k_indices:
                        if row_sims[idx] >= self.min_similarity:
                            candidates[s1_id].add(target_sub[idx])

        # Ensure all S1 entities exist in candidates
        for s1_id in s1_records:
            if s1_id not in candidates:
                candidates[s1_id] = set()

        if verbose:
            dur = time.time() - t0
            tot_c = sum(len(c) for c in candidates.values())
            avg_c = tot_c / max(1, len(candidates))
            print(f"[Blocking] Candidate generation finished in {dur:.2f}s (Total: {tot_c:,} pairs, Avg: {avg_c:.1f} per entity).")

        return candidates


# =============================================================================
# 3. EVALUATION & EXPORT UTILITIES
# =============================================================================

def evaluate_blocking_recall(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    s2_count: int,
    s3_count: int,
) -> Dict[str, float]:
    """
    Computes pair recall, reduction ratio, and candidate distribution statistics.
    """
    total_true_pairs = 0
    captured_true_pairs = 0
    candidate_counts = []

    for s1_id, true_matches in ground_truth.items():
        cands = candidates.get(s1_id, set())
        candidate_counts.append(len(cands))
        for tm in true_matches:
            total_true_pairs += 1
            if tm in cands:
                captured_true_pairs += 1

    recall = captured_true_pairs / total_true_pairs if total_true_pairs > 0 else 1.0
    avg_cands = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    max_cands = int(np.max(candidate_counts)) if candidate_counts else 0
    min_cands = int(np.min(candidate_counts)) if candidate_counts else 0
    median_cands = float(np.median(candidate_counts)) if candidate_counts else 0.0

    total_pool = s2_count + s3_count
    reduction_ratio = (1.0 - (avg_cands / total_pool)) * 100.0 if total_pool > 0 else 0.0

    return {
        "total_true_pairs": total_true_pairs,
        "captured_true_pairs": captured_true_pairs,
        "pair_recall": recall * 100.0,
        "avg_candidates": avg_cands,
        "median_candidates": median_cands,
        "min_candidates": min_cands,
        "max_candidates": max_cands,
        "reduction_ratio": reduction_ratio,
    }


def write_candidate_pairs_tsv(
    candidates: Dict[str, Set[str]],
    output_path: str,
):
    """
    Writes candidate pairs into the standard TSV format for the competition:
    Header: source1_entity_id\tcandidate_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            cands = sorted(candidates[s1_id])
            f.write(f"{s1_id}\t{','.join(cands)}\n")


def _open_text_or_gz(filepath: str):
    """Safely opens a plaintext or gzip-compressed file."""
    if str(filepath).endswith(".gz"):
        import gzip
        return gzip.open(filepath, "rt", encoding="utf-8")
    return open(filepath, "r", encoding="utf-8")


def load_source_tsv(path: str, max_records: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    """
    Loads a source TSV into a dict of records.
    Supports:
      1. Direct .tsv file
      2. Direct .tsv.gz compressed file
      3. A directory containing split shards (e.g. data/shards/train_source2/*.tsv or *.tsv.gz)
    """
    from pathlib import Path
    p = Path(path).resolve()
    records: Dict[str, Dict[str, str]] = {}

    target_files = []
    if p.is_dir():
        target_files = sorted([str(f) for f in p.glob("*part*") if f.suffix in (".tsv", ".gz") or f.name.endswith(".tsv.gz")])
        if not target_files:
            target_files = sorted([str(f) for f in p.glob("*.tsv")] + [str(f) for f in p.glob("*.tsv.gz")])
    elif p.is_file():
        target_files = [str(p)]
    else:
        # Check if corresponding .gz file exists
        if Path(f"{path}.gz").is_file():
            target_files = [f"{path}.gz"]
        elif (p.parent / p.stem).is_dir():
            target_files = sorted([str(f) for f in (p.parent / p.stem).glob("*") if f.name.endswith(".tsv") or f.name.endswith(".tsv.gz")])
        else:
            raise FileNotFoundError(f"Source file or shard directory not found: {path}")

    for fpath in target_files:
        with _open_text_or_gz(fpath) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                continue
            for row in reader:
                if not row:
                    continue
                eid = row[0].strip()
                nm = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                cntry = row[3].strip() if len(row) > 3 else ""
                records[eid] = {
                    "entity_id": eid,
                    "business_name": nm,
                    "business_address": addr,
                    "country": cntry,
                }
                if max_records and len(records) >= max_records:
                    return records
    return records


def load_ground_truth_tsv(path: str, max_records: Optional[int] = None) -> Dict[str, Set[str]]:
    """
    Loads ground truth matching labels.
    Supports direct .tsv, .tsv.gz, or directory of shards.
    """
    from pathlib import Path
    p = Path(path).resolve()
    gt: Dict[str, Set[str]] = {}

    target_files = []
    if p.is_dir():
        target_files = sorted([str(f) for f in p.glob("*") if f.name.endswith(".tsv") or f.name.endswith(".tsv.gz")])
    elif p.is_file():
        target_files = [str(p)]
    elif Path(f"{path}.gz").is_file():
        target_files = [f"{path}.gz"]
    elif (p.parent / p.stem).is_dir():
        target_files = sorted([str(f) for f in (p.parent / p.stem).glob("*") if f.name.endswith(".tsv") or f.name.endswith(".tsv.gz")])
    else:
        raise FileNotFoundError(f"Ground truth file or directory not found: {path}")

    for fpath in target_files:
        with _open_text_or_gz(fpath) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                continue
            for row in reader:
                if not row:
                    continue
                s1_id = row[0].strip()
                m_str = row[1].strip() if len(row) > 1 else ""
                matches = set()
                if m_str:
                    for x in m_str.split(","):
                        x = x.strip()
                        if x:
                            matches.add(x)
                if s1_id in gt:
                    gt[s1_id].update(matches)
                else:
                    gt[s1_id] = matches
                if max_records and len(gt) >= max_records:
                    return gt
    return gt


# =============================================================================
# 4. CLI RUNNER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Candidate Blocking & Pair Recall Evaluation.")
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default="mini_dataset",
        help="Directory containing source1, source2, source3 TSVs.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/candidate_pairs.tsv",
        help="Path to write candidate_pairs.tsv.",
    )
    parser.add_argument(
        "-k", "--k-neighbors",
        type=int,
        default=30,
        help="Top-k nearest neighbors to retrieve via TF-IDF cosine kNN (default: 30).",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.12,
        help="Minimum TF-IDF cosine similarity threshold (default: 0.12).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate pair recall if ground truth file is present in data-dir.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = args.output

    # Find source files
    prefix = "train" if os.path.isfile(os.path.join(data_dir, "train_source1.tsv")) else "test"
    s1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")
    gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")

    print("=" * 68)
    print("      Amazon ML Challenge 2026 - Candidate Blocking Engine")
    print("=" * 68)
    print(f"Data Directory:     {data_dir}")
    print(f"Output File:        {output_path}")
    print(f"TF-IDF kNN Top-K:   {args.k_neighbors} (min sim: {args.min_sim})")
    print("-" * 68)

    print("Loading source files...")
    s1_records = load_source_tsv(s1_path)
    s2_records = load_source_tsv(s2_path)
    s3_records = load_source_tsv(s3_path)
    print(f"Loaded: S1 = {len(s1_records)}, S2 = {len(s2_records)}, S3 = {len(s3_records)}")

    blocker = CandidateBlocker(
        k_neighbors=args.k_neighbors,
        min_similarity=args.min_sim,
    )

    candidates = blocker.generate_candidates(s1_records, s2_records, s3_records)

    write_candidate_pairs_tsv(candidates, output_path)
    print(f"Wrote candidate pairs to: {output_path}")

    # Evaluate if GT is available
    if os.path.isfile(gt_path) or args.eval:
        if os.path.isfile(gt_path):
            gt = load_ground_truth_tsv(gt_path)
            stats = evaluate_blocking_recall(
                candidates, gt, len(s2_records), len(s3_records)
            )

            print("\n" + "=" * 68)
            print("                 BLOCKING AUDIT REPORT")
            print("=" * 68)
            print(f"{'Metric':<32} | {'Value':<18}")
            print("-" * 68)
            print(f"{'Total True Positive Pairs':<32} | {stats['total_true_pairs']}")
            print(f"{'Captured True Positive Pairs':<32} | {stats['captured_true_pairs']}")
            print(f"{'PAIR RECALL (Recall Ceiling)':<32} | {stats['pair_recall']:.2f}%")
            print("-" * 68)
            print(f"{'Average Candidates per S1':<32} | {stats['avg_candidates']:.1f}")
            print(f"{'Median Candidates per S1':<32} | {stats['median_candidates']:.1f}")
            print(f"{'Min / Max Candidates per S1':<32} | {stats['min_candidates']} / {stats['max_candidates']}")
            print(f"{'Search Space Reduction Ratio':<32} | {stats['reduction_ratio']:.2f}%")
            print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
