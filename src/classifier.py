#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: classifier

Pairwise Feature Engineering & GBDT Classifier for Entity Resolution.
Features engineered per candidate pair using rapidfuzz:
  1. Name Metrics:
     - Jaro-Winkler, token_sort_ratio, token_set_ratio, ratio, char n-gram Jaccard
     - TF-IDF character n-gram cosine similarity
     - Canonical legal-suffix match (Inc, LLC, Ltd, Corp, Pvt Ltd, SARL, etc.)
     - Exact match, sorted token exact match, length diff, length ratio, first word match
  2. Address Metrics:
     - Street number exact match (1.0 match, 0.0 mismatch, -1.0 missing)
     - Postal / ZIP code exact match (1.0 match, 0.0 mismatch, -1.0 missing)
     - Street name similarity (Jaro-Winkler on stripped street name)
     - Secondary unit / apartment match (1.0 match, 0.0 mismatch, -1.0 missing)
     - Full address Jaro-Winkler, token_sort_ratio, token_set_ratio, ratio, char Jaccard
     - Address TF-IDF cosine similarity & address null indicator
     - State code exact match
  3. Phonetic Metrics:
     - Name Double Metaphone equality
     - Street Double Metaphone equality
     - Name Soundex equality & Street Soundex equality
  4. Source & Domain Indicators:
     - Source indicator (S1 x S2 vs S1 x S3 pairs behave differently)
     - Optional dual-model mode (--split-source-models)
     - Country indicators (US, India, France)
     - Combined full-text token set ratio & full-text TF-IDF cosine

Model:
  - LightGBM binary classifier trained on candidate pairs
  - Threshold optimization specifically for Macro F_0.5 score
  - Exports 'matching_results.tsv' and validates submission rules
"""

import argparse
import csv
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# System path bootstrap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from rapidfuzz import distance, fuzz
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
    from src.blocking import (
        CandidateBlocker,
        soundex,
        load_source_tsv,
        load_ground_truth_tsv,
        write_candidate_pairs_tsv,
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
    from blocking import (
        CandidateBlocker,
        soundex,
        load_source_tsv,
        load_ground_truth_tsv,
        write_candidate_pairs_tsv,
    )

try:
    from src.decision_layer import DecisionLayer
except ImportError:
    from decision_layer import DecisionLayer


# =============================================================================
# 1. FEATURE NAMES & EXTRACTION
# =============================================================================
FEATURE_NAMES: List[str] = [
    # 1. Name features
    "nm_jw",                   # Jaro-Winkler similarity on normalized name
    "nm_sort_ratio",           # Token sort ratio (order-invariant)
    "nm_set_ratio",            # Token set ratio (handles extra tokens/abbreviations)
    "nm_ratio",                # Standard Levenshtein ratio
    "nm_exact",                # Binary exact normalized name match
    "nm_sort_exact",           # Binary exact sorted token match
    "nm_char_jaccard",         # Character 3-gram Jaccard similarity
    "nm_tfidf_cosine",         # TF-IDF cosine similarity of name character n-grams
    "nm_legal_suffix_match",   # Legal suffix match (1.0=match, 0.0=mismatch, -1.0=missing)
    "nm_len_diff",             # Absolute character length difference
    "nm_len_ratio",            # Length ratio min(l1,l2)/max(l1,l2)
    "nm_first_word_match",     # Binary exact match on first token
    # 2. Address features
    "ad_jw",                   # Jaro-Winkler similarity on normalized address
    "ad_sort_ratio",           # Token sort ratio on address
    "ad_set_ratio",            # Token set ratio on address
    "ad_ratio",                # Standard Levenshtein ratio on address
    "ad_char_jaccard",         # Character 3-gram Jaccard similarity of address
    "ad_tfidf_cosine",         # TF-IDF cosine similarity of normalized address (0.0 if missing)
    "ad_street_similarity",    # Jaro-Winkler similarity on extracted street name
    "ad_unit_match",           # Secondary unit/apt match (1.0=match, 0.0=mismatch, -1.0=missing)
    "ad_street_num_match",     # Street/house number exact match (1.0=match, 0.0=mismatch, -1.0=missing)
    "ad_state_match",          # State code exact match (1.0=match, 0.0=mismatch, -1.0=missing)
    "ad_zip_match",            # Postal/ZIP code match (1.0=match, 0.0=mismatch, -1.0=missing)
    "ad_is_null",              # Indicator if candidate address is null/empty
    # 3. Phonetic features
    "nm_metaphone_match",      # Double Metaphone match on primary name word
    "street_metaphone_match",  # Double Metaphone match on primary street word
    "nm_soundex_match",        # Soundex match on primary name word
    "street_soundex_match",    # Soundex match on primary street word
    # 4. Source & Domain Indicators
    "comb_set_ratio",          # Token set ratio on combined full_text
    "full_tfidf_cosine",       # TF-IDF cosine on combined full_text
    "is_s2",                   # Indicator if candidate is from Source 2
    "is_s3",                   # Indicator if candidate is from Source 3
    "is_us",                   # Country indicator: United States
    "is_india",                # Country indicator: India
    "is_france",               # Country indicator: France
]


def char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Generates set of character n-grams from a string."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def fallback_char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Computes cosine similarity of character n-grams as a vectorizer-free fallback."""
    if not s1 or not s2:
        return 0.0
    g1 = char_ngrams(s1, n)
    g2 = char_ngrams(s2, n)
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / math.sqrt(len(g1) * len(g2))


def extract_pair_features(
    r1: Dict[str, any],
    r2: Dict[str, any],
    nm_tfidf_cos: Optional[float] = None,
    ad_tfidf_cos: Optional[float] = None,
    full_tfidf_cos: Optional[float] = None,
) -> List[float]:
    """
    Extracts dense pairwise similarity features between Source 1 record (r1)
    and candidate Source 2 or Source 3 record (r2).
    """
    n1 = r1["name_clean"]
    n2 = r2["name_clean"]
    n1_sort = r1["name_sorted"]
    n2_sort = r2["name_sorted"]

    a1 = r1["addr_clean"]
    a2 = r2["addr_clean"]

    # -------------------------------------------------------------
    # 1. Name Features
    # -------------------------------------------------------------
    if n1 == n2 and n1:
        nm_jw = nm_sort_ratio = nm_set_ratio = nm_ratio = nm_exact = nm_sort_exact = nm_char_jaccard = 1.0
        nm_len_diff = 0.0
        nm_len_ratio = 1.0
        nm_first_word_match = 1.0
    else:
        nm_jw = distance.JaroWinkler.similarity(n1, n2) if n1 and n2 else 0.0
        nm_sort_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        nm_set_ratio = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        nm_ratio = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        nm_exact = 0.0
        nm_sort_exact = 1.0 if n1_sort == n2_sort and n1_sort else 0.0

        g1 = r1.get("name_ngrams")
        if g1 is None:
            g1 = char_ngrams(n1, 3)
        g2 = r2.get("name_ngrams")
        if g2 is None:
            g2 = char_ngrams(n2, 3)
        if g1 and g2:
            inter = len(g1.intersection(g2))
            nm_char_jaccard = inter / (len(g1) + len(g2) - inter)
        else:
            nm_char_jaccard = 0.0

        len1, len2 = len(n1), len(n2)
        nm_len_diff = float(abs(len1 - len2))
        nm_len_ratio = min(len1, len2) / max(len1, len2, 1)

        w1 = r1.get("first_word") or (n1.split()[0] if n1.split() else "")
        w2 = r2.get("first_word") or (n2.split()[0] if n2.split() else "")
        nm_first_word_match = 1.0 if w1 and w2 and w1 == w2 else 0.0

    # TF-IDF Cosine
    if nm_tfidf_cos is not None:
        nm_tfidf = float(nm_tfidf_cos)
    else:
        nm_tfidf = fallback_char_ngram_cosine(n1, n2, 3)

    # Legal Suffix Match (1.0 = match, 0.0 = mismatch, -1.0 = either missing)
    suf1 = r1.get("legal_suffix", "")
    suf2 = r2.get("legal_suffix", "")
    if suf1 and suf2:
        nm_legal_suffix_match = 1.0 if suf1 == suf2 else 0.0
    else:
        nm_legal_suffix_match = -1.0

    # -------------------------------------------------------------
    # 2. Address Features
    # -------------------------------------------------------------
    if a1 == a2 and a1:
        ad_jw = ad_sort_ratio = ad_set_ratio = ad_ratio = ad_char_jaccard = 1.0
        ad_is_null = 0.0
    else:
        ad_jw = distance.JaroWinkler.similarity(a1, a2) if a1 and a2 else 0.0
        ad_sort_ratio = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        ad_set_ratio = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        ad_ratio = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

        ga1 = r1.get("addr_ngrams")
        if ga1 is None:
            ga1 = char_ngrams(a1, 3)
        ga2 = r2.get("addr_ngrams")
        if ga2 is None:
            ga2 = char_ngrams(a2, 3)
        if ga1 and ga2:
            inter_a = len(ga1.intersection(ga2))
            ad_char_jaccard = inter_a / (len(ga1) + len(ga2) - inter_a)
        else:
            ad_char_jaccard = 0.0
        ad_is_null = 1.0 if not a2 else 0.0

    # Address TF-IDF Cosine
    if ad_tfidf_cos is not None:
        ad_tfidf = float(ad_tfidf_cos) if a1 and a2 else 0.0
    else:
        ad_tfidf = fallback_char_ngram_cosine(a1, a2, 3) if a1 and a2 else 0.0

    # Street Name Similarity
    st1 = r1.get("street_name", "")
    st2 = r2.get("street_name", "")
    ad_street_similarity = distance.JaroWinkler.similarity(st1, st2) if st1 and st2 else 0.0

    # Secondary Unit / Apt Match (1.0 = match, 0.0 = mismatch, -1.0 = missing)
    u1 = r1.get("unit", "")
    u2 = r2.get("unit", "")
    if u1 and u2:
        ad_unit_match = 1.0 if u1 == u2 else 0.0
    else:
        ad_unit_match = -1.0

    # Street / House number exact match
    nums1 = r1.get("numbers", [])
    nums2 = r2.get("numbers", [])
    if nums1 and nums2:
        ad_street_num_match = 1.0 if nums1[0] == nums2[0] else 0.0
    else:
        ad_street_num_match = -1.0

    # State code match
    state1 = r1.get("state", "")
    state2 = r2.get("state", "")
    if state1 and state2:
        ad_state_match = 1.0 if state1 == state2 else 0.0
    else:
        ad_state_match = -1.0

    # Postal code match
    postal1 = r1.get("postal", "")
    postal2 = r2.get("postal", "")
    if postal1 and postal2:
        ad_zip_match = 1.0 if postal1 == postal2 else 0.0
    else:
        ad_zip_match = -1.0

    # -------------------------------------------------------------
    # 3. Phonetic Features (Metaphone & Soundex)
    # -------------------------------------------------------------
    # Name Metaphone match
    m1 = r1.get("nm_metaphone", ("", ""))
    m2 = r2.get("nm_metaphone", ("", ""))
    if (m1[0] and (m1[0] == m2[0] or m1[0] == m2[1])) or (m1[1] and (m1[1] == m2[0] or m1[1] == m2[1])):
        nm_metaphone_match = 1.0
    else:
        nm_metaphone_match = 0.0

    # Street Metaphone match
    sm1 = r1.get("street_metaphone", ("", ""))
    sm2 = r2.get("street_metaphone", ("", ""))
    if (sm1[0] and (sm1[0] == sm2[0] or sm1[0] == sm2[1])) or (sm1[1] and (sm1[1] == sm2[0] or sm1[1] == sm2[1])):
        street_metaphone_match = 1.0
    else:
        street_metaphone_match = 0.0

    # Soundex matches
    sx1 = r1.get("nm_soundex", "")
    sx2 = r2.get("nm_soundex", "")
    nm_soundex_match = 1.0 if sx1 and sx2 and sx1 == sx2 else 0.0

    ssx1 = r1.get("street_soundex", "")
    ssx2 = r2.get("street_soundex", "")
    street_soundex_match = 1.0 if ssx1 and ssx2 and ssx1 == ssx2 else 0.0

    # -------------------------------------------------------------
    # 4. Combined & Contextual Features
    # -------------------------------------------------------------
    comb1 = r1["full_text"]
    comb2 = r2["full_text"]
    if comb1 == comb2 and comb1:
        comb_set_ratio = 1.0
    else:
        comb_set_ratio = fuzz.token_set_ratio(comb1, comb2) / 100.0 if comb1 and comb2 else 0.0

    # Full text TF-IDF Cosine
    if full_tfidf_cos is not None:
        full_tfidf = float(full_tfidf_cos)
    else:
        full_tfidf = fallback_char_ngram_cosine(comb1, comb2, 3)

    # Source Indicator (S1 x S2 vs S1 x S3 behavior)
    cid = r2["entity_id"]
    is_s2 = 1.0 if cid.startswith("S2-") else 0.0
    is_s3 = 1.0 if cid.startswith("S3-") else 0.0

    # Country Indicator
    cntry = (r1.get("country", "")).upper()
    is_us = 1.0 if cntry == "US" else 0.0
    is_in = 1.0 if cntry == "INDIA" else 0.0
    is_fr = 1.0 if cntry == "FRANCE" else 0.0

    return [
        nm_jw, nm_sort_ratio, nm_set_ratio, nm_ratio, nm_exact, nm_sort_exact,
        nm_char_jaccard, nm_tfidf, nm_legal_suffix_match, nm_len_diff, nm_len_ratio,
        nm_first_word_match,
        ad_jw, ad_sort_ratio, ad_set_ratio, ad_ratio, ad_char_jaccard, ad_tfidf,
        ad_street_similarity, ad_unit_match, ad_street_num_match, ad_state_match,
        ad_zip_match, ad_is_null,
        nm_metaphone_match, street_metaphone_match, nm_soundex_match, street_soundex_match,
        comb_set_ratio, full_tfidf, is_s2, is_s3, is_us, is_in, is_fr
    ]


# =============================================================================
# 2. EVALUATION METRIC (MACRO F_0.5 SCORE)
# =============================================================================
def compute_macro_f05(
    pred_matches_dict: Dict[str, Set[str]],
    true_matches_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
) -> float:
    """
    Computes macro-averaged F_0.5 score across all Source 1 entities,
    strictly adhering to competition rules (singletons included, beta=0.5).
    """
    scores = []
    for s1_id in all_s1_ids:
        pred_set = pred_matches_dict.get(s1_id, set())
        true_set = true_matches_dict.get(s1_id, set())

        # Singleton case (true matches is empty)
        if not true_set:
            if not pred_set:
                scores.append(1.0)  # Correctly identified singleton
            else:
                scores.append(0.0)  # False merge on singleton
            continue

        # Matched entity case
        if not pred_set:
            scores.append(0.0)  # Missed all true links
            continue

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = 0.25 * precision + recall
        if denom > 0:
            f05 = (1.25 * precision * recall) / denom
        else:
            f05 = 0.0
        scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# 3. ENTITY RESOLUTION CLASSIFIER MODEL
# =============================================================================
class EntityResolutionClassifier:
    """
    LightGBM pairwise matching classifier for entity resolution.
    Supports unified model (with source indicator features) or split models.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        split_source_models: bool = False,
        use_decision_layer: bool = True,
        n_jobs: int = -1,
    ):
        self.threshold = threshold
        self.threshold_s2 = threshold
        self.threshold_s3 = threshold
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.split_source_models = split_source_models
        self.use_decision_layer = use_decision_layer
        self.n_jobs = n_jobs
        self.decision_layer = DecisionLayer()

        self.model: Optional[lgb.LGBMClassifier] = None
        self.model_s2: Optional[lgb.LGBMClassifier] = None
        self.model_s3: Optional[lgb.LGBMClassifier] = None

        self.tfidf_name = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True
        )
        self.tfidf_addr = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True
        )
        self.tfidf_full = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True
        )
        self.is_vectorizer_fitted = False

        self.blocker = CandidateBlocker(k_neighbors=30, min_similarity=0.12)
        
        # High-throughput inference caches for target (S2/S3) records
        self._target_prep_cache: Dict[str, Dict[str, any]] = {}
        self._target_tfidf_cache: Dict[str, Tuple[any, any, any]] = {}
        self._target_tfidf_nm: Optional[sp.csr_matrix] = None
        self._target_tfidf_ad: Optional[sp.csr_matrix] = None
        self._target_tfidf_fl: Optional[sp.csr_matrix] = None
        self._target_cid_to_row: Dict[str, int] = {}

    def clear_inference_cache(self):
        """Clears cached preprocessed and vectorized target records to free memory."""
        self._target_prep_cache.clear()
        self._target_tfidf_cache.clear()
        self._target_tfidf_nm = None
        self._target_tfidf_ad = None
        self._target_tfidf_fl = None
        self._target_cid_to_row.clear()

    def _fit_vectorizers(self, all_prep_records: List[Dict[str, any]]):
        """Fits TF-IDF character n-gram vectorizers on record corpora."""
        names = [r["name_clean"] for r in all_prep_records if r["name_clean"]]
        addrs = [r["addr_clean"] for r in all_prep_records if r["addr_clean"]]
        fulls = [r["full_text"] for r in all_prep_records if r["full_text"]]

        self.tfidf_name.fit(names if names else [""])
        self.tfidf_addr.fit(addrs if addrs else [""])
        self.tfidf_full.fit(fulls if fulls else [""])
        self.is_vectorizer_fitted = True

    def _compute_vectorized_matrices(self, all_prep_records: List[Dict[str, any]]):
        """Transforms records into sparse TF-IDF matrices."""
        names = [r["name_clean"] for r in all_prep_records]
        addrs = [r["addr_clean"] for r in all_prep_records]
        fulls = [r["full_text"] for r in all_prep_records]

        X_name = self.tfidf_name.transform(names)
        X_addr = self.tfidf_addr.transform(addrs)
        X_full = self.tfidf_full.transform(fulls)
        return X_name, X_addr, X_full

    def fit(
        self,
        s1_records: Dict[str, Dict[str, str]],
        s2_records: Dict[str, Dict[str, str]],
        s3_records: Dict[str, Dict[str, str]],
        ground_truth: Dict[str, Set[str]],
        tune_threshold: bool = True,
        verbose: bool = True,
    ):
        """
        Extracts candidate pairs, computes 35 pairwise features, trains LightGBM,
        and tunes decision threshold for optimal Macro F0.5.
        """
        t0 = time.time()
        if verbose:
            print("[Model] 1/3 Preprocessing records & generating candidate pairs...")

        candidates = self.blocker.generate_candidates(
            s1_records, s2_records, s3_records, verbose=False
        )

        s1_prep = {eid: self.blocker.preprocess_record(r) for eid, r in s1_records.items()}
        
        # Only preprocess and vectorize target records that appear in candidate pairs or ground truth
        needed_target_ids = set()
        for cset in candidates.values():
            needed_target_ids.update(cset)
        for s1_id, gt_set in ground_truth.items():
            needed_target_ids.update(gt_set)

        all_targets = {**s2_records, **s3_records}
        target_recs = {eid: all_targets[eid] for eid in needed_target_ids if eid in all_targets}
        target_prep = {eid: self.blocker.preprocess_record(r) for eid, r in target_recs.items()}

        all_records = list(s1_prep.values()) + list(target_prep.values())
        self._fit_vectorizers(all_records)
        X_name, X_addr, X_full = self._compute_vectorized_matrices(all_records)
        id_to_idx = {r["entity_id"]: i for i, r in enumerate(all_records)}

        if verbose:
            print("[Model] 2/3 Building feature matrix on candidate pairs (35 features)...")

        # Collect pair metadata and compute TF-IDF dot products
        pair_list = []
        for s1_id, cand_set in candidates.items():
            for cid in cand_set:
                pair_list.append((s1_id, cid))

        if not pair_list:
            raise ValueError("No candidate pairs generated by blocker!")

        idx1_list = [id_to_idx[p[0]] for p in pair_list]
        idx2_list = [id_to_idx[p[1]] for p in pair_list]

        # Vectorized dot products across all candidate pairs
        nm_sims = np.asarray(X_name[idx1_list].multiply(X_name[idx2_list]).sum(axis=1)).ravel()
        ad_sims = np.asarray(X_addr[idx1_list].multiply(X_addr[idx2_list]).sum(axis=1)).ravel()
        full_sims = np.asarray(X_full[idx1_list].multiply(X_full[idx2_list]).sum(axis=1)).ravel()

        X: List[List[float]] = []
        y: List[int] = []

        for (s1_id, cid), nm_s, ad_s, full_s in zip(pair_list, nm_sims, ad_sims, full_sims):
            r1 = s1_prep[s1_id]
            r2 = target_prep[cid]
            feat = extract_pair_features(
                r1, r2,
                nm_tfidf_cos=nm_s,
                ad_tfidf_cos=ad_s if r1["addr_clean"] and r2["addr_clean"] else 0.0,
                full_tfidf_cos=full_s
            )
            label = 1 if cid in ground_truth.get(s1_id, set()) else 0
            X.append(feat)
            y.append(label)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        if verbose:
            print(f"[Model] Feature matrix shape: {X_arr.shape} ({n_pos} positive, {n_neg} negative pairs)")
            print("[Model] 3/3 Training LightGBM GBDT classifier...")

        if not self.split_source_models:
            # Single unified model with source indicator features
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
                n_jobs=self.n_jobs,
            )
            self.model.fit(X_arr, y_arr)
            probs = self.model.predict_proba(X_arr)[:, 1]
        else:
            # Dual models: S1xS2 vs S1xS3
            is_s2_mask = np.array([p[1].startswith("S2-") for p in pair_list])
            is_s3_mask = ~is_s2_mask

            if verbose:
                print(f"[Model] Dual mode: {np.sum(is_s2_mask)} S1xS2 pairs, {np.sum(is_s3_mask)} S1xS3 pairs")

            self.model_s2 = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
                n_jobs=self.n_jobs,
            )
            self.model_s2.fit(X_arr[is_s2_mask], y_arr[is_s2_mask])

            self.model_s3 = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
                n_jobs=self.n_jobs,
            )
            self.model_s3.fit(X_arr[is_s3_mask], y_arr[is_s3_mask])

            probs = np.zeros(len(pair_list), dtype=np.float32)
            if np.sum(is_s2_mask) > 0:
                probs[is_s2_mask] = self.model_s2.predict_proba(X_arr[is_s2_mask])[:, 1]
            if np.sum(is_s3_mask) > 0:
                probs[is_s3_mask] = self.model_s3.predict_proba(X_arr[is_s3_mask])[:, 1]

        # Group candidate probabilities per S1 entity
        cand_probs_by_s1 = defaultdict(list)
        for (s1_id, cid), p in zip(pair_list, probs):
            cand_probs_by_s1[s1_id].append((cid, float(p)))
        for s1_id in list(s1_records.keys()):
            cand_probs_by_s1[s1_id].sort(key=lambda x: -x[1])

        if tune_threshold:
            if self.use_decision_layer:
                if verbose:
                    print("[Model] Tuning Stage-2 Decision Layer & Asymmetric Thresholds for Macro F_0.5...")
                self.decision_layer.fit_and_tune(cand_probs_by_s1, ground_truth, verbose=verbose)
                self.threshold = self.decision_layer.policy.t_first
                self.threshold_s2 = self.decision_layer.policy.t_first
                self.threshold_s3 = self.decision_layer.policy.t_first
            else:
                if verbose:
                    print("[Model] Tuning threshold for Macro F_0.5 score...")
                best_thresh = self.threshold
                best_f05 = -1.0
                candidate_thresholds = np.linspace(0.20, 0.85, 27)

                for th in candidate_thresholds:
                    preds_by_s1 = defaultdict(set)
                    for (s1_id, cid), p in zip(pair_list, probs):
                        if p >= th:
                            preds_by_s1[s1_id].add(cid)

                    score = compute_macro_f05(preds_by_s1, ground_truth, list(s1_records.keys()))
                    if score > best_f05:
                        best_f05 = score
                        best_thresh = float(th)

                self.threshold = best_thresh
                self.threshold_s2 = best_thresh
                self.threshold_s3 = best_thresh
                if verbose:
                    print(f"[Model] Optimal threshold: {self.threshold:.3f} (Train Macro F0.5 = {best_f05:.4f})")

        dur = time.time() - t0
        if verbose:
            print(f"[Model] Training completed in {dur:.2f}s.")

    def predict(
        self,
        s1_records: Dict[str, Dict[str, str]],
        s2_records: Dict[str, Dict[str, str]],
        s3_records: Dict[str, Dict[str, str]],
        candidates: Optional[Dict[str, Set[str]]] = None,
        verbose: bool = True,
    ) -> Dict[str, Set[str]]:
        """
        Runs candidate generation (if not provided), scores pairs using 35 features,
        and returns {s1_id: set(matched_s2_s3_ids)}.
        Optimized with target caching and multi-threaded parallel feature extraction.
        """
        if self.model is None and self.model_s2 is None:
            raise ValueError("Model must be trained before predicting.")

        if candidates is None:
            candidates = self.blocker.generate_candidates(
                s1_records, s2_records, s3_records, verbose=verbose
            )

        # 1. Preprocess S1 records for this chunk
        s1_prep = {eid: self.blocker.preprocess_record(r) for eid, r in s1_records.items()}

        # 2. Preprocess target records with cross-chunk cache
        needed_cids = set()
        for cset in candidates.values():
            needed_cids.update(cset)

        # Only prune cache under severe memory pressure (< 2.5 GB free RAM)
        try:
            from src.utils.system import get_available_ram_gb
            if get_available_ram_gb() < 2.5:
                self.clear_inference_cache()
        except Exception:
            if len(self._target_prep_cache) > 1500000:
                self.clear_inference_cache()

        missing_cids = [cid for cid in needed_cids if cid not in self._target_prep_cache]
        for cid in missing_cids:
            if cid in s2_records:
                self._target_prep_cache[cid] = self.blocker.preprocess_record(s2_records[cid])
            elif cid in s3_records:
                self._target_prep_cache[cid] = self.blocker.preprocess_record(s3_records[cid])

        target_prep = {cid: self._target_prep_cache[cid] for cid in needed_cids if cid in self._target_prep_cache}

        # 3. Assemble candidate pair list
        pair_list = []
        for s1_id, cand_set in candidates.items():
            for cid in sorted(cand_set):
                if cid in target_prep:
                    pair_list.append((s1_id, cid))

        predictions: Dict[str, Set[str]] = {eid: set() for eid in s1_records}
        if not pair_list:
            return predictions

        # 4. High-performance pre-stacked target TF-IDF indexing (0.029s vs 1.15s)
        missing_tfidf_cids = [cid for cid in needed_cids if cid in target_prep and cid not in self._target_cid_to_row]
        if missing_tfidf_cids:
            missing_target_prep = [target_prep[cid] for cid in missing_tfidf_cids]
            t_nm, t_ad, t_fl = self._compute_vectorized_matrices(missing_target_prep)
            start_row = len(self._target_cid_to_row)
            for idx, cid in enumerate(missing_tfidf_cids):
                self._target_cid_to_row[cid] = start_row + idx
            if self._target_tfidf_nm is None:
                self._target_tfidf_nm = t_nm
                self._target_tfidf_ad = t_ad
                self._target_tfidf_fl = t_fl
            else:
                self._target_tfidf_nm = sp.vstack([self._target_tfidf_nm, t_nm], format="csr")
                self._target_tfidf_ad = sp.vstack([self._target_tfidf_ad, t_ad], format="csr")
                self._target_tfidf_fl = sp.vstack([self._target_tfidf_fl, t_fl], format="csr")

        # Vectorize S1 chunk records
        s1_prep_list = list(s1_prep.values())
        X_name_s1, X_addr_s1, X_full_s1 = self._compute_vectorized_matrices(s1_prep_list)
        s1_id_to_idx = {r["entity_id"]: i for i, r in enumerate(s1_prep_list)}

        idx1_list = [s1_id_to_idx[p[0]] for p in pair_list]
        idx2_list = [self._target_cid_to_row[p[1]] for p in pair_list]

        # Fast O(1) slice multiplication without Python sp.vstack row iteration
        nm_sims = np.asarray(X_name_s1[idx1_list].multiply(self._target_tfidf_nm[idx2_list]).sum(axis=1)).ravel()
        ad_sims = np.asarray(X_addr_s1[idx1_list].multiply(self._target_tfidf_ad[idx2_list]).sum(axis=1)).ravel()
        full_sims = np.asarray(X_full_s1[idx1_list].multiply(self._target_tfidf_fl[idx2_list]).sum(axis=1)).ravel()

        # 5. Multi-threaded feature extraction
        effective_n_jobs = self.n_jobs
        if effective_n_jobs == -1:
            effective_n_jobs = min(os.cpu_count() or 1, 16)

        n_pairs = len(pair_list)
        if n_pairs > 500 and effective_n_jobs > 1:
            batch_size = max(500, n_pairs // (effective_n_jobs * 4))
            pair_data = list(zip(pair_list, nm_sims, ad_sims, full_sims))
            sub_chunks = [pair_data[i:i + batch_size] for i in range(0, n_pairs, batch_size)]

            def _extract_subchunk(sub_items):
                out = []
                for (s1_id, cid), nm_s, ad_s, full_s in sub_items:
                    r1 = s1_prep[s1_id]
                    r2 = target_prep[cid]
                    out.append(extract_pair_features(
                        r1, r2,
                        nm_tfidf_cos=nm_s,
                        ad_tfidf_cos=ad_s if r1["addr_clean"] and r2["addr_clean"] else 0.0,
                        full_tfidf_cos=full_s
                    ))
                return out

            chunk_features = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
                delayed(_extract_subchunk)(c) for c in sub_chunks
            )
            X = [f for sub in chunk_features for f in sub]
        else:
            X = []
            for (s1_id, cid), nm_s, ad_s, full_s in zip(pair_list, nm_sims, ad_sims, full_sims):
                r1 = s1_prep[s1_id]
                r2 = target_prep[cid]
                X.append(extract_pair_features(
                    r1, r2,
                    nm_tfidf_cos=nm_s,
                    ad_tfidf_cos=ad_s if r1["addr_clean"] and r2["addr_clean"] else 0.0,
                    full_tfidf_cos=full_s
                ))

        X_arr = np.array(X, dtype=np.float32)

        if not self.split_source_models:
            probs = self.model.predict_proba(X_arr)[:, 1]
        else:
            is_s2_mask = np.array([p[1].startswith("S2-") for p in pair_list])
            is_s3_mask = ~is_s2_mask

            probs = np.zeros(len(pair_list), dtype=np.float32)
            if np.sum(is_s2_mask) > 0 and self.model_s2 is not None:
                probs[is_s2_mask] = self.model_s2.predict_proba(X_arr[is_s2_mask])[:, 1]
            if np.sum(is_s3_mask) > 0 and self.model_s3 is not None:
                probs[is_s3_mask] = self.model_s3.predict_proba(X_arr[is_s3_mask])[:, 1]

        # Group candidate probabilities per S1 entity
        cand_probs_by_s1 = defaultdict(list)
        for (s1_id, cid), p in zip(pair_list, probs):
            cand_probs_by_s1[s1_id].append((cid, float(p)))
        for s1_id in list(s1_records.keys()):
            cand_probs_by_s1[s1_id].sort(key=lambda x: -x[1])

        # Stage-2 Decision Layer prediction (handles singleton gating & asymmetric thresholding)
        if self.use_decision_layer and self.decision_layer is not None and self.decision_layer.is_optimized:
            return self.decision_layer.predict(cand_probs_by_s1)

        # Fallback flat thresholding
        for (s1_id, cid), p in zip(pair_list, probs):
            th = self.threshold_s2 if cid.startswith("S2-") else self.threshold_s3
            if p >= th:
                predictions[s1_id].add(cid)

        return predictions

    def save(self, model_path: str):
        """Saves trained model, vectorizers, and configuration to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        payload = {
            "model": self.model,
            "model_s2": self.model_s2,
            "model_s3": self.model_s3,
            "threshold": self.threshold,
            "threshold_s2": self.threshold_s2,
            "threshold_s3": self.threshold_s3,
            "split_source_models": self.split_source_models,
            "use_decision_layer": self.use_decision_layer,
            "decision_layer": self.decision_layer,
            "tfidf_name": self.tfidf_name,
            "tfidf_addr": self.tfidf_addr,
            "tfidf_full": self.tfidf_full,
            "feature_names": FEATURE_NAMES,
        }
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, model_path: str, n_jobs: int = -1) -> "EntityResolutionClassifier":
        """Loads trained model, vectorizers, and configuration from disk."""
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        clf = cls(
            threshold=payload["threshold"],
            split_source_models=payload.get("split_source_models", False),
            use_decision_layer=payload.get("use_decision_layer", True),
            n_jobs=n_jobs,
        )
        clf.model = payload.get("model")
        clf.model_s2 = payload.get("model_s2")
        clf.model_s3 = payload.get("model_s3")
        clf.threshold_s2 = payload.get("threshold_s2", payload["threshold"])
        clf.threshold_s3 = payload.get("threshold_s3", payload["threshold"])
        clf.decision_layer = payload.get("decision_layer")
        clf.tfidf_name = payload["tfidf_name"]
        clf.tfidf_addr = payload["tfidf_addr"]
        clf.tfidf_full = payload["tfidf_full"]
        clf.is_vectorizer_fitted = True

        eff_jobs = min(os.cpu_count() or 1, 16) if n_jobs == -1 else n_jobs
        for m in [clf.model, clf.model_s2, clf.model_s3]:
            if m is not None and hasattr(m, "set_params"):
                try:
                    m.set_params(n_jobs=eff_jobs)
                except Exception:
                    pass
        return clf


# =============================================================================
# 4. EXPORT MATCHING RESULTS TSV
# =============================================================================
def write_matching_results_tsv(
    predictions: Dict[str, Set[str]],
    output_path: str,
):
    """
    Writes matching results into official TSV format:
    Header: source1_entity_id\tmatched_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(predictions.keys()):
            matches = sorted(predictions[s1_id])
            f.write(f"{s1_id}\t{','.join(matches)}\n")


# =============================================================================
# 5. CLI RUNNER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pairwise Feature GBDT Entity Resolver.")
    parser.add_argument(
        "--train-dir",
        type=str,
        default="dataset_split/train",
        help="Training dataset directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset_split/test",
        help="Test dataset directory.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="Output directory for matching_results.tsv and candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/lgbm_entity_resolver.pkl",
        help="Path to save trained LightGBM model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Initial probability threshold (tuned automatically if train labels available).",
    )
    parser.add_argument(
        "--split-source-models",
        action="store_true",
        help="Train two separate GBDT models for S1xS2 and S1xS3 candidate pairs.",
    )
    parser.add_argument(
        "--no-decision-layer",
        action="store_true",
        help="Disable Stage-2 Decision Layer and use flat thresholding.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("   Amazon ML Challenge 2026 - GBDT Entity Resolution Pipeline")
    print("=" * 72)
    print(f"Train Directory:       {args.train_dir}")
    print(f"Test Directory:        {args.test_dir}")
    print(f"Output Directory:      {args.output_dir}")
    print(f"Model Save Path:       {args.model_path}")
    print(f"Split Source Models:   {args.split_source_models}")
    print(f"Use Decision Layer:    {not args.no_decision_layer}")
    print("-" * 72)

    # 1. Load Training Data
    print("Loading training data...")
    s1_tr = load_source_tsv(os.path.join(args.train_dir, "train_source1.tsv"))
    s2_tr = load_source_tsv(os.path.join(args.train_dir, "train_source2.tsv"))
    s3_tr = load_source_tsv(os.path.join(args.train_dir, "train_source3.tsv"))
    gt_tr = load_ground_truth_tsv(os.path.join(args.train_dir, "train_ground_truth.tsv"))

    # 2. Train Model
    clf = EntityResolutionClassifier(
        threshold=args.threshold,
        split_source_models=args.split_source_models,
        use_decision_layer=not args.no_decision_layer,
    )
    clf.fit(s1_tr, s2_tr, s3_tr, gt_tr, tune_threshold=True, verbose=True)
    clf.save(args.model_path)
    print(f"Saved trained model to {args.model_path}")

    # Decision Layer Policy Summary
    if clf.use_decision_layer and clf.decision_layer is not None and clf.decision_layer.is_optimized:
        p = clf.decision_layer.policy
        print("\nTrained Stage-2 Decision Layer Policy:")
        print(f"  * Estimated Singleton Prior:   {clf.decision_layer.gatekeeper.singleton_prior * 100:.2f}%")
        print(f"  * Singleton Hurdle (tau_gate): {p.tau_gate:.3f}")
        print(f"  * Primary Match Bar (t_first): {p.t_first:.3f}")
        print(f"  * Multi-Match Bar (t_second):  {p.t_second:.3f}")
        print(f"  * Relative Margin (delta):     {p.delta_margin:.3f}")

    # Top Feature Importances
    active_model = clf.model if clf.model is not None else clf.model_s2
    if active_model is not None:
        importances = sorted(
            zip(FEATURE_NAMES, active_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nTop 15 Feature Importances:")
        for fname, imp in importances[:15]:
            print(f"  {fname:<24}: {imp}")

    # 3. Inference on Test Set
    print("\nLoading test data & running candidate generation...")
    prefix = "train" if os.path.isfile(os.path.join(args.test_dir, "train_source1.tsv")) else "test"
    s1_te = load_source_tsv(os.path.join(args.test_dir, f"{prefix}_source1.tsv"))
    s2_te = load_source_tsv(os.path.join(args.test_dir, f"{prefix}_source2.tsv"))
    s3_te = load_source_tsv(os.path.join(args.test_dir, f"{prefix}_source3.tsv"))

    # Generate blocking candidates
    cands_te = clf.blocker.generate_candidates(s1_te, s2_te, s3_te, verbose=True)

    # Save candidate_pairs.tsv
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    write_candidate_pairs_tsv(cands_te, cand_path)
    print(f"Wrote candidate pairs to: {cand_path}")

    # Predict matching records
    print("Running LightGBM pairwise scoring and classification...")
    predictions = clf.predict(s1_te, s2_te, s3_te, candidates=cands_te, verbose=False)

    # Save matching_results.tsv
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    write_matching_results_tsv(predictions, match_path)
    print(f"Wrote matching results to: {match_path}")

    # 4. Evaluate Test Set if Ground Truth Available
    gt_te_path = os.path.join(args.test_dir, f"{prefix}_ground_truth.tsv")
    if os.path.isfile(gt_te_path):
        gt_te = load_ground_truth_tsv(gt_te_path)
        test_macro_f05 = compute_macro_f05(predictions, gt_te, list(s1_te.keys()))

        tp = sum(len(predictions[s] & gt_te[s]) for s in s1_te)
        all_pred = sum(len(predictions[s]) for s in s1_te)
        all_true = sum(len(gt_te[s]) for s in s1_te)

        prec = tp / all_pred if all_pred > 0 else 1.0
        rec = tp / all_true if all_true > 0 else 1.0

        print("\n" + "=" * 72)
        print("               TEST SET EVALUATION REPORT")
        print("=" * 72)
        print(f"{'Metric':<36} | {'Value':<18}")
        print("-" * 72)
        print(f"{'Source 1 Test Companies':<36} | {len(s1_te)}")
        print(f"{'Total True Positive Pairs':<36} | {all_true}")
        print(f"{'Predicted Positive Pairs':<36} | {all_pred}")
        print(f"{'True Positives (Correct Matches)':<36} | {tp}")
        print(f"{'Pairwise Precision':<36} | {prec * 100:.2f}%")
        print(f"{'Pairwise Recall':<36} | {rec * 100:.2f}%")
        print("-" * 72)
        print(f"{'OFFICIAL MACRO F_0.5 SCORE':<36} | {test_macro_f05:.4f}")
        print("=" * 72 + "\n")

    # 5. Run Submission Validator
    validator_path = "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py"
    if os.path.isfile(validator_path):
        print("Running official submission validator on output files...")
        cmd = [
            sys.executable, validator_path,
            "--matching", match_path,
            "--candidate", cand_path,
            "--test-dir", args.test_dir,
            "--check-ids"
        ]
        import subprocess
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(">> Validation Check: PASS (100% compliant with competition rules!)")
        else:
            print(f">> Validator output:\n{res.stdout}\n{res.stderr}")


if __name__ == "__main__":
    main()
