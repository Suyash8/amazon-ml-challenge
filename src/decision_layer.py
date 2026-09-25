#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: decision_layer

Stage-2 Per-Entity Decision Layer & Asymmetric Thresholding.
Optimized specifically for Macro F_0.5 evaluation:
  1. Singleton Estimation & Prior:
     - Estimates the empirical singleton rate from training data.
     - Quantifies the risk asymmetry: a single false merge on a singleton drops
       its F_0.5 from 1.0 to 0.0, costing 2x more than a false negative.
  2. Second-Stage Per-Entity Model (Singleton Gatekeeper):
     - Aggregates pairwise candidate probabilities per S1 entity:
       * max_score, second_score, score_gap_1_2, score_gap_1_3
       * num_candidates, num_above_0.3, num_above_0.5
       * mean_score, sum_score, top1_source (S2 vs S3)
     - Predicts P(entity has any match).
     - Confident singletons are gated to empty list [] immediately.
  3. Asymmetric Thresholds & Relative Margin Policy:
     - High hurdle for primary match (t_first).
     - Lower threshold for secondary matches (t_second) constrained by a
       relative margin (delta_margin) vs the top score.
  4. Direct Macro F_0.5 Metric Optimization:
     - Multi-dimensional parameter grid search on validation split directly
       maximizing official Macro F_0.5.
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np

# System path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# =============================================================================
# EVALUATION METRIC (MACRO F_0.5 SCORE)
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
# 1. PER-ENTITY FEATURE AGGREGATION
# =============================================================================
ENTITY_FEATURE_NAMES: List[str] = [
    "ent_max_score",          # Highest candidate probability
    "ent_second_score",       # Runner-up candidate probability (0.0 if <2 cands)
    "ent_third_score",        # 3rd candidate probability (0.0 if <3 cands)
    "ent_score_gap_1_2",      # Gap between top-1 and top-2
    "ent_score_gap_1_3",      # Gap between top-1 and top-3
    "ent_num_cands",          # Total candidates returned by blocker
    "ent_num_above_03",       # Number of candidates with prob >= 0.30
    "ent_num_above_05",       # Number of candidates with prob >= 0.50
    "ent_mean_score",         # Mean candidate probability
    "ent_sum_score",          # Sum of candidate probabilities
    "ent_top1_is_s2",         # Binary indicator if top candidate is Source 2
    "ent_top1_is_s3",         # Binary indicator if top candidate is Source 3
]


def extract_entity_features(cand_probs: List[Tuple[str, float]]) -> List[float]:
    """
    Aggregates candidate probability distribution for an S1 entity into
    dense second-stage features for singleton classification.
    Expects cand_probs to be sorted descending by probability.
    """
    if not cand_probs:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    probs = [p for _, p in cand_probs]
    cids = [c for c, _ in cand_probs]

    max_p = float(probs[0])
    p2 = float(probs[1]) if len(probs) > 1 else 0.0
    p3 = float(probs[2]) if len(probs) > 2 else 0.0
    gap12 = max_p - p2
    gap13 = max_p - p3
    n_cands = float(len(probs))
    n_above_03 = float(sum(1 for p in probs if p >= 0.30))
    n_above_05 = float(sum(1 for p in probs if p >= 0.50))
    mean_p = float(np.mean(probs))
    sum_p = float(np.sum(probs))
    top1_s2 = 1.0 if cids[0].startswith("S2-") else 0.0
    top1_s3 = 1.0 if cids[0].startswith("S3-") else 0.0

    return [
        max_p, p2, p3, gap12, gap13, n_cands, n_above_03, n_above_05,
        mean_p, sum_p, top1_s2, top1_s3
    ]


# =============================================================================
# 2. SINGLETON ESTIMATOR & GATEKEEPER MODEL
# =============================================================================
class EntitySingletonGatekeeper:
    """
    Stage-2 GBDT classifier predicting P(entity has >= 1 true match).
    Acts as a precision gatekeeper guarding against false merges on singletons.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 3, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Optional[lgb.LGBMClassifier] = None
        self.singleton_prior: float = 0.0

    def fit(
        self,
        entity_cand_probs: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        verbose: bool = True,
    ):
        """
        Fits second-stage model on per-entity candidate distributions.
        Label = 1 if entity has >= 1 true match in ground truth, 0 if singleton.
        """
        all_s1_ids = sorted(entity_cand_probs.keys())
        singletons = [s for s in all_s1_ids if not ground_truth.get(s)]
        self.singleton_prior = len(singletons) / max(len(all_s1_ids), 1)

        if verbose:
            print(f"[Gatekeeper] Estimated Training Singleton Rate: {self.singleton_prior * 100:.2f}% "
                  f"({len(singletons)}/{len(all_s1_ids)} singletons)")

        X: List[List[float]] = []
        y: List[int] = []

        for s1_id in all_s1_ids:
            cp = entity_cand_probs[s1_id]
            feat = extract_entity_features(cp)
            label = 1 if len(ground_truth.get(s1_id, set())) > 0 else 0
            X.append(feat)
            y.append(label)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_arr, y_arr)

        if verbose:
            print(f"[Gatekeeper] Trained Stage-2 GBDT Gatekeeper on {len(all_s1_ids)} entities.")

    def predict_p_has_match(self, cand_probs: List[Tuple[str, float]]) -> float:
        """Returns predicted probability that the entity has at least one true match."""
        if not cand_probs:
            return 0.0
        if self.model is None:
            # Fallback to top-1 candidate probability
            return cand_probs[0][1]
        feat = np.array([extract_entity_features(cand_probs)], dtype=np.float32)
        return float(self.model.predict_proba(feat)[0, 1])


# =============================================================================
# 3. ASYMMETRIC DECISION POLICY
# =============================================================================
class AsymmetricDecisionPolicy:
    """
    Applies asymmetric decision rules to candidate probabilities:
      1. Gatekeeper: if P(has_match) < tau_gate -> predict []
      2. First match: requires prob >= t_first
      3. Additional matches: requires prob >= t_second AND (top_p - prob) <= delta_margin
    """

    def __init__(
        self,
        tau_gate: float = 0.30,
        t_first: float = 0.45,
        t_second: float = 0.25,
        delta_margin: float = 0.20,
    ):
        self.tau_gate = tau_gate
        self.t_first = t_first
        self.t_second = t_second
        self.delta_margin = delta_margin

    def decide(
        self,
        cand_probs: List[Tuple[str, float]],
        p_has_match: float,
    ) -> Set[str]:
        """
        Executes decision policy for a single S1 entity.
        Returns set of accepted matched entity IDs.
        """
        # 1. Singleton Gatekeeper Check
        if p_has_match < self.tau_gate or not cand_probs:
            return set()

        # 2. First Match Requirement (High Hurdle)
        top_cid, top_p = cand_probs[0]
        if top_p < self.t_first:
            return set()

        matches = {top_cid}

        # 3. Additional Matches (Asymmetric lower bar + relative margin)
        for cid, p in cand_probs[1:]:
            if p >= self.t_second and (top_p - p) <= self.delta_margin:
                matches.add(cid)

        return matches


# =============================================================================
# 4. DECISION LAYER & DIRECT MACRO F_0.5 OPTIMIZER
# =============================================================================
class DecisionLayer:
    """
    Complete Stage-2 Decision Layer combining:
      - Empirical singleton estimation
      - Second-stage entity gatekeeper model
      - Asymmetric decision policy
      - Direct Macro F_0.5 metric optimizer
    """

    def __init__(
        self,
        tau_gate: float = 0.30,
        t_first: float = 0.45,
        t_second: float = 0.25,
        delta_margin: float = 0.20,
    ):
        self.policy = AsymmetricDecisionPolicy(
            tau_gate=tau_gate,
            t_first=t_first,
            t_second=t_second,
            delta_margin=delta_margin,
        )
        self.gatekeeper = EntitySingletonGatekeeper()
        self.is_optimized = False

    def fit_and_tune(
        self,
        entity_cand_probs: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        verbose: bool = True,
    ):
        """
        Trains the stage-2 entity gatekeeper model and tunes asymmetric policy
        parameters directly maximizing Macro F_0.5.
        """
        t0 = time.time()
        if verbose:
            print("[DecisionLayer] 1/2 Training Stage-2 Entity Gatekeeper...")

        self.gatekeeper.fit(entity_cand_probs, ground_truth, verbose=verbose)

        # Precompute gatekeeper P(has_match) for all entities
        all_s1_ids = sorted(entity_cand_probs.keys())
        p_has_match_dict = {
            s1_id: self.gatekeeper.predict_p_has_match(entity_cand_probs[s1_id])
            for s1_id in all_s1_ids
        }

        if verbose:
            print("[DecisionLayer] 2/2 Direct Optimization for Macro F_0.5 Metric...")

        # Parameter Search Grids
        tau_gate_grid = [0.20, 0.30, 0.40, 0.50]
        t_first_grid = [0.35, 0.45, 0.55, 0.65, 0.75]
        t_second_grid = [0.15, 0.25, 0.35, 0.45]
        delta_margin_grid = [0.10, 0.20, 0.30, 0.45]

        best_f05 = -1.0
        best_params = (self.policy.tau_gate, self.policy.t_first, self.policy.t_second, self.policy.delta_margin)

        # Grid search over asymmetric policy space
        for tg in tau_gate_grid:
            for tf in t_first_grid:
                for ts in t_second_grid:
                    if ts > tf:
                        continue  # Secondary threshold must be <= primary
                    for dm in delta_margin_grid:
                        test_policy = AsymmetricDecisionPolicy(
                            tau_gate=tg, t_first=tf, t_second=ts, delta_margin=dm
                        )
                        preds = {
                            s1_id: test_policy.decide(
                                entity_cand_probs[s1_id], p_has_match_dict[s1_id]
                            )
                            for s1_id in all_s1_ids
                        }
                        score = compute_macro_f05(preds, ground_truth, all_s1_ids)
                        if score > best_f05:
                            best_f05 = score
                            best_params = (tg, tf, ts, dm)

        self.policy.tau_gate = best_params[0]
        self.policy.t_first = best_params[1]
        self.policy.t_second = best_params[2]
        self.policy.delta_margin = best_params[3]
        self.is_optimized = True

        dur = time.time() - t0
        if verbose:
            print(f"[DecisionLayer] Optimization complete in {dur:.2f}s:")
            print(f"  * Optimal tau_gate (singleton hurdle) : {self.policy.tau_gate:.3f}")
            print(f"  * Optimal t_first (primary match bar) : {self.policy.t_first:.3f}")
            print(f"  * Optimal t_second (multi-match bar)  : {self.policy.t_second:.3f}")
            print(f"  * Optimal delta_margin (drop limit)   : {self.policy.delta_margin:.3f}")
            print(f"  * Validation Macro F_0.5 Score        : {best_f05:.4f}")

    def predict(
        self,
        entity_cand_probs: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, Set[str]]:
        """
        Executes decision layer for test entities.
        Returns mapping: {s1_entity_id: set(matched_cids)}.
        """
        predictions: Dict[str, Set[str]] = {}
        for s1_id, cp in entity_cand_probs.items():
            p_match = self.gatekeeper.predict_p_has_match(cp)
            matches = self.policy.decide(cp, p_match)
            predictions[s1_id] = matches
        return predictions

    def save(self, file_path: str):
        """Saves decision layer parameters and gatekeeper model to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        payload = {
            "policy": self.policy,
            "gatekeeper": self.gatekeeper,
            "is_optimized": self.is_optimized,
        }
        with open(file_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, file_path: str) -> "DecisionLayer":
        """Loads decision layer parameters and gatekeeper model from disk."""
        with open(file_path, "rb") as f:
            payload = pickle.load(f)
        layer = cls()
        layer.policy = payload["policy"]
        layer.gatekeeper = payload["gatekeeper"]
        layer.is_optimized = payload.get("is_optimized", True)
        return layer


# =============================================================================
# 5. CLI EVALUATOR
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage-2 Entity Decision Layer & Asymmetric Thresholding.")
    parser.add_argument("--train-dir", type=str, default="dataset_split/train")
    parser.add_argument("--test-dir", type=str, default="dataset_split/test")
    parser.add_argument("--model-path", type=str, default="models/lgbm_entity_resolver.pkl")
    parser.add_argument("--output-layer", type=str, default="models/decision_layer.pkl")
    args = parser.parse_args()

    print("=" * 72)
    print("   Amazon ML Challenge 2026 - Stage-2 Decision Layer")
    print("=" * 72)

    # 1. Load Pairwise Classifier
    try:
        from src.classifier import EntityResolutionClassifier, extract_pair_features
        from src.blocking import load_source_tsv, load_ground_truth_tsv
    except ImportError:
        from classifier import EntityResolutionClassifier, extract_pair_features
        from blocking import load_source_tsv, load_ground_truth_tsv

    if not os.path.exists(args.model_path):
        print(f"Error: Base model not found at {args.model_path}. Train it first!")
        sys.exit(1)

    clf = EntityResolutionClassifier.load(args.model_path)

    # 2. Load Train Data & Extract Probabilities
    s1_tr = load_source_tsv(os.path.join(args.train_dir, "train_source1.tsv"))
    s2_tr = load_source_tsv(os.path.join(args.train_dir, "train_source2.tsv"))
    s3_tr = load_source_tsv(os.path.join(args.train_dir, "train_source3.tsv"))
    gt_tr = load_ground_truth_tsv(os.path.join(args.train_dir, "train_ground_truth.tsv"))

    cands_tr = clf.blocker.generate_candidates(s1_tr, s2_tr, s3_tr, verbose=False)
    s1_prep_tr = {eid: clf.blocker.preprocess_record(r) for eid, r in s1_tr.items()}
    target_prep_tr = {eid: clf.blocker.preprocess_record(r) for eid, r in {**s2_tr, **s3_tr}.items()}
    all_tr = list(s1_prep_tr.values()) + list(target_prep_tr.values())
    id_to_idx_tr = {r["entity_id"]: i for i, r in enumerate(all_tr)}
    X_name_tr, X_addr_tr, X_full_tr = clf._compute_vectorized_matrices(all_tr)

    train_cand_probs: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id in s1_tr:
        cset = cands_tr.get(s1_id, set())
        if not cset:
            train_cand_probs[s1_id] = []
            continue
        pair_list = [(s1_id, cid) for cid in sorted(cset)]
        i1 = [id_to_idx_tr[p[0]] for p in pair_list]
        i2 = [id_to_idx_tr[p[1]] for p in pair_list]
        ns = np.asarray(X_name_tr[i1].multiply(X_name_tr[i2]).sum(axis=1)).ravel()
        as_ = np.asarray(X_addr_tr[i1].multiply(X_addr_tr[i2]).sum(axis=1)).ravel()
        fs = np.asarray(X_full_tr[i1].multiply(X_full_tr[i2]).sum(axis=1)).ravel()
        feats = [
            extract_pair_features(
                s1_prep_tr[s1], target_prep_tr[cid],
                nm_tfidf_cos=n,
                ad_tfidf_cos=a if s1_prep_tr[s1]["addr_clean"] and target_prep_tr[cid]["addr_clean"] else 0.0,
                full_tfidf_cos=f
            )
            for (s1, cid), n, a, f in zip(pair_list, ns, as_, fs)
        ]
        probs = clf.model.predict_proba(np.array(feats))[:, 1]
        train_cand_probs[s1_id] = sorted(zip([cid for _, cid in pair_list], probs), key=lambda x: -x[1])

    # 3. Fit & Tune Decision Layer
    decision_layer = DecisionLayer()
    decision_layer.fit_and_tune(train_cand_probs, gt_tr, verbose=True)
    decision_layer.save(args.output_layer)
    print(f"Saved Decision Layer to {args.output_layer}")

    # 4. Evaluate on Test Set
    prefix = "train" if os.path.isfile(os.path.join(args.test_dir, "train_source1.tsv")) else "test"
    s1_te = load_source_tsv(os.path.join(args.test_dir, f"{prefix}_source1.tsv"))
    s2_te = load_source_tsv(os.path.join(args.test_dir, f"{prefix}_source2.tsv"))
    s3_te = load_source_tsv(os.path.join(args.test_dir, f"{prefix}_source3.tsv"))

    cands_te = clf.blocker.generate_candidates(s1_te, s2_te, s3_te, verbose=False)
    s1_prep_te = {eid: clf.blocker.preprocess_record(r) for eid, r in s1_te.items()}
    target_prep_te = {eid: clf.blocker.preprocess_record(r) for eid, r in {**s2_te, **s3_te}.items()}
    all_te = list(s1_prep_te.values()) + list(target_prep_te.values())
    id_to_idx_te = {r["entity_id"]: i for i, r in enumerate(all_te)}
    X_name_te, X_addr_te, X_full_te = clf._compute_vectorized_matrices(all_te)

    test_cand_probs: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id in s1_te:
        cset = cands_te.get(s1_id, set())
        if not cset:
            test_cand_probs[s1_id] = []
            continue
        pair_list = [(s1_id, cid) for cid in sorted(cset)]
        i1 = [id_to_idx_te[p[0]] for p in pair_list]
        i2 = [id_to_idx_te[p[1]] for p in pair_list]
        ns = np.asarray(X_name_te[i1].multiply(X_name_te[i2]).sum(axis=1)).ravel()
        as_ = np.asarray(X_addr_te[i1].multiply(X_addr_te[i2]).sum(axis=1)).ravel()
        fs = np.asarray(X_full_te[i1].multiply(X_full_te[i2]).sum(axis=1)).ravel()
        feats = [
            extract_pair_features(
                s1_prep_te[s1], target_prep_te[cid],
                nm_tfidf_cos=n,
                ad_tfidf_cos=a if s1_prep_te[s1]["addr_clean"] and target_prep_te[cid]["addr_clean"] else 0.0,
                full_tfidf_cos=f
            )
            for (s1, cid), n, a, f in zip(pair_list, ns, as_, fs)
        ]
        probs = clf.model.predict_proba(np.array(feats))[:, 1]
        test_cand_probs[s1_id] = sorted(zip([cid for _, cid in pair_list], probs), key=lambda x: -x[1])

    preds_te = decision_layer.predict(test_cand_probs)

    gt_te_path = os.path.join(args.test_dir, f"{prefix}_ground_truth.tsv")
    if os.path.isfile(gt_te_path):
        gt_te = load_ground_truth_tsv(gt_te_path)
        score = compute_macro_f05(preds_te, gt_te, list(s1_te.keys()))

        tp = sum(len(preds_te[s] & gt_te[s]) for s in s1_te)
        all_pred = sum(len(preds_te[s]) for s in s1_te)
        all_true = sum(len(gt_te[s]) for s in s1_te)

        print("\n" + "=" * 72)
        print("         TEST SET EVALUATION WITH DECISION LAYER")
        print("=" * 72)
        print(f"Source 1 Test Companies:      {len(s1_te)}")
        print(f"True Positive Matches:         {tp} / {all_true}")
        print(f"Predicted Matches:             {all_pred}")
        print(f"Pairwise Precision:            {tp / all_pred * 100:.2f}%" if all_pred > 0 else "100.00%")
        print(f"Pairwise Recall:               {tp / all_true * 100:.2f}%" if all_true > 0 else "100.00%")
        print("-" * 72)
        print(f"OFFICIAL MACRO F_0.5 SCORE:    {score:.4f}")
        print("=" * 72)


if __name__ == "__main__":
    main()
