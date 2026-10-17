"""
Dependency-free Random Forest inference.

Loads the tree structure exported from the real trained sklearn model
(forest.json) and walks each tree in pure Python to produce the same
prediction sklearn would -- without needing sklearn/numpy installed in the
serverless function, which keeps cold starts fast and avoids Vercel build
issues from heavy ML dependencies.
"""
import json
import os

_FOREST = None


def _load_forest():
    global _FOREST
    if _FOREST is None:
        path = os.path.join(os.path.dirname(__file__), "forest.json")
        with open(path, "r") as f:
            _FOREST = json.load(f)
    return _FOREST


def _leaf_proba(tree, features):
    """Walk to a leaf and return its normalized class probabilities, matching
    sklearn's DecisionTreeClassifier.predict_proba (counts / sum(counts))."""
    node = 0
    while tree["left"][node] != -1:  # -1 marks a leaf in sklearn's tree_ arrays
        feat_idx = tree["feature"][node]
        threshold = tree["threshold"][node]
        if features[feat_idx] <= threshold:
            node = tree["left"][node]
        else:
            node = tree["right"][node]
    counts = tree["value"][node]  # [count_stable, count_failure]
    total = counts[0] + counts[1]
    if total == 0:
        return 0.5, 0.5
    return counts[0] / total, counts[1] / total


def predict(features):
    """
    features: [utilization_pct, packet_loss_frac, latency_ms, queue_depth, burst_rate]
    Returns: (prediction: 0|1, confidence: float 0-1, proba: {stable, failure})

    Matches sklearn.ensemble.RandomForestClassifier.predict() exactly: average
    each tree's class-probability vector (not a hard per-tree vote), then take
    the argmax of the averaged probabilities.
    """
    forest = _load_forest()
    sum_stable = 0.0
    sum_failure = 0.0
    for tree in forest["trees"]:
        p_stable, p_failure = _leaf_proba(tree, features)
        sum_stable += p_stable
        sum_failure += p_failure

    n_trees = forest["n_trees"]
    proba_stable = sum_stable / n_trees
    proba_failure = sum_failure / n_trees
    prediction = 1 if proba_failure > proba_stable else 0
    confidence = max(proba_stable, proba_failure)

    return prediction, confidence, {"stable": round(proba_stable, 4), "failure": round(proba_failure, 4)}
