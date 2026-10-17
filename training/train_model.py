"""
AERO-Lite model training.

Improvements over the original aero_train_model.py:
  - 8,000 samples instead of 1,200 (more stable decision boundaries)
  - Correlated features: utilization drives latency/queue/burst upward with
    noise, instead of five independent random draws. Real network links
    behave this way -- a congested link doesn't have high utilization and
    low latency by coincidence.
  - Label has a probabilistic "soft" failure zone near the thresholds
    instead of a hard OR-rule, so the forest learns an actual decision
    boundary rather than memorizing the exact rule used to generate it.
  - Stratified train/test split + 5-fold cross-validation reported for a
    trustworthy accuracy number (the original only reported a single
    train_test_split score).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump


FEATURE_NAMES = ["utilization_pct", "packet_loss_frac", "latency_ms", "queue_depth", "burst_rate"]


def generate_dataset(samples=8000, seed=42):
    rng = np.random.default_rng(seed)

    # Base utilization: beta distribution skewed toward normal operating
    # range, with a heavier tail so congestion events are represented.
    util = rng.beta(2, 4, samples) * 100

    # Latency correlates with utilization (congestion raises latency) plus
    # base propagation delay and noise.
    latency = 5 + util * 0.55 + rng.normal(0, 8, samples)
    latency = np.clip(latency, 1, None)

    # Packet loss rises sharply once utilization crosses ~80%, modeling
    # queue-overflow behavior, plus a small baseline noise floor.
    loss = np.where(
        util > 80,
        rng.normal(0.06, 0.03, samples) + (util - 80) * 0.004,
        rng.normal(0.005, 0.006, samples),
    )
    loss = np.clip(loss, 0, 1)

    # Queue depth correlates with utilization and loss.
    queue = util * 0.9 + loss * 200 + rng.normal(0, 10, samples)
    queue = np.clip(queue, 0, 150)

    # Burst rate (fraction of traffic arriving in short spikes) correlates
    # loosely with utilization.
    burst = np.clip(util / 100 * 0.7 + rng.normal(0, 0.15, samples), 0, 1)

    X = np.column_stack([util, loss, latency, queue, burst])

    # Soft-boundary label: a continuous "risk score" built from the same
    # signals a real operator would watch, pushed through a sigmoid so the
    # boundary isn't a brittle hard cutoff, then sampled as a Bernoulli
    # trial so there's genuine class overlap near the boundary (realistic
    # -- near-threshold cases aren't always failures).
    risk = (
        (util - 50) / 14
        + (loss - 0.02) / 0.025
        + (latency - 33) / 15
        + (queue - 48) / 22
    )
    prob_failure = 1 / (1 + np.exp(-risk))
    y = rng.binomial(1, prob_failure)

    return X, y


def main():
    X, y = generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"Held-out test accuracy: {test_score:.3f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["stable", "failure-risk"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:20s} {imp:.3f}")

    dump(model, "model.pkl")
    print("\nSaved model.pkl")


if __name__ == "__main__":
    main()
