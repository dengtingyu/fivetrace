import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,f1_score,roc_auc_score
from .config import SEED

def fit_linear(x, y, regularization):
    estimator = make_pipeline(StandardScaler(), LogisticRegression(
        C=regularization, max_iter=2000, class_weight="balanced", random_state=SEED,
    ))
    return estimator.fit(x, y)


def thresholds(y, scores, target_error=.05):
    """Empirical thresholds selected exclusively on the validation set."""
    human = np.sort(scores[y == 0])
    ai = np.sort(scores[y == 1])
    index = max(0, int(np.ceil((1-target_error) * len(human))) - 1)
    high = float(np.nextafter(human[index], 1.))
    low = min(float(np.quantile(ai, target_error, method="lower")), high)
    return {"human_below": low, "ai_at_or_above": high, "validation_target_fpr": target_error}


def binary_thresholds(y, scores, target_error=.05):
    """A single validation-selected cutoff, always returning human or AI."""
    limits = thresholds(y, scores, target_error)
    return dict(limits, human_below=limits["ai_at_or_above"], policy="validation_binary_cutoff")


def evaluate(y, scores, limits, generators):
    high, low = limits["ai_at_or_above"], limits["human_below"]
    prediction = (scores >= high).astype(int)
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(y, prediction, labels=[0, 1]).ravel())
    accepted = (scores < low) | (scores >= high)
    return {
        "samples": len(y), "human_samples": tn + fp, "ai_samples": tp + fn,
        "accuracy": float(accuracy_score(y, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")), "auroc": float(roc_auc_score(y, scores)),
        "human_false_positive_rate": fp / max(1, tn+fp), "ai_recall": tp / max(1, tp+fn),
        "brier_score": float(brier_score_loss(y, scores)),
        "confusion_matrix": {"true_human": tn, "human_flagged_ai": fp, "ai_missed": fn, "true_ai": tp},
        "abstention_rate": float(1-accepted.mean()),
        "selective_accuracy": float(accuracy_score(y[accepted], prediction[accepted])) if accepted.any() else None,
        "three_way_counts": {
            label: {"human": int(np.sum((y == target) & (scores < low))),
                    "uncertain": int(np.sum((y == target) & (scores >= low) & (scores < high))),
                    "ai": int(np.sum((y == target) & (scores >= high)))}
            for label, target in [("true_human", 0), ("true_ai", 1)]
        },
        "ai_recall_by_generator": {
            name: float(prediction[(generators == name) & (y == 1)].mean())
            for name in sorted(set(generators)) if name != "human" and ((generators == name) & (y == 1)).any()
        },
    }
