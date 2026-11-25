import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate


def compute_gmean(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(rec_pos * rec_neg))


def compute_oob_roc_auc(rf_model, y_train) -> float:
    return float(roc_auc_score(y_train, rf_model.oob_decision_function_[:, 1]))


def compute_root_information_gain(dt_model) -> float:
    t = dt_model.tree_
    n0 = t.weighted_n_node_samples[0]
    left, right = t.children_left[0], t.children_right[0]
    if left == -1 or right == -1:
        return 0.0
    return float(
        t.impurity[0]
        - (t.weighted_n_node_samples[left] / n0) * t.impurity[left]
        - (t.weighted_n_node_samples[right] / n0) * t.impurity[right]
    )


def compute_mdi_importance(tree_model, feature_names: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_names, "importance": tree_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_perm_importance(
    model, X_val, y_val, feature_names: list[str], scoring: str = "roc_auc", n_repeats: int = 5, random_state: int = 42
) -> pd.DataFrame:
    res = permutation_importance(
        model, X_val, y_val, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return (
        pd.DataFrame({"feature": feature_names, "perm_importance": res.importances_mean})
        .sort_values("perm_importance", ascending=False)
        .reset_index(drop=True)
    )


def cross_validate_model(
    pipeline, X_train, y_train, cv_splits: int = 5, sample_weight=None, n_jobs: int = -1
) -> dict:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    kwargs = {"params": {"clf__sample_weight": sample_weight}} if sample_weight is not None else {}
    res = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=["roc_auc", "f1", "recall"], n_jobs=n_jobs, **kwargs
    )
    return {
        "cv_roc_auc": float(res["test_roc_auc"].mean()),
        "cv_f1": float(res["test_f1"].mean()),
        "cv_recall": float(res["test_recall"].mean()),
    }