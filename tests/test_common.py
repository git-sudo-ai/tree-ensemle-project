import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.common import build_preprocessor, evaluate, get_feature_names, split_data
from src.metrics import compute_gmean, compute_root_information_gain


def _sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({
        "age": rng.integers(20, 65, size=n),
        "balance": rng.integers(-200, 3000, size=n),
        "day_of_week": rng.integers(1, 29, size=n),
        "duration": rng.integers(30, 800, size=n),
        "campaign": rng.integers(1, 5, size=n),
        "pdays": rng.choice([-1, 120, 200], size=n),
        "previous": rng.integers(0, 3, size=n),
        "job": rng.choice(["admin.", "technician", None], size=n),
        "marital": rng.choice(["married", "single"], size=n),
        "education": rng.choice(["secondary", "tertiary", None], size=n),
        "default": rng.choice(["no", "yes"], size=n, p=[0.9, 0.1]),
        "housing": rng.choice(["yes", "no"], size=n),
        "loan": rng.choice(["no", "yes"], size=n),
        "contact": rng.choice(["cellular", None], size=n),
        "month": rng.choice(["may", "jun", "aug"], size=n),
        "poutcome": rng.choice(["failure", "success", None], size=n),
        "y": ["yes"] * 10 + ["no"] * 30,
    })


def test_split_data_stratified():
    df = _sample_df()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    assert len(X_train) == 32
    assert len(X_test) == 8
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_build_preprocessor_no_nans():
    df = _sample_df()
    X_train, X_test, _, _ = split_data(df, test_size=0.2, random_state=42)
    pre = build_preprocessor(scale_numeric=False)
    Xt = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    names = get_feature_names(pre)
    assert not np.isnan(Xt).any()
    assert not np.isnan(Xte).any()
    assert Xt.shape[1] == len(names)


def test_compute_gmean():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    # rec_pos = 3/4, rec_neg = 3/4 -> gmean = 0.75
    assert np.isclose(compute_gmean(y_true, y_pred), 0.75)


def test_information_gain_positive():
    X = np.array([[10], [20], [30], [400], [500], [600]])
    y = np.array([0, 0, 0, 1, 1, 1])
    dt = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=42).fit(X, y)
    ig = compute_root_information_gain(dt)
    assert np.isclose(ig, 0.5)


def test_evaluate_returns_expected_keys():
    df = _sample_df()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    pre = build_preprocessor()
    X_tr_p = pre.fit_transform(X_train)
    X_te_p = pre.transform(X_test)
    dt = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_tr_p, y_train)
    res = evaluate(dt, X_te_p, y_test)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "gmean", "confusion_matrix"):
        assert k in res
    assert res["confusion_matrix"].shape == (2, 2)
    assert 0.0 <= res["roc_auc"] <= 1.0