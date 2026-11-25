from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import plot_tree as sk_plot_tree

UCI_URL = "https://archive.ics.uci.edu/static/public/222/data.csv"

NUMERIC_COLS = ["age", "balance", "day_of_week", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"
]


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    if csv_path is not None:
        candidates.append(Path(csv_path))
    candidates.extend([
        repo_root / "data" / "bank_marketing.csv",
        repo_root.parent.parent / "bank_marketing.csv",
    ])
    for cand in candidates:
        if cand.exists():
            return pd.read_csv(cand)
    local_target = repo_root / "data" / "bank_marketing.csv"
    local_target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(UCI_URL, local_target)
    return pd.read_csv(local_target)


def split_data(
    df: pd.DataFrame, target_col: str = "y", test_size: float = 0.2, random_state: int = 42
):

    X = df.drop(columns=[target_col])
    y = (df[target_col] == "yes").astype(int) if df[target_col].dtype == object or str(df[target_col].dtype) == "str" else df[target_col].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    if scale_numeric:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ])
    else:
        num_pipe = SimpleImputer(strategy="median")
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ])


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    return list(NUMERIC_COLS) + ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(rec_pos),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "gmean": float(np.sqrt(rec_pos * rec_neg)),
        "confusion_matrix": cm,
    }


def plot_tree(dt_model, feature_names: list[str], class_names=None, out_path=None):
    if class_names is None:
        class_names = ["Отказ", "Депозит"]
    fig, ax = plt.subplots(figsize=(15.8, 7.6))
    sk_plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        proportion=True,
        ax=ax,
    )
    if out_path:
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    return fig


def plot_confusion(model, X_test, y_test, display_labels=None, out_path=None):
    if display_labels is None:
        display_labels = ["Отказ", "Депозит"]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=display_labels, cmap="Blues", ax=ax
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.grid(False)
    _ = disp
    if out_path:
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    return fig