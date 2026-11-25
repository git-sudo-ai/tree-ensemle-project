from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.common import build_preprocessor


def build_decision_tree(
    max_depth: int | None = 3,
    criterion: str = "gini",
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(scale_numeric=False)),
        ("clf", DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int = 16,
    min_samples_leaf: int = 4,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(scale_numeric=False)),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            oob_score=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )),
    ])


def build_gradient_boosting(
    n_estimators: int = 150,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(scale_numeric=False)),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )),
    ])


def build_voting_ensemble(random_state: int = 42) -> Pipeline:
    voting = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=16, min_samples_leaf=4, n_jobs=1, random_state=random_state
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=random_state
            )),
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
        ],
        voting="soft",
        weights=[2, 3, 1],
    )
    return Pipeline([
        ("pre", build_preprocessor(scale_numeric=True)),
        ("clf", voting),
    ])