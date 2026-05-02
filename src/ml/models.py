from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_bagging_model():
    base_model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )

    model = BaggingClassifier(
        estimator=base_model,
        n_estimators=50,
        random_state=42
    )

    return model


def build_boosting_model():
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )

    return model


def build_adaboost_model():
    base_model = DecisionTreeClassifier(
        max_depth=2,
        random_state=42
    )

    model = AdaBoostClassifier(
        estimator=base_model,
        n_estimators=80,
        learning_rate=0.08,
        random_state=42
    )

    return model


def build_stacking_model():
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svm", SVC(probability=True, kernel="rbf", random_state=42))
    ]

    final_estimator = LogisticRegression(max_iter=1000)

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5
    )

    return model