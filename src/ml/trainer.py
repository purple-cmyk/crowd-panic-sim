import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from config.settings import (
    BAGGING_MODEL_PATH,
    BOOSTING_MODEL_PATH,
    STACKING_MODEL_PATH,
    SCALER_PATH
)

from src.ml.models import (
    build_bagging_model,
    build_boosting_model,
    build_stacking_model
)


def train_all_models(X, y):
    os.makedirs("data/models", exist_ok=True)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None
    )

    models = {
        "Bagging": build_bagging_model(),
        "Boosting": build_boosting_model(),
        "Stacking": build_stacking_model()
    }

    paths = {
        "Bagging": BAGGING_MODEL_PATH,
        "Boosting": BOOSTING_MODEL_PATH,
        "Stacking": STACKING_MODEL_PATH
    }

    for name, model in models.items():
        print("\nTraining:", name)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))

        joblib.dump(model, paths[name])

    joblib.dump(scaler, SCALER_PATH)

    print("\nModels saved successfully.")