import joblib
import numpy as np

from config.settings import (
    BAGGING_MODEL_PATH,
    BOOSTING_MODEL_PATH,
    STACKING_MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES
)


def load_ml_model(model_type="stacking"):
    scaler = joblib.load(SCALER_PATH)

    if model_type == "bagging":
        model = joblib.load(BAGGING_MODEL_PATH)
    elif model_type == "boosting":
        model = joblib.load(BOOSTING_MODEL_PATH)
    else:
        model = joblib.load(STACKING_MODEL_PATH)

    return model, scaler


def predict_risk_from_features(features, model, scaler):
    row = np.array([[features[name] for name in FEATURE_NAMES]])

    row_scaled = scaler.transform(row)

    label = int(model.predict(row_scaled)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row_scaled)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 0.0

    return label, confidence


def label_to_risk(label):
    if label == 0:
        return "Safe", (0, 255, 0), 20

    if label == 1:
        return "Alert", (0, 255, 255), 45

    if label == 2:
        return "High Risk", (0, 165, 255), 70

    return "Panic", (0, 0, 255), 90