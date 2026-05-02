VIDEO_PATH = "data/raw/sample.mp4"

FEATURE_CSV_PATH = "data/processed/features.csv"
SCORE_LOG_PATH = "outputs/score_log.csv"

RULE_BASED_OUTPUT_VIDEO = "outputs/processed_rule_based.mp4"
ML_BASED_OUTPUT_VIDEO = "outputs/processed_ml_based.mp4"

BAGGING_MODEL_PATH = "data/models/bagging_model.pkl"
BOOSTING_MODEL_PATH = "data/models/boosting_model.pkl"
STACKING_MODEL_PATH = "data/models/stacking_model.pkl"
SCALER_PATH = "data/models/scaler.pkl"

CALIBRATION_FRAMES = 90

FEATURE_NAMES = [
    "avg_speed",
    "speed_variation",
    "direction_variance",
    "acceleration",
    "motion_density",
    "flow_entropy"
]

WEIGHTS = {
    "avg_speed": 0.25,
    "speed_variation": 0.20,
    "direction_variance": 0.20,
    "acceleration": 0.15,
    "motion_density": 0.10,
    "flow_entropy": 0.10
}

DEFAULT_LIMITS = {
    "avg_speed": 10.0,
    "speed_variation": 5.0,
    "direction_variance": 3.14,
    "acceleration": 5.0,
    "motion_density": 1.0,
    "flow_entropy": 2.5
}