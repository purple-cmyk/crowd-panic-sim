import numpy as np
from config.settings import DEFAULT_LIMITS


def compute_calibration_limits(feature_rows):
    if len(feature_rows) == 0:
        return DEFAULT_LIMITS

    limits = {}

    keys = feature_rows[0].keys()

    for key in keys:
        values = np.array([row[key] for row in feature_rows], dtype=float)

        mean_value = np.mean(values)
        std_value = np.std(values)

        limit = mean_value + 3 * std_value

        if limit <= 0:
            limit = DEFAULT_LIMITS[key]

        limits[key] = float(max(limit, DEFAULT_LIMITS[key] * 0.4))

    return limits