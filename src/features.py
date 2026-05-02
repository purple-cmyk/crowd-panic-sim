import numpy as np
from scipy.stats import entropy


def extract_features(magnitude, angle, previous_magnitude=None):
    avg_speed = float(np.mean(magnitude))

    speed_variation = float(np.std(magnitude))

    direction_variance = float(np.std(angle))

    if previous_magnitude is not None:
        acceleration = float(np.mean(np.abs(magnitude - previous_magnitude)))
    else:
        acceleration = 0.0

    motion_density = float(np.sum(magnitude > 2.0) / magnitude.size)

    hist, _ = np.histogram(angle, bins=10, range=(0, 2 * np.pi))

    flow_entropy = float(entropy(hist + 1e-8))

    return {
        "avg_speed": avg_speed,
        "speed_variation": speed_variation,
        "direction_variance": direction_variance,
        "acceleration": acceleration,
        "motion_density": motion_density,
        "flow_entropy": flow_entropy
    }