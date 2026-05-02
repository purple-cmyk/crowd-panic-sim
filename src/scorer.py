from config.settings import WEIGHTS, DEFAULT_LIMITS


def normalize_feature(value, limit):
    if limit <= 0:
        return 0.0

    normalized = value / limit

    if normalized < 0:
        return 0.0

    if normalized > 1:
        return 1.0

    return normalized


def calculate_panic_score(features, limits=None):
    if limits is None:
        limits = DEFAULT_LIMITS

    score = 0.0

    for feature_name, weight in WEIGHTS.items():
        value = features.get(feature_name, 0.0)
        limit = limits.get(feature_name, DEFAULT_LIMITS[feature_name])

        normalized_value = normalize_feature(value, limit)
        score += normalized_value * weight

    return round(score * 100, 2)


def get_risk_level(score):
    if score <= 30:
        return "Safe", (0, 255, 0)

    if score <= 60:
        return "Alert", (0, 255, 255)

    if score <= 80:
        return "High Risk", (0, 165, 255)

    return "Panic", (0, 0, 255)


def get_pseudo_label(score):
    if score <= 30:
        return 0

    if score <= 60:
        return 1

    if score <= 80:
        return 2

    return 3


def explain_score(features, limits=None):
    if limits is None:
        limits = DEFAULT_LIMITS

    reasons = []

    normalized = {}

    for feature_name in WEIGHTS.keys():
        value = features.get(feature_name, 0.0)
        limit = limits.get(feature_name, DEFAULT_LIMITS[feature_name])
        normalized[feature_name] = normalize_feature(value, limit)

    if normalized["avg_speed"] > 0.7:
        reasons.append("high average crowd movement speed")

    if normalized["speed_variation"] > 0.7:
        reasons.append("large variation in movement speed")

    if normalized["direction_variance"] > 0.7:
        reasons.append("chaotic or inconsistent movement direction")

    if normalized["acceleration"] > 0.7:
        reasons.append("sudden acceleration or movement surge")

    if normalized["motion_density"] > 0.7:
        reasons.append("large portion of the scene is actively moving")

    if normalized["flow_entropy"] > 0.7:
        reasons.append("complex and disordered motion field")

    if len(reasons) == 0:
        reasons.append("movement appears stable and controlled")

    return reasons