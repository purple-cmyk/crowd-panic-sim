import os
import pandas as pd


def append_score_log(path, frame_number, features, score, risk_level, label):
    row = {
        "frame": frame_number,
        **features,
        "panic_score": score,
        "risk_level": risk_level,
        "pseudo_label": label
    }

    df = pd.DataFrame([row])

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)


def save_features_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)