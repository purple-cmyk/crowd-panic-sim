import os


def ensure_directories():
    folders = [
        "data/raw",
        "data/processed",
        "data/models",
        "outputs"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)