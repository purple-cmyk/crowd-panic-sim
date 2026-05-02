crowd-panic-estimator/
│
├── app.py
├── run_video.py
├── train_models.py
├── predict_video_ml.py
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/
│   │   └── sample.mp4
│   │
│   ├── processed/
│   │   └── features.csv
│   │
│   └── models/
│       ├── bagging_model.pkl
│       ├── boosting_model.pkl
│       ├── stacking_model.pkl
│       └── scaler.pkl
│
├── outputs/
│   ├── processed_rule_based.mp4
│   ├── processed_ml_based.mp4
│   └── score_log.csv
│
├── src/
│   ├── __init__.py
│   ├── optical_flow.py
│   ├── features.py
│   ├── calibration.py
│   ├── scorer.py
│   ├── visualizer.py
│   ├── logger.py
│   ├── utils.py
│   │
│   └── ml/
│       ├── __init__.py
│       ├── dataset.py
│       ├── models.py
│       ├── trainer.py
│       └── predictor.py
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css




    # Crowd Panic Risk Estimator

This project estimates crowd panic risk using a data-independent hybrid approach.

It contains three major modes:

1. Manual rule-based panic estimation
2. Optical flow-based video panic estimation
3. ML-based panic classification using Bagging, Boosting, and Stacking

## Why This Project Does Not Need a Real Dataset Initially

Real panic datasets are rare, sensitive, and ethically difficult to collect.  
Instead of directly training a model on labeled panic footage, this project first extracts measurable motion features from videos using optical flow.

The rule-based system generates a panic score using physical and behavioral indicators:

- Average speed
- Speed variation
- Direction variance
- Sudden acceleration
- Motion density
- Flow entropy

These scores are then converted into pseudo-labels.

The pseudo-labeled data can be used to train ML models.

## Project Modes

### 1. Manual Mode

Uses manually entered crowd conditions:

- Crowd density
- Movement speed
- Direction disorder
- Threat level
- Exit blockage
- Noise level

Run:

```bash
python app.py