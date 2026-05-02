import cv2

from config.settings import (
    VIDEO_PATH,
    RULE_BASED_OUTPUT_VIDEO,
    FEATURE_CSV_PATH,
    SCORE_LOG_PATH,
    CALIBRATION_FRAMES
)

from src.optical_flow import compute_dense_flow
from src.features import extract_features
from src.calibration import compute_calibration_limits
from src.scorer import calculate_panic_score, get_risk_level, get_pseudo_label, explain_score
from src.visualizer import draw_overlay
from src.logger import append_score_log, save_features_csv
from src.utils import ensure_directories


def main():
    ensure_directories()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    ret, first_frame = cap.read()

    if not ret:
        print("Could not read first frame.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        RULE_BASED_OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    previous_magnitude = None
    frame_number = 0

    calibration_rows = []
    all_rows = []

    limits = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        magnitude, angle = compute_dense_flow(prev_gray, curr_gray)

        features = extract_features(magnitude, angle, previous_magnitude)

        if frame_number < CALIBRATION_FRAMES:
            calibration_rows.append(features)

            score = 0.0
            risk_level = "Calibrating"
            color = (255, 255, 255)
            reasons = ["recording normal motion baseline"]

        else:
            if limits is None:
                limits = compute_calibration_limits(calibration_rows)
                print("Calibration limits:", limits)

            score = calculate_panic_score(features, limits)
            risk_level, color = get_risk_level(score)
            label = get_pseudo_label(score)
            reasons = explain_score(features, limits)

            row = {
                "frame": frame_number,
                **features,
                "panic_score": score,
                "risk_level": risk_level,
                "pseudo_label": label
            }

            all_rows.append(row)

            append_score_log(
                SCORE_LOG_PATH,
                frame_number,
                features,
                score,
                risk_level,
                label
            )

        output_frame = draw_overlay(frame, score, risk_level, color, reasons)

        writer.write(output_frame)

        cv2.imshow("Crowd Panic Estimation", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        previous_magnitude = magnitude
        prev_gray = curr_gray
        frame_number += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    save_features_csv(FEATURE_CSV_PATH, all_rows)

    print("Rule-based video saved:", RULE_BASED_OUTPUT_VIDEO)
    print("Feature CSV saved:", FEATURE_CSV_PATH)


if __name__ == "__main__":
    main()