import cv2

from config.settings import VIDEO_PATH, ML_BASED_OUTPUT_VIDEO
from src.optical_flow import compute_dense_flow
from src.features import extract_features
from src.visualizer import draw_overlay
from src.ml.predictor import load_ml_model, predict_risk_from_features, label_to_risk


def main():
    model, scaler = load_ml_model(model_type="stacking")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    ret, first_frame = cap.read()

    if not ret:
        print("Could not read video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        ML_BASED_OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    previous_magnitude = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        magnitude, angle = compute_dense_flow(prev_gray, curr_gray)

        features = extract_features(magnitude, angle, previous_magnitude)

        label, confidence = predict_risk_from_features(features, model, scaler)

        risk_level, color, score = label_to_risk(label)

        reasons = [f"ML prediction confidence: {confidence:.2f}"]

        output_frame = draw_overlay(frame, score, risk_level, color, reasons)

        writer.write(output_frame)

        cv2.imshow("ML Crowd Panic Estimation", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        previous_magnitude = magnitude
        prev_gray = curr_gray

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("ML-based video saved:", ML_BASED_OUTPUT_VIDEO)


if __name__ == "__main__":
    main()