import cv2


def draw_overlay(frame, score, risk_level, color, reasons=None):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Panic Score: {score:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Risk Level: {risk_level}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    if reasons:
        text = "Reason: " + reasons[0]
        cv2.putText(
            frame,
            text[:80],
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    return frame