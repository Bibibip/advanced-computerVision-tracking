import cv2
import numpy as np


def bgr_to_hex(b, g, r):
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    return f"#{r:02x}{g:02x}{b:02x}"


def extract_clothes_color(frame, box):

    x1, y1, x2, y2 = map(int, box)

    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    person_crop = frame[y1:y2, x1:x2]

    if person_crop.size == 0:
        return {
            "upper": "#cccccc",
            "lower": "#555555"
        }

    h, w, _ = person_crop.shape

    upper_half = person_crop[int(h*0.2):int(h*0.6)]
    lower_half = person_crop[int(h*0.6):]

    up_b, up_g, up_r = np.mean(
        upper_half,
        axis=(0, 1)
    )

    low_b, low_g, low_r = np.mean(
        lower_half,
        axis=(0, 1)
    )

    return {
        "upper": bgr_to_hex(up_b, up_g, up_r),
        "lower": bgr_to_hex(low_b, low_g, low_r)
    }