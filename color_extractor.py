import numpy as np
import cv2

# def bgr_to_hex(b, g, r):
#     r, g, b = max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))
#     return f"#{r:02x}{g:02x}{b:02x}"

# def hsv_to_color_name(b, g, r):

#     hsv = cv2.cvtColor(
#         np.uint8([[[b, g, r]]]),
#         cv2.COLOR_BGR2HSV
#     )[0][0]

#     h = hsv[0]
#     s = hsv[1]
#     v = hsv[2]

#     # 검정
#     if v < 60:
#         return "#000000"
#     # 흰색
#     if s < 40 and v > 180:
#         return "#ffffff"
#     # 회색
#     if s < 40:
#         return "#808080"
#     # 빨강
#     if h < 10 or h > 170:
#         return "#ff0000"
#     # 노랑
#     elif h < 30:
#         return "#ffff00"
#     # 초록
#     elif h < 85:
#         return "#00ff00"
#     # 파랑
#     elif h < 130:
#         return "#0000ff"
#     # 보라
#     else:
#         return "#8000ff"

def get_person_colors(frame, x1, y1, x2, y2, frame_width, frame_height):
    cy1_c, cy2_c = max(0, y1), min(frame_height, y2)
    cx1_c, cx2_c = max(0, x1), min(frame_width, x2)
    p_crop = frame[cy1_c:cy2_c, cx1_c:cx2_c]
    
    if p_crop.size == 0:
        return {"upper": "#cccccc", "lower": "#555555"}
    
    h, w, _ = p_crop.shape
    
    # ★ 좌우 가장자리(배경) 제외하고 중앙 60%만 사용
    x_start, x_end = int(w * 0.2), int(w * 0.8)
    
    # ★ 머리(상단 ~25%)를 제외하고, 상의는 25~55% 구간, 하의는 60~90% 구간만 사용
    upper_y_start, upper_y_end = int(h * 0.25), int(h * 0.55)
    lower_y_start, lower_y_end = int(h * 0.60), int(h * 0.90)
    
    upper_region = p_crop[upper_y_start:upper_y_end, x_start:x_end]
    lower_region = p_crop[lower_y_start:lower_y_end, x_start:x_end]
    
    def region_color(region, default):

        if region.size == 0:
            return default

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        mean_s = np.mean(s)
        mean_v = np.mean(v)

        if mean_v < 70:
            return "#000000"

        if mean_s < 30 and mean_v > 180:
            return "#ffffff"

        if mean_s < 30:
            return "#808080"

        valid_h = h[s > 40]

        if len(valid_h) == 0:
            return "#808080"

        dominant_h = int(np.median(valid_h))

        if dominant_h < 10 or dominant_h > 170:
            return "#ff0000"
        elif dominant_h < 30:
            return "#ffff00"
        elif dominant_h < 85:
            return "#00ff00"
        elif dominant_h < 130:
            return "#0000ff"
        else:
            return "#8000ff"

    return {
        "upper": region_color(upper_region, "#cccccc"),
        "lower": region_color(lower_region, "#555555"),
    }