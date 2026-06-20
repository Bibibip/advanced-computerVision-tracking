import numpy as np

def bgr_to_hex(b, g, r):
    r, g, b = max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"

def get_person_colors(frame, x1, y1, x2, y2, frame_width, frame_height):
    cy1_c, cy2_c = max(0, y1), min(frame_height, y2)
    cx1_c, cx2_c = max(0, x1), min(frame_width, x2)
    p_crop = frame[cy1_c:cy2_c, cx1_c:cx2_c]
    
    if p_crop.size > 0:
        h, w, _ = p_crop.shape
        upper_half = p_crop[0:h//2, :]   
        lower_half = p_crop[h//2:h, :]   
        
        up_b, up_g, up_r = np.mean(upper_half, axis=(0,1)) if upper_half.size > 0 else (204,204,204)
        low_b, low_g, low_r = np.mean(lower_half, axis=(0,1)) if lower_half.size > 0 else (85,85,85)
        
        return {
            "upper": bgr_to_hex(up_b, up_g, up_r),
            "lower": bgr_to_hex(low_b, low_g, low_r)
        }
    return {"upper": "#cccccc", "lower": "#555555"}