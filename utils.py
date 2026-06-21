#핵심 연산 및 유틸리티 함수
# utils.py
import torch
import torchvision

# 기존 UI 테마 및 CSS 스타일 시트 정의 (100% 동일)
CSS_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght=400;500;600;700;800&display=swap');

* {
    font-family: 'Pretendard', sans-serif;
}

.stApp {
    background: #f8fafc;
    color: #0f172a;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: #0f172a;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a !important;
    margin-bottom: 6px;
}

.sub-title {
    font-size: 15px;
    color: #64748b !important;
    margin-bottom: 34px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a !important;
    margin-bottom: 16px;
}

.video-empty {
    height: 300px;
    border-radius: 18px;
    border: 1px dashed #cbd5e1;
    background: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #64748b !important;
    font-weight: 600;
}

.result-box {
    padding: 18px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
}

.result-label {
    color: #64748b !important;
    font-size: 13px;
    font-weight: 700;
}

.result-value {
    color: #0f172a !important;
    font-size: 24px;
    font-weight: 800;
    margin-top: 6px;
}

.alert-box {
    padding: 18px;
    border-radius: 16px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    line-height: 1.7;
}

.alert-title {
    color: #1e40af !important;
    font-weight: 800;
}

.alert-text {
    color: #1e3a8a !important;
}

.log-box {
    padding: 18px;
    border-radius: 16px;
    background: #0f172a;
    color: #e2e8f0 !important;
    line-height: 1.7;
    font-size: 14px;
}

.log-box * {
    color: #e2e8f0 !important;
}

.stButton > button {
    height: 45px;
    border-radius: 12px;
    background: #2563eb;
    color: white !important;
    border: none;
    font-weight: 700;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white !important;
}
</style>
"""

def merge_results(res_base, res_custom, iou_thresh=0.5):
    boxes_list, scores_list, cls_list, id_list = [], [], [], []
    
    if res_base and res_base[0].boxes is not None:
        for box in res_base[0].boxes:
            boxes_list.append(box.xyxy[0].cpu())
            scores_list.append(box.conf[0].cpu())
            cls_list.append(box.cls[0].cpu())
            track_id = int(box.id[0].item()) if box.id is not None else -1
            id_list.append(track_id)
            
    if res_custom and res_custom[0].boxes is not None:
        for box in res_custom[0].boxes:
            boxes_list.append(box.xyxy[0].cpu())
            scores_list.append(box.conf[0].cpu())
            cls_list.append((box.cls[0] + 1000).cpu())
            track_id = int(box.id[0].item()) if box.id is not None else -1
            id_list.append(track_id)
            
    if not boxes_list: 
        return [], [], [], []
        
    boxes_t = torch.stack(boxes_list)
    scores_t = torch.stack(scores_list)
    cls_t = torch.stack(cls_list)
    ids_t = torch.tensor(id_list, device=boxes_t.device)
    
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_thresh)
    return boxes_t[keep], scores_t[keep], cls_t[keep], ids_t[keep]

def get_area_direction(x, y, width, height):
    if x < width / 2 and y < height / 2:
        return "좌상단_출입문방향"
    elif x >= width / 2 and y < height / 2:
        return "우상단_화장실방향"
    elif x >= width / 2 and y >= height / 2:
        return "우하단_화장실방향"
    else:
        return "좌하단_복도방향"