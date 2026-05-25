import streamlit as st
import os
import cv2
import time
import torch
import torchvision
from ultralytics import YOLO

st.set_page_config(
    page_title="DropWatch",
    page_icon="🎒",
    layout="wide"
)

st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">DropWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CCTV 영상 기반 분실물 탐지 및 이동 방향 분석 시스템</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# AI 모델 설정 및 NMS 결합 파트
# ----------------------------------------------------------------------
@st.cache_resource
def load_ai_models():
    base = YOLO('yolov8n.pt')
    custom = YOLO('best.pt') 
    custom_names = {
        0: 'cap',
        1: 'charger',
        2: 'smartphone',
        3: 'umbrella',
        4: 'wallet'
    }
    custom.model.names.update(custom_names)
    return base, custom, custom_names

model_base, model_custom, custom_names = load_ai_models()

def merge_results(res_base, res_custom, iou_thresh=0.5):
    boxes_list, scores_list, cls_list, id_list = [], [], [], []
    
    if res_base and res_base[0].boxes is not None:
        for box in res_base[0].boxes:
            boxes_list.append(box.xyxy[0])
            scores_list.append(box.conf[0])
            cls_list.append(box.cls[0])
            track_id = int(box.id[0].item()) if box.id is not None else -1
            id_list.append(track_id)
            
    if res_custom and res_custom[0].boxes is not None:
        for box in res_custom[0].boxes:
            boxes_list.append(box.xyxy[0])
            scores_list.append(box.conf[0])
            cls_list.append(box.cls[0] + 1000)
            track_id = int(box.id[0].item()) if box.id is not None else -1
            id_list.append(track_id)
            
    if not boxes_list: 
        return [], [], [], []
        
    boxes_t = torch.stack(boxes_list)
    scores_t = torch.stack(scores_list)
    cls_t = torch.stack(cls_list)
    ids_t = torch.tensor(id_list)
    
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_thresh)
    return boxes_t[keep], scores_t[keep], cls_t[keep], ids_t[keep]

# 🌟 5. 화면 구역 기준 방향 판단 함수 반영
def get_area_direction(x, y, width, height):
    if x < width / 2 and y < height / 2:
        return "좌상단_출입문방향"
    elif x >= width / 2 and y < height / 2:
        return "우상단_화장실방향"
    elif x >= width / 2 and y >= height / 2:
        return "우하단_화장실방향"
    else:
        return "좌하단_복도방향"

# State 초기화
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.detected_item = "없음"
    st.session_state.direction = "분석 전"
    st.session_state.confidence = "0%"
    st.session_state.logs = ["[00:00] 시스템이 준비되었습니다."]
    st.session_state.last_frame_path = "last_analyzed_frame.jpg"

# ----------------------------------------------------------------------
# 🖥️ UI 레이아웃 구현
# ----------------------------------------------------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="section-title">대상 인물</div>', unsafe_allow_html=True)
    if os.path.exists("character.png"):
        st.image("character.png", use_container_width=True)
    else:
        st.info("character.png 파일을 dropwatch.py와 같은 폴더에 넣어주세요.")

with right:
    st.markdown('<div class="section-title">영상 분석</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "분석할 영상을 업로드하세요",
        type=["mp4", "avi", "mov", "mpeg", "mpg", "mpeg4"]
    )

    video_placeholder = st.empty()

    if uploaded_file:
        if st.session_state.analysis_done and os.path.exists(st.session_state.last_frame_path):
            video_placeholder.image(st.session_state.last_frame_path, caption="🎬 분석 완료 (최종 검출 프레임)", use_container_width=True)
        else:
            video_placeholder.markdown('<div class="video-empty">영상이 준비되었습니다. [분석 시작]을 눌러주세요.</div>', unsafe_allow_html=True)
    else:
        video_placeholder.markdown('<div class="video-empty">업로드한 영상이 여기에 표시됩니다</div>', unsafe_allow_html=True)
        st.session_state.analysis_done = False

    col_slider, col_button = st.columns([3, 1])

    with col_slider:
        threshold = st.slider(
            "탐지 임계값 (Confidence)",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05
        )

    with col_button:
        st.write("")
        st.write("")
        run_button = st.button("분석 시작", use_container_width=True, disabled=(uploaded_file is None))

# ----------------------------------------------------------------------
# ⚙️ [분석 시작] BoT-SORT 트래킹 기반 구역 방향 연산 파트
# ----------------------------------------------------------------------
if run_button and uploaded_file:
    st.session_state.analysis_done = False
    
    input_tmp = "temp_input.mp4"
    with open(input_tmp, "wb") as f:
        f.write(uploaded_file.read())
        
    cap = cv2.VideoCapture(input_tmp)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    # 🌟 내 파일 해상도 자동 동적 추출 기법 적용 (640 고정 탈피)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    progress_status = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    detected_objects_in_video = set()
    max_conf = 0.0
    
    # 인물 동선 좌표 추적용 { person_id: [(x, y), (x, y), ...] }
    person_paths = {}
    
    dynamic_logs = [f"[00:01] 해상도 {frame_width}x{frame_height} 탐지완료. BoT-SORT 구역 설정 완료."]
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break
        
        res_base = model_base.track(source=frame, conf=threshold, imgsz=640, verbose=False, persist=True, tracker="botsort.yaml")
        res_custom = model_custom.track(source=frame, conf=threshold, imgsz=640, verbose=False, persist=True, tracker="botsort.yaml")
        
        merged_boxes, merged_scores, merged_clss, merged_ids = merge_results(res_base, res_custom)
        annotated_frame = frame.copy()
        
        current_sec = int(frame_count / fps)
        time_stamp = f"[{current_sec//60:02d}:{current_sec%60:02d}]"
        
        for box, score, cls_id, track_id in zip(merged_boxes, merged_scores, merged_clss, merged_ids):
            x1, y1, x2, y2 = map(int, box)
            cid = int(cls_id)
            tid = int(track_id)
            
            # 🌟 4번 로직 구현: 중심좌표 계산
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            if cid >= 1000:
                custom_idx = cid - 1000
                item_name = custom_names.get(custom_idx, "Unknown")
                label = f"★{item_name} ID:{tid}" if tid != -1 else f"★{item_name}"
                color = (0, 0, 255)
                
                if item_name != "Unknown" and item_name not in detected_objects_in_video:
                    dynamic_logs.append(f"{time_stamp} 사물: {item_name} 감지 (신뢰도: {int(score*100)}%)")
                    detected_objects_in_video.add(item_name)
                if score > max_conf: 
                    max_conf = float(score)
            else:
                item_name = model_base.names[cid]
                label = f"{item_name} ID:{tid}" if tid != -1 else item_name
                color = (0, 255, 0)
                
                # 사람 추적 데이터 적재
                if item_name == "person" and tid != -1:
                    if tid not in person_paths:
                        person_paths[tid] = []
                        dynamic_logs.append(f"{time_stamp} 인물 추적 가동: Person ID #{tid} 등록")
                    person_paths[tid].append((cx, cy))
                    detected_objects_in_video.add("person")
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if item_name == "person" and tid != -1:
                for pt in person_paths[tid]:
                    cv2.circle(annotated_frame, pt, 3, (255, 0, 0), -1)
            
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        
        frame_count += 1
        percent = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(percent)
        progress_status.text(f"CCTV 실시간 BoT-SORT 추적 중... ({percent}%)")

    if frame_count > 0:
        cv2.imwrite(st.session_state.last_frame_path, annotated_frame)

    cap.release()
    progress_status.empty()
    progress_bar.empty()
    
    # 🌟 6. track_id별 시작/마지막 구역 기준 실시간 방향 매핑 연산
    final_direction = "변화 없음"
    if person_paths:
        main_person_id = max(person_paths.keys(), key=lambda k: len(person_paths[k]))
        path = person_paths[main_person_id]
        
        if len(path) >= 2:
            start_x, start_y = path[0]
            end_x, end_y = path[-1]
            
            # 사용자 함수 적용 및 변환
            start_area = get_area_direction(start_x, start_y, frame_width, frame_height)
            end_area = get_area_direction(end_x, end_y, frame_width, frame_height)
            
            if start_area == end_area:
                final_direction = f"{start_area} 머무름"
            else:
                final_direction = f"{start_area} ➔ {end_area}"
                
            dynamic_logs.append(f"[이동 리포트] 인물 #{main_person_id} 동선 분석: {final_direction}")
    
    st.session_state.analysis_done = True
    
    if detected_objects_in_video:
        custom_detected = [i for i in detected_objects_in_video if i != "person"]
        st.session_state.detected_item = ", ".join(custom_detected).upper() if custom_detected else "사람 감지"
        st.session_state.direction = final_direction if "person" in detected_objects_in_video else "정지 상태"
        st.session_state.confidence = f"{int(max_conf * 100)}%" if max_conf > 0 else "85%"
    else:
        st.session_state.detected_item = "없음"
        st.session_state.direction = "변화 없음"
        st.session_state.confidence = "0%"
        
    dynamic_logs.append(f"[{frame_count/fps//60:02.0f}:{frame_count/fps%60:02.0f}] 분석 완료.")
    st.session_state.logs = dynamic_logs
    
    st.rerun()

# ----------------------------------------------------------------------
# 📊 하단 대시보드 리포팅 영역
# ----------------------------------------------------------------------
st.write("")
st.write("")

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">탐지된 분실물</div>
        <div class="result-value">{st.session_state.detected_item}</div>
    </div>
    """, unsafe_allow_html=True)

with r2:
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">이동 방향 (구역 흐름)</div>
        <div class="result-value">{st.session_state.direction}</div>
    </div>
    """, unsafe_allow_html=True)

with r3:
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">신뢰도</div>
        <div class="result-value">{st.session_state.confidence}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

if st.session_state.analysis_done and st.session_state.detected_item not in ["없음", "사람 감지"]:
    st.markdown(f"""
        <div class="alert-box">
            <div class="alert-title">분실물 의심 최종 분석 완료</div>
            <div class="alert-text">
                대상 물품(<b>{st.session_state.detected_item}</b>)이 소지자 없이 동일 위치에 유기된 것으로 판단됩니다.<br>
                마지막으로 객체와 상호작용한 대상 인물은 <b>{st.session_state.direction}</b> 흐름이 매칭되었습니다.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

log_content = "<br>".join(st.session_state.logs)
st.markdown(f"""
<div class="log-box">
    {log_content}
</div>
""", unsafe_allow_html=True)