# YOLO 및 Tracking
# inference.py
import streamlit as st
import cv2
import os
from ultralytics import YOLO
from utils import merge_results, get_area_direction

@st.cache_resource
def load_ai_models():
    base = YOLO('models/yolov8n.pt')
    custom = YOLO('models/best.pt') 
    custom_names = {
        0: 'cap',
        1: 'charger',
        2: 'smartphone',
        3: 'umbrella',
        4: 'wallet'
    }
    custom.model.names.update(custom_names)
    return base, custom, custom_names

def run_video_analysis(uploaded_file, threshold, video_placeholder):
    model_base, model_custom, custom_names = load_ai_models()
    
    st.session_state.analysis_done = False
    
    input_tmp = "temp_input.mp4"
    with open(input_tmp, "wb") as f:
        f.write(uploaded_file.read())
        
    cap = cv2.VideoCapture(input_tmp)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    progress_status = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    detected_objects_in_video = set()
    max_conf = 0.0
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
    
    final_direction = "변화 없음"
    if person_paths:
        main_person_id = max(person_paths.keys(), key=lambda k: len(person_paths[k]))
        path = person_paths[main_person_id]
        
        if len(path) >= 2:
            start_x, start_y = path[0]
            end_x, end_y = path[-1]
            
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