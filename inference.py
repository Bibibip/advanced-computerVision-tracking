import streamlit as st
import cv2
import os
import numpy as np              
from ultralytics import YOLO
from utils import merge_results, get_area_direction
TRACKER_PERSON_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort_person.yaml")
TRACKER_ITEM_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort_item.yaml")

# 분리한 모듈 임포트
from color_extractor import get_person_colors
from owner_detector import find_overlapping_person, update_ownership
from lost_detector import check_lost_status, check_recovered_status
from object_manager import match_items 

@st.cache_resource
def load_ai_models():
    base = YOLO('models/yolov8n.pt')
    custom = YOLO('models/best.pt') 
    custom_names = {
        0: 'cap', 1: 'charger', 2: 'smartphone', 3: 'umbrella', 4: 'wallet'
    }
    custom.model.names.update(custom_names)
    print("모델이 인식하는 클래스 이름들:", custom.model.names)
    return base, custom, custom_names

def run_video_analysis(uploaded_file, threshold, video_placeholder, character_placeholder):
    model_base, model_custom, custom_names = load_ai_models()
     # ★ 추가: 매 분석 시작 시 트래커 상태 초기화
    for m in (model_base, model_custom):
        if hasattr(m, "predictor") and m.predictor is not None:
            m.predictor.trackers[0].reset()
        else:
            m.predictor = None  # 다음 track() 호출 시 새로 생성되게

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
    
    best_crop = None                    
    crop_path = "cropped_item.jpg"      
    
    item_states = {}   
    person_colors = {} 
    st.session_state.lost_owner_colors = None 
    
    dynamic_logs = [f"[00:01] 해상도 {frame_width}x{frame_height} 탐지완료. 고해상도 모드 가동."]
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break
        
        res_base = model_base.track(source=frame, conf=threshold, imgsz=1024, verbose=False, persist=True, tracker=TRACKER_PERSON_CFG)
        res_custom = model_custom.track(source=frame, conf=threshold, imgsz=1024, verbose=False, persist=True, tracker=TRACKER_ITEM_CFG)
        
        merged_boxes, merged_scores, merged_clss, merged_ids = merge_results(res_base, res_custom)
        annotated_frame = frame.copy()
        
        current_sec_exact = frame_count / fps  
        current_sec = int(current_sec_exact)
        time_stamp = f"[{current_sec//60:02d}:{current_sec%60:02d}]"
        
        current_persons = {}
        
        # 1. 사람 탐지 및 경로/색상 추출
        for box, score, cls_id, track_id in zip(merged_boxes, merged_scores, merged_clss, merged_ids):
            x1, y1, x2, y2 = map(int, box)
            cid = int(cls_id)
            tid = int(track_id)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if cid < 1000 and model_base.names[cid] == "person" and tid != -1:
                current_persons[tid] = (x1, y1, x2, y2)
                
                color = (0, 255, 0)
                label = f"person ID:{tid}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if tid not in person_paths:
                    person_paths[tid] = []
                    dynamic_logs.append(f"{time_stamp} 인물 추적 가동: Person ID #{tid} 등록")
                person_paths[tid].append((cx, cy))
                detected_objects_in_video.add("person")
                
                person_colors[tid] = get_person_colors(frame, x1, y1, x2, y2, frame_width, frame_height)

        # 2. 사물 탐지
        current_items = []
        for box, score, cls_id, track_id in zip(merged_boxes, merged_scores, merged_clss, merged_ids):
            x1, y1, x2, y2 = map(int, box)
            cid = int(cls_id)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if cid >= 1000:
                custom_idx = cid - 1000
                item_name = custom_names.get(custom_idx, "Unknown")
                max_conf = max(max_conf, float(score))
                
                if item_name != "Unknown" and item_name not in detected_objects_in_video:
                    detected_confidence = float(score)
                    dynamic_logs.append(
                        f"{time_stamp} 사물: {item_name} 감지 (신뢰도: {int(score*100)}%)"
                    )
                    detected_objects_in_video.add(item_name)
                current_items.append({'name': item_name, 'cx': cx, 'cy': cy, 'box': (x1, y1, x2, y2), 'score': score})

        # 3. 사물 ID 매칭
        matched_ids = match_items(current_items, item_states)

        # 4. 소유권 및 분실 판별
        for my_id, state in item_states.items():
            if my_id not in matched_ids:
                continue 

            cx, cy = state['cx'], state['cy']
            item_name = state['name']
            x1, y1, x2, y2 = state['box']

            color = (0, 0, 255)
            label = f"★{item_name} ID:{my_id}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

             # 4-1. 소유자 판별 (거리 기반 및 최초 소지자 즉시 매칭)
            overlapping_person = find_overlapping_person(cx, cy, current_persons, margin=120)
            is_new = state.pop('just_created', False)  # 한 번만 True, 그 이후엔 False        
            just_owned = update_ownership(state, overlapping_person, current_sec_exact, is_new_item=is_new)
                     
            if just_owned:
                state["owner_color"] = person_colors.get(
                    overlapping_person,
                    {
                        "upper": "#cccccc",
                        "lower": "#555555"
                    }
                )

            # ★ 추가: "소유자 본인"이 여전히 겹쳐있는지로 분리 여부 판단
            if state['owner_id'] is not None:
                owner_still_overlapping = (
                    state['owner_id'] in current_persons and
                    find_overlapping_person(cx, cy, {state['owner_id']: current_persons[state['owner_id']]}, margin=50) is not None
                )
                effective_overlap = state['owner_id'] if owner_still_overlapping else None
            else:
                effective_overlap = overlapping_person

            # 4-2. 분리 및 분실 감별
            if effective_overlap is None:
                just_dropped, just_lost = check_lost_status(state, cx, cy, current_sec_exact)
                
                if just_dropped:
                    owner = state['owner_id']
                    owner_colors = person_colors.get(owner, {"upper": "#cccccc", "lower": "#555555"})

                    st.session_state.lost_owner_colors = owner_colors
                    
                    upper_c, lower_c = owner_colors["upper"], owner_colors["lower"]
                    dynamic_svg = f"""
                    <div style="display: flex; justify-content: center; align-items: center; background-color:#e8eef5; border-radius:10px; padding:20px;">
                        <svg viewBox="0 0 100 200" style="width: 100%; max-width: 150px;">
                            <circle cx="50" cy="30" r="20" fill="#fcdbb6" />
                            <path d="M 20 60 C 20 50, 80 50, 80 60 L 85 120 L 15 120 Z" fill="{upper_c}" />
                            <path d="M 20 120 L 45 120 L 45 190 L 20 190 Z" fill="{lower_c}" />
                            <path d="M 55 120 L 80 120 L 80 190 L 55 190 Z" fill="{lower_c}" />
                        </svg>
                    </div>
                    """
                    character_placeholder.markdown(dynamic_svg, unsafe_allow_html=True)
                    
                if just_lost:
                    msg = f"{time_stamp} 🚨 [최종 판정] {item_name} 분실! (소유자 ID: {state['owner_id']})"
                    cy1_crop, cy2_crop = max(0, y1), min(frame_height, y2)
                    cx1_crop, cx2_crop = max(0, x1), min(frame_width, x2)

                    best_crop = frame[
                        cy1_crop:cy2_crop,
                        cx1_crop:cx2_crop
                    ].copy()

                    if msg not in dynamic_logs:
                        dynamic_logs.append(msg)
            # 회수 판정
            if effective_overlap is not None:

                just_recovered, is_owner_recovery = (
                    check_recovered_status(
                        state,
                        effective_overlap,
                        current_sec_exact
                    )
                )

                if (
                    just_recovered
                    and is_owner_recovery
                    and not state.get("recovered_logged", False)
                ):
                    state["retrieved"] = True
                    state["recovered_logged"] = True

                    dynamic_logs.append(
                        f"{time_stamp} ✅ "
                        f"{item_name} 회수 완료 "
                        f"(소유자 ID:{state['owner_id']})"
                    )

                    st.session_state.recovery_message = (
                        f"✅ 원래 소유자가 "
                        f"{item_name}을 회수했습니다."
                    )

                    cv2.putText(
                        annotated_frame,
                        "RECOVERED",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        3
                    )
        
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        
        frame_count += 1
        percent = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(percent)
        progress_status.text(f"CCTV 실시간 이벤트 분석 추적 중... ({percent}%)")

    if frame_count > 0:
        cv2.imwrite(st.session_state.last_frame_path, annotated_frame)

    cap.release()
    progress_status.empty()
    progress_bar.empty()
    
    if best_crop is not None and best_crop.size > 0:
        cv2.imwrite(crop_path, best_crop)
        st.session_state.cropped_item_path = crop_path
    else:
        st.session_state.cropped_item_path = None
    
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
    
    st.session_state.analysis_done = True
    
    if detected_objects_in_video:
        custom_detected = [i for i in detected_objects_in_video if i != "person"]
        detected_name = (
            ", ".join(custom_detected).upper()
            if custom_detected
            else "사람 감지"
        )

        recovered = any(
            state.get("retrieved", False)
            for state in item_states.values()
        )

        if recovered:
            detected_name += " ✅ 회수완료"

        st.session_state.detected_item = detected_name
        st.session_state.direction = final_direction if "person" in detected_objects_in_video else "정지 상태"
        st.session_state.confidence = f"{int(detected_confidence * 100)}%" if detected_confidence > 0 else "85%"
    else:
        st.session_state.detected_item = "없음"
        st.session_state.direction = "변화 없음"
        st.session_state.confidence = "0%"
        
    dynamic_logs.append(f"[{frame_count/fps//60:02.0f}:{frame_count/fps%60:02.0f}] 시스템 분석 완료.")
    st.session_state.logs = dynamic_logs