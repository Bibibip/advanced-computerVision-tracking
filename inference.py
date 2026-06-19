# inference.py
import streamlit as st          # 웹 대시보드 UI를 구성하기 위한 라이브러리
import cv2                      # 영상 처리 및 화면에 박스/글씨를 그리기 위한 OpenCV 라이브러리
import os                       # 파일 경로 등 운영체제 제어를 위한 라이브러리 (현재 코드에선 안 쓰임)
import math                     
import numpy as np              
from ultralytics import YOLO    # YOLOv8 모델을 불러오고 실행하기 위한 라이브러리
from utils import merge_results, get_area_direction  # 직접 만드신 후처리 함수들 (utils.py에 있음)

# ★ 수정된 함수: BGR 값을 한글 텍스트가 아닌 웹용 HEX 색상 코드(#RRGGBB)로 변환
def bgr_to_hex(b, g, r):
    # numpy float 값을 정수로 변환 후 hex 문자열로 포맷팅 (0~255 보장)
    r, g, b = max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"

@st.cache_resource
def load_ai_models():
    # 1. 사람을 탐지하기 위한 기본 YOLO 모델 (가벼운 n버전)
    base = YOLO('models/yolov8n.pt')
    
    # 2. 분실물(5종)을 탐지하기 위해 직접 학습시킨 커스텀 모델
    custom = YOLO('models/best.pt') 
    
    # 커스텀 모델의 클래스 번호(0~4)에 해당하는 물건 이름 매핑
    custom_names = {
        0: 'cap',           # 모자
        1: 'charger',       # 충전기
        2: 'smartphone',    # 스마트폰
        3: 'umbrella',      # 우산
        4: 'wallet'         # 지갑
    }
    # 모델 내부의 클래스 이름 정보도 우리가 정의한 이름으로 업데이트해 줍니다.
    custom.model.names.update(custom_names)
    
    # 기본 모델, 커스텀 모델, 이름 사전을 튜플 형태로 반환
    return base, custom, custom_names

# ★ 수정: 함수 인자에 character_placeholder 통로 추가!
def run_video_analysis(uploaded_file, threshold, video_placeholder, character_placeholder):
    # 위에서 정의한 캐싱된 함수를 호출해 모델들을 가져옵니다. (매우 빠름)
    model_base, model_custom, custom_names = load_ai_models()
    
    # 분석이 진행 중임을 상태값에 저장 (UI 컨트롤용)
    st.session_state.analysis_done = False
    
    # 사용자가 올린 영상을 임시 파일(temp_input.mp4)로 저장합니다.
    # OpenCV는 파일 경로를 통해서 영상을 읽어오기 때문입니다.
    input_tmp = "temp_input.mp4"
    with open(input_tmp, "wb") as f:
        f.write(uploaded_file.read())
        
    # OpenCV로 방금 저장한 영상을 불러옵니다.
    cap = cv2.VideoCapture(input_tmp)
    
    # 영상의 초당 프레임(FPS)을 가져오되, 못 가져오면 기본값 30으로 설정합니다.
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    # 영상의 가로, 세로 해상도를 가져옵니다. (나중에 이동 방향 계산할 때 필요)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 웹 화면에 보여줄 진행 상태 텍스트와 프로그레스 바 영역을 미리 만듭니다.
    progress_status = st.empty()
    progress_bar = st.progress(0)
    
    # 영상의 총 프레임 수를 가져옵니다. (진행률 계산용)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 변수 초기화
    frame_count = 0                  # 현재 몇 번째 프레임인지 세는 변수
    detected_objects_in_video = set()# 영상 내에서 한 번이라도 발견된 객체 이름을 담는 바구니 (중복 제거 위해 set 사용)
    max_conf = 0.0                   # 지금까지 발견된 객체 중 가장 높은 신뢰도(정확도) 저장
    person_paths = {}                # 사람의 이동 궤적을 저장하는 사전 {사람ID: [(x, y), (x, y)...]}
    
    best_crop = None                    
    crop_path = "cropped_item.jpg"      
    
    item_states = {}   
    person_colors = {} # 이제 한글 텍스트가 아닌 {"upper": "#FFFFFF", "lower": "#000000"} 딕셔너리로 저장됨
    st.session_state.lost_owner_colors = None # 대시보드로 색상 HEX 코드를 넘기기 위한 세션 (이름 변경)
    
    # 웹 화면에 띄워줄 분석 로그 리스트 (시작 메시지 추가)
    dynamic_logs = [f"[00:01] 해상도 {frame_width}x{frame_height} 탐지완료. BoT-SORT 구역 설정 완료."]
    
    # ================= [본격적인 영상 프레임 분석 시작 (While Loop)] =================
    while cap.isOpened():
        success, frame = cap.read() # 영상에서 프레임 1장을 읽어옵니다.
        if not success:             # 영상이 끝났으면 반복문 종료
            break
        
        # 기본 모델(사람 등 탐지) 트래킹 실행
        # persist=True (이전 프레임의 객체를 기억함), tracker="botsort.yaml" (BoT-SORT 추적기 사용)
        res_base = model_base.track(source=frame, conf=threshold, imgsz=640, verbose=False, persist=True, tracker="botsort.yaml")
        
        # 커스텀 모델(분실물 5종 탐지) 트래킹 실행
        res_custom = model_custom.track(source=frame, conf=threshold, imgsz=640, verbose=False, persist=True, tracker="botsort.yaml")
        
        # 두 모델의 결과를 하나로 합칩니다. (utils.py에 있는 커스텀 함수)
        merged_boxes, merged_scores, merged_clss, merged_ids = merge_results(res_base, res_custom)
        
        # 화면에 박스를 그릴 도화지(현재 프레임 복사본)를 준비합니다.
        annotated_frame = frame.copy()
        
        # 현재 프레임의 시간(초)을 계산하여 [00:00] 형식의 문자열로 만듭니다.
        current_sec_exact = frame_count / fps  
        current_sec = int(current_sec_exact)
        time_stamp = f"[{current_sec//60:02d}:{current_sec%60:02d}]"
        
        current_persons = {}
        for box, score, cls_id, track_id in zip(merged_boxes, merged_scores, merged_clss, merged_ids):
            if int(cls_id) < 1000 and model_base.names[int(cls_id)] == "person" and int(track_id) != -1:
                px1, py1, px2, py2 = map(int, box)
                current_persons[int(track_id)] = (px1, py1, px2, py2)

        # 병합된 탐지 결과들을 하나씩 꺼내서 확인합니다.
        for box, score, cls_id, track_id in zip(merged_boxes, merged_scores, merged_clss, merged_ids):
            x1, y1, x2, y2 = map(int, box)  # 바운딩 박스의 좌상단(x1, y1), 우하단(x2, y2) 좌표
            cid = int(cls_id)               # 클래스 번호
            tid = int(track_id)             # 트래킹 ID (동일 인물/물건 유지 번호)
            
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) # 박스의 정중앙 좌표 (이동 궤적용)
            
            # [분실물 처리 로직]
            # (아마 merge_results 함수에서 커스텀 모델의 클래스 번호에 1000을 더해서 넘겨주도록 코딩하신 것 같습니다. 겹침 방지용)
            if cid >= 1000:
                custom_idx = cid - 1000  # 원래 클래스 번호(0~4)로 복구
                item_name = custom_names.get(custom_idx, "Unknown")
                
                # 라벨 텍스트 만들기 (추적 ID가 있으면 ID도 표시)
                label = f"★{item_name} ID:{tid}" if tid != -1 else f"★{item_name}"
                color = (0, 0, 255) # 분실물은 빨간색 박스(BGR 방식이라 R이 맨 끝)
                
                # 처음 발견된 분실물이라면 로그에 기록하고 세트에 추가
                if item_name != "Unknown" and item_name not in detected_objects_in_video:
                    dynamic_logs.append(f"{time_stamp} 사물: {item_name} 감지 (신뢰도: {int(score*100)}%)")
                    detected_objects_in_video.add(item_name)
                    
                # 역대 최고 신뢰도 갱신
                if score > max_conf: 
                    max_conf = float(score)
                    cy1_crop, cy2_crop = max(0, y1), min(frame_height, y2)
                    cx1_crop, cx2_crop = max(0, x1), min(frame_width, x2)
                    best_crop = frame[cy1_crop:cy2_crop, cx1_crop:cx2_crop]
                
                if tid != -1:
                    # 상태 변수 초기화 시 2초 검증용 타이머(candidate_owner, overlap_start) 추가
                    if tid not in item_states:
                        item_states[tid] = {
                            'owner_id': None, 
                            'status': 'idle', 
                            'drop_time': 0.0, 
                            'drop_pos': (cx, cy),
                            'candidate_owner': None,
                            'overlap_start': 0.0
                        }
                    
                    state = item_states[tid]
                    
                    # 50픽셀 마진(margin) 적용하여 넓게 겹침 판별
                    margin = 50 
                    overlapping_person = None
                    for p_tid, (px1, py1, px2, py2) in current_persons.items():
                        if (px1 - margin) <= cx <= (px2 + margin) and (py1 - margin) <= cy <= (py2 + margin):
                            overlapping_person = p_tid
                            break
                    
                    if overlapping_person is not None:
                        # 2초 이상 지속 겹침(소유) 판별 로직
                        if state['owner_id'] is None:
                            if state['candidate_owner'] != overlapping_person:
                                state['candidate_owner'] = overlapping_person
                                state['overlap_start'] = current_sec_exact
                            else:
                                if (current_sec_exact - state['overlap_start']) >= 2.0:
                                    state['owner_id'] = overlapping_person
                                    state['status'] = 'held'
                                    dynamic_logs.append(f"{time_stamp} [이벤트] 2초 이상 소지 확인. 소유자(#{overlapping_person}) 확정.")
                        else:
                            if state['owner_id'] == overlapping_person:
                                state['status'] = 'held'
                    
                    else:
                        # 겹침이 풀렸을 경우
                        state['candidate_owner'] = None 
                        
                        if state['owner_id'] is not None:
                            owner = state['owner_id']
                            
                            if state['status'] == 'held':
                                state['status'] = 'dropped'
                                state['drop_time'] = current_sec_exact
                                state['drop_pos'] = (cx, cy)
                                dynamic_logs.append(f"{time_stamp} [이벤트] 소유자(#{owner})와 분리 감지. {item_name} 방치 카운트 시작.")
                                
                                # ★ 핵심: 떨어지는 즉시 실시간으로 캐릭터 그림을 덧칠해서 화면에 쏴줍니다!
                                owner_colors = person_colors.get(owner, {"upper": "#cccccc", "lower": "#555555"})
                                st.session_state.lost_owner_colors = owner_colors
                                
                                upper_c, lower_c = owner_colors["upper"], owner_colors["lower"]
                                dynamic_svg = f"""
                                <div style="display: flex; justify-content: center; align-items: center; background-color:#2b2b2b; border-radius:10px; padding:20px;">
                                    <svg viewBox="0 0 100 200" style="width: 100%; max-width: 150px;">
                                        <circle cx="50" cy="30" r="20" fill="#fcdbb6" />
                                        <path d="M 20 60 C 20 50, 80 50, 80 60 L 85 120 L 15 120 Z" fill="{upper_c}" />
                                        <path d="M 20 120 L 45 120 L 45 190 L 20 190 Z" fill="{lower_c}" />
                                        <path d="M 55 120 L 80 120 L 80 190 L 55 190 Z" fill="{lower_c}" />
                                    </svg>
                                </div>
                                """
                                character_placeholder.markdown(dynamic_svg, unsafe_allow_html=True)
                                
                            elif state['status'] == 'dropped':
                                time_passed = current_sec_exact - state['drop_time']
                                dist = math.hypot(cx - state['drop_pos'][0], cy - state['drop_pos'][1])
                                
                                # 방치 후 분실 판정 기준 시간 4.5초 -> 3.0초로 단축
                                if time_passed >= 3.0 and dist < 50:
                                    state['status'] = 'lost'
                                    
                                    # 텍스트 대신 HEX 색상 코드를 딕셔너리 형태로 가져와서 UI 세션으로 넘김
                                    owner_colors = person_colors.get(owner, {"upper": "#cccccc", "lower": "#555555"}) # 기본값(회색계열)
                                    st.session_state.lost_owner_colors = owner_colors
                                    
                                    msg = f"{time_stamp} 🚨 [최종 판정] {item_name} 분실! (소유자 ID: {owner})"
                                    if msg not in dynamic_logs:
                                        dynamic_logs.append(msg)
                    
            # [사람 처리 로직]
            else:
                item_name = model_base.names[cid] # 기본 모델에서 이름 가져옴 (보통 'person')
                label = f"{item_name} ID:{tid}" if tid != -1 else item_name
                color = (0, 255, 0) # 사람은 초록색 박스
                
                # 사람이 트래킹(ID 부여) 되었다면 동선을 기록
                if item_name == "person" and tid != -1:
                    if tid not in person_paths: # 새로운 사람이면 궤적 리스트 생성 및 로그 기록
                        person_paths[tid] = []
                        dynamic_logs.append(f"{time_stamp} 인물 추적 가동: Person ID #{tid} 등록")
                    # 해당 사람의 리스트에 현재 중앙 좌표(cx, cy) 추가
                    person_paths[tid].append((cx, cy))
                    detected_objects_in_video.add("person")
                    
                    # 상/하의 색상 HEX 코드 추출 로직
                    if tid not in person_colors:
                        cy1_c, cy2_c = max(0, y1), min(frame_height, y2)
                        cx1_c, cx2_c = max(0, x1), min(frame_width, x2)
                        p_crop = frame[cy1_c:cy2_c, cx1_c:cx2_c]
                        
                        if p_crop.size > 0:
                            h, w, _ = p_crop.shape
                            upper_half = p_crop[0:h//2, :]   
                            lower_half = p_crop[h//2:h, :]   
                            
                            up_b, up_g, up_r = np.mean(upper_half, axis=(0,1)) if upper_half.size > 0 else (204,204,204)
                            low_b, low_g, low_r = np.mean(lower_half, axis=(0,1)) if lower_half.size > 0 else (85,85,85)
                            
                            # HEX 코드 딕셔너리로 저장
                            person_colors[tid] = {
                                "upper": bgr_to_hex(up_b, up_g, up_r),
                                "lower": bgr_to_hex(low_b, low_g, low_r)
                            }

            # 영상 프레임 위에 네모 박스와 라벨 텍스트를 그립니다.
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 사람이면 걸어온 길(동선)을 파란색 점으로 그려줍니다.
            if item_name == "person" and tid != -1:
                for pt in person_paths[tid]:
                    cv2.circle(annotated_frame, pt, 3, (255, 0, 0), -1)
            
        # 작업이 완료된 1장(프레임)을 웹 화면(video_placeholder)에 덮어씌워서 보여줍니다. 
        # (이게 계속 반복되면서 동영상처럼 보임)
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        
        # 진행률 업데이트 로직
        frame_count += 1
        percent = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(percent)
        progress_status.text(f"CCTV 실시간 이벤트 분석 추적 중... ({percent}%)")

    # ================= [반복문 종료 (영상 분석 끝)] =================

    # 영상이 정상적으로 끝났다면, 마지막 장면을 캡처 파일로 저장해둡니다. (결과 보고용)
    if frame_count > 0:
        cv2.imwrite(st.session_state.last_frame_path, annotated_frame)

    cap.release()               # 메모리에서 동영상 파일 해제
    progress_status.empty()     # 상태 텍스트 지우기
    progress_bar.empty()        # 프로그레스 바 지우기
    
    # 자른 이미지(best_crop)가 존재한다면 파일로 저장하고 UI로 경로를 넘김
    if best_crop is not None and best_crop.size > 0:
        cv2.imwrite(crop_path, best_crop)
        st.session_state.cropped_item_path = crop_path
    else:
        st.session_state.cropped_item_path = None
    
    # ================= [최종 이동 방향(동선) 계산 로직] =================
    final_direction = "변화 없음"
    
    if person_paths: # 추적된 사람이 1명이라도 있다면
        # 가장 점이 많이 찍힌(가장 오래 등장한) 사람의 ID를 찾습니다.
        main_person_id = max(person_paths.keys(), key=lambda k: len(person_paths[k]))
        path = person_paths[main_person_id] # 그 사람의 이동 좌표 리스트
        
        # 찍힌 좌표가 2개 이상이어야 방향을 알 수 있음
        if len(path) >= 2:
            start_x, start_y = path[0]  # 맨 처음 나타난 위치
            end_x, end_y = path[-1]     # 맨 마지막 사라진 위치
            
            # (utils.py) 화면 좌표를 넘겨주면 '좌측', '우측' 등의 글자로 바꿔주는 함수 실행
            start_area = get_area_direction(start_x, start_y, frame_width, frame_height)
            end_area = get_area_direction(end_x, end_y, frame_width, frame_height)
            
            # 처음과 끝 위치가 같으면 머무름, 다르면 A -> B로 이동했다고 판단
            if start_area == end_area:
                final_direction = f"{start_area} 머무름"
            else:
                final_direction = f"{start_area} ➔ {end_area}"
                
            # 로그에 최종 동선 리포트 추가
            dynamic_logs.append(f"[이동 리포트] 인물 #{main_person_id} 최종 동선: {final_direction}")
    
    # ================= [최종 데이터 UI 연동 로직 (Session State 업데이트)] =================
    st.session_state.analysis_done = True
    
    if detected_objects_in_video:
        # 'person'을 제외한 순수 분실물 리스트만 뽑아냄
        custom_detected = [i for i in detected_objects_in_video if i != "person"]
        
        # 분실물이 있으면 "SMARTPHONE, WALLET" 형식으로, 분실물은 없고 사람만 있으면 "사람 감지"로 텍스트화
        st.session_state.detected_item = ", ".join(custom_detected).upper() if custom_detected else "사람 감지"
        
        # 사람이 한 번이라도 잡혔으면 계산된 방향을, 아니면 "정지 상태" 출력
        st.session_state.direction = final_direction if "person" in detected_objects_in_video else "정지 상태"
        
        # 가장 높았던 신뢰도 출력
        st.session_state.confidence = f"{int(max_conf * 100)}%" if max_conf > 0 else "85%"
    else:
        # 영상에 아무것도 없었을 경우
        st.session_state.detected_item = "없음"
        st.session_state.direction = "변화 없음"
        st.session_state.confidence = "0%"
        
    dynamic_logs.append(f"[{frame_count/fps//60:02.0f}:{frame_count/fps%60:02.0f}] 시스템 분석 완료.")
    st.session_state.logs = dynamic_logs # 완성된 로그 기록들을 세션에 저장하여 웹 화면에 출력시킴