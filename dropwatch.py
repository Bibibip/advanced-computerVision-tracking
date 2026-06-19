# dropwatch.py
import streamlit as st
import os
from utils import CSS_STYLE
from inference import run_video_analysis

st.set_page_config(
    page_title="DropWatch",
    page_icon="🎒",
    layout="wide"
)

# CSS 스타일시트 인젝션
st.markdown(CSS_STYLE, unsafe_allow_html=True)

st.markdown('<div class="main-title">DropWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CCTV 영상 기반 분실물 탐지 및 이동 방향 분석 시스템</div>', unsafe_allow_html=True)


# State 초기화
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.detected_item = "없음"
    st.session_state.direction = "분석 전"
    st.session_state.confidence = "0%"
    st.session_state.logs = ["[00:00] 시스템이 준비되었습니다."]
    st.session_state.last_frame_path = "last_analyzed_frame.jpg"
    st.session_state.cropped_item_path = None  
    st.session_state.lost_owner_colors = None  # ★ 새로 추가됨: 옷 색상 정보 저장용

# ----------------------------------------------------------------------
# 🖥️ UI 레이아웃 구현
# ----------------------------------------------------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="section-title">대상 인물 (소유자)</div>', unsafe_allow_html=True)
    
    # ★ 핵심 수정: 실시간으로 색상이 변할 수 있도록 투명 액자(placeholder) 생성!
    character_placeholder = st.empty()
    
    colors = st.session_state.get("lost_owner_colors")
    if colors:
        upper_c = colors["upper"]
        lower_c = colors["lower"]
    else:
        # 분실 이벤트 판정 전 기본 색상 (회색/검정)
        upper_c = "#888888" 
        lower_c = "#444444" 
        
    svg_character = f"""
    <div style="display: flex; justify-content: center; align-items: center; background-color:#2b2b2b; border-radius:10px; padding:20px;">
        <svg viewBox="0 0 100 200" style="width: 100%; max-width: 150px;">
            <circle cx="50" cy="30" r="20" fill="#fcdbb6" />
            <path d="M 20 60 C 20 50, 80 50, 80 60 L 85 120 L 15 120 Z" fill="{upper_c}" />
            <path d="M 20 120 L 45 120 L 45 190 L 20 190 Z" fill="{lower_c}" />
            <path d="M 55 120 L 80 120 L 80 190 L 55 190 Z" fill="{lower_c}" />
        </svg>
    </div>
    """
    # 아까 만든 투명 액자에 기본 캐릭터 그림을 먼저 띄워둠
    character_placeholder.markdown(svg_character, unsafe_allow_html=True)

    # ★ 탐지된 물건 크롭 이미지 영역 (기존 유지)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown('<div class="section-title">탐지된 물건</div>', unsafe_allow_html=True)
    
    if st.session_state.analysis_done and st.session_state.cropped_item_path and os.path.exists(st.session_state.cropped_item_path):
        st.image(st.session_state.cropped_item_path, caption=f"분실물: {st.session_state.detected_item}", use_container_width=True)
    else:
        st.markdown("""
        <div style="background-color:#2b2b2b; border-radius:10px; padding:40px; text-align:center; color:#888; font-size: 14px;">
            영상 분석 후<br>탐지된 물건이 표시됩니다.
        </div>
        """, unsafe_allow_html=True)

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
# ⚙️ [분석 시작] 외부 파일 모듈화 파트 호출
# ----------------------------------------------------------------------
if run_button and uploaded_file:
    # ★ 핵심 수정 2: AI 분석 함수(inference.py)에 실시간으로 그림을 갈아끼울 액자(character_placeholder)를 전달!
    run_video_analysis(uploaded_file, threshold, video_placeholder, character_placeholder)
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