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
# ⚙️ [분석 시작] 외부 파일 모듈화 파트 호출
# ----------------------------------------------------------------------
if run_button and uploaded_file:
    run_video_analysis(uploaded_file, threshold, video_placeholder)
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