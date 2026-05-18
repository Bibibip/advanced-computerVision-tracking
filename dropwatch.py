import streamlit as st
import os

st.set_page_config(
    page_title="DropWatch",
    page_icon="🎒",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

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

/* Streamlit 기본 글씨 강제 표시 */
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
st.markdown(
    '<div class="sub-title">CCTV 영상 기반 분실물 탐지 및 이동 방향 분석 시스템</div>',
    unsafe_allow_html=True
)

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

    if uploaded_file:
        st.video(uploaded_file)
    else:
        st.markdown(
            '<div class="video-empty">업로드한 영상이 여기에 표시됩니다</div>',
            unsafe_allow_html=True
        )

    col_slider, col_button = st.columns([3, 1])

    with col_slider:
        threshold = st.slider(
            "탐지 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05
        )

    with col_button:
        st.write("")
        st.write("")
        run_button = st.button("분석 시작", use_container_width=True)

st.write("")
st.write("")

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("""
    <div class="result-box">
        <div class="result-label">탐지된 분실물</div>
        <div class="result-value">우산</div>
    </div>
    """, unsafe_allow_html=True)

with r2:
    st.markdown("""
    <div class="result-box">
        <div class="result-label">이동 방향</div>
        <div class="result-value">계단 방향</div>
    </div>
    """, unsafe_allow_html=True)

with r3:
    st.markdown("""
    <div class="result-box">
        <div class="result-label">신뢰도</div>
        <div class="result-value">87%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

st.markdown("""
<div class="alert-box">
    <div class="alert-title">분실물 의심 이벤트 발생</div>
    <div class="alert-text">
        우산이 일정 시간 이상 같은 위치에 머물러 있으며,
        마지막으로 겹쳤던 인물이 화면 밖으로 이동한 것으로 분석되었습니다.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

st.markdown("""
<div class="log-box">
    [00:13] person ID #3 감지<br>
    [00:15] object: umbrella 감지<br>
    [00:18] person ID #3 화면 밖 이탈<br>
    [00:23] umbrella 위치 변화 없음<br>
    [00:25] 분실물 의심 이벤트 생성
</div>
""", unsafe_allow_html=True)