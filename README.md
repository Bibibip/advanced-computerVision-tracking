DropWatch: AI 기반 분실물 탐지 시스템

1. 프로젝트 소개

 DropWatch는 CCTV 영상 내 인물과 사물을 실시간으로 탐지하고 추적하여 분실물을 자동으로 식별하는 AI 기반 분실물 탐지 시스템이다.
 본 시스템은 YOLOv8 객체 탐지 모델과 BoT-SORT 추적 알고리즘을 활용하여 사람과 사물의 위치를 지속적으로 추적한다. 또한 최초 소유자를 추정하고, 소유자와 물체가 일정 시간 이상 분리된 상태를 유지할 경우 분실 이벤트를 발생시킨다.
 분실 발생 시 마지막 소유자의 이동 방향과 의상 색상 정보를 함께 제공하여 관리자가 보다 빠르게 분실자를 식별할 수 있도록 지원한다.

 (실행 화면)

2. 개발 환경 및 의존성
## Development Environment
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4060 (8GB)
- CUDA: 13.1
- Python: 3.11.0
- PyTorch: 2.4.1 (CUDA 12.4)
- Ultralytics: 8.4.72
- ONNX Runtime GPU: 1.20.1
- OpenCV: 4.13.0
- Streamlit: 1.58.0

설치: pip install -r requirements.txt

3. 상세 설치/실행 방법
1) 저장소 Clone
git clone <repository_url>
cd advanced-computerVision-tracking
2) 가상환경 생성
python -m venv .venv
3) 가상환경 활성화

PowerShell
.\.venv\Scripts\Activate.ps1
4) 패키지 설치
pip install -r requirements.txt
5) Streamlit 실행
streamlit run dropwatch.py

## 모델 다운로드

아래 링크에서 모델 파일을 다운로드한 뒤 models 폴더에 저장하세요.

- best.pt : Google Drive 링크
- best.onnx : Google Drive 링크

프로젝트 구조
models/
 ├─ best.pt
 ├─ best.onnx
 ├─ yolov8n.pt
 └─ yolov8n.onnx
 * 실제 실행은 FPS가 더 높게 측정된 pt 파일을 사용하고 있습니다.
 
4. 데이터 파이프라인
CCTV 영상 입력
↓
YOLO 객체 탐지
(Person, Cap, Umbrella, Wallet, Smartphone, Charger)
↓
BoT-SORT 객체 추적
(Person ID 부여)
↓
사람-물체 매칭
(소유자 추정)
↓
분리 이벤트 감지
↓
분실 여부 판정
↓
분실자 의상 색상 추출
↓
Streamlit 시각화
↓
최종 결과 출력

5. 팀원별 역할 분담
이솔희: 데이터 수집 및 라벨링, 데이터 전처리, streamlit UI 구현, BoT-SORT 인물 트래킹 통합 ID 유지 로직 구현, 보고서 작성
선비: 데이터 수집 및 라벨링, YOLO Baseline 학습, streamlit UI 구현, BoT-SORT 인물 트래킹 통합 ID 유지 로직 구현, 최종 코드 제출
