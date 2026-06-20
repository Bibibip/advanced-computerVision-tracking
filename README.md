DropWatch: AI 기반 분실물 탐지 시스템

1. 프로젝트 소개

 DropWatch는 CCTV 영상 내 인물과 사물을 실시간으로 탐지하고 추적하여 분실물을 자동으로 식별하는 AI 기반 분실물 탐지 시스템이다.
 본 시스템은 YOLOv8 객체 탐지 모델과 BoT-SORT 추적 알고리즘을 활용하여 사람과 사물의 위치를 지속적으로 추적한다. 또한 최초 소유자를 추정하고, 소유자와 물체가 일정 시간 이상 분리된 상태를 유지할 경우 분실 이벤트를 발생시킨다.

 실행 화면

2. 개발 환경 및 의존성


3. 상세 설치/실행 방법


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
