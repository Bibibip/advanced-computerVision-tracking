from ultralytics import YOLO

# 사람 탐지 모델
base = YOLO('./yolov8n.pt')
base.export(format='onnx', imgsz=1024, simplify=True, half=True, device=0)

# 물건 탐지 모델
custom = YOLO('./best.pt')
custom.export(format='onnx', imgsz=1024, simplify=True, half=True, device=0)