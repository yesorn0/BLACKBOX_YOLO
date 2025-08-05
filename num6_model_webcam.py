import cv2
from ultralytics import YOLO
import torch

# 1. 디바이스 설정 (CUDA 우선)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"YOLO 디바이스: {device}")

# 2. 모델 로드
# model = YOLO("./runs/train/optimized_exp/weights/best.pt")
model = YOLO("yolov8s.pt")    # YOLOv8 medium
model.to(device)

# 3. 웹캠 연결 (0번 카메라, 외장카메라는 1, 2로 시도)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception('웹캠 열기 실패! 번호를 바꿔보세요 (0,1,2...)')

# 4. 실시간 탐지 루프
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패!")
            break

        # YOLO 추론 (conf=0.4, iou=0.5 등 조정 가능)
        results = model(
            frame,
            conf=0.4,
            iou=0.5,
            imgsz=640,
            device=device,
            verbose=False
        )

        # 결과 시각화 (annotated_frame: 탐지 결과 오버레이)
        annotated_frame = results[0].plot()

        # 화면 출력
        cv2.imshow("YOLOv8 실시간 웹캠", annotated_frame)

        # ESC(27) 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
