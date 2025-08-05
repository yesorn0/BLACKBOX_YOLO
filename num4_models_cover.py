from ultralytics import YOLO
import cv2
import torch

model_path = "./runs/train/optimized_exp/weights/best.pt"
video_path = "./videos/youtube_download.mp4"
result_path = "./videos/youtube_result_batch.mp4"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 디바이스: {device}")

model = YOLO(model_path)
model.to(device)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(result_path, fourcc, fps, (w, h))

batch_size = 8  # 원하는 배치 크기(6GB GPU면 2~4 적당)
frame_batch = []
frame_indices = []

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_batch.append(frame)
    frame_indices.append(frame_idx)
    frame_idx += 1

    # batch_size만큼 쌓이면 한 번에 추론
    if len(frame_batch) == batch_size:
        # 신뢰도(conf) 0.3 이상만 출력
        results = model(frame_batch, device=device, conf=0.3)
        for i, res in enumerate(results):
            annotated_frame = res.plot().copy()
            out.write(annotated_frame)
        frame_batch.clear()
        frame_indices.clear()

# 남은 프레임(끝부분)도 처리
if frame_batch:
    results = model(frame_batch, device=device, conf=0.3)
    for i, res in enumerate(results):
        annotated_frame = res.plot().copy()
        out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"저장 완료: {result_path}")
