from ultralytics import YOLO
import cv2
import time
import torch
import gc
from video_control import VideoPlayer

# GPU 메모리 관리 설정
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # GPU 메모리 사용량 제한 (3060 6GB의 80% 사용)
    torch.cuda.set_per_process_memory_fraction(0.8)
    device = 'cuda'
    print(f"GPU 메모리 정보: {torch.cuda.get_device_name(0)}")
    print(f"사용 가능한 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
else:
    device = 'cpu'

print(f"사용 디바이스: {device}")

# 1. 모델 로드 및 최적화
model = YOLO("./runs/train/optimized_exp/weights/best.pt")
# model = YOLO("./yolov8m.pt")
model.to(device)

# 모델 최적화 (TensorRT 사용 가능시)
try:
    if device == 'cuda':
        # 첫 번째 더미 추론으로 모델 워밍업
        dummy_img = torch.randn(1, 3, 640, 640).to(device)
        _ = model(dummy_img, verbose=False)
        print("모델 워밍업 완료")
except:
    print("모델 워밍업 실패, 계속 진행...")

# 2. 비디오 설정
video_path = "./videos/youtube_download.mp4"
result_path = "./videos/result_optimized.mp4"

cap = cv2.VideoCapture(video_path)

# 비디오 정보 가져오기
original_fps = cap.get(cv2.CAP_PROP_FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"원본 해상도: {original_width}x{original_height}")
print(f"원본 FPS: {original_fps:.2f}")
print(f"총 프레임 수: {total_frames}")

# VideoPlayer 초기화 (적당한 창 크기로 설정)
player = VideoPlayer(window_width=960, window_height=540, control_height=80)
player.set_video_info(total_frames, original_fps)

# 성능 최적화 설정
CONF_THRESHOLD = 0.4      # 모델 신뢰도 임계값
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
TARGET_FPS = 30  # 목표 FPS

# 비디오 저장 설정
save_output = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(result_path, fourcc, TARGET_FPS,
                      (player.WINDOW_WIDTH, player.WINDOW_HEIGHT)) if save_output else None

# 성능 측정
start_time = time.time()
fps_list = []
last_time = time.time()

# OpenCV 윈도우 생성
window_name = "객체 탐지 비디오 플레이어"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# 컨트롤 사용법 출력
player.print_controls_help()

print(f"\n=== 처리 시작 ===")
print(f"목표 FPS: {TARGET_FPS}")
print(f"화면 크기: {player.WINDOW_WIDTH}x{player.WINDOW_HEIGHT}")

try:
    while True:
        current_time = time.time()

        # FPS 제한 (30fps 목표)
        if player.is_playing:
            elapsed_time = current_time - last_time
            target_frame_time = 1.0 / TARGET_FPS

            if elapsed_time < target_frame_time:
                time.sleep(target_frame_time - elapsed_time)
                current_time = time.time()

        # 시킹 중이 아닐 때만 다음 프레임으로 진행
        if player.is_playing and not player.seeking:
            ret, frame = cap.read()
            if not ret:
                break

            player.update_current_frame(cap)
        else:
            # 일시정지 상태이거나 시킹 중일 때는 현재 프레임 유지
            cap.set(cv2.CAP_PROP_POS_FRAMES, player.current_frame)
            ret, frame = cap.read()
            if not ret:
                break

        # 추론 시작 시간
        infer_start = time.time()

        # 객체 탐지
        results = model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
            half=True if device == 'cuda' else False
        )

        # 추론 시간 계산
        infer_time = time.time() - infer_start
        current_fps = 1.0 / infer_time if infer_time > 0 else 0
        fps_list.append(current_fps)

        # 시각화
        annotated_frame = results[0].plot().copy()

        # 성능 정보 오버레이
        info_y = 30
        cv2.putText(annotated_frame, f"추론 FPS: {current_fps:.1f}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:]) if fps_list else 0
        cv2.putText(annotated_frame, f"평균 FPS: {avg_fps:.1f}",
                    (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 재생 상태 표시
        status = "재생중" if player.is_playing else "일시정지"
        cv2.putText(annotated_frame, f"상태: {status}",
                    (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # GPU 메모리 사용량 표시 (CUDA 사용시)
        if device == 'cuda':
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3
            cv2.putText(annotated_frame, f"GPU: {gpu_mem:.1f}GB",
                        (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # VideoPlayer로 프레임 준비 (크기 조정 및 컨트롤 패널 추가)
        final_frame, seekbar_info = player.prepare_frame(annotated_frame)

        # 마우스 콜백 설정
        cv2.setMouseCallback(window_name, player.mouse_callback, (cap, seekbar_info))

        # 저장
        if save_output and out is not None and player.is_playing:
            # 컨트롤 패널 제외하고 저장
            save_frame = final_frame[:player.WINDOW_HEIGHT, :]
            out.write(save_frame)

        # 화면 출력
        cv2.imshow(window_name, final_frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break
        elif key != 255:  # 키가 눌렸을 때
            player.handle_keyboard(key, cap)

            # 추가 키 처리
            if key == ord('q'):
                break
            elif key == ord('r'):  # 저장 토글
                save_output = not save_output
                print(f"저장 {'켜짐' if save_output else '꺼짐'}")

        # 메모리 정리 (주기적으로)
        if len(fps_list) % 100 == 0:
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        last_time = current_time

except KeyboardInterrupt:
    print("\n중단됨")

finally:
    # 정리
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # 최종 통계
    total_time = time.time() - start_time
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"\n=== 최종 통계 ===")
        print(f"총 실행 시간: {total_time:.2f}초")
        print(f"평균 추론 FPS: {avg_fps:.2f}")
        print(f"처리된 프레임: {len(fps_list)}")

        if device == 'cuda':
            print(f"최대 GPU 메모리 사용량: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

        if save_output:
            print(f"결과 저장됨: {result_path}")

    # GPU 메모리 정리
    if device == 'cuda':
        torch.cuda.empty_cache()

    print("완료!")