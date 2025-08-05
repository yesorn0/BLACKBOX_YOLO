import cv2
import numpy as np


class VideoPlayer:
    def __init__(self, window_width=960, window_height=540, control_height=80):
        self.WINDOW_WIDTH = window_width
        self.WINDOW_HEIGHT = window_height
        self.CONTROL_HEIGHT = control_height

        # 플레이어 상태 변수들
        self.is_playing = True
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.seeking = False
        self.seek_pos = 0

    def set_video_info(self, total_frames, fps):
        """비디오 정보 설정"""
        self.total_frames = total_frames
        self.fps = fps

    def format_time(self, frame_num):
        """프레임 번호를 시:분:초 형식으로 변환"""
        seconds = int(frame_num / self.fps)
        minutes = seconds // 60
        hours = minutes // 60
        return f"{hours:02d}:{minutes % 60:02d}:{seconds % 60:02d}"

    def draw_controls(self, img):
        """컨트롤 패널 그리기"""
        height, width = img.shape[:2]

        # 컨트롤 패널 배경
        cv2.rectangle(img, (0, height - self.CONTROL_HEIGHT), (width, height), (40, 40, 40), -1)

        # 시간 정보
        current_time = self.format_time(self.current_frame)
        total_time = self.format_time(self.total_frames)
        time_text = f"{current_time} / {total_time}"
        cv2.putText(img, time_text, (10, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 재생/일시정지 버튼
        button_x = 10
        button_y = height - 35
        button_size = 25

        if self.is_playing:
            # 일시정지 아이콘 (두 개의 세로선)
            cv2.rectangle(img, (button_x + 5, button_y), (button_x + 10, button_y + button_size), (255, 255, 255), -1)
            cv2.rectangle(img, (button_x + 15, button_y), (button_x + 20, button_y + button_size), (255, 255, 255), -1)
        else:
            # 재생 아이콘 (삼각형)
            triangle = np.array([[button_x + 5, button_y],
                                 [button_x + 5, button_y + button_size],
                                 [button_x + 20, button_y + button_size // 2]], np.int32)
            cv2.fillPoly(img, [triangle], (255, 255, 255))

        # 시크바
        seekbar_x = 150
        seekbar_y = height - 25
        seekbar_width = width - 200
        seekbar_height = 10

        # 시크바 배경
        cv2.rectangle(img, (seekbar_x, seekbar_y), (seekbar_x + seekbar_width, seekbar_y + seekbar_height),
                      (100, 100, 100), -1)

        # 진행 바
        if self.total_frames > 0:
            progress = self.current_frame / self.total_frames
            progress_width = int(seekbar_width * progress)
            cv2.rectangle(img, (seekbar_x, seekbar_y), (seekbar_x + progress_width, seekbar_y + seekbar_height),
                          (0, 120, 255), -1)

            # 시크 핸들
            handle_x = seekbar_x + progress_width
            cv2.circle(img, (handle_x, seekbar_y + seekbar_height // 2), 8, (255, 255, 255), -1)
            cv2.circle(img, (handle_x, seekbar_y + seekbar_height // 2), 6, (0, 120, 255), -1)

        return seekbar_x, seekbar_y, seekbar_width, seekbar_height

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리"""
        cap, seekbar_info = param
        seekbar_x, seekbar_y, seekbar_width, seekbar_height = seekbar_info

        if event == cv2.EVENT_LBUTTONDOWN:
            # 재생/일시정지 버튼 클릭
            img_height = self.WINDOW_HEIGHT + self.CONTROL_HEIGHT
            if 10 <= x <= 35 and img_height - 35 <= y <= img_height - 10:
                self.is_playing = not self.is_playing
            # 시크바 클릭
            elif seekbar_x <= x <= seekbar_x + seekbar_width and seekbar_y <= y <= seekbar_y + seekbar_height:
                self.seeking = True
                progress = (x - seekbar_x) / seekbar_width
                self.seek_pos = int(progress * self.total_frames)

        elif event == cv2.EVENT_MOUSEMOVE and self.seeking:
            # 시크바 드래그
            if seekbar_x <= x <= seekbar_x + seekbar_width:
                progress = (x - seekbar_x) / seekbar_width
                self.seek_pos = int(progress * self.total_frames)

        elif event == cv2.EVENT_LBUTTONUP and self.seeking:
            # 시크 완료
            self.seeking = False
            self.current_frame = self.seek_pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def handle_keyboard(self, key, cap):
        """키보드 입력 처리"""
        if key == ord(' '):  # 스페이스바로 재생/일시정지
            self.is_playing = not self.is_playing
        elif key == ord('a') or key == 81:  # 'a' 또는 왼쪽 화살표 - 5초 뒤로
            self.current_frame = max(0, self.current_frame - int(5 * self.fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        elif key == ord('d') or key == 83:  # 'd' 또는 오른쪽 화살표 - 5초 앞으로
            self.current_frame = min(self.total_frames - 1, self.current_frame + int(5 * self.fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        elif key == ord('s') or key == 82:  # 's' 또는 위쪽 화살표 - 1프레임 앞으로
            if not self.is_playing:
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        elif key == ord('w') or key == 84:  # 'w' 또는 아래쪽 화살표 - 1프레임 뒤로
            if not self.is_playing:
                self.current_frame = max(0, self.current_frame - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def prepare_frame(self, annotated_frame):
        """프레임 크기 조정 및 컨트롤 패널 추가"""
        # annotated_frame resize
        resized_frame = cv2.resize(annotated_frame, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        # 컨트롤 패널을 포함한 최종 이미지 생성
        final_frame = np.zeros((self.WINDOW_HEIGHT + self.CONTROL_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        final_frame[:self.WINDOW_HEIGHT, :] = resized_frame

        # 컨트롤 패널 그리기
        seekbar_info = self.draw_controls(final_frame)

        return final_frame, seekbar_info

    def update_current_frame(self, cap):
        """현재 프레임 번호 업데이트"""
        self.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    def print_controls_help(self):
        """컨트롤 사용법 출력"""
        print("\n=== 플레이어 컨트롤 사용법 ===")
        print("마우스:")
        print("- 재생/일시정지 버튼 클릭")
        print("- 시크바 클릭/드래그로 구간 이동")
        print("\n키보드:")
        print("- 스페이스바: 재생/일시정지")
        print("- A/왼쪽 화살표: 5초 뒤로")
        print("- D/오른쪽 화살표: 5초 앞으로")
        print("- W/위쪽 화살표: 1프레임 뒤로 (일시정지 상태에서)")
        print("- S/아래쪽 화살표: 1프레임 앞으로 (일시정지 상태에서)")
        print("- ESC: 종료")