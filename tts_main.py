# tts_main.py
# TTS 보행 도우미 시스템 메인 실행 파일

import os
import time
import logging
from tts_config import LOGGING_LEVEL, LOGGING_FORMAT
from tts_settings import TTSNavigationSystem

# 로깅 설정
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL), format=LOGGING_FORMAT)

def main():
    """메인 실행 함수"""
    print("=== TTS Navigation System 시작 ===")
    
    # 현재 작업 디렉토리 확인
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # 시스템 초기화
    tts_system = TTSNavigationSystem()
    
    # 디렉토리 생성 확인
    print(f"오디오 디렉토리 존재: {os.path.exists(tts_system.AUDIO_DIR)}")
    print(f"텍스트 디렉토리 존재: {os.path.exists(tts_system.TEXT_DIR)}")
    print(f"텍스트 파일 존재: {os.path.exists(tts_system.txt_path)}")
    
    # 기본 TTS 기능 테스트
    tts_system.test_tts_basic()
    
    # 우선순위 큐 처리 시작
    tts_system.start_queue_processor()
    
    # 초기 상태 출력
    tts_system.show_text_file_status()

    time.sleep(2)
    tts_system.show_text_file_status()
    
    # 탐지 시뮬레이션
    detection_sequence = [
        "car",
        "green", 
        "red",
        "left_turn",
        "right_turn"

    ]
    
    print("\n=== 탐지 시뮬레이션 ===")
    for i, detection in enumerate(detection_sequence):
        print(f"\n[{i+1}] 탐지: {detection}")
        tts_system.announce_detection(detection, force_play=True)  # 즉시 재생으로 변경
        time.sleep(4)  # 4초 간격으로 대기
        tts_system.show_text_file_status()
    
    print("\n=== 프로그램 완료 ===")
    tts_system.shutdown()


if __name__ == "__main__":
    main()