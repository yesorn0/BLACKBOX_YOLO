# tts_config.py
# TTS 보행 도우미 시스템 설정 파일
# 모든 설정값과 상수들을 관리
# 탐지 키워드와 메시지 매핑
# 오디오 플레이어 목록
# TTS 및 로깅 설정 

import os

# 기본 디렉토리 설정
DEFAULT_AUDIO_DIR = "audio"
DEFAULT_TEXT_DIR = "text"
DEFAULT_TXT_FILENAME = "important.txt"

# 메시지 우선순위 및 쿨다운 설정
MESSAGE_COOLDOWN = 3  # 초
MAX_WORKERS = 2

# 객체 탐지 결과 → 안내 문구 매핑 (우선순위 포함)
DETECTION2TEXT = {
    # 긴급상황 (최우선)
    "collision": {"text": "위험! 충돌 위험!", "priority": 1},
    "emergency": {"text": "긴급상황입니다!", "priority": 1},
    
    # 신호등 상태 (높은 우선순위)
    "red": {"text": "빨간불입니다. 정지하세요.", "priority": 2},
    "green": {"text": "초록불입니다. 지나가세요 .", "priority": 2},
    "yellow": {"text": "노란불입니다. 주의하세요.", "priority": 2},
    "changed": {"text": "신호가 바뀌었습니다.", "priority": 2},
    
    # 보행자/장애물 (중간 우선순위)
    "person": {"text": "앞에 사람이 있습니다.", "priority": 3},
    "obstacle": {"text": "장애물이 감지되었습니다.", "priority": 3},
    "bicycle": {"text": "자전거가 접근하고 있습니다.", "priority": 3},
    "car": {"text": "차량이 접근하고 있습니다.", "priority": 3},
    
    # 방향 안내 (낮은 우선순위)
    "left_turn": {"text": "좌회전 하세요.", "priority": 4},
    "right_turn": {"text": "우회전 하세요.", "priority": 4},
    "straight": {"text": "직진하세요.", "priority": 4},
    "crosswalk": {"text": "전방에 횡단보도입니다.", "priority": 4},
}

# 오디오 플레이어 목록 (우선순위 순)
AUDIO_PLAYERS = ['mpg321', 'mpg123', 'play', 'afplay', 'vlc']

# TTS 설정
TTS_LANGUAGE = 'ko'
TTS_SLOW = False

# 로깅 설정
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# 파일 인코딩
FILE_ENCODING = "utf-8"

# # 테스트 메시지
# TEST_MESSAGE = "안녕하세요. 테스트 메시지입니다."
# TEST_MP3_FILENAME = "test_basic.mp3"

# 큐 처리 간격
QUEUE_SLEEP_INTERVAL = 0.1  # 초
MESSAGE_INTERVAL = 0.5  # 메시지 간 간격(초)