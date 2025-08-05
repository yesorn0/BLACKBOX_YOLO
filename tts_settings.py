# tts_settings.py
# TTS 보행 도우미 시스템 핵심 클래스

# TTSNavigationSystem 클래스의 모든 기능
# 파일 관리, TTS 생성, 오디오 재생
# 우선순위 큐 처리
# 탐지 결과 안내 시스템


import os
import threading
import time
import logging
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor

from tts_config import *

class TTSNavigationSystem:
    def __init__(self, audio_dir=None, text_dir=None, txt_filename=None):
        self.AUDIO_DIR = audio_dir or DEFAULT_AUDIO_DIR
        self.TEXT_DIR = text_dir or DEFAULT_TEXT_DIR
        self.TXT_NAME = txt_filename or DEFAULT_TXT_FILENAME
        self.txt_path = os.path.join(self.TEXT_DIR, self.TXT_NAME)
        
        # 디렉토리 생성
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.TEXT_DIR, exist_ok=True)
        
        # 텍스트 파일 초기화
        self._initialize_text_file()
        
        # 설정 불러오기
        self.DETECTION2TEXT = DETECTION2TEXT
        self.message_cooldown = MESSAGE_COOLDOWN
        
        # 최근 재생된 메시지 추적 (중복 방지)
        self.recent_messages = {}
        
        # 우선순위 큐
        self.priority_queue = []
        self.queue_lock = threading.Lock()
        
        # 스레드풀
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # 파일 접근 락
        self.file_lock = threading.Lock()
        
        logging.info("TTS Navigation System 초기화 완료")

    def _initialize_text_file(self):
        """텍스트 파일 초기화 (없으면 생성)"""
        try:
            # 디렉토리 경로 출력
            print(f"텍스트 디렉토리: {os.path.abspath(self.TEXT_DIR)}")
            print(f"텍스트 파일 경로: {os.path.abspath(self.txt_path)}")
            
            if not os.path.exists(self.txt_path):
                with open(self.txt_path, "w", encoding=FILE_ENCODING) as f:
                    f.write("")  # 빈 파일 생성
                print(f"✓ 텍스트 파일 생성됨: {self.txt_path}")
                logging.info(f"텍스트 파일 생성: {self.txt_path}")
            else:
                print(f"✓ 텍스트 파일 이미 존재: {self.txt_path}")
                
        except Exception as e:
            print(f"✗ 텍스트 파일 초기화 오류: {e}")
            logging.error(f"텍스트 파일 초기화 오류: {e}")

    def _read_text_lines(self):
        """텍스트 파일에서 모든 줄 읽기"""
        with self.file_lock:
            try:
                with open(self.txt_path, "r", encoding=FILE_ENCODING) as f:
                    lines = [line.strip() for line in f if line.strip()]
                return lines
            except Exception as e:
                logging.error(f"파일 읽기 오류: {e}")
                return []

    def _add_text_to_file(self, text):
        """텍스트 파일에 새 문구 추가"""
        with self.file_lock:
            try:
                # 파일이 없으면 다시 생성 시도
                if not os.path.exists(self.txt_path):
                    print(f"파일이 없어서 다시 생성: {self.txt_path}")
                    self._initialize_text_file()
                
                with open(self.txt_path, "a", encoding=FILE_ENCODING) as f:
                    f.write(text + "\n")
                print(f"✓ 새 문구 추가됨: {text}")
                logging.info(f"새 문구 추가: {text}")
                
                # 파일 내용 확인
                if os.path.exists(self.txt_path):
                    with open(self.txt_path, "r", encoding=FILE_ENCODING) as f:
                        lines = f.readlines()
                    print(f"현재 파일 총 줄 수: {len(lines)}")
                    
            except Exception as e:
                print(f"✗ 파일 쓰기 오류: {e}")
                logging.error(f"파일 쓰기 오류: {e}")

    def _find_or_add_text(self, situation_text):
        """문구를 찾거나 추가하고 줄 번호 반환"""
        lines = self._read_text_lines()
        
        # 기존 문구가 있는지 확인
        try:
            line_number = lines.index(situation_text) + 1  # 1-based index
            logging.info(f"기존 문구 발견: {line_number}번째 줄")
            return line_number
        except ValueError:
            # 없으면 추가
            self._add_text_to_file(situation_text)
            new_line_number = len(lines) + 1
            logging.info(f"새 문구 추가됨: {new_line_number}번째 줄")
            return new_line_number

    def _generate_mp3_filename(self, keyword_or_line_number):
        """MP3 파일명 생성: 키워드 또는 줄 번호 기반"""
        if isinstance(keyword_or_line_number, int):
            # 줄 번호인 경우
            return f"line_{keyword_or_line_number}.mp3"
        else:
            # 키워드인 경우
            safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in str(keyword_or_line_number))
            return f"{safe_keyword}.mp3"

    def _generate_audio_file(self, text, mp3_path):
        """오디오 파일 생성"""
        try:
            print(f"[DEBUG] TTS 생성 시도: {text} -> {mp3_path}")
            
            if not os.path.exists(mp3_path):
                tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)
                tts.save(mp3_path)
                print(f"✓ MP3 파일 생성 완료: {mp3_path}")
                logging.info(f"MP3 파일 생성: {mp3_path}")
                return True
            else:
                print(f"✓ MP3 파일 이미 존재: {mp3_path}")
                logging.info(f"MP3 파일 이미 존재: {mp3_path}")
                return True
        except Exception as e:
            print(f"✗ TTS 생성 실패: {e}")
            logging.error(f"TTS 생성 오류: {e}")
            return False

    def _play_audio(self, mp3_path):
        """오디오 재생"""
        try:
            print(f"[DEBUG] 오디오 재생 시도: {mp3_path}")
            
            # 파일 존재 확인
            if not os.path.exists(mp3_path):
                print(f"✗ 파일이 존재하지 않음: {mp3_path}")
                return False
            
            # 여러 플레이어 시도
            for player in AUDIO_PLAYERS:
                # 플레이어가 설치되어 있는지 확인
                if os.system(f"which {player} > /dev/null 2>&1") == 0:
                    print(f"[DEBUG] {player} 사용하여 재생 시도")
                    result = os.system(f"{player} \"{mp3_path}\" > /dev/null 2>&1")
                    if result == 0:
                        print(f"✓ {player}로 재생 완료: {os.path.basename(mp3_path)}")
                        logging.info(f"{player}로 재생 완료: {os.path.basename(mp3_path)}")
                        return True
                    else:
                        print(f"✗ {player}로 재생 실패 (return code: {result})")
                        logging.warning(f"{player}로 재생 실패")
            
            print("✗ 사용 가능한 오디오 플레이어를 찾을 수 없습니다.")
            print(f"  다음 플레이어 중 하나를 설치해주세요: {', '.join(AUDIO_PLAYERS)}")
            logging.warning("사용 가능한 오디오 플레이어를 찾을 수 없습니다.")
            return False
            
        except Exception as e:
            print(f"✗ 오디오 재생 오류: {e}")
            logging.error(f"오디오 재생 오류: {e}")
            return False

    def is_recently_played(self, text):
        """최근에 재생된 메시지인지 확인"""
        current_time = time.time()
        
        if text in self.recent_messages:
            if current_time - self.recent_messages[text] < self.message_cooldown:
                return True
        
        self.recent_messages[text] = current_time
        return False

    def play_situation_from_txt(self, situation_text, keyword=None):
        """상황 문구를 텍스트 파일 기반으로 재생"""
        try:
            print(f"\n[DEBUG] 처리할 문구: {situation_text}")
            
            # 1. 텍스트 파일에서 줄 번호 찾기 또는 추가
            line_number = self._find_or_add_text(situation_text)
            print(f"[DEBUG] 줄 번호: {line_number}")
            
            # 2. MP3 파일명 생성 (키워드가 있으면 키워드 사용, 없으면 줄번호 사용)
            if keyword:
                mp3_filename = self._generate_mp3_filename(keyword)
            else:
                mp3_filename = self._generate_mp3_filename(line_number)
            
            mp3_path = os.path.join(self.AUDIO_DIR, mp3_filename)
            print(f"[DEBUG] MP3 파일: {mp3_filename}")
            print(f"[DEBUG] MP3 경로: {mp3_path}")
            
            # 3. MP3 파일 생성 (없으면)
            if self._generate_audio_file(situation_text, mp3_path):
                # 4. 재생
                return self._play_audio(mp3_path)
            else:
                return False
                
        except Exception as e:
            print(f"✗ 음성 재생 처리 오류: {e}")
            logging.error(f"음성 재생 처리 오류: {e}")
            return False

    def add_to_priority_queue(self, text, priority, keyword=None):
        """우선순위 큐에 메시지 추가"""
        with self.queue_lock:
            # 중복 제거
            self.priority_queue = [
                (p, t, k) for p, t, k in self.priority_queue 
                if not (p == priority and t == text)
            ]
            
            self.priority_queue.append((priority, text, keyword))
            # 우선순위로 정렬 (낮은 숫자 = 높은 우선순위)
            self.priority_queue.sort(key=lambda x: x[0])

    def process_priority_queue(self):
        """우선순위 큐 처리"""
        while True:
            with self.queue_lock:
                if self.priority_queue:
                    priority, text, keyword = self.priority_queue.pop(0)
                else:
                    time.sleep(QUEUE_SLEEP_INTERVAL)
                    continue
            
            if not self.is_recently_played(text):
                self.play_situation_from_txt(text, keyword)
            
            time.sleep(MESSAGE_INTERVAL)  # 메시지 간 간격

    def announce_detection(self, detected_key, force_play=False):
        """탐지 결과를 음성으로 안내"""
        try:
            if detected_key in self.DETECTION2TEXT:
                info = self.DETECTION2TEXT[detected_key]
                text = info["text"]
                priority = info["priority"]
                keyword = detected_key  # 키워드는 탐지된 키 그대로 사용
            else:
                # 매핑되지 않은 키워드는 그대로 사용
                text = f"{detected_key}가 감지되었습니다."
                priority = 5
                keyword = detected_key
            
            if force_play:
                # 즉시 재생
                self.play_situation_from_txt(text, keyword)
            else:
                # 우선순위 큐에 추가
                self.add_to_priority_queue(text, priority, keyword)
            
            logging.info(f"안내 메시지: {text} (키워드: {keyword}, 우선순위: {priority})")
            
        except Exception as e:
            logging.error(f"탐지 안내 오류: {e}")

    def start_queue_processor(self):
        """우선순위 큐 처리 스레드 시작"""
        queue_thread = threading.Thread(target=self.process_priority_queue, daemon=True)
        queue_thread.start()
        logging.info("우선순위 큐 처리 스레드 시작")

    def emergency_announce(self, text, keyword="emergency"):
        """긴급 상황 즉시 안내"""
        self.play_situation_from_txt("긴급상황입니다!", "emergency")
        time.sleep(0.5)
        self.play_situation_from_txt(text, keyword)

    def show_text_file_status(self):
        """텍스트 파일 상태 출력"""
        lines = self._read_text_lines()
        print(f"\n=== {self.TXT_NAME} 파일 상태 ===")
        if lines:
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}. {line}")
            
            # MP3 파일들 상태 확인
            print(f"\n=== audio 디렉토리 MP3 파일들 ===")
            if os.path.exists(self.AUDIO_DIR):
                mp3_files = [f for f in os.listdir(self.AUDIO_DIR) if f.endswith('.mp3')]
                if mp3_files:
                    for mp3_file in sorted(mp3_files):
                        mp3_path = os.path.join(self.AUDIO_DIR, mp3_file)
                        file_size = os.path.getsize(mp3_path)
                        print(f"✓ {mp3_file} ({file_size} bytes)")
                else:
                    print("MP3 파일이 없습니다.")
            else:
                print("audio 디렉토리가 없습니다.")
        else:
            print("파일이 비어있습니다.")
        print("=" * 40)

    def test_tts_basic(self):
        """기본 TTS 기능 테스트"""
        pass
        
    #     try:
            # TTS 객체 생성 테스트
            # print("1. TTS 객체 생성 테스트...")
            # tts = gTTS(text=TEST_MESSAGE, lang=TTS_LANGUAGE, slow=TTS_SLOW)
            # print("✓ TTS 객체 생성 성공")
            
            # 파일 저장 테스트
            # test_mp3_path = os.path.join(self.AUDIO_DIR, TEST_MP3_FILENAME)
            # print(f"2. MP3 파일 저장 테스트: {test_mp3_path}")
            # tts.save(test_mp3_path)
            
            # if os.path.exists(test_mp3_path):
            #     file_size = os.path.getsize(test_mp3_path)
            #     print(f"✓ MP3 파일 저장 성공 ({file_size} bytes)")
                
            #     # 재생 테스트
            #     print("3. 재생 테스트...")
            #     self._play_audio(test_mp3_path)
        #     else:
        #         print("✗ MP3 파일 저장 실패")
                
        # except Exception as e:
        #     print(f"✗ 기본 TTS 테스트 실패: {e}")

    def shutdown(self):
        """시스템 종료"""
        self.executor.shutdown(wait=True)
        logging.info("TTS Navigation System 종료")