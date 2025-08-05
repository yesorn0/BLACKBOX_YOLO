import sys
import os
import gc
import torch
import time
from pathlib import Path
from importlib.util import find_spec


def check_dependencies():
    """Check if required packages are installed and importable."""
    pkgs = {
        "numpy": "numpy",
        "torch": "torch",
        "ultralytics": "ultralytics",
        "cv2": "opencv-python",
        "yaml": "pyyaml"
    }
    missing = []
    for name, pip_name in pkgs.items():
        if find_spec(name) is None:
            missing.append(pip_name)
            print(f"\033[91m✗ {name} 미설치\033[0m")
        else:
            print(f"\033[92m✓ {name} 설치됨\033[0m")
    if missing:
        print(f"\n아래 명령으로 부족한 패키지 설치:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True


def optimize_system_for_6gb():
    """RTX 3060 6GB 전용 시스템 최적화"""
    print("RTX 3060 6GB 환경에 맞춘 최적화 적용...")

    # GPU 메모리 적극적 관리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

        # 6GB GPU를 위한 메모리 할당 최적화
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'

        # cuDNN 최적화 (메모리 효율성 우선)
        torch.backends.cudnn.benchmark = False  # 메모리 절약
        torch.backends.cudnn.deterministic = True

        # 메모리 프래그멘테이션 방지
        torch.cuda.set_per_process_memory_fraction(0.85)  # 6GB의 85%만 사용

    # 멀티프로세싱 최적화 (Windows PyCharm 환경)
    if sys.platform.startswith('win'):
        os.environ['OMP_NUM_THREADS'] = '2'  # 6GB GPU에 맞춘 조정
        torch.set_num_threads(2)

    # Python 가비지 컬렉션 최적화
    gc.set_threshold(700, 10, 10)  # 더 자주 메모리 정리


def print_system_info():
    """시스템 정보 출력"""
    import torch, numpy
    print(f"\n=== 시스템 정보 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {numpy.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU: {gpu_name}")
        print(f"GPU 총 메모리: {gpu_memory:.1f}GB")

        # 현재 GPU 사용량
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f"GPU 할당됨: {allocated:.2f}GB / 예약됨: {reserved:.2f}GB")

        # 6GB GPU 경고
        if gpu_memory <= 6.5:
            print(f"⚠️  6GB GPU 감지 - 메모리 최적화 모드 활성화")
    else:
        print("❌ CUDA 미사용 - CPU 모드 (매우 느림)")


def get_rtx3060_6gb_params():
    """RTX 3060 6GB 전용 최적화 파라미터"""
    if not torch.cuda.is_available():
        return {
            'epochs': 50,
            'imgsz': 320,  # CPU는 매우 작은 이미지
            'batch': 2,
            'workers': 1,
            'amp': False,
            'cache': False
        }

    # GPU 메모리 확인
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    # RTX 3060 6GB 전용 설정
    if gpu_memory <= 6.5:  # 6GB GPU
        params = {
            # 기본 훈련 설정
            'epochs': 50,  # 더 많은 에포크로 성능 보완
            'imgsz': 384,  # 6GB에 최적화된 이미지 크기
            'batch': 8,  # 6GB에 안전한 배치 크기
            'workers': 6,  # PyCharm 가상환경에 적합
            'amp': True,  # 혼합 정밀도로 메모리 절약
            'cache': False,  # 메모리 절약을 위해 캐시 비활성화
            'device': 0,
            'patience': 30,  # 더 긴 patience
            'save_period': 3,  # 3 에포크마다 저장

            # 옵티마이저 설정 (6GB 최적화)
            'optimizer': 'AdamW',
            'lr0': 0.0008,  # 약간 낮은 학습률
            'lrf': 0.05,  # 더 강한 스케줄링
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # 손실 함수 가중치 (6GB 환경 최적화)
            'box': 7.5,
            'cls': 0.3,  # 분류 손실 약간 감소
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,  # 객체성 손실 감소

            # 데이터 증강 (메모리 효율적)
            'hsv_h': 0.01,  # 색상 증강 감소
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'degrees': 0.0,
            'translate': 0.05,  # 이동 증강 감소
            'scale': 0.8,  # 스케일 증강 감소
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.8,  # 모자이크 확률 감소
            'mixup': 0.0,  # mixup 비활성화 (메모리 절약)
            'copy_paste': 0.0  # copy_paste 비활성화
        }
    else:
        # 더 큰 GPU용 설정
        params = {
            'epochs': 50,
            'imgsz': 384,
            'batch': 8,
            'workers': 8,
            'amp': True,
            'cache': 'ram',
            'device': 0,
            'patience': 20,
            'save_period': 3
        }

    return params


def validate_dataset(data_path):
    """데이터셋 유효성 검사"""
    if not data_path.exists():
        print(f"\033[91m❌ {data_path} 파일을 찾을 수 없습니다!\033[0m")
        return False

    try:
        import yaml
        with open(data_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data_config:
                print(f"\033[91m❌ data.yaml에 '{key}' 키가 없습니다!\033[0m")
                return False

        print(f"\033[92m✓ 데이터셋 설정 확인 완료\033[0m")
        print(f"  - 클래스 수: {data_config['nc']}")
        print(f"  - 클래스명: {data_config['names']}")

        # 이미지 경로 확인
        train_path = Path(data_config['train'])
        val_path = Path(data_config['val'])

        if not train_path.exists():
            print(f"\033[93m⚠️  훈련 데이터 경로를 찾을 수 없습니다: {train_path}\033[0m")
        if not val_path.exists():
            print(f"\033[93m⚠️  검증 데이터 경로를 찾을 수 없습니다: {val_path}\033[0m")

        return True

    except Exception as e:
        print(f"\033[91m❌ data.yaml 읽기 실패: {e}\033[0m")
        return False


def cleanup_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def monitor_memory_usage():
    """메모리 사용량 모니터링"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU 메모리: {allocated:.2f}GB 사용 / {reserved:.2f}GB 예약 / {total:.1f}GB 총량")

        # 메모리 사용률이 높으면 경고
        if reserved / total > 0.9:
            print("⚠️  GPU 메모리 사용률이 높습니다. 배치 크기를 줄이는 것을 권장합니다.")


def cleanup_runs_folder():
    """이전 실행 결과 정리"""
    runs_path = Path('runs')
    if runs_path.exists():
        import shutil
        existing_runs = list(runs_path.glob('train/*'))
        if len(existing_runs) > 3:
            print(f"이전 학습 결과가 {len(existing_runs)}개 있습니다.")
            print("정리하시겠습니까? (y/N): ", end='')
            try:
                if input().lower() == 'y':
                    backup_name = f'runs_backup_{int(time.time())}'
                    shutil.move(str(runs_path), backup_name)
                    runs_path.mkdir()
                    print(f"백업 완료: {backup_name}")
            except Exception as e:
                print(f"폴더 정리 실패: {e}")


def main():
    print("=" * 60)
    print("🚀 RTX 3060 6GB 최적화 YOLO 학습 스크립트 v3.0")
    print("💡 PyCharm 가상환경 + 6GB GPU 전용 최적화")
    print("=" * 60)

    # 1. 패키지 확인
    print("\n1️⃣ 필수 패키지 확인...")
    if not check_dependencies():
        print("\n패키지 설치 후 다시 실행해주세요.")
        print("PyCharm Terminal에서: pip install ultralytics opencv-python pyyaml")
        sys.exit(1)

    # 2. 시스템 최적화
    print("\n2️⃣ RTX 3060 6GB 최적화 적용...")
    optimize_system_for_6gb()
    print_system_info()

    # 초기 메모리 상태 확인
    monitor_memory_usage()

    # 3. 데이터셋 확인
    print("\n3️⃣ 데이터셋 확인...")
    data_path = Path('data.yaml')
    if not validate_dataset(data_path):
        print("\n데이터셋 설정을 확인한 후 다시 실행해주세요.")
        return

    # 4. 이전 결과 정리
    print("\n4️⃣ 이전 학습 결과 확인...")
    cleanup_runs_folder()

    try:
        from ultralytics import YOLO

        print("\n5️⃣ YOLO 모델 로딩...")
        # RTX 3060 6GB에 최적화된 모델 선택
        print("RTX 3060 6GB에 최적화된 YOLOv8s 모델 사용")
        model = YOLO('yolov8m.pt')  # Small 모델이 6GB에 최적

        print("\n6️⃣ RTX 3060 6GB 최적화 파라미터 설정...")
        params = get_rtx3060_6gb_params()

        print(f"\n📊 RTX 3060 6GB 최적화 학습 설정:")
        for key, value in params.items():
            print(f"  - {key}: {value}")

        # 예상 시간 계산 (더 정확한 추정)
        estimated_time_per_epoch = 8566 / params['batch'] * 0.8  # 6GB GPU 보정
        total_estimated_time = estimated_time_per_epoch * params['epochs'] / 60
        print(f"\n⏱️  예상 학습 시간: {total_estimated_time:.0f}분 ({total_estimated_time / 60:.1f}시간)")
        print("📌 6GB GPU 환경에서는 시간이 더 걸릴 수 있습니다.")

        # 메모리 정리 후 시작
        cleanup_gpu_memory()

        print("\n🏁 YOLO 학습 시작!")
        print("💡 메모리 부족 시 자동으로 배치 크기가 조정됩니다.")
        print("-" * 50)

        start_time = time.time()

        # 학습 실행 (에러 핸들링 강화)
        try:
            results = model.train(
                data='data.yaml',
                project='runs/train',
                name='optimized_exp',
                exist_ok=True,
                resume=False,
                verbose=True,
                **params
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n❌ GPU 메모리 부족! 자동으로 배치 크기를 줄입니다...")
                params['batch'] = max(4, params['batch'] // 2)
                params['workers'] = max(2, params['workers'] // 2)
                print(f"새로운 배치 크기: {params['batch']}")

                cleanup_gpu_memory()

                # 재시도
                results = model.train(
                    data='data.yaml',
                    exist_ok=True,
                    resume='runs/train/optimized_exp',
                    verbose=True,
                    **params
                )
            else:
                raise e

        end_time = time.time()
        training_time = end_time - start_time

        print("\n" + "=" * 60)
        print("🎉 RTX 3060 6GB 최적화 학습 완료!")
        print(f"⏰ 총 학습 시간: {training_time / 3600:.1f}시간")
        print(f"📁 결과 저장 위치: {results.save_dir}")
        print(f"🏆 최고 모델: {results.save_dir}/weights/best.pt")
        print(f"📊 마지막 모델: {results.save_dir}/weights/last.pt")
        print("\n💡 PyCharm에서 결과 확인:")
        print(f"   - TensorBoard: tensorboard --logdir {results.save_dir}")
        print(f"   - 결과 이미지: {results.save_dir}/")
        print("=" * 60)

        # 최종 메모리 상태
        monitor_memory_usage()

    except KeyboardInterrupt:
        print("\n\n❌ 사용자에 의해 학습이 중단되었습니다.")
        print("💡 resume=True 옵션으로 이어서 학습할 수 있습니다.")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ RTX 3060 6GB 메모리 부족!")
            print("🔧 해결 방법:")
            print("1. 배치 크기 더 줄이기: batch=8 또는 batch=4")
            print("2. 이미지 크기 줄이기: imgsz=320 또는 imgsz=384")
            print("3. 워커 수 줄이기: workers=2")
            print("4. 혼합 정밀도 확인: amp=True")
            print("5. PyCharm에서 다른 프로그램 종료")
        else:
            print(f"\n❌ RuntimeError: {e}")

    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        print("🔍 PyCharm 콘솔에서 전체 에러 로그를 확인해주세요.")

    finally:
        # 항상 GPU 메모리 정리
        cleanup_gpu_memory()
        print("\n🧹 GPU 메모리 정리 완료")


if __name__ == "__main__":
    main()