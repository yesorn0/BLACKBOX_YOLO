from ultralytics import YOLO
import matplotlib.pyplot as plt
import platform

def run_val():
    # 폰트 세팅
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

    # 모델 로드
    model = YOLO('runs/train/optimized_exp/weights/best.pt')
    # 결과 생성
    results = model.val(data='data.yaml', project='runs/val', name='font_test', save=True)

if __name__ == "__main__":
    run_val()
