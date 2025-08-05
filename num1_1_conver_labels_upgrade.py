import os
import json
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def polygon_to_bbox_vectorized(seg: List[float]) -> List[float]:
    """폴리곤을 bbox로 변환 (NumPy 사용)"""
    if not seg or len(seg) < 4:
        return [0, 0, 0, 0]
    coords = np.array(seg)
    xs = coords[::2]
    ys = coords[1::2]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def process_single_json(args: Tuple[str, str, Dict[int, int], int]) -> Tuple[str, List[str], int]:
    """단일 JSON 파일 처리 (bbox/segmentation 둘 중 하나만 있어도 변환)"""
    json_path, out_label_dir, categories_map, total_files = args

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("images") or not data["images"]:
            logger.warning(f"이미지 정보가 없는 파일: {json_path}")
            return json_path, [], 0
        img_info = data["images"][0]
        img_w, img_h = img_info["width"], img_info["height"]
        img_name = img_info["file_name"]

        yolo_labels = []
        processed_count = 0

        for anno in data.get("annotations", []):
            # 카테고리 인덱스 변환 (YOLO는 0부터)
            category_id = anno.get("category_id")
            if category_id not in categories_map:
                continue
            class_id = categories_map[category_id]

            # 1. bbox 우선 사용
            bbox = anno.get("bbox", [])
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
            else:
                # 2. bbox 없으면 segmentation→bbox 변환
                seg = anno.get("segmentation", [])
                if not seg or (isinstance(seg, list) and len(seg) == 0):
                    continue
                # 리스트의 리스트일 경우 첫번째 polygon만
                if isinstance(seg[0], list):
                    seg = seg[0]
                bbox = polygon_to_bbox_vectorized(seg)
                x, y, w, h = bbox

            # 유효성 체크
            if w <= 0 or h <= 0:
                continue

            # YOLO 좌표계 변환 (중심/정규화)
            xc = (x + w * 0.5) / img_w
            yc = (y + h * 0.5) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # 0~1 범위 벗어나면 skip
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < norm_w <= 1 and 0 < norm_h <= 1):
                continue

            yolo_labels.append(f"{class_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")
            processed_count += 1

        # YOLO 라벨 저장 (빈 라벨도 txt 생성)
        txt_name = Path(img_name).stem + ".txt"
        out_path = os.path.join(out_label_dir, txt_name)
        with open(out_path, "w", encoding="utf-8") as out:
            if yolo_labels:
                out.write("\n".join(yolo_labels))

        return json_path, yolo_labels, processed_count

    except Exception as e:
        logger.error(f"파일 처리 중 오류 발생 {json_path}: {str(e)}")
        return json_path, [], 0

def build_categories_map(json_files: List[str]) -> Dict[int, int]:
    """카테고리 매핑 (category_id - 1)"""
    all_categories = {}
    for json_path in json_files[:5]:  # 처음 몇 개 파일만 확인
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cat in data.get("categories", []):
                cat_id = cat["id"]
                if cat_id not in all_categories:
                    all_categories[cat_id] = cat_id - 1
        except Exception as e:
            logger.warning(f"카테고리 정보 읽기 실패 {json_path}: {str(e)}")
            continue
    logger.info(f"총 {len(all_categories)}개 카테고리 발견")
    return all_categories

def convert_dir_optimized(json_dir: str, out_label_dir: str, img_dir: Optional[str] = None,
                          max_workers: Optional[int] = None) -> Dict[str, int]:
    """최적화된 디렉토리 변환 함수"""
    os.makedirs(out_label_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        logger.warning(f"JSON 파일을 찾을 수 없습니다: {json_dir}")
        return {"processed_files": 0, "total_annotations": 0, "failed_files": 0}

    logger.info(f"처리할 JSON 파일 수: {len(json_files)}")
    categories_map = build_categories_map(json_files)
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(json_files))
    logger.info(f"워커 수: {max_workers}")

    tasks = [(json_path, out_label_dir, categories_map, len(json_files))
             for json_path in json_files]
    stats = {
        "processed_files": 0,
        "total_annotations": 0,
        "failed_files": 0,
        "empty_files": 0
    }

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_json, task) for task in tasks]
        for i, future in enumerate(as_completed(futures)):
            try:
                json_path, yolo_labels, processed_count = future.result()
                stats["processed_files"] += 1
                stats["total_annotations"] += processed_count
                if processed_count == 0:
                    stats["empty_files"] += 1
                if (i + 1) % 100 == 0 or (i + 1) == len(futures):
                    logger.info(f"진행률: {i + 1}/{len(futures)} ({(i + 1) / len(futures) * 100:.1f}%)")
            except Exception as e:
                stats["failed_files"] += 1
                logger.error(f"작업 실행 중 오류: {str(e)}")

    logger.info(f"""
변환 완료!
- 입력 디렉토리: {json_dir}
- 출력 디렉토리: {out_label_dir}
- 처리된 파일: {stats['processed_files']}
- 총 어노테이션: {stats['total_annotations']}
- 빈 파일: {stats['empty_files']}
- 실패한 파일: {stats['failed_files']}
- 카테고리 수: {len(categories_map)}
    """)
    return stats

def convert_with_validation(json_dir: str, out_label_dir: str, img_dir: Optional[str] = None):
    """검증 기능이 포함된 변환 함수"""
    stats = convert_dir_optimized(json_dir, out_label_dir, img_dir)
    json_count = len(glob.glob(os.path.join(json_dir, "*.json")))
    txt_count = len(glob.glob(os.path.join(out_label_dir, "*.txt")))
    if json_count != txt_count:
        logger.warning(f"파일 수 불일치: JSON {json_count}개, TXT {txt_count}개")
    else:
        logger.info("✅ 모든 JSON 파일이 성공적으로 변환되었습니다!")
    return stats

# 사용 예시
if __name__ == "__main__":
    # 예시
    train_stats = convert_with_validation("./jsons/train", "./labels/train", "./images/train")
    val_stats = convert_with_validation("./jsons/val", "./labels/val", "./images/val")
    # 필요시 커스텀 워커 수 사용 가능
    # convert_dir_optimized("./jsons/train", "./labels/train", max_workers=4)
