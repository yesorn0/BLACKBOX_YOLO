import os
import json
import glob

def polygon_to_bbox(seg):
    xs = seg[::2]
    ys = seg[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

def convert_dir(json_dir, out_label_dir, img_dir):
    os.makedirs(out_label_dir, exist_ok=True)
    for json_path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_info = data["images"][0]
        img_w, img_h = img_info["width"], img_info["height"]
        img_name = img_info["file_name"]
        categories = {c["id"]: i for i, c in enumerate(data["categories"])}
        yolo_labels = []
        for anno in data["annotations"]:
            if not anno["segmentation"]:
                continue
            bbox = polygon_to_bbox(anno["segmentation"])
            x, y, w, h = bbox
            xc = (x + w/2) / img_w
            yc = (y + h/2) / img_h
            w /= img_w
            h /= img_h
            class_id = categories[anno["category_id"]]
            yolo_labels.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        out_path = os.path.join(out_label_dir, txt_name)
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("\n".join(yolo_labels))
    print(f"{json_dir} → {out_label_dir} 변환 완료")

# 예시 사용법
convert_dir("./jsons/train", "./labels/train", "./images/train")
convert_dir("./jsons/val", "./labels/val", "./images/val")
