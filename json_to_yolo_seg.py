"""
Labelme JSON -> YOLOv8-seg 학습용 데이터셋 변환 스크립트

사용법:
    python json_to_yolo_seg.py

폴더 구조 가정 (스크립트와 같은 위치):
    labeling_project/
        images/   <- 원본 jpg 395장
        labels/   <- labelme json (obstacle 없는 이미지는 json 없음)

결과물 (YOLO 학습 바로 가능한 구조):
    yolo_dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt   (obstacle 폴리곤, 정규화 좌표)
        labels/val/*.txt
        data.yaml

주의:
- obstacle이 없는 이미지도 images/train(or val)에 포함되고, 대응 label txt는 만들지 않습니다.
  (YOLO는 label 파일이 없으면 그 이미지를 배경/negative sample로 취급 -> 의도한 동작)
- class는 obstacle 하나만 사용 (id=0). floor는 배경이라 별도 라벨 없음.
"""

import json
import os
import glob
import random
import shutil

# ---- 경로 설정 ----
IMAGES_DIR = "images"
LABELS_DIR = "labels"          # labelme json 폴더
OUT_DIR = "yolo_dataset"
OBSTACLE_LABEL_NAME = "obstacle"
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
if not image_files:
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))

random.shuffle(image_files)
n_val = int(len(image_files) * VAL_RATIO)
val_set = set(image_files[:n_val])

stats = {"total": len(image_files), "train": 0, "val": 0, "with_obstacle": 0, "background_only": 0, "skipped_other_label": 0}

def get_image_size(json_data, img_path):
    h = json_data.get("imageHeight")
    w = json_data.get("imageWidth")
    if h and w:
        return h, w
    import cv2
    img = cv2.imread(img_path)
    return img.shape[0], img.shape[1]

for img_path in image_files:
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    json_path = os.path.join(LABELS_DIR, stem + ".json")

    split = "val" if img_path in val_set else "train"
    stats[split] += 1

    # 이미지 복사
    shutil.copy(img_path, os.path.join(OUT_DIR, "images", split, fname))

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        h, w = get_image_size(data, img_path)
        shapes = data.get("shapes", [])

        lines = []
        for shape in shapes:
            label = shape.get("label", "")
            if label != OBSTACLE_LABEL_NAME:
                stats["skipped_other_label"] += 1
                continue

            points = shape["points"]  # [[x1,y1],[x2,y2],...]
            shape_type = shape.get("shape_type", "polygon")

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]  # 사각형 4점으로 변환

            if len(points) < 3:
                continue  # 폴리곤이 되려면 점 3개 이상 필요

            norm_coords = []
            for (x, y) in points:
                norm_coords.append(x / w)
                norm_coords.append(y / h)
            norm_coords = [max(0.0, min(1.0, c)) for c in norm_coords]  # 0~1 클리핑

            coord_str = " ".join(f"{c:.6f}" for c in norm_coords)
            lines.append(f"0 {coord_str}")  # class_id=0 (obstacle)

        if lines:
            stats["with_obstacle"] += 1
            with open(os.path.join(OUT_DIR, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
        else:
            stats["background_only"] += 1
        # lines가 비었으면(폴리곤 없었으면) label txt 안 만듦 -> 배경 이미지 취급
    else:
        stats["background_only"] += 1
        # json 자체가 없으면 = obstacle 없는 프레임 = label txt 없음 (배경 이미지)

# data.yaml 생성
yaml_content = f"""path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val

names:
  0: obstacle
"""
with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("\n=== 변환 완료 ===")
for k, v in stats.items():
    print(f"{k}: {v}")
print(f"\n결과 폴더: {os.path.abspath(OUT_DIR)}")
print("이 폴더(yolo_dataset) 전체를 zip으로 묶어서 Colab에 업로드하세요.")
