"""
Labelme JSON -> obstacle/floor 이진 마스크 변환 스크립트

사용법:
    python json_to_mask.py

폴더 구조 가정 (스크립트와 같은 위치, 혹은 아래 경로 직접 수정):
    labeling_project/
        images/   <- 원본 jpg 395장
        labels/   <- labelme가 저장한 json (라벨링 안 한 이미지는 json 없음)
        masks/    <- 이 스크립트가 생성할 mask png (자동 생성됨)

결과:
    masks/box_0001.png 등, 원본과 같은 해상도의 1채널 png
    obstacle 영역 = 255, floor(배경) = 0
"""

import json
import os
import glob
import numpy as np
import cv2

# ---- 경로 설정 (본인 환경에 맞게 수정) ----
IMAGES_DIR = "images"
LABELS_DIR = "labels"
MASKS_DIR = "masks"
OBSTACLE_LABEL_NAME = "obstacle"  # labels.txt에 적은 이름과 동일해야 함

os.makedirs(MASKS_DIR, exist_ok=True)

image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
if not image_files:
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))

stats = {"total_images": len(image_files), "with_json": 0, "empty_floor": 0, "polygons_drawn": 0, "skipped_other_label": 0}

for img_path in image_files:
    fname = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(LABELS_DIR, fname + ".json")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[경고] 이미지를 못 읽음: {img_path}")
        continue
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # 기본 floor(0)

    if os.path.exists(json_path):
        stats["with_json"] += 1
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if len(shapes) == 0:
            stats["empty_floor"] += 1

        for shape in shapes:
            label = shape.get("label", "")
            if label != OBSTACLE_LABEL_NAME:
                stats["skipped_other_label"] += 1
                continue

            points = np.array(shape["points"], dtype=np.float64)
            shape_type = shape.get("shape_type", "polygon")

            if shape_type == "polygon":
                pts = points.round().astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
                stats["polygons_drawn"] += 1
            elif shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
                stats["polygons_drawn"] += 1
            else:
                # circle, line 등 다른 타입 쓰신 경우 여기 추가 처리 필요
                print(f"[알림] 처리 안 된 shape_type '{shape_type}' in {fname}")
    else:
        # JSON 자체가 없음 = 장애물 없는 프레임 = 전체 floor
        stats["empty_floor"] += 1

    out_path = os.path.join(MASKS_DIR, fname + ".png")
    cv2.imwrite(out_path, mask)

print("\n=== 변환 완료 ===")
for k, v in stats.items():
    print(f"{k}: {v}")
print(f"\n마스크 저장 위치: {os.path.abspath(MASKS_DIR)}")