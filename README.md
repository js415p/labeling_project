# Labeling Project — Obstacle Segmentation & Depth Map Projector

> LabelMe 기반 장애물 라벨링 → 이진 마스크 / YOLOv8-seg 데이터셋 생성 → **좌표 변환 노드(`depth_map_projector.py`)**로 Depth+Mask를 3D PointCloud로 변환하는 ROS 2 파이프라인

## 목차
- [개요](#개요)
- [폴더 구조](#폴더-구조)
- [데이터셋](#데이터셋)
- [스크립트 설명](#스크립트-설명)
- [핵심: 좌표 변환 노드 `depth_map_projector.py`](#핵심-좌표-변환-노드-depth_map_projectorpy)
- [테스트용 노드](#테스트용-노드)
- [YOLOv8-seg 학습](#yolov8-seg-학습)
- [실행 예시](#실행-예시)
- [파라미터 레퍼런스](#파라미터-레퍼런스)
- [RViz 확인](#rviz-확인)
- [Troubleshooting](#troubleshooting)

---

## 개요

이 프로젝트는 주행 로봇의 **바닥(floor) vs 장애물(obstacle)** 을 구분하기 위한 전체 워크플로우를 담는다.

```
[ LabelMe 라벨링 ] → [ json_to_mask.py / json_to_yolo_seg.py ] → [ YOLOv8-seg 학습 ]
                                                            ↓
                    [ 카메라 RGB + Aligned Depth ] → [ YOLO 추론 → obstacle mask (mono8) ]
                                                            ↓
                                          ┌─────────────────┴─────────────────┐
                                          │  depth_map_projector.py (좌표 변환 노드)  │
                                          │  mask + depth + CameraInfo → 3D 복원 → │
                                          │  tf2 변환(target_frame=map) → PointCloud2 │
                                          └─────────────────┬─────────────────┘
                                                            ↓
                                                  /vision_obstacles (PointCloud2)
                                                  → Nav2 / Costmap / RViz
```

핵심은 `depth_map_projector.py` 이며, **2D Segmentation 결과를 3D 공간으로 역투영 + TF 변환**하는 역할을 한다. 실제 주행에서는 이 PointCloud를 Costmap에 넣어 회피에 사용한다.

---

## 폴더 구조

```
labeling_project/
├── images/                     # 원본 이미지 395장 (box_*.jpg, cable_*.jpg)
├── labels/                     # LabelMe JSON 112개 (장애물 없는 프레임은 JSON 없음)
├── labels.txt                  # LabelMe 라벨 정의: floor, obstacle
├── masks/                      # json_to_mask.py가 생성한 이진 마스크 (obstacle=255, floor=0)
├── yolo_dataset/               # json_to_yolo_seg.py가 생성한 YOLOv8-seg 학습 데이터셋
│   ├── images/train (336장)
│   ├── images/val   (59장)
│   ├── labels/train (97개 txt, obstacle 있는 이미지만)
│   ├── labels/val   (15개 txt)
│   └── data.yaml
├── depth_map_projector.py      # ★ 좌표 변환 노드 (메인)
├── fake_mask_depth_publisher.py# 테스트용 합성 depth/mask/camera_info Publisher
├── test_static_tf_publisher.py # 테스트용 Static TF Broadcaster
├── debug_pointcloud_echo.py    # PointCloud2 좌표 디버그용 Echo 노드
├── json_to_mask.py             # LabelMe JSON → 이진 마스크 변환
├── json_to_yolo_seg.py         # LabelMe JSON → YOLOv8-seg 데이터셋 변환
├── dataset.zip / yolo_dataset.zip / labels.zip # 배포용 압축본
└── README.md
```

> `labelenv/`는 Python venv이며 `.gitignore`로 추적 제외됨.

---

## 데이터셋

| 항목 | 수량/설명 |
|------|-----------|
| **원본 이미지** | `images/` 395장 |
| **LabelMe JSON** | `labels/` 112개 — `obstacle` 폴리곤/사각형만 저장, `floor`는 배경(라벨 없음)으로 처리 |
| **마스크** | `masks/` 395개 PNG — `json_to_mask.py` 생성, 1채널 `255=obstacle, 0=floor` |
| **YOLO 데이터셋** | `yolo_dataset/` — 85:15 랜덤 분할(`seed=42`), 장애물 없는 이미지는 `label txt 없음` = YOLO negative sample |

`labels.txt`:
```
__ignore__
floor obstacle
```

> 라벨링 팁: 장애물이 없는 프레임은 JSON을 만들지 않는다. 그러면 `json_to_mask.py`가 전체를 `0(바닥)`으로, `json_to_yolo_seg.py`가 `배경 이미지`로 처리한다.

---

## 스크립트 설명

### 1. `json_to_mask.py` — LabelMe → 이진 마스크
```bash
python json_to_mask.py
```
- `images/`와 `labels/`를 매칭해 `masks/<이름>.png` 생성
- `polygon` → `cv2.fillPoly`, `rectangle` → `cv2.rectangle` 로 255 채움
- JSON이 없거나 shapes가 비어있으면 전체 0 (floor)
- 통계 출력: `total_images`, `with_json`, `empty_floor`, `polygons_drawn`

### 2. `json_to_yolo_seg.py` — LabelMe → YOLOv8-seg
```bash
python json_to_yolo_seg.py
```
- `yolo_dataset/images/train|val`, `labels/train|val`, `data.yaml` 생성
- `rectangle`은 4개 꼭짓점으로 전개, 모든 좌표는 `x/w, y/h` 정규화 후 0~1 클리핑
- 포맷: `0 x1 y1 x2 y2 ...` (class 0 = obstacle, segmentation polygon)
- 배경 이미지(장애물 없음)는 `images/`에만 복사하고 `labels/*.txt`는 생성 안 함
- `data.yaml`은 절대경로 `path:` 포함 — Colab에서 그대로 `yolo train data=.../data.yaml` 가능

---

## 핵심: 좌표 변환 노드 `depth_map_projector.py`

> **좌표 변환 노드** — 2D 마스크 픽셀을 3D 카메라 좌표로 복원하고 `tf2`로 목표 프레임(`map`)까지 변환해 `PointCloud2`로 퍼블리시

### 입력 / 출력

| 구분 | 토픽 (파라미터로 변경 가능) | 타입 | 설명 |
|------|------------------------------|------|------|
| 입력 | `mask_topic` (`/obstacle_mask`) | `sensor_msgs/Image` `mono8` | YOLO 추론 결과 등, 255=장애물 |
| 입력 | `depth_topic` (`/camera/aligned_depth_to_color/image_raw`) | `sensor_msgs/Image` `16UC1` or `32FC1` | **반드시 mask와 aligned** (같은 픽셀=같은 3D 점) |
| 입력 | `camera_info_topic` | `sensor_msgs/CameraInfo` | `K[0]=fx, K[4]=fy, K[2]=cx, K[5]=cy` |
| 출력 | `output_topic` (`/vision_obstacles`) | `sensor_msgs/PointCloud2` | `target_frame` 기준 3D 점군 |
| TF | `target_frame` (`map`) | `tf2` | `depth_msg.header.frame_id → target_frame` 변환 필요 |

### 동작 원리

1. **CameraInfo 캐싱**: 첫 `CameraInfo` 수신 시 `fx, fy, cx, cy` 저장
2. **시간 동기화**: `message_filters.ApproximateTimeSynchronizer`로 mask/depth 동기화 (`slop=0.05s`)
3. **마스크 필터링**: `mask >= mask_threshold(127)` 인 픽셀만 추출 → `point_stride(4)` 간격으로 다운샘플
4. **Depth → 미터 변환**: `depth * depth_scale(0.001)` — RealSense는 mm→m
5. **핀홀 역투영**:
   ```
   z = depth[v,u]
   x = (u - cx) * z / fx
   y = (v - cy) * z / fy
   ```
   → `camera_color_optical_frame` 기준 `(x,y,z)` (optical: x=right, y=down, z=forward)
6. **유효성 검사**: `min_depth(0.1) < z < max_depth(5.0)` && `isfinite(z)` 만 유지
7. **PointCloud2 생성** (`x,y,z` FLOAT32, `point_step=12`)
8. **TF 변환**: `tf_buffer.lookup_transform(target_frame, frame_id, stamp)` → `do_transform_cloud()`로 변환 후 퍼블리시

> 실전에서는 URDF + `robot_state_publisher` + SLAM/AMCL이 `map → odom → base_link → camera_link → camera_color_optical_frame` 트리를 제공한다.

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `mask_topic` | `/obstacle_mask` | 입력 마스크 토픽 |
| `depth_topic` | `/camera/aligned_depth_to_color/image_raw` | 입력 depth 토픽 (aligned 필수) |
| `camera_info_topic` | `/camera/aligned_depth_to_color/camera_info` | CameraInfo 토픽 |
| `output_topic` | `/vision_obstacles` | 출력 PointCloud2 토픽 |
| `target_frame` | `map` | 변환 목표 프레임 (RViz Fixed Frame과 일치시킬 것) |
| `depth_scale` | `0.001` | depth 원시값 → 미터 스케일 (RealSense 16UC1은 0.001, 32FC1(m)은 1.0) |
| `min_depth` | `0.1` | 최소 유효 거리 [m] |
| `max_depth` | `5.0` | 최대 유효 거리 [m] |
| `point_stride` | `4` | N개당 1개 포인트만 사용 (성능/대역폭 절약) |
| `mask_threshold` | `127` | 이 값 이상을 장애물로 간주 |

---

## 테스트용 노드

실제 카메라/모델 없이 좌표 변환 로직을 검증하기 위한 스텁 3종.

### `test_static_tf_publisher.py`
```
map(identity) → odom(identity) → base_link → camera_link(앞 0.1m, 위 0.3m, pitch +15°) → camera_color_optical_frame(광학 변환)
```
- `StaticTransformBroadcaster`로 고정 TF 발행
- 실제 로봇에서는 `robot_state_publisher` + SLAM이 대체

### `fake_mask_depth_publisher.py`
- 640×480 합성 프레임 생성, 하단 중앙 직사각형(`0.4w:0.6w, 0.75h:0.95h`)을 장애물로 설정
- `depth`는 해당 영역 0.5m(mm=500), 배경 4.0m, `mask`는 255
- `CameraInfo`는 `fx=fy=525, cx=320, cy=240`
- **검증 포인트**: `test_static_tf_publisher.py` 조건에서 `obstacle_depth_m=0.5`일 때 `map` 기준 예상 출력은 대략 `x≈0.54m, y≈0m, z≈0.02m (바닥 근처)` — 이 근처로 나오면 변환 정상

### `debug_pointcloud_echo.py`
- `/vision_obstacles` 구독 → 콘솔에 `frame_id`, 총 포인트 수, 앞 5개 점의 `x,y,z` 출력
- RViz는 시각적 확인용, 숫자는 이 노드로 확인

---

## YOLOv8-seg 학습

```bash
# 로컬 또는 Colab
pip install ultralytics
yolo segment train data=yolo_dataset/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
```

- `yolo_dataset.zip`을 Colab에 업로드 후 압축 해제하면 바로 학습 가능
- 추론 결과 마스크를 `mono8`로 변환해 `depth_map_projector.py`의 `mask_topic`으로 퍼블리시하면 파이프라인 연결

---

## 실행 예시

### 1. 마스크 / YOLO 데이터셋 생성
```bash
python json_to_mask.py        # masks/ 생성
python json_to_yolo_seg.py    # yolo_dataset/ 생성
```

### 2. 좌표 변환 노드 단독 실행 (실제 로봇)
```bash
ros2 run labeling_project depth_map_projector --ros-args \
  -p mask_topic:=/obstacle_mask \
  -p depth_topic:=/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/aligned_depth_to_color/camera_info \
  -p target_frame:=map \
  -p depth_scale:=0.001 \
  -p point_stride:=4
```

### 3. 카메라 없이 전체 파이프라인 테스트
```bash
# 터미널 1: 가짜 TF
ros2 run labeling_project test_static_tf_publisher

# 터미널 2: 가짜 depth/mask
ros2 run labeling_project fake_mask_depth_publisher --ros-args \
  -p depth_topic:=/test/depth \
  -p mask_topic:=/test/mask \
  -p camera_info_topic:=/test/camera_info

# 터미널 3: 좌표 변환 노드 (테스트 토픽으로 오버라이드)
ros2 run labeling_project depth_map_projector --ros-args \
  -p mask_topic:=/test/mask \
  -p depth_topic:=/test/depth \
  -p camera_info_topic:=/test/camera_info \
  -p target_frame:=map \
  -p output_topic:=/vision_obstacles

# 터미널 4: 디버그 출력
ros2 run labeling_project debug_pointcloud_echo --ros-args -p topic:=/vision_obstacles

# 터미널 5: RViz
rviz2  # Fixed Frame=map, Add PointCloud2 /vision_obstacles
```

---

## 파라미터 레퍼런스

| 노드 | 주요 파라미터 |
|------|---------------|
| `depth_map_projector` | `mask_topic`, `depth_topic`, `camera_info_topic`, `output_topic`, `target_frame`, `depth_scale`, `min/max_depth`, `point_stride`, `mask_threshold` |
| `fake_mask_depth_publisher` | `depth_topic`, `mask_topic`, `camera_info_topic`, `frame_id`, `width/height`, `fx/fy`, `obstacle_depth_m`, `publish_rate_hz` |
| `debug_pointcloud_echo` | `topic` |
| `test_static_tf_publisher` | (고정값, 코드 내 수정) |

---

## RViz 확인

1. `Fixed Frame`을 `target_frame`과 동일하게 설정 (기본 `map`)
2. `Add → PointCloud2`, Topic을 `output_topic` (`/vision_obstacles`)로 설정
3. 로버를 이동시키며 장애물 위치에 점군이 바닥(z≈0)에 맞게 찍히는지 확인
4. 정확한 수치는 `debug_pointcloud_echo.py` 콘솔 로그로 대조

> ⚠️ `mask`와 `depth`는 **반드시 정렬(aligned)** 되어야 한다. RealSense라면 `aligned_depth_to_color` 토픽을, 다른 카메라는 `depth-to-rgb` 정렬 노드를 사용한다. 정렬이 안 되면 같은 픽셀이 다른 3D 점을 가리켜 점군이 엉뚱한 곳에 찍힌다.

---

## Troubleshooting

| 증상 | 원인 / 해결 |
|------|-------------|
| `아직 camera_info를 못 받음` 경고 | `camera_info_topic` 리매핑 확인, 카메라 드라이버가 `CameraInfo` 퍼블리시 중인지 `ros2 topic echo`로 확인 |
| `TF lookup 실패` | `target_frame`이 TF 트리에 존재하는지 `ros2 run tf2_ros tf2_echo map camera_color_optical_frame`로 확인. `test_static_tf_publisher` 또는 `robot_state_publisher` 실행 여부 확인 |
| 점군이 허공에 뜸 / 바닥에 안 붙음 | `depth_scale` 확인 (16UC1 mm → 0.001, 32FC1 m → 1.0), 카메라 설치 높이/각도 TF 오차, depth 정렬 여부 확인 |
| 포인트 수 0 | `mask_threshold`가 너무 높거나, depth가 `min/max_depth` 밖, `point_stride` 과도, 마스크가 실제로 비어있는지 `ros2 topic echo /obstacle_mask` 확인 |
| RViz에 점이 안 보임 | Fixed Frame mismatch, PointCloud2 decay time, `output_topic` 이름 불일치 확인 |

---

## 라이선스 & 기여

- LabelMe 라벨은 `labels/`에, 학습용 데이터는 `yolo_dataset/`에 포함
- 이슈/개선은 PR 환영 — 특히 `depth_map_projector.py`의 stride/필터 파라미터 튜닝, YOLO 추론 노드 연동 예시 추가 등

