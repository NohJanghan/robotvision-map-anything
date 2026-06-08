# input/<scene> Task 1-3 재현 튜토리얼

이 문서는 `input/bike`와 같은 디렉터리 구조를 가진 `input/<scene>` 캡처를
COLMAP/MapAnything Task 1-3 산출물로 변환하는 절차를 정리한다. 실행 결과 수치와
해석은 이 문서에 적지 않고 `result.md`에만 기록한다.

colmap 실행을 위해서 [colmap docker tutorial](colmap_docker_tutorial.md)을 참고한다.

## 전제

- 작업 디렉터리: `/home/acsl/projects/robotvision-map-anything`
- conda 환경: `rkv-mapanything`
- 입력 구조:
  - `input/<scene>/rgb.mp4`
  - `input/<scene>/odometry.csv`
  - `input/<scene>/camera_matrix.csv`
  - `input/<scene>/depth/`
  - `input/<scene>/confidence/`
- `<scene>`은 `bike`, `lounge`처럼 run 이름으로도 사용한다.
- Codex sandbox 안에서는 GPU가 보이지 않을 수 있다. MapAnything과 Docker COLMAP GPU
  실행은 권한 상승으로 실행한다.

## 핵심 원칙

- Task 1은 RGB sequence만으로 COLMAP intrinsics/poses를 추정한다.
  `--camera-params`로 외부 calibration을 주입하지 않는다.
- COLMAP dense MVS는 기본 실행에서 제외한다. 과제 필수 산출물은 sparse SfM export,
  sparse point cloud, camera trajectory, registered image count, mean reprojection
  error이다.
- 가능한 많은 데이터를 쓰되, VRAM 사용량은 MapAnything의 `--max-images`와
  `--view-stride`로 제한한다. 시간은 오래 걸려도 되므로 먼저 더 많은 frame으로
  COLMAP 입력을 만들고, MapAnything에서 필요한 경우 view 수만 줄인다.
- 결과 수치, 표, 해석은 `result.md`에 기록한다.

## 1. 환경 확인

GPU와 MapAnything CUDA 인식을 확인한다. Codex에서 실행할 때는 두 명령 모두 권한
상승으로 실행해야 다음 단계에서도 GPU가 잡힌다.

```bash
nvidia-smi
```

```bash
conda run -n rkv-mapanything python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

COLMAP은 Docker wrapper를 기본으로 사용한다.

```bash
scripts/colmap/colmap_docker.py -h
```

정상이라면 `COLMAP ... with CUDA`가 보인다. Docker를 사용할 수 없고 CPU-only 로컬
COLMAP을 꼭 써야 할 때만 Task 1 명령에 `--colmap colmap --use-gpu 0`을 명시한다.

## 2. 입력 규모 확인

새 장면을 처리하기 전에 원본 프레임 수, FPS, 해상도, odometry row 수를 확인한다.

```bash
conda run -n rkv-mapanything python -c "import cv2, pathlib; scene='<scene>'; cap=cv2.VideoCapture(f'input/{scene}/rgb.mp4'); print('opened', cap.isOpened()); print('frames', int(cap.get(cv2.CAP_PROP_FRAME_COUNT))); print('fps', cap.get(cv2.CAP_PROP_FPS)); print('width', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))); print('height', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))); p=pathlib.Path(f'input/{scene}/odometry.csv'); print('odometry_rows', sum(1 for _ in p.open())-1 if p.exists() else None)"
```

프레임 선택 기준:

- `--max-frames`: RTX 4090 24GB 환경에서는 256 view를 high-data 기준으로 사용한다.
  MapAnything inference에는 그중 128 view를 넣고, 나머지 128 view는 holdout 평가에
  사용한다.
- `--stride`: 원본 전체 구간을 대략 덮도록 정한다.
  예: `stride ~= floor((source_frames - 1) / (max_frames - 1))`.
- `--max-side 1600`: COLMAP과 MapAnything 모두에서 재사용한 안정 해상도이다.

예시:

- `bike`: source_frames 1176이면 256 view 기준 `stride ~= floor((1176 - 1) / (256 - 1)) = 4`
- `lounge`: source_frames 1412이면 256 view 기준 `stride ~= floor((1412 - 1) / (256 - 1)) = 5`

## 3. 입력 전처리

`prepare_capture_inputs.py`를 사용한다. `prepare_bike_inputs.py`는 과거 호환용
entrypoint이며 같은 구현을 사용한다.

```bash
conda run -n rkv-mapanything python scripts/mapanything/prepare_capture_inputs.py \
  --input-dir input/<scene> \
  --stride <stride> \
  --max-frames 256 \
  --max-side 1600 \
  --overwrite
```

필요하면 출력 경로를 명시한다.

```bash
conda run -n rkv-mapanything python scripts/mapanything/prepare_capture_inputs.py \
  --input-dir input/<scene> \
  --image-output-dir data/raw/rgb_sequences/<scene> \
  --ar-output-file data/raw/ar_captures/<scene>/poses.json \
  --stride <stride> \
  --max-frames 256 \
  --max-side 1600 \
  --overwrite
```

생성물:

- `data/raw/rgb_sequences/<scene>/*.jpg`
- `data/raw/rgb_sequences/<scene>/manifest.json`
- `data/raw/ar_captures/<scene>/poses.json`

주의:

- 이미지 파일명은 원본 frame id를 유지한다. 예: `000000.jpg`, `000009.jpg`
- `odometry.csv`의 `frame` 값과 이미지 파일명이 맞아야 한다.
- AR/VIO pose JSON의 기본 convention은 `arkit_cam2world`이다.
- intrinsics는 resize scale에 맞춰 함께 축소된다.

## 4. Task 1: COLMAP

먼저 sequential matcher로 실행한다.

```bash
conda run -n rkv-mapanything python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/<scene> \
  --run-name <scene> \
  --overwrite \
  --use-gpu 1 \
  --use-xvfb \
  --camera-model PINHOLE \
  --max-image-size 1600 \
  --max-num-features 8192 \
  --matcher sequential \
  --sequential-overlap 15
```

설정 이유:

- `--camera-model PINHOLE`: MapAnything Config B/C가 3x3 pinhole intrinsics를 바로
  사용할 수 있게 한다.
- `--max-num-features 8192`: 256-frame 실행에서도 안정적인 track 연결성을 얻기 위해
  충분한 feature를 사용한다.
- `--matcher sequential`: 비디오에서 추출한 순차 프레임에 적합한 기본값이다.
- `--sequential-overlap 15`: 넉넉한 인접 프레임 matching으로 track 연결성을 높인다.
- `--use-xvfb`: headless 환경의 OpenGL context 문제를 피한다.

등록 이미지 수가 너무 낮으면 frame 수를 바꾸기 전에 같은 입력으로 exhaustive
matching을 시도한다.

```bash
conda run -n rkv-mapanything python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/<scene> \
  --run-name <scene> \
  --overwrite \
  --use-gpu 1 \
  --use-xvfb \
  --camera-model PINHOLE \
  --max-image-size 1600 \
  --max-num-features 8192 \
  --matcher exhaustive
```

주의:

- COLMAP 4 Docker는 `FeatureExtraction.*`, `FeatureMatching.*`,
  `Mapper.ba_global_frames_ratio` 옵션명을 사용한다. `run_pipeline.py`는 COLMAP 3/4
  옵션명을 자동 감지한다.
- COLMAP이 여러 sparse submodel을 만들 수 있다. `run_pipeline.py`는 등록 이미지 수와
  point 수가 가장 큰 model을 export한다.

Task 1 산출물:

- `data/processed/colmap_exports/<scene>/cameras.txt`
- `data/processed/colmap_exports/<scene>/images.txt`
- `data/processed/colmap_exports/<scene>/points3D.txt`
- `outputs/colmap/visualizations/<scene>/points3D.ply`
- `outputs/colmap/visualizations/<scene>/camera_trajectory.png`
- `outputs/colmap/logs/<scene>/metrics.json`
- `outputs/colmap/logs/<scene>/run_summary.json`

## 5. Task 2/3: MapAnything

COLMAP export와 AR/VIO pose를 모두 넘겨 네 config를 실행한다. 기본 설정은 256개
입력 중 짝수 index 128개를 MapAnything inference에 사용하고, 홀수 index 128개를
holdout 평가 target으로 사용한다.

```bash
conda run -n rkv-mapanything python scripts/mapanything/run_pipline.py \
  --config configs/mapanything/task2_pipeline.json \
  --run-name <scene> \
  --image-dir data/raw/rgb_sequences/<scene> \
  --colmap-export-dir data/processed/colmap_exports/<scene> \
  --ar-pose-file data/raw/ar_captures/<scene>/poses.json \
  --view-start 0 \
  --view-stride 2 \
  --max-images 128 \
  --eval-holdout \
  --eval-holdout-start 1 \
  --eval-holdout-stride 2 \
  --eval-holdout-max-images 128 \
  --require-gpu \
  --overwrite \
  --continue-on-error
```

동작 방식:

- Config A: image-only
- Config B: image + COLMAP intrinsics
- Config C: image + COLMAP intrinsics + COLMAP poses
- Config D: image + intrinsics + AR/VIO poses
- COLMAP 등록 이미지가 있으면 `registered_or_all` 선택 규칙에 따라 등록된 view를
  기준으로 네 config가 같은 train/holdout view split을 사용한다.
- holdout 평가는 train prediction을 COLMAP reference pose 기준으로 similarity-align한
  뒤, 모든 train view의 predicted depth에서 얻은 valid points를 하나의 global point
  cloud로 합쳐 inference에 쓰지 않은 target view pose로 point-splat rendering한다.
  기본 `point_stride`는 2이며, summary 표의 `Eval` 열이 `holdout`이면 holdout metric을
  사용한 것이다.
- `configs/mapanything/task2_pipeline.json`은 `memory_efficient_inference: true`,
  `minibatch_size: 1`, `use_amp: true`, `amp_dtype: bf16`로 설정되어 있어 실행 시간이
  늘어나는 대신 VRAM 사용량을 낮춘다.

VRAM이 부족하면 전처리/COLMAP 결과는 그대로 두고 MapAnything view 수만 줄인다.
128 train view가 OOM이면 다음 순서로 train/holdout view 수를 같이 줄여 재실행한다.

```bash
conda run -n rkv-mapanything python scripts/mapanything/run_pipline.py \
  --config configs/mapanything/task2_pipeline.json \
  --run-name <scene> \
  --image-dir data/raw/rgb_sequences/<scene> \
  --colmap-export-dir data/processed/colmap_exports/<scene> \
  --ar-pose-file data/raw/ar_captures/<scene>/poses.json \
  --view-start 0 \
  --view-stride 2 \
  --max-images 96 \
  --eval-holdout \
  --eval-holdout-start 1 \
  --eval-holdout-stride 2 \
  --eval-holdout-max-images 96 \
  --require-gpu \
  --overwrite \
  --continue-on-error
```

그래도 OOM이면 `--max-images 64 --eval-holdout-max-images 64`를 사용한다. 장면 전체
구간을 더 넓게 유지하고 싶을 때는 `--view-stride`와 `--eval-holdout-stride`를 함께
키운다.

Task 2/3 산출물:

- `outputs/mapanything/<scene>/config_a_uncalibrated_image_only/`
- `outputs/mapanything/<scene>/config_b_calibrated_image_only/`
- `outputs/mapanything/<scene>/config_c_calibrated_colmap_poses/`
- `outputs/mapanything/<scene>/config_d_calibrated_ar_poses/`
- `outputs/mapanything/<scene>_task2_summary.json`
- `outputs/mapanything/<scene>_task2_summary.md`

각 config 디렉터리에는 `predicted_poses.json`, `reconstruction_points.ply`,
`reconstruction.glb`, `predictions/`, `depth/`, `renders/`, `renders_holdout/`가
생성된다.

## 6. 검증

문법 검사:

```bash
conda run -n rkv-mapanything python -m py_compile \
  scripts/colmap/run_pipeline.py \
  scripts/colmap/colmap_docker.py \
  scripts/mapanything/run_pipline.py \
  scripts/mapanything/prepare_capture_inputs.py \
  scripts/mapanything/prepare_bike_inputs.py
```

diff whitespace 검사:

```bash
git diff --check
```

COLMAP 지표 확인:

```bash
jq '{registered_images, cameras, points, observations, mean_track_length, mean_observations_per_image, mean_reprojection_error_px}' \
  outputs/colmap/logs/<scene>/metrics.json
```

MapAnything 요약 확인:

```bash
sed -n '1,220p' outputs/mapanything/<scene>_task2_summary.md
```

산출물 일관성 확인:

```bash
jq '{image_count, paths, metrics}' outputs/colmap/logs/<scene>/run_summary.json
jq '.results[] | {config, status, view_count, holdout_view_count, runtime_seconds, summary_metrics}' \
  outputs/mapanything/<scene>_task2_summary.json
```

## 7. 결과 기록

튜토리얼에는 실행 절차만 유지한다. 다음 항목은 `result.md`에 기록한다.

- scene 이름과 입력 frame 설정
- COLMAP registered image count, sparse point count, reprojection error
- COLMAP intrinsics line
- MapAnything A/B/C/D metric table
- A vs B, B vs C, C vs D 비교
- 관찰된 실패 사례와 해석
