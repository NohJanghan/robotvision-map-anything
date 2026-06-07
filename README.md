# Robot Kinematics and Vision Term Project

## 프로젝트 목표

- RGB 이미지 시퀀스에 대해 COLMAP을 실행하여 카메라 내부 파라미터와 포즈를 추정한다.
- MapAnything을 네 가지 입력 설정으로 실행하여 입력 정보의 차이가 3D 복원 결과에 미치는 영향을 분석한다.
- 모바일 기기에서 ARKit 또는 ARCore를 사용하여 VIO pose를 얻고 이를 MapAnything의 입력으로 사용한다.

## Environment Settings

```bash
conda env create -f environment.yml
conda activate rkv-mapanything
```

Map-Anything 설치를 위해서
```bash
conda activate rkv-mapanything
git clone https://github.com/facebookresearch/map-anything.git ./third_party/map-anything
cd third_party/map-anything
pip install -e .
```

COLMAP은 conda 패키지 대신 공식 GPU 지원 Docker 이미지를 사용한다. 호스트에는
NVIDIA driver, Docker 19.03+, NVIDIA Container Toolkit이 필요하다.

```bash
# 공식 colmap/colmap:latest 이미지 또는 로컬 colmap:latest 이미지로 COLMAP 확인
scripts/colmap/colmap_docker.py -h
```

로컬에서 공식 Dockerfile로 직접 빌드하려면 다음을 실행한다.

```bash
scripts/colmap/build_colmap_docker.sh
```

`build_colmap_docker.sh`는 `https://github.com/colmap/colmap`을
`third_party/colmap/`에 체크아웃하고, 공식 `docker/Dockerfile`로
`colmap:latest` 이미지를 만든다. 이후 `scripts/colmap/colmap_docker.py`는 로컬
`colmap:latest`를 우선 사용하고, 없으면 공식 `colmap/colmap:latest` 이미지를
사용한다.

## 디렉토리 구조

```text
.
├── configs/                  # COLMAP / MapAnything 실행 설정
│   ├── colmap/
│   └── mapanything/
├── data/                     # Git으로 관리하지 않는 로컬 데이터
│   ├── raw/
│   │   ├── rgb_sequences/    # 원본 RGB 이미지 시퀀스
│   │   └── ar_captures/      # 선택 과제용 ARKit/ARCore 캡처
│   └── processed/
│       ├── colmap_exports/   # cameras.txt, images.txt, points3D.txt
│       └── mapanything_inputs/
├── outputs/                  # Git으로 관리하지 않는 실험 산출물
│   ├── colmap/
│   │   ├── databases/
│   │   ├── dense/
│   │   ├── exports/
│   │   ├── sparse/
│   │   ├── visualizations/
│   │   └── logs/
│   └── mapanything/
│       ├── config_a_uncalibrated_image_only/
│       ├── config_b_calibrated_image_only/
│       ├── config_c_calibrated_colmap_poses/
│       └── config_d_calibrated_ar_poses/
├── scripts/                  # COLMAP, MapAnything, 평가 자동화 스크립트
│   ├── colmap/
│   ├── mapanything/
│   ├── evaluation/
│   └── utils/
└── third_party/
    └── map-anything/         # facebookresearch/map-anything 로컬 체크아웃
```

`data/`, `outputs/`, `third_party/map-anything/`의 실제 내용물은 Git에서 제외한다.
대신 `.gitkeep`과 README 파일만 남겨 필요한 디렉토리 구조를 보이게 한다. 보고서는
코드 및 대용량 산출물과 분리해 `reports/` 아래에서 관리한다.

## Task 1: COLMAP 기반 Classical SfM

Task 1의 목표는 이후 MapAnything 입력으로 활용할 카메라 내부 파라미터와 포즈를
COLMAP으로 추정하는 것이다.

MP4 영상을 사용하는 경우 먼저 COLMAP 입력 형식인 이미지 디렉토리로 변환한다.

```bash
conda activate rkv-mapanything
python scripts/colmap/preprocess_video.py /path/to/scene.mp4 \
  --output-dir data/raw/rgb_sequences/scene \
  --fps 2 \
  --max-side 1600
```

이후 COLMAP 파이프라인을 실행한다.

```bash
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene
```

기본값은 `scripts/colmap/colmap_docker.py`를 통해 GPU 지원 Docker COLMAP을
실행한다. CPU-only 로컬 바이너리를 꼭 써야 할 때만
`--colmap colmap --use-gpu 0`을 명시한다.

실행 결과는 `data/processed/colmap_exports/<run-name>/`의 `cameras.txt`,
`images.txt`, `points3D.txt`와 `outputs/colmap/` 아래의 sparse model, PLY,
camera trajectory plot, metrics/log 파일로 저장된다. 기본 COLMAP 설정은 이후
MapAnything Config B/C에서 바로 쓸 수 있도록 pinhole intrinsics 중심으로 맞춰져
있다.

COLMAP dense MVS 결과는 과제 필수 산출물이 아니다. Project Guide의 Task 1은
feature extraction, matching, incremental mapping, TXT export, sparse point cloud
및 camera trajectory visualization, 등록 이미지 수와 평균 reprojection error를
요구한다. Dense fused point cloud가 추가 스크린샷에 필요할 때만 다음 옵션을 붙인다.

```bash
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene \
  --run-dense
```

이 경우 `outputs/colmap/dense/<run-name>/fused.ply`가 추가로 저장된다.

수행 절차:

1. Google Colab 또는 Elice cloud GPU 환경에서 COLMAP을 빌드하고 실행한다.
2. 직접 준비한 RGB 이미지 시퀀스를 입력으로 사용한다.
3. Feature extraction, matching, incremental mapping을 순서대로 수행한다.
4. COLMAP 결과 파일을 내보낸다.
   - `cameras.txt`: 카메라 내부 파라미터
   - `images.txt`: 카메라 포즈
   - `points3D.txt`: 3D 포인트
5. 복원된 포인트 클라우드와 카메라 궤적을 시각화한다.
6. 등록된 이미지 수, 평균 reprojection error, 복원 결과 스크린샷을 정리한다.

## Task 2: MapAnything 기반 Feed-Forward 3D Reconstruction

Task 2의 목표는 MapAnything을 네 가지 입력 설정으로 실행하고, calibration 정보와
pose 정보가 결과에 어떤 영향을 주는지 비교하는 것이다.

MapAnything은 이미지뿐 아니라 카메라 내부 파라미터, 포즈, 깊이 등 추가적인 기하
정보를 선택적으로 입력받을 수 있는 feed-forward metric 3D reconstruction 모델이다.
본 프로젝트에서는 다음 네 가지 설정을 비교한다.

| 설정 | 입력 이미지 | 내부 파라미터 | 포즈 | 목적 |
| --- | --- | --- | --- | --- |
| Config A | RGB 이미지 | 없음 | 없음 | Uncalibrated image-only baseline |
| Config B | RGB 이미지 | COLMAP intrinsics | 없음 | Calibration 효과 분석 |
| Config C | RGB 이미지 | COLMAP intrinsics | COLMAP poses | Pose 입력 효과 분석 |
| Config D | RGB 이미지 | 카메라 intrinsics | ARKit/ARCore poses | 선택 과제: AR pose와 COLMAP pose 비교 |

각 설정에 대해 수행할 작업:

- MapAnything reconstruction 실행
- Novel view rendering 수행
- 실제 이미지와 렌더링 이미지 비교
- PSNR, SSIM 등 정량 지표 계산
- 예측 pose error와 실행 시간 등 추가 지표 정리
- A vs B, B vs C, 선택적으로 C vs D 비교표 정리
- 실패 사례, artefact, 관찰된 품질 차이 분석

기본 Task 2 파이프라인은 다음과 같이 실행한다. 설정은
`configs/mapanything/task2_pipeline.json`에서 읽는다.

```bash
conda activate rkv-mapanything
python scripts/mapanything/run_pipline.py \
  --config configs/mapanything/task2_pipeline.json
```

입력 검증과 manifest 작성만 먼저 확인하려면 다음을 실행한다.

```bash
python scripts/mapanything/run_pipline.py --prepare-only
```

결과는 각 설정별 `outputs/mapanything/<run-name>/config_*` 디렉토리와
`outputs/mapanything/<run-name>_task2_summary.json`에 저장된다. Markdown 요약에는
각 config별 metric 표와 함께 A vs B calibration 효과, B vs C pose 입력 효과,
선택 과제 C vs D pose source 비교가 자동으로 추가된다.

## Task 3: ARKit 또는 ARCore 기반 Pose 입력, 선택 과제

Task 3은 선택 과제이다. 모바일 기기에서 ARKit 또는 ARCore를 사용하여 VIO pose를
얻고, 이를 MapAnything 입력으로 사용하는 방식이다.

수행 절차:

1. 정적인 장면을 대상으로 짧은 AR scan을 촬영한다.
2. 저장된 AR pose를 MapAnything 좌표계에 맞게 변환한다.
3. 변환 과정을 문서화한다.
4. 동일한 장면에 대해 MapAnything을 실행한다.
5. COLMAP pose를 사용한 Config C와 AR pose를 사용한 Config D를 비교한다.

보고 항목:

- ARKit 또는 ARCore 캡처 환경
- AR pose를 MapAnything 좌표계로 변환한 방법
- PSNR, SSIM, pose drift, runtime 비교
- COLMAP pose 대비 AR pose의 장단점 분석

## 참고 자료

- MapAnything project page: <https://map-anything.github.io/>
- MapAnything code: <https://github.com/facebookresearch/map-anything>
- MapAnything paper: <https://arxiv.org/abs/2509.13414>
- COLMAP 공식 튜토리얼: <https://colmap.github.io/tutorial.html>
- COLMAP & MapAnything overview video: <https://www.youtube.com/watch?v=z4L9NLHKScM>
- Build & run tutorial video: <https://youtu.be/Z8kJ86JdxB8>
- AR capture tutorial video: <https://www.youtube.com/watch?v=H3ssOq7W-ho>
- ARKit documentation: <https://developer.apple.com/documentation/arkit>
- ARCore documentation: <https://developers.google.com/ar/develop>
