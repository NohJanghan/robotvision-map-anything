# Task 2 MapAnything Pipeline

이 디렉토리는 Task 2의 MapAnything 실행을 자동화하는 스크립트를 담고 있다.
주 실행 파일은 요청된 파일명 그대로 `run_pipline.py`이다.

```bash
conda activate rkv-mapanything
python scripts/mapanything/run_pipline.py \
  --config configs/mapanything/task2_pipeline.json
```

## 목적

`run_pipline.py`는 RGB 이미지 시퀀스에 대해 MapAnything을 네 가지 입력 설정으로
실행하고, reconstruction artifact와 정량 지표를 저장한다.

| Config | 입력 |
| --- | --- |
| `config_a` | RGB 이미지만 사용 |
| `config_b` | RGB 이미지 + COLMAP intrinsics |
| `config_c` | RGB 이미지 + COLMAP intrinsics + COLMAP poses |
| `config_d` | RGB 이미지 + ARKit/ARCore poses + AR 또는 COLMAP intrinsics |

Config D는 Task 3과 연결되는 선택 설정이다. AR pose 파일이 없으면 skip될 수 있다.

## 입력 준비

기본 JSON 설정은 `configs/mapanything/task2_pipeline.json`이다.

기본 입력 경로:

```text
data/raw/rgb_sequences/<run-name>/          # RGB 이미지 시퀀스
data/processed/colmap_exports/<run-name>/  # Task 1 COLMAP export
data/raw/ar_captures/<run-name>/poses.json # 선택: AR pose
```

Config B/C를 실행하려면 다음 COLMAP TXT export가 필요하다.

```text
data/processed/colmap_exports/<run-name>/cameras.txt
data/processed/colmap_exports/<run-name>/images.txt
data/processed/colmap_exports/<run-name>/points3D.txt
```

이 파일들은 Task 1의 `scripts/colmap/run_pipeline.py`가 생성한다.

## 빠른 점검

설정 목록만 확인:

```bash
python scripts/mapanything/run_pipline.py --list-configs
```

MapAnything 모델을 로드하지 않고 입력 manifest만 생성:

```bash
python scripts/mapanything/run_pipline.py --prepare-only
```

특정 config만 점검:

```bash
python scripts/mapanything/run_pipline.py --only config_a --prepare-only
python scripts/mapanything/run_pipline.py --only config_b,config_c --prepare-only
```

입력이 일부 없어도 가능한 config를 계속 점검:

```bash
python scripts/mapanything/run_pipline.py --prepare-only --continue-on-error
```

## 주요 옵션

| 옵션 | 설명 |
| --- | --- |
| `--config` | 사용할 Task 2 JSON 설정 파일 |
| `--run-name` | 출력과 manifest에 사용할 run 이름 |
| `--image-dir` | RGB 이미지 디렉토리 override |
| `--colmap-export-dir` | COLMAP TXT export 디렉토리 override |
| `--ar-pose-file` | ARKit/ARCore pose JSON override |
| `--input-root` | MapAnything input manifest 저장 루트 |
| `--output-root` | MapAnything 결과 저장 루트 |
| `--only` | 특정 config만 실행 |
| `--skip` | 특정 config 제외 |
| `--prepare-only` | 모델 inference 없이 입력 검증만 수행 |
| `--overwrite` | 이전 생성 결과를 지우고 다시 실행 |
| `--continue-on-error` | config 하나가 실패해도 다음 config 계속 처리 |

`--overwrite`는 `run_pipline.py`가 만든 marker 파일이 있는 출력 디렉토리만 삭제한다.
임의 디렉토리 삭제를 막기 위한 안전장치이다.

## 출력 구조

기본 출력 루트는 `outputs/mapanything`이다.

```text
outputs/mapanything/
  <run-name>/
    config_a_uncalibrated_image_only/
    config_b_calibrated_image_only/
    config_c_calibrated_colmap_poses/
    config_d_calibrated_ar_poses/
  <run-name>_task2_summary.json
  <run-name>_task2_summary.md
```

각 config 디렉토리에는 가능한 경우 다음 파일들이 저장된다.

```text
predictions/view_0000.npz       # depth, intrinsics, pose, mask 등
depth/view_0000.png             # depth visualization
predicted_poses.json            # 예측 intrinsics / cam2world poses
reconstruction_points.ply       # point cloud
reconstruction.glb              # GLB scene, export 실패 시 warning 처리
renders/*_render.png            # point-splat novel/re-target view rendering
renders/*_target.png
renders/*_mask.png
renders/*_diff.png
metrics/run_summary.json        # config별 metric summary
```

입력 manifest는 별도 루트에 저장된다.

```text
data/processed/mapanything_inputs/<run-name>/<config-id>/manifest.json
```

## 평가 지표

스크립트가 저장하는 주요 지표:

- runtime
- render PSNR / SSIM / coverage
- full-image PSNR / SSIM
- pose ATE RMSE
- relative rotation error
- intrinsics error
- Task 2 coverage review

렌더링 평가는 예측 depth/pose로 만든 point-splat 결과를 target view와 비교한다.
`evaluation.render.min_coverage`보다 coverage가 낮은 pair는 평균 PSNR/SSIM 집계에서
제외되고, low-coverage pair 수가 summary에 기록된다.

## AR Pose JSON

Config D는 다음 형태의 AR pose JSON을 지원한다.

```json
{
  "frames": [
    {
      "image": "000000.jpg",
      "cam2world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
      "pose_convention": "arkit_cam2world"
    }
  ]
}
```

`poses` 배열도 `frames`와 동일하게 처리된다. 단순히 이미지 이름을 4x4 pose에
매핑하는 JSON도 사용할 수 있다.

지원하는 pose convention:

- `opencv_cam2world`
- `world2cam_opencv`
- `arkit_cam2world`
- `arkit_world2cam`
- `arcore_cam2world`
- `arcore_world2cam`
- `opengl_cam2world`
- `opengl_world2cam`

MapAnything 입력 pose는 OpenCV/RDF 기준 cam2world로 변환된다.

## 주의사항

- Config B/C는 COLMAP `cameras.txt`와 `images.txt`가 없으면 실행할 수 없다.
- COLMAP distortion 계수는 3x3 pinhole intrinsics로 축약된다. 왜곡 모델을 사용한
  경우에는 가능하면 undistorted image를 입력으로 사용하는 것이 좋다.
- `exports.save_colmap`은 기본값이 `false`이다. 이 옵션을 켜려면 MapAnything의
  COLMAP exporter가 요구하는 `open3d`, `pycolmap` 설치가 필요하다.
- GLB export는 dense points/images/masks를 한 번에 stack하므로 이미지가 많으면
  메모리를 많이 쓸 수 있다. 긴 시퀀스에서는 `view_selection.stride` 또는
  `view_selection.max_images`를 조정한다.
- 정성 분석, 실패 사례 설명, artifact 해석은 최종 보고서에서 사람이 직접
  정리해야 한다. 이 스크립트는 그 분석에 필요한 산출물과 지표를 생성한다.
