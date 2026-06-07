# Task 1-3 Results

이 문서는 `input/bike`와 `input/lounge`로 수행한 COLMAP / MapAnything 결과를
정리한다. 재현 명령과 절차는 `tutorial.md`를 참고한다.

## Summary

| Scene | RGB input | COLMAP result | MapAnything views | Best config |
| --- | --- | --- | ---: | --- |
| bike | 128 frames, stride 9, 1600x1200 | 128/128 registered, 68,365 points, 0.587909 px | 128 | Config C |
| lounge | 128 frames, stride 11, 1600x1200 | 96/128 registered, 11,246 points, 0.824094 px | 96 | Config C |

## Bike

### Input

| Item | Value |
| --- | --- |
| Source video | `input/bike/rgb.mp4` |
| Source size | 1920 x 1440 |
| Source frames / FPS | 1176 / 30.0 |
| Extracted frames | 128 |
| Stride | 9 |
| Output image size | 1600 x 1200 |
| AR pose JSON | `data/raw/ar_captures/bike/poses.json` |

### Task 1: COLMAP

| Metric | Value |
| --- | ---: |
| Registered images | 128 / 128 |
| Cameras | 1 |
| Sparse points | 68,365 |
| Observations | 892,038 |
| Mean track length | 13.048168 |
| Mean observations per image | 6969.046875 |
| Mean reprojection error | 0.587909 px |

COLMAP intrinsics:

```text
1 PINHOLE 1600 1200 1185.612602634478 1192.720415692288 800 600
```

Key files:

- `data/processed/colmap_exports/bike/cameras.txt`
- `data/processed/colmap_exports/bike/images.txt`
- `data/processed/colmap_exports/bike/points3D.txt`
- `outputs/colmap/visualizations/bike/points3D.ply`
- `outputs/colmap/visualizations/bike/camera_trajectory.png`
- `outputs/colmap/logs/bike/metrics.json`

### Task 2/3: MapAnything

| Config | Input | Views | Runtime (s) | PSNR | SSIM | Coverage | Pose ATE RMSE | Rel. Rot. Err. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | image-only | 128 | 17.6487 | 21.5523 | 0.9296 | 0.2137 | 0.1050 | 0.5407 |
| B | intrinsics | 128 | 17.6056 | 21.5120 | 0.9310 | 0.2127 | 0.1332 | 0.5029 |
| C | intrinsics + COLMAP poses | 128 | 17.6443 | 24.6081 | 0.9658 | 0.2150 | 0.0190 | 0.0729 |
| D | intrinsics + AR poses | 128 | 17.6652 | 17.4384 | 0.8292 | 0.2048 | 0.7199 | 2.1540 |

Comparisons:

| Comparison | Meaning | PSNR d | SSIM d | Coverage d | Pose ATE d | Runtime d (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A vs B | Calibration effect | -0.0403 | +0.0014 | -0.0010 | +0.0282 | -0.0431 |
| B vs C | COLMAP pose input effect | +3.0961 | +0.0348 | +0.0022 | -0.1142 | +0.0387 |
| C vs D | COLMAP pose vs AR pose | -7.1696 | -0.1366 | -0.0102 | +0.7009 | +0.0209 |

## Lounge

### Input

| Item | Value |
| --- | --- |
| Source video | `input/lounge/rgb.mp4` |
| Source size | 1920 x 1440 |
| Source frames / FPS | 1412 / 29.957567 |
| Extracted frames | 128 |
| Stride | 11 |
| Output image size | 1600 x 1200 |
| AR pose JSON | `data/raw/ar_captures/lounge/poses.json` |

### Task 1: COLMAP

Sequential matching registered only 52 / 128 images. The final run kept the same
128-frame input but used exhaustive GPU matching, which registered 96 images.

| Metric | Value |
| --- | ---: |
| Registered images | 96 / 128 |
| Cameras | 1 |
| Sparse points | 11,246 |
| Observations | 49,243 |
| Mean track length | 4.378712 |
| Mean observations per image | 512.947917 |
| Mean reprojection error | 0.824094 px |

COLMAP intrinsics:

```text
1 PINHOLE 1600 1200 1186.6213611960807 1188.4832835811933 800 600
```

Key files:

- `data/processed/colmap_exports/lounge/cameras.txt`
- `data/processed/colmap_exports/lounge/images.txt`
- `data/processed/colmap_exports/lounge/points3D.txt`
- `outputs/colmap/visualizations/lounge/points3D.ply`
- `outputs/colmap/visualizations/lounge/camera_trajectory.png`
- `outputs/colmap/logs/lounge/metrics.json`

### Task 2/3: MapAnything

MapAnything used the 96 COLMAP-registered views for all configurations.

| Config | Input | Views | Runtime (s) | PSNR | SSIM | Coverage | Pose ATE RMSE | Rel. Rot. Err. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | image-only | 96 | 12.3343 | 22.5297 | 0.9321 | 0.2099 | 2.3931 | 17.0925 |
| B | intrinsics | 96 | 12.3604 | 22.1657 | 0.9262 | 0.2103 | 2.4468 | 17.7398 |
| C | intrinsics + COLMAP poses | 96 | 12.2362 | 25.4805 | 0.9652 | 0.2100 | 0.0613 | 0.4070 |
| D | intrinsics + AR poses | 96 | 12.3784 | 12.8721 | 0.3627 | 0.1986 | 0.6717 | 12.6126 |

Comparisons:

| Comparison | Meaning | PSNR d | SSIM d | Coverage d | Pose ATE d | Runtime d (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A vs B | Calibration effect | -0.3640 | -0.0059 | +0.0005 | +0.0538 | +0.0261 |
| B vs C | COLMAP pose input effect | +3.3148 | +0.0390 | -0.0003 | -2.3855 | -0.1242 |
| C vs D | COLMAP pose vs AR pose | -12.6084 | -0.6026 | -0.0114 | +0.6104 | +0.1422 |

## Interpretation

- Config C is best for both scenes. Supplying COLMAP poses gives the largest
  improvement in PSNR, SSIM, and pose error.
- Calibration alone is not consistently beneficial in these runs. Config B is
  close to Config A and sometimes slightly worse in PSNR.
- Config D completes for both scenes, but AR/VIO poses are much worse than
  COLMAP poses. The likely causes are coordinate convention mismatch, scale
  mismatch, VIO drift, or residual assumptions in the AR-to-MapAnything pose
  conversion.
