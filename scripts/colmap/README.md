# COLMAP Pipeline

Task 1 expects COLMAP intrinsics, poses, sparse points, a reconstruction
visualization, and summary numbers such as registered image count and mean
reprojection error.

## 1. Extract Frames From Video

```bash
conda activate rkv-mapanything
python scripts/colmap/preprocess_video.py /path/to/scene.mp4 \
  --output-dir data/raw/rgb_sequences/scene \
  --fps 2 \
  --max-side 1600
```

The output directory contains sequential images named `000000.jpg`,
`000001.jpg`, ... plus `manifest.json`. This is the image folder passed to
COLMAP.

Useful alternatives:

```bash
# Keep every 10th decoded frame instead of sampling by time.
python scripts/colmap/preprocess_video.py /path/to/scene.mp4 \
  --output-dir data/raw/rgb_sequences/scene_stride10 \
  --stride 10

# Extract a 20 second window starting at 5 seconds.
python scripts/colmap/preprocess_video.py /path/to/scene.mp4 \
  --output-dir data/raw/rgb_sequences/scene_clip \
  --start-time 5 \
  --duration 20 \
  --fps 3
```

## 2. Run COLMAP

```bash
conda activate rkv-mapanything
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene
```

For a CPU-only environment, add `--use-gpu 0`.

When running over SSH or on a headless server, the script sets
`QT_QPA_PLATFORM=offscreen` for COLMAP by default so the Qt-based CLI binary
does not try to open an X display. If your COLMAP build needs a different
backend, pass `--qt-qpa-platform minimal`.

For unordered image collections instead of video frames, use exhaustive
matching:

```bash
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene_exhaustive \
  --matcher exhaustive
```

## Outputs

The pipeline writes:

- `data/processed/colmap_exports/<run-name>/cameras.txt`
- `data/processed/colmap_exports/<run-name>/images.txt`
- `data/processed/colmap_exports/<run-name>/points3D.txt`
- `outputs/colmap/sparse/<run-name>/` binary COLMAP sparse model
- `outputs/colmap/visualizations/<run-name>/points3D.ply`
- `outputs/colmap/visualizations/<run-name>/camera_trajectory.png`
- `outputs/colmap/logs/<run-name>/metrics.json`
- `outputs/colmap/logs/<run-name>/run_summary.json`

`metrics.json` is intended for the final report and includes values parsed from
`colmap model_analyzer`, including registered images and mean reprojection
error when COLMAP reports them.
