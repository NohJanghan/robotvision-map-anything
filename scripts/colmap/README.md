# COLMAP Pipeline

Task 1 expects COLMAP intrinsics, poses, sparse points, a reconstruction
visualization, and summary numbers such as registered image count and mean
reprojection error.

The optimized default is sparse SfM, not dense MVS. The project guide asks for
`cameras.txt`, `images.txt`, `points3D.txt`, a point-cloud/trajectory
visualization, registered image count, and reprojection error. Dense COLMAP MVS
(`image_undistorter -> patch_match_stereo -> stereo_fusion`) is therefore kept
as an optional extra for qualitative screenshots only.

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

The defaults are tuned for a short RGB video/image sequence used later by
MapAnything:

- `--camera-model PINHOLE` keeps the exported intrinsics compatible with the
  3x3 pinhole matrix consumed by MapAnything Config B/C.
- `--max-image-size 1600` controls SIFT memory/runtime on Colab/Elice while
  preserving enough features for typical phone or webcam sequences.
- `--mapper-init-min-tri-angle 8.0` and the global BA ratios follow a
  video-sequence-friendly SfM setup for lower-parallax adjacent frames.

If your raw camera has strong lens distortion and `PINHOLE` registers too few
images, rerun with `--camera-model SIMPLE_RADIAL` and mention in the report that
MapAnything receives the pinhole part of the COLMAP intrinsics.

When running over SSH or on a headless server, the script sets
`QT_QPA_PLATFORM=offscreen` for COLMAP by default so the Qt-based CLI binary
does not try to open an X display. If your COLMAP build needs a different
backend, pass `--qt-qpa-platform minimal`.

If GPU SIFT fails with an OpenGL context error on a headless machine, run the
COLMAP commands through `xvfb-run`:

```bash
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene \
  --overwrite \
  --use-gpu 1 \
  --use-xvfb
```

The `--use-xvfb` option removes the default `QT_QPA_PLATFORM=offscreen` setting
for COLMAP subprocesses so the temporary X display created by `xvfb-run` can be
used.

For unordered image collections instead of video frames, use exhaustive
matching:

```bash
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene_exhaustive \
  --matcher exhaustive
```

## Optional Dense COLMAP MVS

Dense output is not required for Task 1 or as input to Task 2. If you want an
additional COLMAP dense fused point cloud for screenshots, run:

```bash
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene \
  --run-dense
```

This adds:

- `outputs/colmap/dense/<run-name>/fused.ply`
- `outputs/colmap/dense/<run-name>/images/` undistorted dense workspace images
- dense command logs under `outputs/colmap/logs/<run-name>/`

Use dense sparingly on cloud GPUs because PatchMatch stereo is much slower and
more memory-intensive than the sparse SfM outputs required by the project.

## Outputs

The pipeline writes:

- `data/processed/colmap_exports/<run-name>/cameras.txt`
- `data/processed/colmap_exports/<run-name>/images.txt`
- `data/processed/colmap_exports/<run-name>/points3D.txt`
- `outputs/colmap/sparse/<run-name>/` binary COLMAP sparse model
- optional `outputs/colmap/dense/<run-name>/fused.ply` when `--run-dense` is set
- `outputs/colmap/visualizations/<run-name>/points3D.ply`
- `outputs/colmap/visualizations/<run-name>/camera_trajectory.png`
- `outputs/colmap/logs/<run-name>/metrics.json`
- `outputs/colmap/logs/<run-name>/run_summary.json`

`metrics.json` is intended for the final report and includes values parsed from
`colmap model_analyzer`, including registered images and mean reprojection
error when COLMAP reports them.
