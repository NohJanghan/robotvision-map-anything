# Scripts

Automation scripts for running and evaluating the project pipeline.

- `colmap/`: feature extraction, matching, mapping, and export helpers
- `mapanything/`: launchers and input adapters for the four MapAnything runs
- `evaluation/`: PSNR, SSIM, pose error, and runtime aggregation
- `utils/`: shared conversion and filesystem helpers

Implemented COLMAP helpers:

- `colmap/preprocess_video.py`: extract `000000.jpg`, `000001.jpg`, ... frames
  from an MP4/MOV video for COLMAP input.
- `colmap/run_pipeline.py`: run feature extraction, matching, mapper, TXT export,
  PLY export, trajectory plotting, metrics logging, and optional dense MVS via
  `--run-dense`.
- `mapanything/run_pipline.py`: run Task 2 MapAnything configurations from
  `configs/mapanything/task2_pipeline.json`, write per-config manifests,
  reconstruction artifacts, render/pose/intrinsics metrics, required/optional
  comparison deltas, and a Task 2 summary.
