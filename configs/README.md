# Configs

Project-level configuration files for COLMAP and MapAnything runs.

- `colmap/`: COLMAP command options and workspace presets
- `mapanything/`: per-configuration MapAnything options

Current COLMAP preset:

- `colmap/default_pipeline.json`: reference settings matching the defaults in
  `scripts/colmap/run_pipeline.py`. Dense COLMAP MVS is disabled by default
  because Task 1 and Task 2 consume sparse SfM exports; enable `--run-dense`
  only for extra qualitative COLMAP screenshots.

Current MapAnything preset:

- `mapanything/task2_pipeline.json`: Task 2 A/B/C/D configuration matrix for
  `scripts/mapanything/run_pipline.py`, including summary metrics for the
  required A vs B and B vs C comparisons plus optional C vs D.
