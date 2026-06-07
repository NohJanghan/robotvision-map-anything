#!/usr/bin/env python3
"""Run the Task 1 COLMAP SfM pipeline for an RGB image sequence."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NoReturn


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TEXT_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")
DENSE_TASK_DECISION = {
    "required_for_project_pdf": False,
    "reason": (
        "Task 1 asks for COLMAP intrinsics, camera poses, sparse points, "
        "trajectory visualization, registered image count, and mean "
        "reprojection error. Dense MVS is useful for an extra qualitative "
        "point-cloud screenshot, but it is not required for the Task 2 "
        "MapAnything inputs."
    ),
    "required_outputs": ["cameras.txt", "images.txt", "points3D.txt", "points3D.ply"],
    "optional_dense_output": "outputs/colmap/dense/<run-name>/fused.ply",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run COLMAP feature extraction, matching, incremental mapping, "
            "TXT export, lightweight visualization, and optional dense MVS."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing input RGB images.",
    )
    parser.add_argument(
        "--run-name",
        help="Name used under outputs/colmap and data/processed exports.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/colmap"),
        help="Root directory for COLMAP outputs. Default: outputs/colmap.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("data/processed/colmap_exports"),
        help="Root directory for exported TXT models. Default: data/processed/colmap_exports.",
    )
    parser.add_argument(
        "--colmap",
        default="colmap",
        help="COLMAP executable path or command name. Default: colmap.",
    )
    parser.add_argument(
        "--camera-model",
        default="PINHOLE",
        help=(
            "COLMAP ImageReader camera model. Default: PINHOLE, so Task 2 can "
            "reuse a 3x3 pinhole intrinsics matrix without silently dropping "
            "distortion parameters."
        ),
    )
    parser.add_argument(
        "--camera-params",
        help="Optional comma-separated ImageReader.camera_params value.",
    )
    parser.add_argument(
        "--single-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat all images as one camera. Default: true.",
    )
    parser.add_argument(
        "--use-gpu",
        type=int,
        choices=(0, 1),
        default=1,
        help="Set COLMAP SIFT GPU usage. Default: 1.",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="SIFT max number of features per image. Default: 8192.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="SIFT max image side in pixels. Default: 1600.",
    )
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive", "spatial"),
        default="sequential",
        help="Feature matcher. Use sequential for video/image sequences. Default: sequential.",
    )
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        default=10,
        help="Sequential matcher overlap. Default: 10.",
    )
    parser.add_argument(
        "--loop-detection",
        action="store_true",
        help="Enable loop detection for sequential matching.",
    )
    parser.add_argument(
        "--mapper-min-num-matches",
        type=int,
        default=15,
        help="Mapper minimum number of matches. Default: 15.",
    )
    parser.add_argument(
        "--mapper-init-min-tri-angle",
        type=float,
        default=8.0,
        help=(
            "Mapper initial pair minimum triangulation angle in degrees. "
            "Default: 8.0, a video-sequence-friendly value."
        ),
    )
    parser.add_argument(
        "--mapper-ba-global-images-ratio",
        type=float,
        default=1.4,
        help="Mapper global BA image growth ratio. Default: 1.4.",
    )
    parser.add_argument(
        "--mapper-ba-global-points-ratio",
        type=float,
        default=1.4,
        help="Mapper global BA point growth ratio. Default: 1.4.",
    )
    parser.add_argument(
        "--run-dense",
        action="store_true",
        help=(
            "Also run optional COLMAP dense MVS: image_undistorter, "
            "patch_match_stereo, and stereo_fusion. Not required by the "
            "project PDF or by MapAnything inputs."
        ),
    )
    parser.add_argument(
        "--dense-max-image-size",
        type=int,
        default=1600,
        help="Dense MVS max image side in pixels. Default: 1600.",
    )
    parser.add_argument(
        "--dense-geom-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use geometric consistency for dense stereo/fusion. Default: true.",
    )
    parser.add_argument(
        "--qt-qpa-platform",
        default="offscreen",
        help=(
            "Qt platform backend for headless/SSH COLMAP runs. Set to an empty "
            "string to inherit the current environment. Default: offscreen."
        ),
    )
    parser.add_argument(
        "--use-xvfb",
        action="store_true",
        help=(
            "Run COLMAP commands through xvfb-run. Useful when GPU SIFT needs an "
            "X display in a headless terminal."
        ),
    )
    parser.add_argument(
        "--xvfb-run",
        default="xvfb-run",
        help="xvfb-run executable path or command name. Default: xvfb-run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove this run's generated output/export directories first.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature extraction and reuse an existing database.",
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip matching and reuse existing matches in the database.",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Skip mapper and reuse an existing binary sparse model.",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_tool(command: str, install_hint: str | None = None) -> str:
    if Path(command).exists():
        return command
    path = shutil.which(command)
    if path is None:
        if install_hint is None:
            install_hint = "Install it and ensure it is available on PATH."
        fail(f"'{command}' was not found on PATH. {install_hint}")
    return path


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def colmap_environment(qt_qpa_platform: str | None, use_xvfb: bool) -> dict[str, str]:
    env = os.environ.copy()
    if use_xvfb and qt_qpa_platform == "offscreen":
        env.pop("QT_QPA_PLATFORM", None)
        return env
    if qt_qpa_platform:
        env["QT_QPA_PLATFORM"] = qt_qpa_platform
    return env


def xvfb_command_prefix(args: argparse.Namespace) -> list[str]:
    if not args.use_xvfb:
        return []
    xvfb_run = require_tool(
        args.xvfb_run,
        "Install Xvfb, e.g. `sudo apt install xvfb`, or pass --xvfb-run /path/to/xvfb-run.",
    )
    return [xvfb_run, "-a"]


def run_command(command: list[str], log_file: Path, env: dict[str, str]) -> str:
    print(" ".join(command))
    result = subprocess.run(
        command,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(f"\n$ {' '.join(command)}\n")
        if "QT_QPA_PLATFORM" in env:
            fp.write(f"QT_QPA_PLATFORM={env['QT_QPA_PLATFORM']}\n")
        fp.write(result.stdout)
        if result.stdout and not result.stdout.endswith("\n"):
            fp.write("\n")
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        fail(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return result.stdout


def safe_remove(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        fail(f"refusing to remove path outside allowed root: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def latest_sparse_model(sparse_dir: Path) -> Path:
    candidates = [
        p for p in sparse_dir.iterdir() if p.is_dir() and (p / "cameras.bin").exists()
    ]
    if not candidates:
        fail(f"no sparse COLMAP model found under {sparse_dir}")

    def sort_key(path: Path) -> tuple[int, str]:
        return (int(path.name), path.name) if path.name.isdigit() else (sys.maxsize, path.name)

    return sorted(candidates, key=sort_key)[0]


def parse_model_analyzer(output: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    patterns = {
        "registered_images": r"Registered images:\s+(\d+)",
        "cameras": r"Cameras:\s+(\d+)",
        "points": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([0-9.]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([0-9.]+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s+([0-9.]+)\s*px",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            continue
        value = match.group(1)
        metrics[key] = int(value) if value.isdigit() else float(value)
    return metrics


def parse_images_txt(images_txt: Path) -> list[tuple[str, tuple[float, float, float]]]:
    cameras: list[tuple[str, tuple[float, float, float]]] = []
    if not images_txt.exists():
        return cameras
    lines = [
        line
        for line in images_txt.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    for line in lines[0::2]:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0])
            qw, qx, qy, qz = (float(v) for v in parts[1:5])
            tx, ty, tz = (float(v) for v in parts[5:8])
            int(parts[8])
        except ValueError:
            continue
        name = " ".join(parts[9:])
        rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        center = camera_center(rotation, (tx, ty, tz))
        cameras.append((name, center))
    return cameras


def quaternion_to_rotation_matrix(
    qw: float, qx: float, qy: float, qz: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
        ),
        (
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qx * qw,
        ),
        (
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx * qx - 2 * qy * qy,
        ),
    )


def camera_center(
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    translation: tuple[float, float, float],
) -> tuple[float, float, float]:
    # COLMAP stores world-to-camera transform; camera center is -R^T t.
    return (
        -sum(rotation[row][0] * translation[row] for row in range(3)),
        -sum(rotation[row][1] * translation[row] for row in range(3)),
        -sum(rotation[row][2] * translation[row] for row in range(3)),
    )


def write_trajectory_plot(images_txt: Path, output_path: Path) -> None:
    cameras = parse_images_txt(images_txt)
    if len(cameras) < 2:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [center[0] for _, center in cameras]
    zs = [center[2] for _, center in cameras]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xs, zs, marker="o", linewidth=1.5, markersize=3)
    ax.scatter([xs[0]], [zs[0]], c="green", label="start")
    ax.scatter([xs[-1]], [zs[-1]], c="red", label="end")
    ax.set_title("COLMAP camera trajectory (top-down X/Z)")
    ax.set_xlabel("world X")
    ax.set_ylabel("world Z")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def copy_text_model(export_dir: Path, processed_export_dir: Path) -> None:
    processed_export_dir.mkdir(parents=True, exist_ok=True)
    for filename in TEXT_MODEL_FILES:
        source = export_dir / filename
        if not source.exists():
            fail(f"expected export file missing: {source}")
        shutil.copy2(source, processed_export_dir / filename)


def write_run_summary(
    path: Path,
    args: argparse.Namespace,
    image_count: int,
    paths: dict[str, Path],
    metrics: dict[str, object],
    commands: Iterable[str],
) -> None:
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "image_dir": str(args.image_dir.resolve()),
        "image_count": image_count,
        "settings": {
            "camera_model": args.camera_model,
            "camera_params": args.camera_params,
            "single_camera": args.single_camera,
            "use_gpu": args.use_gpu,
            "max_num_features": args.max_num_features,
            "max_image_size": args.max_image_size,
            "matcher": args.matcher,
            "sequential_overlap": args.sequential_overlap,
            "loop_detection": args.loop_detection,
            "mapper_min_num_matches": args.mapper_min_num_matches,
            "mapper_init_min_tri_angle": args.mapper_init_min_tri_angle,
            "mapper_ba_global_images_ratio": args.mapper_ba_global_images_ratio,
            "mapper_ba_global_points_ratio": args.mapper_ba_global_points_ratio,
            "run_dense": args.run_dense,
            "dense_max_image_size": args.dense_max_image_size,
            "dense_geom_consistency": args.dense_geom_consistency,
            "qt_qpa_platform": args.qt_qpa_platform,
            "use_xvfb": args.use_xvfb,
            "xvfb_run": args.xvfb_run,
        },
        "colmap_dense_decision": DENSE_TASK_DECISION,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "metrics": metrics,
        "commands": list(commands),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> list[Path]:
    if not args.image_dir.exists():
        fail(f"image directory does not exist: {args.image_dir}")
    if not args.image_dir.is_dir():
        fail(f"image path is not a directory: {args.image_dir}")
    images = list_images(args.image_dir)
    if not images:
        fail(f"no RGB image files found in {args.image_dir}")
    if args.max_num_features < 1:
        fail("--max-num-features must be positive")
    if args.max_image_size < 1:
        fail("--max-image-size must be positive")
    if args.sequential_overlap < 1:
        fail("--sequential-overlap must be positive")
    if args.mapper_min_num_matches < 1:
        fail("--mapper-min-num-matches must be positive")
    if args.mapper_init_min_tri_angle < 0:
        fail("--mapper-init-min-tri-angle must be non-negative")
    if args.mapper_ba_global_images_ratio <= 1:
        fail("--mapper-ba-global-images-ratio must be greater than 1")
    if args.mapper_ba_global_points_ratio <= 1:
        fail("--mapper-ba-global-points-ratio must be greater than 1")
    if args.dense_max_image_size < 1:
        fail("--dense-max-image-size must be positive")
    if args.run_name is None:
        args.run_name = args.image_dir.resolve().name
    return images


def main() -> None:
    args = parse_args()
    images = validate_args(args)
    colmap = require_tool(
        args.colmap,
        "Activate rkv-mapanything and install COLMAP, e.g. "
        "`conda install -c conda-forge colmap`.",
    )
    xvfb_prefix = xvfb_command_prefix(args)

    output_root = args.output_root
    run_name = args.run_name
    database_dir = output_root / "databases" / run_name
    database_path = database_dir / "database.db"
    sparse_dir = output_root / "sparse" / run_name
    dense_dir = output_root / "dense" / run_name
    visual_dir = output_root / "visualizations" / run_name
    log_dir = output_root / "logs" / run_name
    export_dir = output_root / "exports" / run_name
    processed_export_dir = args.export_root / run_name

    if args.overwrite:
        for path, allowed_root in (
            (database_dir, output_root),
            (sparse_dir, output_root),
            (dense_dir, output_root),
            (visual_dir, output_root),
            (log_dir, output_root),
            (export_dir, output_root),
            (processed_export_dir, args.export_root),
        ):
            safe_remove(path, allowed_root)

    for path in (database_dir, sparse_dir, visual_dir, log_dir, export_dir, processed_export_dir):
        path.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    env = colmap_environment(args.qt_qpa_platform, args.use_xvfb)

    def run(command: list[str], log_name: str) -> str:
        full_command = [*xvfb_prefix, *command]
        commands.append(" ".join(full_command))
        return run_command(full_command, log_dir / log_name, env)

    if not args.skip_features:
        feature_command = [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(args.image_dir),
            "--ImageReader.camera_model",
            args.camera_model,
            "--ImageReader.single_camera",
            "1" if args.single_camera else "0",
            "--SiftExtraction.use_gpu",
            str(args.use_gpu),
            "--SiftExtraction.max_num_features",
            str(args.max_num_features),
            "--SiftExtraction.max_image_size",
            str(args.max_image_size),
        ]
        if args.camera_params:
            feature_command.extend(["--ImageReader.camera_params", args.camera_params])
        run(feature_command, "01_feature_extractor.log")

    if not args.skip_matching:
        matcher_command = [
            colmap,
            f"{args.matcher}_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            str(args.use_gpu),
        ]
        if args.matcher == "sequential":
            matcher_command.extend(
                [
                    "--SequentialMatching.overlap",
                    str(args.sequential_overlap),
                    "--SequentialMatching.loop_detection",
                    "1" if args.loop_detection else "0",
                ]
            )
        run(matcher_command, "02_matcher.log")

    if not args.skip_mapping:
        run(
            [
                colmap,
                "mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(args.image_dir),
                "--output_path",
                str(sparse_dir),
                "--Mapper.min_num_matches",
                str(args.mapper_min_num_matches),
                "--Mapper.init_min_tri_angle",
                str(args.mapper_init_min_tri_angle),
                "--Mapper.ba_global_images_ratio",
                str(args.mapper_ba_global_images_ratio),
                "--Mapper.ba_global_points_ratio",
                str(args.mapper_ba_global_points_ratio),
            ],
            "03_mapper.log",
        )

    model_dir = latest_sparse_model(sparse_dir)
    run(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(export_dir),
            "--output_type",
            "TXT",
        ],
        "04_model_converter_txt.log",
    )
    copy_text_model(export_dir, processed_export_dir)

    ply_path = visual_dir / "points3D.ply"
    run(
        [
            colmap,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(ply_path),
            "--output_type",
            "PLY",
        ],
        "05_model_converter_ply.log",
    )

    analyzer_output = run(
        [colmap, "model_analyzer", "--path", str(model_dir)],
        "06_model_analyzer.log",
    )
    metrics = parse_model_analyzer(analyzer_output)
    metrics_path = log_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    trajectory_path = visual_dir / "camera_trajectory.png"
    write_trajectory_plot(processed_export_dir / "images.txt", trajectory_path)

    paths = {
        "database": database_path,
        "binary_sparse_model": model_dir,
        "txt_export": processed_export_dir,
        "ply_point_cloud": ply_path,
        "trajectory_plot": trajectory_path,
        "metrics": metrics_path,
        "logs": log_dir,
    }

    if args.run_dense:
        # Preserve the required sparse-SfM summary even if optional dense MVS fails.
        write_run_summary(log_dir / "run_summary.json", args, len(images), paths, metrics, commands)
        dense_dir.mkdir(parents=True, exist_ok=True)
        dense_fused_path = dense_dir / "fused.ply"
        run(
            [
                colmap,
                "image_undistorter",
                "--image_path",
                str(args.image_dir),
                "--input_path",
                str(model_dir),
                "--output_path",
                str(dense_dir),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(args.dense_max_image_size),
            ],
            "07_image_undistorter.log",
        )
        run(
            [
                colmap,
                "patch_match_stereo",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "1" if args.dense_geom_consistency else "0",
                "--PatchMatchStereo.max_image_size",
                str(args.dense_max_image_size),
            ],
            "08_patch_match_stereo.log",
        )
        run(
            [
                colmap,
                "stereo_fusion",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric" if args.dense_geom_consistency else "photometric",
                "--output_path",
                str(dense_fused_path),
                "--StereoFusion.max_image_size",
                str(args.dense_max_image_size),
            ],
            "09_stereo_fusion.log",
        )
        paths["dense_workspace"] = dense_dir
        paths["dense_fused_point_cloud"] = dense_fused_path

    write_run_summary(log_dir / "run_summary.json", args, len(images), paths, metrics, commands)

    print("\nCOLMAP pipeline complete")
    print(f"registered images: {metrics.get('registered_images', 'unknown')} / {len(images)}")
    print(f"mean reprojection error: {metrics.get('mean_reprojection_error_px', 'unknown')} px")
    print(f"exported TXT model: {processed_export_dir}")
    print(f"visualizations: {visual_dir}")
    if args.run_dense:
        print(f"optional dense fused point cloud: {dense_dir / 'fused.ply'}")
    else:
        print("dense MVS: skipped (not required for the project PDF or MapAnything inputs)")


if __name__ == "__main__":
    main()
