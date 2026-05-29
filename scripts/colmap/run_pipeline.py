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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run COLMAP feature extraction, matching, incremental mapping, "
            "TXT export, and lightweight visualization."
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
        default="SIMPLE_RADIAL",
        help="COLMAP ImageReader camera model. Default: SIMPLE_RADIAL.",
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
        "--qt-qpa-platform",
        default="offscreen",
        help=(
            "Qt platform backend for headless/SSH COLMAP runs. Set to an empty "
            "string to inherit the current environment. Default: offscreen."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove this run's generated database/model/export directories first.",
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


def require_tool(command: str) -> str:
    if Path(command).exists():
        return command
    path = shutil.which(command)
    if path is None:
        fail(
            f"'{command}' was not found on PATH. Activate rkv-mapanything and "
            "install COLMAP, e.g. `conda install -c conda-forge colmap`."
        )
    return path


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def colmap_environment(qt_qpa_platform: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if qt_qpa_platform:
        env["QT_QPA_PLATFORM"] = qt_qpa_platform
    return env


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
            "matcher": args.matcher,
            "sequential_overlap": args.sequential_overlap,
            "loop_detection": args.loop_detection,
            "mapper_min_num_matches": args.mapper_min_num_matches,
            "qt_qpa_platform": args.qt_qpa_platform,
        },
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
    if args.sequential_overlap < 1:
        fail("--sequential-overlap must be positive")
    if args.mapper_min_num_matches < 1:
        fail("--mapper-min-num-matches must be positive")
    if args.run_name is None:
        args.run_name = args.image_dir.resolve().name
    return images


def main() -> None:
    args = parse_args()
    images = validate_args(args)
    colmap = require_tool(args.colmap)

    output_root = args.output_root
    run_name = args.run_name
    database_dir = output_root / "databases" / run_name
    database_path = database_dir / "database.db"
    sparse_dir = output_root / "sparse" / run_name
    visual_dir = output_root / "visualizations" / run_name
    log_dir = output_root / "logs" / run_name
    export_dir = output_root / "exports" / run_name
    processed_export_dir = args.export_root / run_name

    if args.overwrite:
        for path, allowed_root in (
            (database_dir, output_root),
            (sparse_dir, output_root),
            (visual_dir, output_root),
            (log_dir, output_root),
            (export_dir, output_root),
            (processed_export_dir, args.export_root),
        ):
            safe_remove(path, allowed_root)

    for path in (database_dir, sparse_dir, visual_dir, log_dir, export_dir, processed_export_dir):
        path.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    env = colmap_environment(args.qt_qpa_platform)

    def run(command: list[str], log_name: str) -> str:
        commands.append(" ".join(command))
        return run_command(command, log_dir / log_name, env)

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
    write_run_summary(log_dir / "run_summary.json", args, len(images), paths, metrics, commands)

    print("\nCOLMAP pipeline complete")
    print(f"registered images: {metrics.get('registered_images', 'unknown')} / {len(images)}")
    print(f"mean reprojection error: {metrics.get('mean_reprojection_error_px', 'unknown')} px")
    print(f"exported TXT model: {processed_export_dir}")
    print(f"visualizations: {visual_dir}")


if __name__ == "__main__":
    main()
