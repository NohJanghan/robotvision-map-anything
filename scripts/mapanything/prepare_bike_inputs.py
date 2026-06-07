#!/usr/bin/env python3
"""Prepare an RGB+AR/VIO capture for COLMAP and MapAnything.

The historical filename is kept for compatibility with the first bike run. The
script itself is scene-agnostic: it extracts a manageable RGB frame subset while
preserving original frame names, then writes an AR pose JSON that
run_pipline.py can consume for Config D.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RGB frames and AR/VIO poses.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Capture directory containing rgb.mp4 and odometry.csv.",
    )
    parser.add_argument(
        "--scene-name",
        help="Scene/run name. Default: the input directory name.",
    )
    parser.add_argument(
        "--pose-csv",
        type=Path,
        help="AR/VIO pose CSV. Default: <input-dir>/odometry.csv.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        help="Output image directory. Default: data/raw/rgb_sequences/<scene-name>.",
    )
    parser.add_argument(
        "--ar-output-file",
        type=Path,
        help="Output AR pose JSON. Default: data/raw/ar_captures/<scene-name>/poses.json.",
    )
    parser.add_argument(
        "--pose-convention",
        default="arkit_cam2world",
        help="Pose convention written in the JSON. Default: arkit_cam2world.",
    )
    parser.add_argument("--stride", type=int, default=10, help="Keep every Nth source frame.")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


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


def load_pose_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        fail(f"pose CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows: dict[str, dict[str, str]] = {}
        for raw_row in csv.DictReader(fp):
            row = {
                key.strip(): value.strip()
                for key, value in raw_row.items()
                if key is not None and value is not None
            }
            if "frame" not in row:
                fail(f"pose CSV is missing a frame column: {path}")
            rows[row["frame"]] = row
        return rows


def load_camera_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    matrix = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if matrix.shape != (3, 3):
        fail(f"camera matrix must be 3x3: {path}")
    return matrix


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0:
        fail("zero-norm quaternion in pose CSV")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float32,
    )


def has_row_intrinsics(row: dict[str, str]) -> bool:
    return all(row.get(key, "").strip() for key in ("fx", "fy", "cx", "cy"))


def scaled_intrinsics(
    row: dict[str, str],
    static_intrinsics: np.ndarray | None,
    scale: float,
) -> list[list[float]]:
    if has_row_intrinsics(row):
        fx = float(row["fx"]) * scale
        fy = float(row["fy"]) * scale
        cx = float(row["cx"]) * scale
        cy = float(row["cy"]) * scale
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    if static_intrinsics is None:
        fail("pose CSV has no fx/fy/cx/cy columns and camera_matrix.csv is missing")
    scaled = static_intrinsics.astype(np.float32).copy()
    scaled[0, :] *= scale
    scaled[1, :] *= scale
    return scaled.astype(float).tolist()


def pose_from_row(row: dict[str, str]) -> list[list[float]]:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = quaternion_to_rotation_matrix(
        float(row["qx"]),
        float(row["qy"]),
        float(row["qz"]),
        float(row["qw"]),
    )
    pose[:3, 3] = [float(row["x"]), float(row["y"]), float(row["z"])]
    return pose.astype(float).tolist()


def resize_frame(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_side) / float(max(width, height)))
    if scale >= 1.0:
        return frame, 1.0
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale


def validate_args(args: argparse.Namespace) -> None:
    if args.stride < 1:
        fail("--stride must be positive")
    if args.max_frames < 1:
        fail("--max-frames must be positive")
    if args.max_side < 64:
        fail("--max-side must be at least 64")
    if not 1 <= args.jpeg_quality <= 100:
        fail("--jpeg-quality must be between 1 and 100")
    if not (args.input_dir / "rgb.mp4").exists():
        fail(f"rgb.mp4 is missing under {args.input_dir}")


def resolve_scene_name(args: argparse.Namespace) -> str:
    if args.scene_name:
        return args.scene_name
    return args.input_dir.name


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    scene_name = resolve_scene_name(args)
    pose_csv = args.pose_csv or args.input_dir / "odometry.csv"
    image_output_dir = args.image_output_dir or Path("data/raw/rgb_sequences") / scene_name
    ar_output_file = args.ar_output_file or Path("data/raw/ar_captures") / scene_name / "poses.json"
    return pose_csv, image_output_dir, ar_output_file


def main() -> None:
    args = parse_args()
    validate_args(args)
    pose_csv, image_output_dir, ar_output_file = resolve_outputs(args)

    if args.overwrite:
        safe_remove(image_output_dir, Path("data/raw/rgb_sequences"))
        safe_remove(ar_output_file.parent, Path("data/raw/ar_captures"))

    image_output_dir.mkdir(parents=True, exist_ok=True)
    ar_output_file.parent.mkdir(parents=True, exist_ok=True)

    pose_rows = load_pose_rows(pose_csv)
    static_intrinsics = load_camera_matrix(args.input_dir / "camera_matrix.csv")
    capture = cv2.VideoCapture(str(args.input_dir / "rgb.mp4"))
    if not capture.isOpened():
        fail(f"failed to open video: {args.input_dir / 'rgb.mp4'}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[dict[str, Any]] = []
    frame_index = 0
    while len(frames) < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % args.stride != 0:
            frame_index += 1
            continue
        frame_name = f"{frame_index:06d}.jpg"
        odom_key = f"{frame_index:06d}"
        row = pose_rows.get(odom_key)
        if row is None:
            fail(f"pose row is missing for frame {odom_key}")
        resized, scale = resize_frame(frame, args.max_side)
        output_path = image_output_dir / frame_name
        cv2.imwrite(
            str(output_path),
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        frames.append(
            {
                "image": frame_name,
                "source_frame": odom_key,
                "timestamp": float(row["timestamp"]),
                "cam2world": pose_from_row(row),
                "intrinsics": scaled_intrinsics(row, static_intrinsics, scale),
                "pose_convention": args.pose_convention,
            }
        )
        frame_index += 1
    capture.release()

    if not frames:
        fail("no frames were extracted")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str((args.input_dir / "rgb.mp4").resolve()),
        "source_pose_csv": str(pose_csv.resolve()),
        "source_camera_matrix": (
            str((args.input_dir / "camera_matrix.csv").resolve())
            if (args.input_dir / "camera_matrix.csv").exists()
            else None
        ),
        "image_output_dir": str(image_output_dir.resolve()),
        "ar_output_file": str(ar_output_file.resolve()),
        "source_video_size": [source_width, source_height],
        "source_fps": source_fps,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "max_side": args.max_side,
        "extracted_frames": [frame["image"] for frame in frames],
    }
    (image_output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ar_output_file.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": str(pose_csv),
                "pose_convention": args.pose_convention,
                "frames": frames,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"extracted frames: {len(frames)}")
    print(f"image dir: {image_output_dir}")
    print(f"AR poses: {ar_output_file}")


if __name__ == "__main__":
    main()
