#!/usr/bin/env python3
"""Run the Task 2 MapAnything pipeline from a JSON configuration.

The filename intentionally keeps the requested spelling: run_pipline.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mapanything" / "task2_pipeline.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GENERATED_MARKER = ".task2_mapanything_generated"
SUMMARY_FIELDS = (
    "render_psnr_db",
    "render_ssim",
    "render_coverage",
    "pose_ate_rmse",
    "relative_rotation_error_deg",
)
COMPARISON_FIELDS = (
    "render_psnr_db",
    "render_ssim",
    "render_coverage",
    "pose_ate_rmse",
    "relative_rotation_error_deg",
    "runtime_seconds",
)
TASK_COMPARISON_SPECS = (
    {
        "id": "a_vs_b",
        "name": "A vs B",
        "baseline": "config_a",
        "candidate": "config_b",
        "question": "Calibration effect",
    },
    {
        "id": "b_vs_c",
        "name": "B vs C",
        "baseline": "config_b",
        "candidate": "config_c",
        "question": "Pose input effect",
    },
    {
        "id": "c_vs_d",
        "name": "C vs D",
        "baseline": "config_c",
        "candidate": "config_d",
        "question": "COLMAP pose vs AR pose",
    },
)


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]
    intrinsics: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    cam2world: np.ndarray


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, Camera]
    images_by_name: dict[str, ColmapImage]
    registered_names: list[str]


@dataclass(frozen=True)
class PoseTable:
    poses_by_name: dict[str, np.ndarray]
    intrinsics_by_name: dict[str, np.ndarray]


@dataclass(frozen=True)
class ViewSpec:
    image_path: Path
    image_name: str
    intrinsics: np.ndarray | None
    camera_pose: np.ndarray | None
    pose_metric_scale: bool


@dataclass(frozen=True)
class PreparedConfig:
    entry: dict[str, Any]
    view_specs: list[ViewSpec]
    holdout_view_specs: list[ViewSpec]
    manifest_path: Path
    output_dir: Path
    stats: dict[str, Any]


@dataclass(frozen=True)
class PredictionData:
    image_name: str
    image_float: np.ndarray
    image_uint8: np.ndarray
    depth_z: np.ndarray
    intrinsics: np.ndarray
    camera_pose: np.ndarray
    mask: np.ndarray
    world_points: np.ndarray


@dataclass(frozen=True)
class RenderView:
    image_name: str
    image_uint8: np.ndarray
    intrinsics: np.ndarray
    camera_pose: np.ndarray
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 2 MapAnything configs from configs/mapanything/*.json."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Pipeline JSON path. Default: {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)}",
    )
    parser.add_argument("--run-name", help="Override run_name from the JSON config.")
    parser.add_argument("--image-dir", type=Path, help="Override image_dir from the JSON config.")
    parser.add_argument(
        "--colmap-export-dir",
        type=Path,
        help="Override colmap_export_dir from the JSON config.",
    )
    parser.add_argument(
        "--ar-pose-file",
        type=Path,
        help="Override ar_pose_file from the JSON config.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Override input_root from the JSON config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override output_root from the JSON config.",
    )
    parser.add_argument(
        "--model-name",
        help="Override model.name from the JSON config.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before loading MapAnything if CUDA is not available.",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use facebook/map-anything-apache regardless of the JSON model setting.",
    )
    parser.add_argument(
        "--view-start",
        type=int,
        help="Override view_selection.start from the JSON config.",
    )
    parser.add_argument(
        "--view-stride",
        type=int,
        help="Override view_selection.stride from the JSON config.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Override view_selection.max_images from the JSON config.",
    )
    parser.add_argument(
        "--eval-holdout",
        action="store_true",
        help="Enable evaluation.holdout rendering against views that were not used for inference.",
    )
    parser.add_argument(
        "--eval-holdout-start",
        type=int,
        help="Override evaluation.holdout.start.",
    )
    parser.add_argument(
        "--eval-holdout-stride",
        type=int,
        help="Override evaluation.holdout.stride.",
    )
    parser.add_argument(
        "--eval-holdout-max-images",
        type=int,
        help="Override evaluation.holdout.max_images.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="CONFIG",
        help="Run only selected config ids/names. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="CONFIG",
        help="Skip selected config ids/names. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print configs found in the JSON and exit.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate inputs and write manifests without loading MapAnything.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --prepare-only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove generated Task 2 output directories for selected configs first.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next config if one config fails.",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"config JSON does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"failed to parse JSON config {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"top-level config must be a JSON object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_path(value: str | Path | None, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def path_string(value: str | Path | None) -> str | None:
    path = resolve_path(value)
    return None if path is None else str(path)


def split_names(values: list[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                names.add(item)
    return names


def safe_remove(path: Path, allowed_root: Path, require_marker: bool = False) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        fail(f"refusing to remove path outside allowed root: {path}")
    if require_marker and path.is_dir() and not (path / GENERATED_MARKER).exists():
        fail(
            f"refusing to remove unmarked output directory: {path}. "
            f"Expected generated marker file '{GENERATED_MARKER}'."
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def mark_generated_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    marker = path / GENERATED_MARKER
    if not marker.exists():
        marker.write_text(f"generated_by=scripts/mapanything/run_pipline.py\ncreated_at={now_utc()}\n", encoding="utf-8")


def list_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.exists():
        fail(f"image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        fail(f"image path is not a directory: {image_dir}")
    images = {
        path.name: path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not images:
        fail(f"no RGB images found in {image_dir}")
    return images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qw, qx, qy, qz = (qvec / norm).astype(np.float64)
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


def invert_pose(pose: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float32)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def colmap_camera_matrix(model: str, params: list[float]) -> np.ndarray:
    model = model.upper()
    if model in {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
    }:
        if len(params) < 3:
            fail(f"camera model {model} needs at least 3 params")
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE",
    }:
        if len(params) < 4:
            fail(f"camera model {model} needs at least 4 params")
        fx, fy, cx, cy = params[:4]
    elif len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif len(params) >= 3:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fail(f"unsupported camera model with too few params: {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def parse_cameras_txt(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    if not path.exists():
        return cameras
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(value) for value in parts[4:]]
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
            intrinsics=colmap_camera_matrix(model, params),
        )
    return cameras


def parse_images_txt(path: Path) -> tuple[dict[str, ColmapImage], list[str]]:
    images: dict[str, ColmapImage] = {}
    order: list[str] = []
    if not path.exists():
        return images, order
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float32)
        tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float32)
        camera_id = int(parts[8])
        name = " ".join(parts[9:])

        world2cam = np.eye(4, dtype=np.float32)
        world2cam[:3, :3] = qvec_to_rotmat(qvec)
        world2cam[:3, 3] = tvec
        cam2world = invert_pose(world2cam)
        images[name] = ColmapImage(
            image_id=image_id,
            name=name,
            camera_id=camera_id,
            qvec=qvec,
            tvec=tvec,
            cam2world=cam2world,
        )
        order.append(name)
        # COLMAP image records are two lines; the point line can be blank.
        if index < len(lines):
            index += 1
    return images, order


def load_colmap_model(colmap_export_dir: Path | None) -> ColmapModel | None:
    if colmap_export_dir is None or not colmap_export_dir.exists():
        return None
    cameras = parse_cameras_txt(colmap_export_dir / "cameras.txt")
    images_by_name, registered_names = parse_images_txt(colmap_export_dir / "images.txt")
    if not cameras and not images_by_name:
        return None
    return ColmapModel(
        cameras=cameras,
        images_by_name=images_by_name,
        registered_names=registered_names,
    )


def setup_warnings(
    colmap_export_dir: Path | None,
    colmap_model: ColmapModel | None,
    ar_pose_file: Path | None,
    ar_table: PoseTable | None,
) -> list[str]:
    warnings: list[str] = []
    if colmap_export_dir is None or not colmap_export_dir.exists():
        warnings.append("COLMAP export directory is missing; Config B/C cannot run until Task 1 exports are present.")
    else:
        for filename in ("cameras.txt", "images.txt"):
            if not (colmap_export_dir / filename).exists():
                warnings.append(f"COLMAP export file is missing: {colmap_export_dir / filename}")
    if colmap_model is not None:
        distorted_models = {
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
            "FULL_OPENCV",
            "FOV",
            "SIMPLE_RADIAL_FISHEYE",
            "THIN_PRISM_FISHEYE",
            "RAD_TAN_THIN_PRISM_FISHEYE",
        }
        distorted = sorted(
            {
                camera.model
                for camera in colmap_model.cameras.values()
                if camera.model.upper() in distorted_models
            }
        )
        if distorted:
            warnings.append(
                "COLMAP distortion parameters are ignored because MapAnything "
                f"expects pinhole intrinsics; undistort images first. Models: {', '.join(distorted)}"
            )
    if ar_pose_file is not None and ar_pose_file.exists() and ar_table is None:
        warnings.append(f"AR pose file exists but no usable frames were parsed: {ar_pose_file}")
    return warnings


def normalize_pose_matrix(matrix: Any, convention: str) -> np.ndarray:
    pose = np.array(matrix, dtype=np.float32)
    if pose.shape != (4, 4):
        fail(f"pose matrix must be 4x4, got {pose.shape}")

    convention = convention.lower()
    if convention in {"opencv_cam2world", "colmap_cam2world"}:
        return pose
    if convention in {"world2cam_opencv", "opencv_world2cam", "colmap_world2cam"}:
        return invert_pose(pose)
    if convention in {"arkit_cam2world", "arcore_cam2world", "opengl_cam2world"}:
        conversion = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        return pose @ conversion
    if convention in {"arkit_world2cam", "arcore_world2cam", "opengl_world2cam"}:
        conversion = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        return invert_pose(pose) @ conversion
    fail(
        "unsupported pose convention "
        f"'{convention}'. Expected opencv_cam2world, arkit_cam2world, "
        "opengl_cam2world, or a world2cam variant."
    )


def first_existing_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def parse_ar_pose_file(path: Path | None, default_convention: str) -> PoseTable | None:
    if path is None or not path.exists():
        return None
    data = read_json(path)
    frames: list[dict[str, Any]]
    if "frames" in data and isinstance(data["frames"], list):
        frames = data["frames"]
    elif "poses" in data and isinstance(data["poses"], list):
        frames = data["poses"]
    elif all(isinstance(value, list) for value in data.values()):
        frames = [{"image": key, "pose": value} for key, value in data.items()]
    else:
        fail(
            f"unsupported AR pose JSON format: {path}. Use frames/poses list or "
            "a mapping from image name to 4x4 pose."
        )

    poses: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    for frame in frames:
        name = first_existing_key(
            frame,
            ("image", "image_name", "file_name", "filename", "name", "file_path"),
        )
        matrix = first_existing_key(
            frame,
            ("cam2world", "camera_pose", "pose", "transform_matrix", "transform"),
        )
        if name is None or matrix is None:
            continue
        image_name = Path(str(name)).name
        convention = str(frame.get("pose_convention", default_convention))
        poses[image_name] = normalize_pose_matrix(matrix, convention)
        intrinsics_value = frame.get("intrinsics")
        if intrinsics_value is not None:
            k_matrix = np.array(intrinsics_value, dtype=np.float32)
            if k_matrix.shape == (3, 3):
                intrinsics[image_name] = k_matrix
    return PoseTable(poses_by_name=poses, intrinsics_by_name=intrinsics)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.run_name:
        config["run_name"] = args.run_name
    if args.image_dir:
        config["image_dir"] = str(args.image_dir)
    if args.colmap_export_dir:
        config["colmap_export_dir"] = str(args.colmap_export_dir)
    if args.ar_pose_file:
        config["ar_pose_file"] = str(args.ar_pose_file)
    if args.input_root:
        config["input_root"] = str(args.input_root)
    if args.output_root:
        config["output_root"] = str(args.output_root)
    if args.model_name:
        config.setdefault("model", {})["name"] = args.model_name
        config.setdefault("model", {})["apache"] = False
    if args.require_gpu:
        config.setdefault("model", {})["require_gpu"] = True
    if args.apache:
        config.setdefault("model", {})["name"] = "facebook/map-anything-apache"
        config.setdefault("model", {})["apache"] = True
    if args.view_start is not None:
        config.setdefault("view_selection", {})["start"] = args.view_start
    if args.view_stride is not None:
        config.setdefault("view_selection", {})["stride"] = args.view_stride
    if args.max_images is not None:
        config.setdefault("view_selection", {})["max_images"] = args.max_images
    if (
        args.eval_holdout
        or args.eval_holdout_start is not None
        or args.eval_holdout_stride is not None
        or args.eval_holdout_max_images is not None
    ):
        holdout_overrides = config.setdefault("evaluation", {}).setdefault("holdout", {})
        if args.eval_holdout:
            holdout_overrides["enabled"] = True
        if args.eval_holdout_start is not None:
            holdout_overrides["start"] = args.eval_holdout_start
        if args.eval_holdout_stride is not None:
            holdout_overrides["stride"] = args.eval_holdout_stride
        if args.eval_holdout_max_images is not None:
            holdout_overrides["max_images"] = args.eval_holdout_max_images
    if args.continue_on_error:
        config["continue_on_error"] = True


def selected_entries(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    entries = config.get("configs")
    if not isinstance(entries, list) or not entries:
        fail("config JSON must contain a non-empty 'configs' list")

    only = split_names(args.only)
    skip = split_names(args.skip)
    selected: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            fail("each item in 'configs' must be an object")
        config_id = str(entry.get("id", ""))
        config_name = str(entry.get("name", config_id))
        names = {config_id, config_name}
        if only and not names.intersection(only):
            continue
        if names.intersection(skip):
            continue
        if not only and entry.get("enabled", True) is False:
            continue
        selected.append(entry)
    if not selected:
        fail("no configs selected")
    return selected


def print_configs(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        config_id = entry.get("id", entry.get("name", "unnamed"))
        name = entry.get("name", config_id)
        label = entry.get("label", "")
        status = "enabled" if entry.get("enabled", True) else "disabled"
        print(f"{config_id}: {name} [{status}] {label}")


def select_image_names(
    image_paths: dict[str, Path],
    colmap_model: ColmapModel | None,
    view_selection: dict[str, Any],
    entry: dict[str, Any],
) -> list[str]:
    mode = str(entry.get("image_selection", view_selection.get("mode", "registered_or_all")))
    if mode == "registered":
        if colmap_model is None or not colmap_model.registered_names:
            fail("image_selection='registered' requires COLMAP images.txt")
        names = [name for name in colmap_model.registered_names if name in image_paths]
    elif mode == "registered_or_all":
        if colmap_model is not None and colmap_model.registered_names:
            names = [name for name in colmap_model.registered_names if name in image_paths]
        else:
            names = sorted(image_paths)
    elif mode == "all":
        names = sorted(image_paths)
    else:
        fail(f"unsupported image selection mode: {mode}")

    start = int(entry.get("start", view_selection.get("start", 0)) or 0)
    stride = int(entry.get("stride", view_selection.get("stride", 1)) or 1)
    max_images = entry.get("max_images", view_selection.get("max_images"))
    if stride < 1:
        fail("view_selection.stride must be positive")
    names = names[start::stride]
    if max_images is not None:
        names = names[: int(max_images)]
    return names


def colmap_intrinsics_for_name(name: str, colmap_model: ColmapModel | None) -> np.ndarray | None:
    if colmap_model is None or not colmap_model.cameras:
        return None
    image = colmap_model.images_by_name.get(name)
    if image is not None:
        camera = colmap_model.cameras.get(image.camera_id)
        if camera is not None:
            return camera.intrinsics.copy()
    if len(colmap_model.cameras) == 1:
        return next(iter(colmap_model.cameras.values())).intrinsics.copy()
    return None


def intrinsics_for_name(
    name: str,
    source: str,
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
) -> np.ndarray | None:
    source = source.lower()
    if source == "colmap":
        return colmap_intrinsics_for_name(name, colmap_model)
    if source == "ar":
        if ar_table is None:
            return None
        return ar_table.intrinsics_by_name.get(name)
    if source == "ar_or_colmap":
        if ar_table is not None and name in ar_table.intrinsics_by_name:
            return ar_table.intrinsics_by_name[name]
        return colmap_intrinsics_for_name(name, colmap_model)
    if source == "colmap_or_ar":
        intrinsics = colmap_intrinsics_for_name(name, colmap_model)
        if intrinsics is not None:
            return intrinsics
        if ar_table is None:
            return None
        return ar_table.intrinsics_by_name.get(name)
    fail(f"unsupported intrinsics source: {source}")


def pose_for_name(
    name: str,
    source: str,
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
) -> np.ndarray | None:
    source = source.lower()
    if source in {"none", "", "null"}:
        return None
    if source == "colmap":
        if colmap_model is None:
            return None
        image = colmap_model.images_by_name.get(name)
        return None if image is None else image.cam2world.copy()
    if source == "ar":
        if ar_table is None:
            return None
        pose = ar_table.poses_by_name.get(name)
        return None if pose is None else pose.copy()
    fail(f"unsupported pose source: {source}")


def prepare_holdout_view_specs(
    config: dict[str, Any],
    image_paths: dict[str, Path],
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
    train_names: set[str],
) -> tuple[list[ViewSpec], dict[str, Any]]:
    evaluation_config = dict(config.get("evaluation", {}))
    holdout_config = dict(evaluation_config.get("holdout", {}))
    if not evaluation_config.get("enabled", True) or not holdout_config.get("enabled", False):
        return [], {"enabled": False}

    pose_source = str(
        holdout_config.get("pose_source", evaluation_config.get("reference_pose_source", "colmap"))
    )
    intrinsics_source = str(
        holdout_config.get(
            "intrinsics_source",
            evaluation_config.get("reference_intrinsics_source", "colmap"),
        )
    )
    selected_names = select_image_names(
        image_paths=image_paths,
        colmap_model=colmap_model,
        view_selection=holdout_config,
        entry={},
    )
    if holdout_config.get("exclude_input_views", True):
        selected_names = [name for name in selected_names if name not in train_names]

    skipped_without_required_data: list[str] = []
    view_specs: list[ViewSpec] = []
    for name in selected_names:
        intrinsics = intrinsics_for_name(name, intrinsics_source, colmap_model, ar_table)
        camera_pose = pose_for_name(name, pose_source, colmap_model, ar_table)
        if intrinsics is None or camera_pose is None:
            skipped_without_required_data.append(name)
            continue
        view_specs.append(
            ViewSpec(
                image_path=image_paths[name],
                image_name=name,
                intrinsics=intrinsics,
                camera_pose=camera_pose,
                pose_metric_scale=False,
            )
        )

    return view_specs, {
        "enabled": True,
        "selected_images": len(selected_names),
        "loaded_views": len(view_specs),
        "skipped_without_required_data": skipped_without_required_data,
        "pose_source": pose_source,
        "intrinsics_source": intrinsics_source,
        "exclude_input_views": bool(holdout_config.get("exclude_input_views", True)),
    }


def prepare_config(
    config: dict[str, Any],
    entry: dict[str, Any],
    image_paths: dict[str, Path],
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
) -> PreparedConfig | dict[str, Any]:
    config_id = str(entry.get("id", entry.get("name", "config")))
    config_name = str(entry.get("name", config_id))
    run_name = str(config.get("run_name", "scene"))
    output_dir = output_root / run_name / config_name
    manifest_path = input_root / run_name / config_id / "manifest.json"

    if overwrite:
        safe_remove(output_dir, output_root, require_marker=True)
        safe_remove(manifest_path.parent, input_root)

    use_intrinsics = bool(entry.get("use_intrinsics", False))
    intrinsics_source = str(entry.get("intrinsics_source", "colmap"))
    pose_source = str(entry.get("pose_source", "none")).lower()
    pose_metric_scale = bool(entry.get("pose_metric_scale", pose_source == "ar"))
    skip_if_missing = bool(entry.get("skip_if_missing_inputs", False))
    selected_names = select_image_names(
        image_paths=image_paths,
        colmap_model=colmap_model,
        view_selection=dict(config.get("view_selection", {})),
        entry=entry,
    )

    missing_inputs: list[str] = []
    skipped_without_required_data: list[str] = []
    view_specs: list[ViewSpec] = []
    for name in selected_names:
        intrinsics = None
        if use_intrinsics:
            intrinsics = intrinsics_for_name(name, intrinsics_source, colmap_model, ar_table)
            if intrinsics is None:
                skipped_without_required_data.append(name)
                continue

        camera_pose = None
        if pose_source not in {"none", "", "null"}:
            camera_pose = pose_for_name(name, pose_source, colmap_model, ar_table)
            if camera_pose is None:
                skipped_without_required_data.append(name)
                continue

        view_specs.append(
            ViewSpec(
                image_path=image_paths[name],
                image_name=name,
                intrinsics=intrinsics,
                camera_pose=camera_pose,
                pose_metric_scale=pose_metric_scale,
            )
        )

    if use_intrinsics and not view_specs:
        missing_inputs.append(f"intrinsics_source={intrinsics_source}")
    if pose_source not in {"none", "", "null"} and not view_specs:
        missing_inputs.append(f"pose_source={pose_source}")
    if not view_specs:
        message = (
            f"{config_id} has no runnable views after input filtering "
            f"(missing: {', '.join(missing_inputs) or 'required data'})"
        )
        if skip_if_missing:
            return {
                "config": config_id,
                "name": config_name,
                "label": entry.get("label", ""),
                "status": "skipped",
                "reason": message,
            }
        if config.get("continue_on_error", False):
            return {
                "config": config_id,
                "name": config_name,
                "label": entry.get("label", ""),
                "status": "failed",
                "reason": message,
            }
        fail(message)

    holdout_view_specs, holdout_stats = prepare_holdout_view_specs(
        config=config,
        image_paths=image_paths,
        colmap_model=colmap_model,
        ar_table=ar_table,
        train_names={spec.image_name for spec in view_specs},
    )
    mark_generated_dir(output_dir)
    stats = {
        "selected_images": len(selected_names),
        "loaded_views": len(view_specs),
        "skipped_without_required_data": skipped_without_required_data,
        "with_intrinsics": sum(1 for spec in view_specs if spec.intrinsics is not None),
        "with_poses": sum(1 for spec in view_specs if spec.camera_pose is not None),
        "holdout": holdout_stats,
    }
    write_input_manifest(
        manifest_path=manifest_path,
        config=config,
        entry=entry,
        view_specs=view_specs,
        holdout_view_specs=holdout_view_specs,
        stats=stats,
    )
    return PreparedConfig(
        entry=entry,
        view_specs=view_specs,
        holdout_view_specs=holdout_view_specs,
        manifest_path=manifest_path,
        output_dir=output_dir,
        stats=stats,
    )


def write_input_manifest(
    manifest_path: Path,
    config: dict[str, Any],
    entry: dict[str, Any],
    view_specs: list[ViewSpec],
    holdout_view_specs: list[ViewSpec],
    stats: dict[str, Any],
) -> None:
    def matrix_or_none(matrix: np.ndarray | None) -> list[list[float]] | None:
        return None if matrix is None else matrix.astype(float).tolist()

    def view_payload(spec: ViewSpec) -> dict[str, Any]:
        return {
            "image_name": spec.image_name,
            "image_path": str(spec.image_path.resolve()),
            "intrinsics": matrix_or_none(spec.intrinsics),
            "camera_pose": matrix_or_none(spec.camera_pose),
            "pose_metric_scale": spec.pose_metric_scale,
        }

    manifest = {
        "created_at": now_utc(),
        "task": "Task 2 MapAnything",
        "run_name": config.get("run_name"),
        "config": entry.get("id"),
        "name": entry.get("name"),
        "label": entry.get("label"),
        "description": entry.get("description"),
        "image_dir": path_string(config.get("image_dir")),
        "colmap_export_dir": path_string(config.get("colmap_export_dir")),
        "ar_pose_file": path_string(config.get("ar_pose_file")),
        "uses_intrinsics": bool(entry.get("use_intrinsics", False)),
        "intrinsics_source": entry.get("intrinsics_source"),
        "pose_source": entry.get("pose_source", "none"),
        "pose_metric_scale": bool(entry.get("pose_metric_scale", False)),
        "input_stats": stats,
        "views": [view_payload(spec) for spec in view_specs],
        "holdout_views": [view_payload(spec) for spec in holdout_view_specs],
    }
    write_json(manifest_path, manifest)


def load_runtime() -> dict[str, Any]:
    third_party = PROJECT_ROOT / "third_party" / "map-anything"
    if third_party.exists() and str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from PIL import Image
        from mapanything.models import MapAnything
        from mapanything.utils.colmap_export import export_predictions_to_colmap
        from mapanything.utils.device import get_device
        from mapanything.utils.geometry import depthmap_to_world_frame
        from mapanything.utils.image import preprocess_inputs, rgb
        from mapanything.utils.viz import predictions_to_glb
    except ImportError as exc:
        fail(
            "failed to import MapAnything runtime dependencies. Activate "
            "rkv-mapanything and install third_party/map-anything with `pip install -e .`. "
            f"Original error: {exc}"
        )
    return {
        "torch": torch,
        "Image": Image,
        "MapAnything": MapAnything,
        "export_predictions_to_colmap": export_predictions_to_colmap,
        "get_device": get_device,
        "depthmap_to_world_frame": depthmap_to_world_frame,
        "preprocess_inputs": preprocess_inputs,
        "rgb": rgb,
        "predictions_to_glb": predictions_to_glb,
    }


def load_model(runtime: dict[str, Any], model_config: dict[str, Any]) -> tuple[Any, Any, str]:
    torch = runtime["torch"]
    if model_config.get("require_gpu", False) and not torch.cuda.is_available():
        fail("CUDA is not available, but --require-gpu/model.require_gpu was set.")
    device = runtime["get_device"]()
    model_name = str(model_config.get("name", "facebook/map-anything"))
    if model_config.get("apache", False):
        model_name = "facebook/map-anything-apache"
    print(f"Loading MapAnything model: {model_name}")
    print(f"Using device: {device}")
    model = runtime["MapAnything"].from_pretrained(model_name).to(device)
    model.eval()
    return model, device, model_name


def materialize_views(runtime: dict[str, Any], view_specs: list[ViewSpec]) -> list[dict[str, Any]]:
    torch = runtime["torch"]
    Image = runtime["Image"]
    views: list[dict[str, Any]] = []
    for spec in view_specs:
        image = Image.open(spec.image_path).convert("RGB")
        image_array = np.array(image).astype(np.uint8)
        view: dict[str, Any] = {"img": torch.from_numpy(image_array)}
        if spec.intrinsics is not None:
            view["intrinsics"] = torch.from_numpy(spec.intrinsics.astype(np.float32))
        if spec.camera_pose is not None:
            view["camera_poses"] = torch.from_numpy(spec.camera_pose.astype(np.float32))
            view["is_metric_scale"] = torch.tensor([spec.pose_metric_scale])
        views.append(view)
    return views


def preprocess_kwargs(config: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config.get("preprocess", {}))
    kwargs.update(entry.get("preprocess", {}))
    return kwargs


def inference_kwargs(config: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config.get("inference", {}))
    kwargs.update(entry.get("inference", {}))
    pose_source = str(entry.get("pose_source", "none")).lower()
    use_intrinsics = bool(entry.get("use_intrinsics", False))
    pose_metric_scale = bool(entry.get("pose_metric_scale", pose_source == "ar"))
    kwargs.setdefault("ignore_calibration_inputs", not use_intrinsics)
    kwargs.setdefault("ignore_depth_inputs", True)
    kwargs.setdefault("ignore_pose_inputs", pose_source in {"none", "", "null"})
    kwargs.setdefault("ignore_depth_scale_inputs", True)
    kwargs.setdefault("ignore_pose_scale_inputs", not pose_metric_scale)
    return kwargs


def run_inference(
    runtime: dict[str, Any],
    model: Any,
    device: Any,
    prepared: PreparedConfig,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    torch = runtime["torch"]
    raw_views = materialize_views(runtime, prepared.view_specs)
    print(f"Preprocessing {len(raw_views)} views for {prepared.entry.get('id')}")
    processed_views = runtime["preprocess_inputs"](
        raw_views,
        **preprocess_kwargs(config, prepared.entry),
    )
    kwargs = inference_kwargs(config, prepared.entry)
    print(f"Running MapAnything inference for {prepared.entry.get('id')}")
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.infer(processed_views, **kwargs)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - start
    return outputs, processed_views, runtime_seconds


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.array(value)


def processed_intrinsics_table(
    processed_views: list[dict[str, Any]],
    view_specs: list[ViewSpec],
) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for processed_view, spec in zip(processed_views, view_specs):
        intrinsics = processed_view.get("intrinsics")
        if intrinsics is None:
            continue
        intrinsics_np = tensor_to_numpy(intrinsics)
        if intrinsics_np.ndim == 3:
            intrinsics_np = intrinsics_np[0]
        if intrinsics_np.shape == (3, 3):
            table[spec.image_name] = intrinsics_np.astype(np.float32)
    return table


def ensure_output_masks(runtime: dict[str, Any], outputs: list[dict[str, Any]]) -> None:
    torch = runtime["torch"]
    for pred in outputs:
        if "mask" in pred:
            continue
        if "depth_z" not in pred:
            continue
        pred["mask"] = torch.isfinite(pred["depth_z"]) & (pred["depth_z"] > 0)


def extract_predictions(
    runtime: dict[str, Any],
    outputs: list[dict[str, Any]],
    view_specs: list[ViewSpec],
) -> list[PredictionData]:
    predictions: list[PredictionData] = []
    for index, pred in enumerate(outputs):
        depth_t = pred["depth_z"][0].squeeze(-1)
        intrinsics_t = pred["intrinsics"][0]
        pose_t = pred["camera_poses"][0]
        pts3d_t, valid_depth_t = runtime["depthmap_to_world_frame"](
            depth_t,
            intrinsics_t,
            pose_t,
        )
        if "mask" in pred:
            mask = tensor_to_numpy(pred["mask"][0].squeeze(-1)).astype(bool)
        else:
            mask = np.ones_like(tensor_to_numpy(depth_t), dtype=bool)
        mask &= tensor_to_numpy(valid_depth_t).astype(bool)
        image_float = tensor_to_numpy(pred["img_no_norm"][0]).astype(np.float32)
        if image_float.max(initial=0) > 1.5:
            image_float = np.clip(image_float / 255.0, 0.0, 1.0)
        image_uint8 = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
        predictions.append(
            PredictionData(
                image_name=view_specs[index].image_name,
                image_float=image_float,
                image_uint8=image_uint8,
                depth_z=tensor_to_numpy(depth_t).astype(np.float32),
                intrinsics=tensor_to_numpy(intrinsics_t).astype(np.float32),
                camera_pose=tensor_to_numpy(pose_t).astype(np.float32),
                mask=mask,
                world_points=tensor_to_numpy(pts3d_t).astype(np.float32),
            )
        )
    return predictions


def processed_view_to_render_view(
    runtime: dict[str, Any],
    processed_view: dict[str, Any],
    spec: ViewSpec,
) -> RenderView:
    image_float = runtime["rgb"](
        processed_view["img"][0],
        processed_view.get("data_norm_type", ["dinov2"])[0],
    ).astype(np.float32)
    image_uint8 = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
    intrinsics = tensor_to_numpy(processed_view["intrinsics"][0]).astype(np.float32)
    camera_pose = tensor_to_numpy(processed_view["camera_poses"][0]).astype(np.float32)
    mask = np.ones(image_uint8.shape[:2], dtype=bool)
    return RenderView(
        image_name=spec.image_name,
        image_uint8=image_uint8,
        intrinsics=intrinsics,
        camera_pose=camera_pose,
        mask=mask,
    )


def materialize_render_targets(
    runtime: dict[str, Any],
    view_specs: list[ViewSpec],
    config: dict[str, Any],
    entry: dict[str, Any],
) -> list[RenderView]:
    if not view_specs:
        return []
    raw_views = materialize_views(runtime, view_specs)
    processed_views = runtime["preprocess_inputs"](
        raw_views,
        **preprocess_kwargs(config, entry),
    )
    return [
        processed_view_to_render_view(runtime, processed_view, spec)
        for processed_view, spec in zip(processed_views, view_specs)
    ]


def write_depth_png(path: Path, depth: np.ndarray, mask: np.ndarray, Image: Any) -> None:
    valid = depth[mask & np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(valid, [1, 99])
        if high <= low:
            high = low + 1e-6
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        normalized = (normalized * 255).astype(np.uint8)
        normalized[~mask] = 0
    Image.fromarray(normalized).save(path)


def write_prediction_files(
    runtime: dict[str, Any],
    predictions: list[PredictionData],
    prepared: PreparedConfig,
    export_config: dict[str, Any],
) -> dict[str, str]:
    Image = runtime["Image"]
    prediction_dir = prepared.output_dir / "predictions"
    depth_dir = prepared.output_dir / "depth"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if export_config.get("save_npz", True):
        for index, pred in enumerate(predictions):
            payload: dict[str, Any] = {
                "image_name": pred.image_name,
                "image_uint8": pred.image_uint8,
                "depth_z": pred.depth_z,
                "intrinsics": pred.intrinsics,
                "camera_pose": pred.camera_pose,
                "mask": pred.mask,
            }
            if export_config.get("save_dense_points_npz", False):
                payload["world_points"] = pred.world_points
            np.savez_compressed(prediction_dir / f"view_{index:04d}.npz", **payload)
        paths["predictions"] = str(prediction_dir.resolve())

    if export_config.get("save_depth_png", True):
        depth_dir.mkdir(parents=True, exist_ok=True)
        for index, pred in enumerate(predictions):
            write_depth_png(depth_dir / f"view_{index:04d}.png", pred.depth_z, pred.mask, Image)
        paths["depth"] = str(depth_dir.resolve())

    pose_data = {
        "created_at": now_utc(),
        "views": [
            {
                "image_name": pred.image_name,
                "intrinsics": pred.intrinsics.astype(float).tolist(),
                "camera_pose": pred.camera_pose.astype(float).tolist(),
            }
            for pred in predictions
        ],
    }
    pose_path = prepared.output_dir / "predicted_poses.json"
    write_json(pose_path, pose_data)
    paths["predicted_poses"] = str(pose_path.resolve())
    return paths


def write_point_cloud_ply(
    predictions: list[PredictionData],
    path: Path,
    stride: int,
    max_points: int | None,
) -> None:
    stride = max(1, int(stride))
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    for pred in predictions:
        points = pred.world_points[::stride, ::stride].reshape(-1, 3)
        colors = pred.image_uint8[::stride, ::stride].reshape(-1, 3)
        mask = pred.mask[::stride, ::stride].reshape(-1)
        finite = np.isfinite(points).all(axis=1)
        keep = mask & finite
        point_chunks.append(points[keep])
        color_chunks.append(colors[keep])

    if point_chunks:
        points_all = np.concatenate(point_chunks, axis=0)
        colors_all = np.concatenate(color_chunks, axis=0)
    else:
        points_all = np.empty((0, 3), dtype=np.float32)
        colors_all = np.empty((0, 3), dtype=np.uint8)

    if max_points is not None and len(points_all) > int(max_points):
        indices = np.linspace(0, len(points_all) - 1, int(max_points)).astype(np.int64)
        points_all = points_all[indices]
        colors_all = colors_all[indices]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {len(points_all)}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write("end_header\n")
        for point, color in zip(points_all, colors_all):
            fp.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def export_optional_artifacts(
    runtime: dict[str, Any],
    outputs: list[dict[str, Any]],
    processed_views: list[dict[str, Any]],
    predictions: list[PredictionData],
    prepared: PreparedConfig,
    config: dict[str, Any],
    model: Any,
) -> tuple[dict[str, str], list[str]]:
    export_config = dict(config.get("exports", {}))
    export_config.update(prepared.entry.get("exports", {}))
    paths = write_prediction_files(runtime, predictions, prepared, export_config)
    warnings: list[str] = []

    if export_config.get("save_point_cloud_ply", True):
        ply_path = prepared.output_dir / "reconstruction_points.ply"
        write_point_cloud_ply(
            predictions=predictions,
            path=ply_path,
            stride=int(export_config.get("point_cloud_stride", 4)),
            max_points=export_config.get("point_cloud_max_points"),
        )
        paths["point_cloud_ply"] = str(ply_path.resolve())

    if export_config.get("save_glb", True):
        try:
            glb_path = prepared.output_dir / "reconstruction.glb"
            scene = runtime["predictions_to_glb"](
                {
                    "world_points": np.stack([pred.world_points for pred in predictions], axis=0),
                    "images": np.stack([pred.image_float for pred in predictions], axis=0),
                    "final_masks": np.stack([pred.mask for pred in predictions], axis=0),
                },
                as_mesh=bool(export_config.get("glb_as_mesh", True)),
            )
            scene.export(glb_path)
            paths["glb"] = str(glb_path.resolve())
        except Exception as exc:  # noqa: BLE001 - keep long GPU runs from being wasted.
            message = f"GLB export failed: {exc}"
            warnings.append(message)
            if not export_config.get("continue_on_export_error", True):
                raise

    if export_config.get("save_colmap", False):
        try:
            colmap_output_dir = prepared.output_dir / "colmap_export"
            runtime["export_predictions_to_colmap"](
                outputs=outputs,
                processed_views=processed_views,
                image_names=[pred.image_name for pred in predictions],
                output_dir=str(colmap_output_dir),
                voxel_fraction=float(export_config.get("voxel_fraction", 0.01)),
                voxel_size=export_config.get("voxel_size"),
                data_norm_type=model.encoder.data_norm_type,
                save_ply=True,
                skip_point2d=bool(export_config.get("skip_point2d", True)),
            )
            paths["colmap_export"] = str(colmap_output_dir.resolve())
        except Exception as exc:  # noqa: BLE001
            message = f"COLMAP export failed: {exc}"
            warnings.append(message)
            if not export_config.get("continue_on_export_error", True):
                raise

    return paths, warnings


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    if len(source) != len(target) or len(source) == 0:
        return None
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    src_centered = source - source_mean
    tgt_centered = target - target_mean
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if variance <= 1e-12:
        return 1.0, np.eye(3, dtype=np.float32), target_mean - source_mean
    covariance = (src_centered.T @ tgt_centered) / len(source)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt_mat.T @ u_mat.T) < 0:
        correction[-1, -1] = -1
    rotation = vt_mat.T @ correction @ u_mat.T
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (source_mean @ rotation.T)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def apply_similarity_transform(
    points: np.ndarray,
    transform: tuple[float, np.ndarray, np.ndarray],
) -> np.ndarray:
    scale, rotation, translation = transform
    return (scale * (points @ rotation.T) + translation).astype(np.float32)


def similarity_align_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    transform = similarity_transform(source, target)
    if transform is None:
        return source
    return apply_similarity_transform(source, transform)


def transform_prediction_world(
    prediction: PredictionData,
    transform: tuple[float, np.ndarray, np.ndarray],
) -> PredictionData:
    scale, rotation, translation = transform
    camera_pose = prediction.camera_pose.copy()
    camera_pose[:3, :3] = rotation @ camera_pose[:3, :3]
    camera_pose[:3, 3] = scale * (camera_pose[:3, 3] @ rotation.T) + translation
    world_points = apply_similarity_transform(prediction.world_points, transform)
    return PredictionData(
        image_name=prediction.image_name,
        image_float=prediction.image_float,
        image_uint8=prediction.image_uint8,
        depth_z=prediction.depth_z,
        intrinsics=prediction.intrinsics,
        camera_pose=camera_pose.astype(np.float32),
        mask=prediction.mask,
        world_points=world_points,
    )


def align_predictions_to_reference(
    predictions: list[PredictionData],
    reference_poses: dict[str, np.ndarray],
) -> tuple[list[PredictionData], dict[str, Any]]:
    common = [pred for pred in predictions if pred.image_name in reference_poses]
    if len(common) < 2:
        return predictions, {
            "available": False,
            "common_views": len(common),
            "reason": "need at least two common train/reference poses",
        }
    pred_centers = np.stack([pred.camera_pose[:3, 3] for pred in common], axis=0)
    ref_centers = np.stack([reference_poses[pred.image_name][:3, 3] for pred in common], axis=0)
    transform = similarity_transform(pred_centers, ref_centers)
    if transform is None:
        return predictions, {
            "available": False,
            "common_views": len(common),
            "reason": "failed to estimate similarity transform",
        }
    aligned = [transform_prediction_world(pred, transform) for pred in predictions]
    scale, _, _ = transform
    return aligned, {
        "available": True,
        "common_views": len(common),
        "scale": scale,
    }


def compute_pose_metrics(
    predictions: list[PredictionData],
    reference_poses: dict[str, np.ndarray],
    direct_pose_comparison: bool,
) -> dict[str, Any]:
    common = [pred for pred in predictions if pred.image_name in reference_poses]
    if len(common) < 2:
        return {"available": False, "common_views": len(common)}

    pred_centers = np.stack([pred.camera_pose[:3, 3] for pred in common], axis=0)
    ref_centers = np.stack([reference_poses[pred.image_name][:3, 3] for pred in common], axis=0)
    aligned_centers = similarity_align_points(pred_centers, ref_centers)
    ate_errors = np.linalg.norm(aligned_centers - ref_centers, axis=1)

    relative_rotation_errors = []
    for prev, curr in zip(common[:-1], common[1:]):
        pred_rel = prev.camera_pose[:3, :3].T @ curr.camera_pose[:3, :3]
        ref_prev = reference_poses[prev.image_name]
        ref_curr = reference_poses[curr.image_name]
        ref_rel = ref_prev[:3, :3].T @ ref_curr[:3, :3]
        relative_rotation_errors.append(rotation_angle_deg(pred_rel @ ref_rel.T))

    metrics: dict[str, Any] = {
        "available": True,
        "common_views": len(common),
        "ate_rmse": float(np.sqrt(np.mean(ate_errors * ate_errors))),
        "ate_mean": float(np.mean(ate_errors)),
        "ate_median": float(np.median(ate_errors)),
        "relative_rotation_error_deg_mean": float(np.mean(relative_rotation_errors)),
        "relative_rotation_error_deg_median": float(np.median(relative_rotation_errors)),
    }
    if direct_pose_comparison:
        direct_rotation_errors = []
        direct_translation_errors = []
        for pred in common:
            ref = reference_poses[pred.image_name]
            direct_rotation_errors.append(rotation_angle_deg(pred.camera_pose[:3, :3] @ ref[:3, :3].T))
            direct_translation_errors.append(float(np.linalg.norm(pred.camera_pose[:3, 3] - ref[:3, 3])))
        metrics.update(
            {
                "direct_rotation_error_deg_mean": float(np.mean(direct_rotation_errors)),
                "direct_translation_error_mean": float(np.mean(direct_translation_errors)),
            }
        )
    return metrics


def compute_intrinsics_metrics(
    predictions: list[PredictionData],
    reference_intrinsics: dict[str, np.ndarray],
) -> dict[str, Any]:
    common = [pred for pred in predictions if pred.image_name in reference_intrinsics]
    if not common:
        return {"available": False, "common_views": 0}
    fx_errors = []
    fy_errors = []
    principal_errors = []
    for pred in common:
        ref = reference_intrinsics[pred.image_name]
        fx_errors.append(abs(float(pred.intrinsics[0, 0] - ref[0, 0])) / max(abs(float(ref[0, 0])), 1e-6))
        fy_errors.append(abs(float(pred.intrinsics[1, 1] - ref[1, 1])) / max(abs(float(ref[1, 1])), 1e-6))
        principal_errors.append(float(np.linalg.norm(pred.intrinsics[:2, 2] - ref[:2, 2])))
    return {
        "available": True,
        "common_views": len(common),
        "fx_relative_error_mean": float(np.mean(fx_errors)),
        "fy_relative_error_mean": float(np.mean(fy_errors)),
        "principal_point_error_px_mean": float(np.mean(principal_errors)),
    }


def make_render_pairs(count: int, render_config: dict[str, Any]) -> list[tuple[int, int]]:
    if count < 2:
        return []
    strategy = str(render_config.get("pair_strategy", "adjacent"))
    if strategy == "adjacent":
        pairs = [(index, index + 1) for index in range(count - 1)]
    elif strategy == "all":
        pairs = [(src, tgt) for src in range(count) for tgt in range(count) if src != tgt]
    else:
        fail(f"unsupported render pair_strategy: {strategy}")
    if render_config.get("bidirectional", False) and strategy == "adjacent":
        pairs = pairs + [(target, source) for source, target in pairs]
    max_pairs = render_config.get("max_pairs")
    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]
    return pairs


def render_points_to_target(
    points: np.ndarray,
    colors: np.ndarray,
    target: PredictionData | RenderView,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = target.image_uint8.shape[:2]
    render = np.zeros((height, width, 3), dtype=np.uint8)
    mask_out = np.zeros((height, width), dtype=bool)
    if len(points) == 0:
        return render, mask_out

    world2target = invert_pose(target.camera_pose)
    target_points = points @ world2target[:3, :3].T + world2target[:3, 3]
    z = target_points[:, 2]
    in_front = z > 1e-6
    target_points = target_points[in_front]
    colors = colors[in_front]
    z = z[in_front]
    if len(target_points) == 0:
        return render, mask_out

    projected = (target.intrinsics @ target_points.T).T
    u_coord = np.rint(projected[:, 0] / np.maximum(projected[:, 2], 1e-6)).astype(
        np.int32
    )
    v_coord = np.rint(projected[:, 1] / np.maximum(projected[:, 2], 1e-6)).astype(
        np.int32
    )
    in_bounds = (u_coord >= 0) & (u_coord < width) & (v_coord >= 0) & (v_coord < height)
    u_coord = u_coord[in_bounds]
    v_coord = v_coord[in_bounds]
    colors = colors[in_bounds]
    z = z[in_bounds]
    if len(z) == 0:
        return render, mask_out

    flat_indices = v_coord * width + u_coord
    order = np.lexsort((z, flat_indices))
    sorted_flat_indices = flat_indices[order]
    first_for_pixel = np.empty(len(order), dtype=bool)
    first_for_pixel[0] = True
    first_for_pixel[1:] = sorted_flat_indices[1:] != sorted_flat_indices[:-1]
    chosen = order[first_for_pixel]

    flat_render = render.reshape(-1, 3)
    flat_mask = mask_out.reshape(-1)
    flat_render[flat_indices[chosen]] = colors[chosen]
    flat_mask[flat_indices[chosen]] = True
    return render, mask_out


def prediction_points(
    prediction: PredictionData,
    point_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    point_stride = max(1, int(point_stride))
    points = prediction.world_points[::point_stride, ::point_stride].reshape(-1, 3)
    colors = prediction.image_uint8[::point_stride, ::point_stride].reshape(-1, 3)
    mask = prediction.mask[::point_stride, ::point_stride].reshape(-1)
    finite = np.isfinite(points).all(axis=1)
    keep = mask & finite
    return points[keep], colors[keep]


def collect_global_point_cloud(
    predictions: list[PredictionData],
    point_stride: int,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    for prediction in predictions:
        points, colors = prediction_points(prediction, point_stride)
        if len(points) == 0:
            continue
        point_chunks.append(points)
        color_chunks.append(colors)

    if not point_chunks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    points_all = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
    colors_all = np.concatenate(color_chunks, axis=0).astype(np.uint8, copy=False)
    if max_points is not None and len(points_all) > int(max_points):
        indices = np.linspace(0, len(points_all) - 1, int(max_points)).astype(np.int64)
        points_all = points_all[indices]
        colors_all = colors_all[indices]
    return points_all, colors_all


def render_source_to_target(
    source: PredictionData,
    target: PredictionData | RenderView,
    point_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    points, colors = prediction_points(source, point_stride)
    return render_points_to_target(points, colors, target)


def masked_global_ssim(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    a = image_a.astype(np.float64) / 255.0
    b = image_b.astype(np.float64) / 255.0
    values = []
    c1 = 0.01**2
    c2 = 0.03**2
    for channel in range(3):
        x = a[..., channel][mask]
        y = b[..., channel][mask]
        if x.size < 2:
            continue
        mu_x = x.mean()
        mu_y = y.mean()
        var_x = x.var()
        var_y = y.var()
        cov_xy = ((x - mu_x) * (y - mu_y)).mean()
        values.append(((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)))
    if not values:
        return None
    return float(np.mean(values))


def image_metrics(render: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    coverage = float(mask.mean())
    full_diff = (render.astype(np.float32) - target.astype(np.float32)) / 255.0
    full_mse = float(np.mean(full_diff**2))
    full_psnr = float("inf") if full_mse == 0 else float(-10.0 * math.log10(full_mse))
    full_ssim = None
    try:
        from skimage.metrics import structural_similarity

        full_ssim = float(
            structural_similarity(
                target,
                render,
                channel_axis=2,
                data_range=255,
            )
        )
    except Exception:
        full_ssim = masked_global_ssim(render, target, np.ones(mask.shape, dtype=bool))

    if not np.any(mask):
        return {
            "coverage": coverage,
            "mse": None,
            "psnr_db": None,
            "ssim": None,
            "full_image_mse": full_mse,
            "full_image_psnr_db": full_psnr,
            "full_image_ssim": full_ssim,
        }
    diff = (render.astype(np.float32) - target.astype(np.float32)) / 255.0
    mse = float(np.mean(diff[mask] ** 2))
    psnr = None if mse <= 0 else float(-10.0 * math.log10(mse))
    if mse == 0:
        psnr = float("inf")
    return {
        "coverage": coverage,
        "mse": mse,
        "psnr_db": psnr,
        "ssim": masked_global_ssim(render, target, mask),
        "full_image_mse": full_mse,
        "full_image_psnr_db": full_psnr,
        "full_image_ssim": full_ssim,
    }


def evaluate_rendering(
    predictions: list[PredictionData],
    output_dir: Path,
    render_config: dict[str, Any],
    Image: Any,
) -> dict[str, Any]:
    if not render_config.get("enabled", True):
        return {"available": False, "reason": "disabled"}
    min_coverage = float(render_config.get("min_coverage", 0.0) or 0.0)
    pairs = make_render_pairs(len(predictions), render_config)
    if not pairs:
        return {"available": False, "reason": "not enough views"}
    render_dir = output_dir / "renders"
    if render_config.get("save_images", True):
        render_dir.mkdir(parents=True, exist_ok=True)

    pair_metrics: list[dict[str, Any]] = []
    for source_idx, target_idx in pairs:
        source = predictions[source_idx]
        target = predictions[target_idx]
        render, render_mask = render_source_to_target(
            source=source,
            target=target,
            point_stride=int(render_config.get("point_stride", 2)),
        )
        valid_mask = render_mask & target.mask
        metrics = image_metrics(render, target.image_uint8, valid_mask)
        metrics["meets_min_coverage"] = metrics["coverage"] >= min_coverage
        metrics.update(
            {
                "source_index": source_idx,
                "target_index": target_idx,
                "source_image": source.image_name,
                "target_image": target.image_name,
            }
        )
        pair_metrics.append(metrics)
        if render_config.get("save_images", True):
            stem = f"{source_idx:04d}_to_{target_idx:04d}"
            Image.fromarray(render).save(render_dir / f"{stem}_render.png")
            Image.fromarray(target.image_uint8).save(render_dir / f"{stem}_target.png")
            Image.fromarray((valid_mask.astype(np.uint8) * 255)).save(render_dir / f"{stem}_mask.png")
            diff = np.abs(render.astype(np.int16) - target.image_uint8.astype(np.int16)).astype(np.uint8)
            diff[~valid_mask] = 0
            Image.fromarray(diff).save(render_dir / f"{stem}_diff.png")

    qualified_metrics = [item for item in pair_metrics if item["meets_min_coverage"]]
    psnr_values = [
        item["psnr_db"]
        for item in qualified_metrics
        if item["psnr_db"] is not None and math.isfinite(item["psnr_db"])
    ]
    ssim_values = [item["ssim"] for item in qualified_metrics if item["ssim"] is not None]
    full_psnr_values = [
        item["full_image_psnr_db"]
        for item in qualified_metrics
        if item.get("full_image_psnr_db") is not None
        and math.isfinite(item["full_image_psnr_db"])
    ]
    full_ssim_values = [
        item["full_image_ssim"]
        for item in qualified_metrics
        if item.get("full_image_ssim") is not None
    ]
    coverage_values = [item["coverage"] for item in qualified_metrics]
    return {
        "available": bool(qualified_metrics),
        "pair_count": len(pair_metrics),
        "qualified_pair_count": len(qualified_metrics),
        "low_coverage_pair_count": len(pair_metrics) - len(qualified_metrics),
        "min_coverage": min_coverage,
        "pairs": pair_metrics,
        "psnr_db_mean": None if not psnr_values else float(np.mean(psnr_values)),
        "ssim_mean": None if not ssim_values else float(np.mean(ssim_values)),
        "full_image_psnr_db_mean": None if not full_psnr_values else float(np.mean(full_psnr_values)),
        "full_image_ssim_mean": None if not full_ssim_values else float(np.mean(full_ssim_values)),
        "coverage_mean": None if not coverage_values else float(np.mean(coverage_values)),
        "render_dir": str(render_dir.resolve()) if render_config.get("save_images", True) else None,
        "note": "Point-splat rendering from predicted depth/poses; primary PSNR/SSIM are computed on valid splatted pixels, with full-image PSNR/SSIM also reported.",
    }


def make_holdout_target_indices(
    target_count: int,
    holdout_config: dict[str, Any],
) -> list[int]:
    if target_count < 1:
        return []
    strategy = str(holdout_config.get("source_strategy", "global_point_cloud"))
    if strategy != "global_point_cloud":
        fail(f"unsupported holdout source_strategy: {strategy}")

    max_pairs = holdout_config.get("max_pairs")
    target_indices = list(range(target_count))
    if max_pairs is not None:
        target_indices = target_indices[: int(max_pairs)]
    return target_indices


def evaluate_holdout_rendering(
    runtime: dict[str, Any],
    predictions: list[PredictionData],
    prepared: PreparedConfig,
    config: dict[str, Any],
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
) -> dict[str, Any]:
    evaluation_config = dict(config.get("evaluation", {}))
    holdout_config = dict(evaluation_config.get("holdout", {}))
    if not evaluation_config.get("enabled", True) or not holdout_config.get(
        "enabled", False
    ):
        return {"available": False, "reason": "disabled"}
    if not prepared.holdout_view_specs:
        return {"available": False, "reason": "no holdout views"}

    target_views = materialize_render_targets(
        runtime=runtime,
        view_specs=prepared.holdout_view_specs,
        config=config,
        entry=prepared.entry,
    )
    if not target_views:
        return {"available": False, "reason": "no materialized holdout targets"}

    pose_source = str(
        holdout_config.get("pose_source", evaluation_config.get("reference_pose_source", "colmap"))
    )
    reference_poses = reference_pose_table(pose_source, colmap_model, ar_table)
    aligned_predictions, alignment = align_predictions_to_reference(predictions, reference_poses)
    if not alignment.get("available", False):
        return {
            "available": False,
            "reason": alignment.get("reason", "alignment unavailable"),
            "alignment": alignment,
        }

    point_stride = int(holdout_config.get("point_stride", 2))
    global_points, global_colors = collect_global_point_cloud(
        predictions=aligned_predictions,
        point_stride=point_stride,
        max_points=holdout_config.get("global_point_max_points"),
    )
    if len(global_points) == 0:
        return {
            "available": False,
            "reason": "global point cloud is empty",
            "alignment": alignment,
        }

    min_coverage = float(holdout_config.get("min_coverage", 0.0) or 0.0)
    target_indices = make_holdout_target_indices(len(target_views), holdout_config)
    if not target_indices:
        return {"available": False, "reason": "no holdout targets", "alignment": alignment}

    render_dir = prepared.output_dir / "renders_holdout"
    if holdout_config.get("save_images", True):
        render_dir.mkdir(parents=True, exist_ok=True)

    pair_metrics: list[dict[str, Any]] = []
    for target_idx in target_indices:
        target = target_views[target_idx]
        render, render_mask = render_points_to_target(
            points=global_points,
            colors=global_colors,
            target=target,
        )
        valid_mask = render_mask & target.mask
        metrics = image_metrics(render, target.image_uint8, valid_mask)
        metrics["meets_min_coverage"] = metrics["coverage"] >= min_coverage
        metrics.update(
            {
                "target_index": target_idx,
                "source_mode": "global_point_cloud",
                "source_images": len(aligned_predictions),
                "global_point_count": len(global_points),
                "target_image": target.image_name,
            }
        )
        pair_metrics.append(metrics)
        if holdout_config.get("save_images", True):
            target_stem = Path(target.image_name).stem
            stem = f"global_to_holdout_{target_idx:04d}_{target_stem}"
            Image = runtime["Image"]
            Image.fromarray(render).save(render_dir / f"{stem}_render.png")
            Image.fromarray(target.image_uint8).save(render_dir / f"{stem}_target.png")
            Image.fromarray((valid_mask.astype(np.uint8) * 255)).save(
                render_dir / f"{stem}_mask.png"
            )
            diff = np.abs(
                render.astype(np.int16) - target.image_uint8.astype(np.int16)
            ).astype(np.uint8)
            diff[~valid_mask] = 0
            Image.fromarray(diff).save(render_dir / f"{stem}_diff.png")

    qualified_metrics = [item for item in pair_metrics if item["meets_min_coverage"]]
    psnr_values = [
        item["psnr_db"]
        for item in qualified_metrics
        if item["psnr_db"] is not None and math.isfinite(item["psnr_db"])
    ]
    ssim_values = [item["ssim"] for item in qualified_metrics if item["ssim"] is not None]
    full_psnr_values = [
        item["full_image_psnr_db"]
        for item in qualified_metrics
        if item.get("full_image_psnr_db") is not None
        and math.isfinite(item["full_image_psnr_db"])
    ]
    full_ssim_values = [
        item["full_image_ssim"]
        for item in qualified_metrics
        if item.get("full_image_ssim") is not None
    ]
    coverage_values = [item["coverage"] for item in qualified_metrics]
    return {
        "available": bool(qualified_metrics),
        "split": "holdout",
        "source_count": len(predictions),
        "target_count": len(target_views),
        "global_point_count": len(global_points),
        "point_stride": point_stride,
        "pair_count": len(pair_metrics),
        "qualified_pair_count": len(qualified_metrics),
        "low_coverage_pair_count": len(pair_metrics) - len(qualified_metrics),
        "min_coverage": min_coverage,
        "source_strategy": str(
            holdout_config.get("source_strategy", "global_point_cloud")
        ),
        "pairs": pair_metrics,
        "alignment": alignment,
        "psnr_db_mean": None if not psnr_values else float(np.mean(psnr_values)),
        "ssim_mean": None if not ssim_values else float(np.mean(ssim_values)),
        "full_image_psnr_db_mean": None
        if not full_psnr_values
        else float(np.mean(full_psnr_values)),
        "full_image_ssim_mean": None if not full_ssim_values else float(np.mean(full_ssim_values)),
        "coverage_mean": None if not coverage_values else float(np.mean(coverage_values)),
        "render_dir": str(render_dir.resolve())
        if holdout_config.get("save_images", True)
        else None,
        "note": (
            "Holdout point-splat rendering. Train predictions are "
            "similarity-aligned to the reference pose source, fused into one "
            "global point cloud, then rendered into target views that were not "
            "used for MapAnything inference."
        ),
    }


def reference_pose_table(
    source: str,
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
) -> dict[str, np.ndarray]:
    source = source.lower()
    if source == "colmap":
        if colmap_model is None:
            return {}
        return {name: image.cam2world for name, image in colmap_model.images_by_name.items()}
    if source == "ar":
        if ar_table is None:
            return {}
        return ar_table.poses_by_name
    return {}


def reference_intrinsics_table(
    source: str,
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
) -> dict[str, np.ndarray]:
    source = source.lower()
    if source == "colmap":
        if colmap_model is None:
            return {}
        table: dict[str, np.ndarray] = {}
        for name in colmap_model.registered_names:
            intrinsics = colmap_intrinsics_for_name(name, colmap_model)
            if intrinsics is not None:
                table[name] = intrinsics
        return table
    if source == "ar":
        if ar_table is None:
            return {}
        return ar_table.intrinsics_by_name
    return {}


def evaluate_predictions(
    runtime: dict[str, Any],
    predictions: list[PredictionData],
    prepared: PreparedConfig,
    config: dict[str, Any],
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
    input_intrinsics: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    evaluation_config = dict(config.get("evaluation", {}))
    if not evaluation_config.get("enabled", True):
        return {"enabled": False}

    pose_source = str(evaluation_config.get("reference_pose_source", "colmap"))
    intrinsics_source = str(evaluation_config.get("reference_intrinsics_source", "colmap"))
    reference_poses = reference_pose_table(pose_source, colmap_model, ar_table)
    if input_intrinsics is not None:
        reference_intrinsics = input_intrinsics
        effective_intrinsics_source = "preprocessed_input"
    else:
        reference_intrinsics = reference_intrinsics_table(
            intrinsics_source,
            colmap_model,
            ar_table,
        )
        effective_intrinsics_source = intrinsics_source
    direct_pose_comparison = str(prepared.entry.get("pose_source", "none")).lower() == pose_source.lower()

    render_config = dict(evaluation_config.get("render", {}))
    render_metrics = evaluate_rendering(
        predictions=predictions,
        output_dir=prepared.output_dir,
        render_config=render_config,
        Image=runtime["Image"],
    )
    holdout_render_metrics = evaluate_holdout_rendering(
        runtime=runtime,
        predictions=predictions,
        prepared=prepared,
        config=config,
        colmap_model=colmap_model,
        ar_table=ar_table,
    )
    return {
        "pose_reference_source": pose_source,
        "intrinsics_reference_source": effective_intrinsics_source,
        "pose": compute_pose_metrics(predictions, reference_poses, direct_pose_comparison),
        "intrinsics": compute_intrinsics_metrics(predictions, reference_intrinsics),
        "render": render_metrics,
        "render_holdout": holdout_render_metrics,
    }


def flatten_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    holdout_render = metrics.get("render_holdout", {})
    if holdout_render.get("available", False):
        render = holdout_render
        render_eval_split = "holdout"
    else:
        render = metrics.get("render", {})
        render_eval_split = "train"
    pose = metrics.get("pose", {})
    return {
        "render_psnr_db": render.get("psnr_db_mean"),
        "render_ssim": render.get("ssim_mean"),
        "render_coverage": render.get("coverage_mean"),
        "render_eval_split": render_eval_split,
        "pose_ate_rmse": pose.get("ate_rmse"),
        "relative_rotation_error_deg": pose.get("relative_rotation_error_deg_mean"),
    }


def run_prepared_config(
    runtime: dict[str, Any],
    model: Any,
    device: Any,
    prepared: PreparedConfig,
    config: dict[str, Any],
    colmap_model: ColmapModel | None,
    ar_table: PoseTable | None,
) -> dict[str, Any]:
    outputs, processed_views, runtime_seconds = run_inference(runtime, model, device, prepared, config)
    ensure_output_masks(runtime, outputs)
    input_intrinsics = processed_intrinsics_table(processed_views, prepared.view_specs)
    predictions = extract_predictions(runtime, outputs, prepared.view_specs)
    artifact_paths, warnings = export_optional_artifacts(
        runtime=runtime,
        outputs=outputs,
        processed_views=processed_views,
        predictions=predictions,
        prepared=prepared,
        config=config,
        model=model,
    )
    metrics = evaluate_predictions(
        runtime=runtime,
        predictions=predictions,
        prepared=prepared,
        config=config,
        colmap_model=colmap_model,
        ar_table=ar_table,
        input_intrinsics=input_intrinsics,
    )

    config_result = {
        "config": prepared.entry.get("id"),
        "name": prepared.entry.get("name"),
        "label": prepared.entry.get("label"),
        "status": "completed",
        "view_count": len(prepared.view_specs),
        "holdout_view_count": len(prepared.holdout_view_specs),
        "runtime_seconds": runtime_seconds,
        "input_manifest": str(prepared.manifest_path.resolve()),
        "output_dir": str(prepared.output_dir.resolve()),
        "input_stats": prepared.stats,
        "artifacts": artifact_paths,
        "metrics": metrics,
        "summary_metrics": flatten_summary_metrics(metrics),
        "warnings": warnings,
    }
    write_json(prepared.output_dir / "metrics" / "run_summary.json", config_result)
    return config_result


def task2_coverage_review(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    entries = config.get("configs", [])
    result_by_config = {str(result.get("config")): result for result in results}
    required_configs = ("config_a", "config_b", "config_c")

    def has_render_metrics(result: dict[str, Any]) -> bool:
        metrics = result.get("metrics", {})
        return bool(
            metrics.get("render_holdout", {}).get("available", False)
            or metrics.get("render", {}).get("available", False)
        )

    def has_config(use_intrinsics: bool, pose_source: str) -> bool:
        return any(
            bool(entry.get("use_intrinsics", False)) is use_intrinsics
            and str(entry.get("pose_source", "none")).lower() == pose_source
            for entry in entries
        )

    completed = {str(result.get("config")) for result in results if result.get("status") == "completed"}
    required_completed = [
        config_id
        for config_id in required_configs
        if result_by_config.get(config_id, {}).get("status") == "completed"
    ]
    required_render_ready = [
        config_id
        for config_id in required_configs
        if has_render_metrics(result_by_config.get(config_id, {}))
    ]
    required_view_counts = [
        int(result_by_config[config_id]["view_count"])
        for config_id in required_configs
        if result_by_config.get(config_id, {}).get("status") == "completed"
        and result_by_config[config_id].get("view_count") is not None
    ]
    required_skips = {
        config_id: result_by_config.get(config_id, {})
        .get("input_stats", {})
        .get("skipped_without_required_data", [])
        for config_id in required_configs
    }
    checks = [
        {
            "name": "Config A exists",
            "passed": has_config(False, "none"),
            "detail": "Uncalibrated image-only configuration is present.",
        },
        {
            "name": "Config B exists",
            "passed": has_config(True, "none"),
            "detail": "Calibrated image-only configuration is present.",
        },
        {
            "name": "Config C exists",
            "passed": has_config(True, "colmap"),
            "detail": "Calibrated + COLMAP poses configuration is present.",
        },
        {
            "name": "Config D supported",
            "passed": has_config(True, "ar"),
            "detail": "Optional AR pose configuration is present and skipped only when AR input is missing.",
        },
        {
            "name": "Quantitative metrics",
            "passed": bool(config.get("evaluation", {}).get("enabled", True)),
            "detail": "Pose, intrinsics, runtime, and point-splat render metrics are configured.",
        },
        {
            "name": "Reconstruction artifacts",
            "passed": bool(config.get("exports", {}).get("save_point_cloud_ply", True))
            or bool(config.get("exports", {}).get("save_glb", True)),
            "detail": "PLY/GLB reconstruction outputs are configured.",
        },
        {
            "name": "Required configs completed",
            "passed": set(required_completed) == set(required_configs),
            "detail": (
                "Completed required configs: "
                f"{', '.join(required_completed) or 'none'}; required: "
                f"{', '.join(required_configs)}."
            ),
        },
        {
            "name": "Required render metrics available",
            "passed": set(required_render_ready) == set(required_configs),
            "detail": (
                "Render metrics available for: "
                f"{', '.join(required_render_ready) or 'none'}."
            ),
        },
        {
            "name": "Required configs use matching view counts",
            "passed": len(required_view_counts) == len(required_configs)
            and len(set(required_view_counts)) == 1,
            "detail": (
                "Completed required view counts: "
                f"{required_view_counts or 'none'}."
            ),
        },
        {
            "name": "Required configs did not drop required-data views",
            "passed": all(not required_skips[config_id] for config_id in required_configs),
            "detail": (
                "Skipped required-data views by config: "
                + json.dumps(required_skips, ensure_ascii=False)
            ),
        },
        {
            "name": "Optional Config D handled",
            "passed": result_by_config.get("config_d", {}).get("status")
            in {None, "completed", "skipped", "prepared"},
            "detail": (
                "Config D status: "
                f"{result_by_config.get('config_d', {}).get('status', 'not selected')}."
            ),
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def comparison_value(result: dict[str, Any], field: str) -> Any:
    if field == "runtime_seconds":
        return result.get("runtime_seconds")
    return result.get("summary_metrics", {}).get(field)


def build_task_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result_by_config = {str(result.get("config")): result for result in results}
    comparisons: list[dict[str, Any]] = []
    for spec in TASK_COMPARISON_SPECS:
        baseline_id = str(spec["baseline"])
        candidate_id = str(spec["candidate"])
        baseline = result_by_config.get(baseline_id)
        candidate = result_by_config.get(candidate_id)
        comparison: dict[str, Any] = {
            "id": spec["id"],
            "name": spec["name"],
            "question": spec["question"],
            "baseline": baseline_id,
            "candidate": candidate_id,
        }
        if baseline is None or candidate is None:
            comparison.update(
                {
                    "status": "unavailable",
                    "reason": "one or both configs were not selected",
                    "metrics": {},
                }
            )
            comparisons.append(comparison)
            continue
        if baseline.get("status") != "completed" or candidate.get("status") != "completed":
            comparison.update(
                {
                    "status": "unavailable",
                    "reason": (
                        f"{baseline_id} status={baseline.get('status')}; "
                        f"{candidate_id} status={candidate.get('status')}"
                    ),
                    "metrics": {},
                }
            )
            comparisons.append(comparison)
            continue

        metric_deltas: dict[str, dict[str, Any]] = {}
        for field in COMPARISON_FIELDS:
            before = comparison_value(baseline, field)
            after = comparison_value(candidate, field)
            if before is None or after is None:
                delta = None
            else:
                try:
                    delta = float(after) - float(before)
                except (TypeError, ValueError):
                    delta = None
            metric_deltas[field] = {
                "baseline": before,
                "candidate": after,
                "delta": delta,
            }
        comparison.update({"status": "available", "metrics": metric_deltas})
        comparisons.append(comparison)
    return comparisons


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task 2 MapAnything Summary",
        "",
        f"- Created at: {summary['created_at']}",
        f"- Run name: {summary['run_name']}",
        f"- Config JSON: {summary['config_path']}",
        "",
        "| Config | Status | Views | Eval | Runtime (s) | PSNR | SSIM | Coverage | Pose ATE RMSE | Rel. Rot. Err. |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["results"]:
        metrics = result.get("summary_metrics", {})
        values = {
            key: metrics.get(key)
            for key in SUMMARY_FIELDS
        }
        lines.append(
            "| {config} | {status} | {views} | {eval_split} | {runtime} | {psnr} | {ssim} | {coverage} | {ate} | {rot} |".format(
                config=result.get("config", ""),
                status=result.get("status", ""),
                views=result.get("view_count", ""),
                eval_split=metrics.get("render_eval_split", ""),
                runtime=format_float(result.get("runtime_seconds")),
                psnr=format_float(values["render_psnr_db"]),
                ssim=format_float(values["render_ssim"]),
                coverage=format_float(values["render_coverage"]),
                ate=format_float(values["pose_ate_rmse"]),
                rot=format_float(values["relative_rotation_error_deg"]),
            )
        )
    lines.extend(
        [
            "",
            "## Task Comparisons",
            "",
            "| Comparison | Question | PSNR d | SSIM d | Coverage d | Pose ATE d | Runtime d (s) | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for comparison in summary.get("task2_comparisons", []):
        metrics = comparison.get("metrics", {})
        status = comparison.get("status", "")
        if status != "available":
            status = f"{status}: {comparison.get('reason', '')}"
        lines.append(
            "| {name} | {question} | {psnr} | {ssim} | {coverage} | {ate} | {runtime} | {status} |".format(
                name=comparison.get("name", ""),
                question=comparison.get("question", ""),
                psnr=format_delta(metrics.get("render_psnr_db", {}).get("delta")),
                ssim=format_delta(metrics.get("render_ssim", {}).get("delta")),
                coverage=format_delta(metrics.get("render_coverage", {}).get("delta")),
                ate=format_delta(metrics.get("pose_ate_rmse", {}).get("delta")),
                runtime=format_delta(metrics.get("runtime_seconds", {}).get("delta")),
                status=status,
            )
        )
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    lines.extend(["", "## Task 2 Coverage Review", ""])
    for check in summary["task2_coverage_review"]["checks"]:
        mark = "PASS" if check["passed"] else "WARN"
        lines.append(f"- {mark}: {check['name']} - {check['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def format_delta(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:+.4f}"


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.prepare_only = True
    config_path = args.config.resolve()
    config = read_json(config_path)
    apply_cli_overrides(config, args)
    entries = selected_entries(config, args)
    if args.list_configs:
        print_configs(entries)
        return

    image_dir = resolve_path(config.get("image_dir"))
    colmap_export_dir = resolve_path(config.get("colmap_export_dir"))
    ar_pose_file = resolve_path(config.get("ar_pose_file"))
    input_root = resolve_path(config.get("input_root")) or (PROJECT_ROOT / "data" / "processed" / "mapanything_inputs")
    output_root = resolve_path(config.get("output_root")) or (PROJECT_ROOT / "outputs" / "mapanything")
    run_name = str(config.get("run_name", image_dir.name if image_dir else "scene"))
    config["run_name"] = run_name

    image_paths = list_images(image_dir) if image_dir else fail("image_dir is required")
    colmap_model = load_colmap_model(colmap_export_dir)
    default_ar_convention = "arkit_cam2world"
    for entry in entries:
        if str(entry.get("pose_source", "")).lower() == "ar":
            default_ar_convention = str(entry.get("pose_convention", default_ar_convention))
            break
    ar_table = parse_ar_pose_file(ar_pose_file, default_ar_convention)
    warnings = setup_warnings(colmap_export_dir, colmap_model, ar_pose_file, ar_table)

    results: list[dict[str, Any]] = []
    prepared_configs: list[PreparedConfig] = []
    for entry in entries:
        prepared = prepare_config(
            config=config,
            entry=entry,
            image_paths=image_paths,
            colmap_model=colmap_model,
            ar_table=ar_table,
            input_root=input_root,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        if isinstance(prepared, dict):
            print(f"Skipping {prepared.get('config')}: {prepared.get('reason')}")
            results.append(prepared)
        else:
            prepared_configs.append(prepared)
            if args.prepare_only:
                result = {
                    "config": prepared.entry.get("id"),
                    "name": prepared.entry.get("name"),
                    "label": prepared.entry.get("label"),
                    "status": "prepared",
                    "view_count": len(prepared.view_specs),
                    "holdout_view_count": len(prepared.holdout_view_specs),
                    "input_manifest": str(prepared.manifest_path.resolve()),
                    "output_dir": str(prepared.output_dir.resolve()),
                    "input_stats": prepared.stats,
                }
                results.append(result)
                print(f"Prepared {prepared.entry.get('id')}: {prepared.manifest_path}")

    runtime = None
    model = None
    device = None
    model_name = None
    if not args.prepare_only and prepared_configs:
        runtime = load_runtime()
        model, device, model_name = load_model(runtime, dict(config.get("model", {})))
        for prepared in prepared_configs:
            try:
                result = run_prepared_config(
                    runtime=runtime,
                    model=model,
                    device=device,
                    prepared=prepared,
                    config=config,
                    colmap_model=colmap_model,
                    ar_table=ar_table,
                )
                results.append(result)
                print(
                    f"Completed {prepared.entry.get('id')} in "
                    f"{result['runtime_seconds']:.2f}s"
                )
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "config": prepared.entry.get("id"),
                    "name": prepared.entry.get("name"),
                    "label": prepared.entry.get("label"),
                    "status": "failed",
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                    "input_manifest": str(prepared.manifest_path.resolve()),
                    "output_dir": str(prepared.output_dir.resolve()),
                }
                results.append(failure)
                write_json(prepared.output_dir / "metrics" / "failure.json", failure)
                if not config.get("continue_on_error", False):
                    summary = build_summary(config, config_path, results, model_name, warnings)
                    write_summaries(output_root, run_name, summary)
                    raise
                print(f"Failed {prepared.entry.get('id')}: {exc}", file=sys.stderr)

    summary = build_summary(config, config_path, results, model_name, warnings)
    write_summaries(output_root, run_name, summary)
    print("\nTask 2 pipeline complete")
    print(f"summary JSON: {output_root / f'{run_name}_task2_summary.json'}")
    print(f"summary Markdown: {output_root / f'{run_name}_task2_summary.md'}")


def build_summary(
    config: dict[str, Any],
    config_path: Path,
    results: list[dict[str, Any]],
    model_name: str | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "created_at": now_utc(),
        "task": "Task 2 MapAnything",
        "run_name": config.get("run_name"),
        "config_path": str(config_path),
        "model_name": model_name or config.get("model", {}).get("name"),
        "warnings": warnings or [],
        "results": results,
        "task2_comparisons": build_task_comparisons(results),
        "task2_coverage_review": task2_coverage_review(config, results),
    }


def write_summaries(output_root: Path, run_name: str, summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / f"{run_name}_task2_summary.json", summary)
    write_markdown_summary(output_root / f"{run_name}_task2_summary.md", summary)


if __name__ == "__main__":
    main()
