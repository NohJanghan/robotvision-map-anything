#!/usr/bin/env python3
"""Extract a COLMAP-ready image sequence from an MP4 video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NoReturn


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an MP4 video into the image-directory input expected by "
            "COLMAP feature_extractor."
        )
    )
    parser.add_argument("video", type=Path, help="Input video file, e.g. scene.mp4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for extracted frames. Defaults to "
            "data/raw/rgb_sequences/<video_stem>."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Sample frames at this rate. Mutually exclusive with --stride.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="Keep every Nth decoded frame. Mutually exclusive with --fps.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds. Default: 0.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional duration in seconds to extract.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum number of output frames.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        help="Resize frames to fit within max-side x max-side pixels.",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        help="Output image format. Default: jpg.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help=(
            "ffmpeg MJPEG quality scale for JPG output, 2 is high quality and "
            "31 is low quality. Default: 2."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing extracted images in the output directory first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ffmpeg command without extracting frames.",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        fail(
            f"'{name}' was not found on PATH. Install it in the rkv-mapanything "
            "environment, e.g. `conda install -c conda-forge ffmpeg`."
        )
    return path


def existing_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def clean_existing_images(path: Path) -> None:
    for image_path in existing_images(path):
        image_path.unlink()
    manifest = path / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def build_filter_chain(args: argparse.Namespace) -> str | None:
    filters: list[str] = []
    if args.fps is not None:
        filters.append(f"fps={args.fps:g}")
    if args.stride is not None:
        filters.append(f"select=not(mod(n\\,{args.stride}))")
    if args.max_side is not None:
        filters.append(
            f"scale={args.max_side}:{args.max_side}:"
            "force_original_aspect_ratio=decrease"
        )
    return ",".join(filters) if filters else None


def ffprobe_metadata(ffprobe: str, video: Path) -> dict[str, object]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def build_ffmpeg_command(
    ffmpeg: str, args: argparse.Namespace, output_pattern: Path
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if args.overwrite:
        command.append("-y")
    else:
        command.append("-n")
    if args.start_time > 0:
        command.extend(["-ss", f"{args.start_time:g}"])
    command.extend(["-i", str(args.video)])
    if args.duration is not None:
        command.extend(["-t", f"{args.duration:g}"])
    filters = build_filter_chain(args)
    if filters:
        command.extend(["-vf", filters])
    if args.stride is not None:
        command.extend(["-vsync", "0"])
    if args.max_frames is not None:
        command.extend(["-frames:v", str(args.max_frames)])
    if args.format == "jpg":
        command.extend(["-q:v", str(args.jpeg_quality)])
    command.extend(["-start_number", "0", str(output_pattern)])
    return command


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    command: Iterable[str],
    metadata: dict[str, object],
    frames: list[Path],
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(args.video.resolve()),
        "output_dir": str(output_dir.resolve()),
        "frame_count": len(frames),
        "frame_files": [p.name for p in frames],
        "sampling": {
            "fps": args.fps,
            "stride": args.stride,
            "start_time": args.start_time,
            "duration": args.duration,
            "max_frames": args.max_frames,
            "max_side": args.max_side,
            "format": args.format,
        },
        "ffprobe": metadata,
        "ffmpeg_command": list(command),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.video.exists():
        fail(f"input video does not exist: {args.video}")
    if not args.video.is_file():
        fail(f"input video is not a file: {args.video}")
    if args.video.suffix.lower() not in {".mp4", ".mov", ".m4v"}:
        fail("input video should be an MP4/MOV-style file")
    if args.fps is not None and args.stride is not None:
        fail("--fps and --stride are mutually exclusive")
    if args.fps is not None and args.fps <= 0:
        fail("--fps must be positive")
    if args.stride is not None and args.stride < 1:
        fail("--stride must be >= 1")
    if args.start_time < 0:
        fail("--start-time must be >= 0")
    if args.duration is not None and args.duration <= 0:
        fail("--duration must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        fail("--max-frames must be >= 1")
    if args.max_side is not None and args.max_side < 64:
        fail("--max-side must be >= 64")
    if not 2 <= args.jpeg_quality <= 31:
        fail("--jpeg-quality must be between 2 and 31")


def main() -> None:
    args = parse_args()
    validate_args(args)

    ffmpeg = require_tool("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    output_dir = args.output_dir or Path("data/raw/rgb_sequences") / args.video.stem

    current_images = existing_images(output_dir)
    if current_images and not args.overwrite:
        fail(
            f"{output_dir} already contains {len(current_images)} image(s). "
            "Use --overwrite or choose a different --output-dir."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clean_existing_images(output_dir)

    output_pattern = output_dir / f"%06d.{args.format}"
    command = build_ffmpeg_command(ffmpeg, args, output_pattern)
    print(" ".join(command))
    if args.dry_run:
        return

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        fail(f"ffmpeg failed with exit code {exc.returncode}")

    frames = existing_images(output_dir)
    if not frames:
        fail("ffmpeg completed but no frames were extracted")

    metadata = ffprobe_metadata(ffprobe, args.video) if ffprobe else {}
    write_manifest(output_dir, args, command, metadata, frames)
    print(f"wrote {len(frames)} frames to {output_dir}")
    print(f"manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
