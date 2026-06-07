#!/usr/bin/env python3
"""Run COLMAP through the official GPU-enabled Docker image."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


OFF_VALUES = {"0", "false", "no", "none", "off", "cpu"}
DEFAULT_IMAGE = "colmap/colmap:latest"
LOCAL_IMAGE = "colmap:latest"
FORWARDED_ENV = (
    "QT_QPA_PLATFORM",
    "QT_XCB_GL_INTEGRATION",
    "DISPLAY",
    "XAUTHORITY",
)


def docker_image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def selected_image() -> str:
    configured = os.environ.get("COLMAP_DOCKER_IMAGE")
    if configured:
        return configured
    if docker_image_exists(LOCAL_IMAGE):
        return LOCAL_IMAGE
    return DEFAULT_IMAGE


def gpu_args() -> list[str]:
    configured = os.environ.get("COLMAP_DOCKER_GPU", "all").strip()
    if configured.lower() in OFF_VALUES:
        return []
    if configured.startswith("--"):
        return shlex.split(configured)
    if configured.lower() == "runtime":
        return ["--runtime", "nvidia"]
    return ["--gpus", configured]


def path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def add_mount(mounts: list[Path], path: Path) -> None:
    resolved = path.resolve()
    for existing in mounts:
        if path_is_within(resolved, existing):
            return
    mounts[:] = [existing for existing in mounts if not path_is_within(existing, resolved)]
    mounts.append(resolved)


def mount_for_arg(arg: str) -> Path | None:
    if not arg or arg.startswith("-"):
        return None
    path = Path(arg)
    if not path.is_absolute():
        return None
    if path.exists():
        return path if path.is_dir() else path.parent
    if path.parent.exists():
        return path.parent
    return None


def docker_mounts(argv: list[str]) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    mounts: list[Path] = []
    add_mount(mounts, repo_root)
    add_mount(mounts, cwd)
    for arg in argv:
        mount = mount_for_arg(arg)
        if mount is not None:
            add_mount(mounts, mount)
    if os.environ.get("DISPLAY") and Path("/tmp/.X11-unix").exists():
        add_mount(mounts, Path("/tmp/.X11-unix"))
    xauthority = os.environ.get("XAUTHORITY")
    if os.environ.get("DISPLAY") and xauthority and Path(xauthority).exists():
        add_mount(mounts, Path(xauthority).resolve().parent)
    extra_mounts = os.environ.get("COLMAP_DOCKER_MOUNTS", "")
    for raw_mount in extra_mounts.split(os.pathsep):
        if raw_mount:
            add_mount(mounts, Path(raw_mount))
    return mounts


def env_args() -> list[str]:
    args: list[str] = [
        "-e",
        "NVIDIA_VISIBLE_DEVICES=all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
    ]
    for key in FORWARDED_ENV:
        if key in {"DISPLAY", "XAUTHORITY"} and not os.environ.get("DISPLAY"):
            continue
        value = os.environ.get(key)
        if value is not None:
            args.extend(["-e", f"{key}={value}"])
    return args


def user_args() -> list[str]:
    configured = os.environ.get("COLMAP_DOCKER_USER")
    if configured:
        if configured.lower() in OFF_VALUES:
            return []
        return ["--user", configured]
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def main(argv: list[str]) -> int:
    docker = shutil.which("docker")
    if docker is None:
        print(
            "error: docker was not found. Install Docker and NVIDIA Container Toolkit, "
            "or pass --colmap colmap to scripts/colmap/run_pipeline.py to use a local binary.",
            file=sys.stderr,
        )
        return 127

    image = selected_image()
    command = [
        docker,
        "run",
        "--rm",
        *gpu_args(),
        *user_args(),
        *env_args(),
    ]
    if sys.stdin.isatty():
        command.append("-i")
    if sys.stdout.isatty():
        command.append("-t")
    for mount in docker_mounts(argv):
        command.extend(["-v", f"{mount}:{mount}"])
    command.extend(["-w", str(Path.cwd().resolve()), image, "colmap", *argv])

    print(f"[colmap-docker] image={image}", file=sys.stderr)
    print(f"[colmap-docker] {' '.join(shlex.quote(part) for part in command)}", file=sys.stderr)
    result = subprocess.run(command, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
