#!/usr/bin/env python3
"""
Extract frames from stereo MP4 files (side-by-side format) and split into left/right.

For stereo videos with 2560px width, this extracts:
- Left half (0:1280)
- Right half (1280:2560)

And saves them as separate camera views.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List


def extract_video_id_from_filename(filename: str) -> str:
    """Return video id derived from the MP4 filename."""
    base, _ = os.path.splitext(filename)
    return base.rstrip(".")


def run_ffmpeg_extract_stereo(
    input_path: Path,
    output_dir_left: Path,
    output_dir_right: Path,
) -> None:
    """Use ffmpeg to extract frames and split stereo video into left/right."""
    output_dir_left.mkdir(parents=True, exist_ok=True)
    output_dir_right.mkdir(parents=True, exist_ok=True)

    output_pattern_left = str(output_dir_left / "%03d.jpg")
    output_pattern_right = str(output_dir_right / "%03d.jpg")

    # Extract left half
    cmd_left = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-y",
        "-i", str(input_path),
        "-vsync", "0",
        "-start_number", "0",
        "-vf", "crop=in_w/2:in_h:0:0",  # Left half: x=0
        "-q:v", "2",
        output_pattern_left
    ]

    # Extract right half
    cmd_right = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-y",
        "-i", str(input_path),
        "-vsync", "0",
        "-start_number", "0",
        "-vf", "crop=in_w/2:in_h:in_w/2:0",  # Right half: x=in_w/2
        "-q:v", "2",
        output_pattern_right
    ]

    print(f"Extracting left half: {output_dir_left.name}")
    subprocess.run(cmd_left, check=True)

    print(f"Extracting right half: {output_dir_right.name}")
    subprocess.run(cmd_right, check=True)


def frames_already_exist(frames_dir: Path) -> bool:
    return frames_dir.exists() and any(frames_dir.glob("*.jpg"))


def ensure_clean_output_dir(frames_dir: Path, overwrite: bool) -> None:
    if frames_dir.exists() and overwrite:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)


def find_all_mp4s(root: Path) -> List[Path]:
    return list(root.rglob("recordings/MP4/*.mp4"))


def process_single_mp4(
    mp4_path: Path,
    overwrite: bool,
) -> None:
    recordings_dir = mp4_path.parent.parent  # .../recordings
    video_id = extract_video_id_from_filename(mp4_path.name)

    # Create left and right directories
    frames_dir_left = recordings_dir / "frames" / f"{video_id}_left"
    frames_dir_right = recordings_dir / "frames" / f"{video_id}_right"

    if (frames_already_exist(frames_dir_left) and
        frames_already_exist(frames_dir_right) and
        not overwrite):
        print(f"SKIP (exists): {mp4_path}")
        return

    ensure_clean_output_dir(frames_dir_left, overwrite)
    ensure_clean_output_dir(frames_dir_right, overwrite)

    run_ffmpeg_extract_stereo(mp4_path, frames_dir_left, frames_dir_right)

    print(f"DONE: {mp4_path} -> {frames_dir_left.parent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and split stereo MP4 videos into left/right frame sequences"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search (defaults to current working directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate frames if output already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root: Path = args.root.resolve()
    overwrite: bool = bool(args.overwrite)

    mp4_files = find_all_mp4s(root)
    if not mp4_files:
        print(f"No MP4 files found under: {root}")
        return

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("ERROR: ffmpeg not found in PATH")
        return

    print(f"Using ffmpeg at: {ffmpeg_path}")

    for mp4_path in mp4_files:
        try:
            process_single_mp4(mp4_path, overwrite)
        except Exception as exc:
            print(f"ERROR processing {mp4_path}: {exc}")


if __name__ == "__main__":
    main()
