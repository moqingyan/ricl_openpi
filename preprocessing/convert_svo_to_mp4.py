#!/usr/bin/env python3
"""
Convert SVO2 files to MP4 videos.

This script searches for all .svo2 files under recordings/SVO directories
and converts them to MP4 format using the ZED SDK.

Expected structure:
  <demo_folder>/recordings/SVO/<camera_id>.svo2

Output structure:
  <demo_folder>/recordings/MP4/<camera_id>.mp4
  <demo_folder>/recordings/MP4/<camera_id>_timestamps.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

try:
    import pyzed.sl as sl
    import cv2
except ImportError as e:
    print(f"ERROR: Required dependencies not found: {e}")
    print("Please install: pip install opencv-python")
    print("And ensure ZED SDK Python API is installed")
    sys.exit(1)


def convert_svo_to_mp4(svo_path: Path, output_dir: Path, fps: int = 15) -> None:
    """
    Convert a single SVO2 file to MP4.

    Args:
        svo_path: Path to the .svo2 file
        output_dir: Directory to save the MP4 file
        fps: Frames per second for the output video
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_id = svo_path.stem
    mp4_path = output_dir / f"{camera_id}.mp4"
    timestamps_path = output_dir / f"{camera_id}_timestamps.json"

    # Skip if already converted
    if mp4_path.exists() and timestamps_path.exists():
        print(f"SKIP (exists): {mp4_path}")
        return

    print(f"Converting: {svo_path} -> {mp4_path}")

    # Initialize ZED camera
    zed = sl.Camera()

    # Set configuration parameters for SVO playback
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False  # Process as fast as possible
    init_params.coordinate_units = sl.UNIT.METER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR opening SVO file: {err}")
        zed.close()
        return

    # Get video properties
    resolution = zed.get_camera_information().camera_configuration.resolution
    width = resolution.width
    height = resolution.height

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(mp4_path),
        fourcc,
        fps,
        (width, height)
    )

    # Prepare image container
    image = sl.Mat()

    # Storage for timestamps
    timestamps = []
    frame_count = 0

    # Process all frames
    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Get timestamp
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            timestamps.append(timestamp)

            # Convert to numpy array and write to video
            frame = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            video_writer.write(frame_bgr)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print(f"  Completed: {frame_count} frames")
            break
        else:
            print(f"ERROR during frame grab: {err}")
            break

    # Cleanup
    video_writer.release()
    zed.close()

    # Save timestamps
    with open(timestamps_path, 'w') as f:
        json.dump(timestamps, f)

    print(f"DONE: {mp4_path}")


def find_all_svo_files(root: Path) -> List[Path]:
    """Find all .svo2 files under recordings/SVO directories."""
    return list(root.rglob("recordings/SVO/*.svo2"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SVO2 files to MP4 videos"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing demo folders with SVO files"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for output videos (default: 15)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: Root directory does not exist: {root}")
        sys.exit(1)

    svo_files = find_all_svo_files(root)
    if not svo_files:
        print(f"No SVO files found under: {root}")
        return

    print(f"Found {len(svo_files)} SVO files to convert")

    for svo_path in svo_files:
        try:
            recordings_dir = svo_path.parent.parent  # .../recordings
            mp4_dir = recordings_dir / "MP4"
            convert_svo_to_mp4(svo_path, mp4_dir, args.fps)
        except Exception as exc:
            print(f"ERROR processing {svo_path}: {exc}")
            continue

    print("\nAll conversions complete!")


if __name__ == "__main__":
    main()
