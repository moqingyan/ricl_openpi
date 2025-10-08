#!/usr/bin/env python3
"""
Standalone script for converting SVO files to MP4 videos.

Converts all SVO2 files found in recordings/SVO directories to MP4 format.
Saves MP4 files and timestamp JSONs in recordings/MP4 directories.

Expected structure:
  <demo_folder>/recordings/SVO/<camera_id>.svo2

Output structure:
  <demo_folder>/recordings/MP4/<camera_id>.mp4
  <demo_folder>/recordings/MP4/<camera_id>_timestamps.json

Requirements:
  - ZED SDK must be installed
  - pyzed Python package
  - opencv-python

Usage:
  python standalone_svo_to_mp4.py --root /path/to/demos
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
    import numpy as np
except ImportError as e:
    print(f"ERROR: Required dependencies not found: {e}")
    print("Please install:")
    print("  - ZED SDK: https://www.stereolabs.com/developers/release/")
    print("  - pip install opencv-python numpy")
    sys.exit(1)


def convert_svo_to_mp4(
    svo_path: Path,
    mp4_output_path: Path,
    timestamps_output_path: Path,
    fps: int = 15,
    quality: int = 90,
    skip_if_exists: bool = True
) -> bool:
    """
    Convert a single SVO2 file to MP4 with timestamps.

    Args:
        svo_path: Path to the .svo2 file
        mp4_output_path: Path where MP4 will be saved
        timestamps_output_path: Path where timestamps JSON will be saved
        fps: Frames per second for the output video
        quality: JPEG quality for encoding (0-100, higher is better)
        skip_if_exists: Skip conversion if output already exists

    Returns:
        True if successful, False otherwise
    """
    # Skip if already converted
    if skip_if_exists and mp4_output_path.exists() and timestamps_output_path.exists():
        print(f"SKIP (exists): {mp4_output_path}")
        return True

    print(f"Converting: {svo_path.name}")

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
        print(f"  ERROR: Failed to open SVO file: {err}")
        zed.close()
        return False

    # Get video properties
    camera_info = zed.get_camera_information()
    resolution = camera_info.camera_configuration.resolution
    width = resolution.width
    height = resolution.height

    print(f"  Resolution: {width}x{height}")
    print(f"  Target FPS: {fps}")

    # Initialize video writer with H.264 codec
    # Try different codecs in order of preference
    codecs_to_try = [
        ('avc1', 'H.264'),  # H.264
        ('mp4v', 'MPEG-4'),  # MPEG-4
        ('X264', 'X264'),    # X264
    ]

    video_writer = None
    for codec_code, codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            video_writer = cv2.VideoWriter(
                str(mp4_output_path),
                fourcc,
                fps,
                (width, height)
            )
            if video_writer.isOpened():
                print(f"  Using codec: {codec_name}")
                break
            else:
                video_writer.release()
                video_writer = None
        except:
            continue

    if video_writer is None:
        print(f"  ERROR: Could not initialize video writer with any codec")
        zed.close()
        return False

    # Prepare image container
    image = sl.Mat()

    # Storage for timestamps
    timestamps = []
    frame_count = 0
    error_count = 0

    # Get total number of frames for progress
    total_frames = zed.get_svo_number_of_frames()
    print(f"  Total frames: {total_frames}")

    # Process all frames
    try:
        while True:
            err = zed.grab()

            if err == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT)

                # Get timestamp in milliseconds
                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                timestamps.append(timestamp)

                # Convert to numpy array
                frame = image.get_data()

                # Convert from RGBA to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                # Write frame to video
                video_writer.write(frame_bgr)

                frame_count += 1

                # Progress indicator
                if frame_count % 50 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print(f"  Completed: {frame_count} frames processed")
                break
            else:
                error_count += 1
                if error_count > 10:
                    print(f"  ERROR: Too many grab errors ({error_count}), aborting")
                    return False
                continue

    except KeyboardInterrupt:
        print(f"  Interrupted by user")
        return False
    except Exception as e:
        print(f"  ERROR during conversion: {e}")
        return False
    finally:
        # Cleanup
        video_writer.release()
        zed.close()

    # Save timestamps as JSON
    try:
        with open(timestamps_output_path, 'w') as f:
            json.dump(timestamps, f)
        print(f"  Saved timestamps: {timestamps_output_path.name}")
    except Exception as e:
        print(f"  ERROR saving timestamps: {e}")
        return False

    print(f"  ✓ DONE: {mp4_output_path.name}")
    return True


def find_all_svo_files(root: Path) -> List[Path]:
    """Find all .svo2 files under recordings/SVO directories."""
    svo_files = []

    # Search for SVO files in the expected directory structure
    for svo_file in root.rglob("recordings/SVO/*.svo2"):
        svo_files.append(svo_file)

    # Also check for .svo files (older format)
    for svo_file in root.rglob("recordings/SVO/*.svo"):
        svo_files.append(svo_file)

    return sorted(svo_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SVO2 files to MP4 videos with timestamps"
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
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="Video quality 0-100 (default: 90)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 files"
    )

    args = parser.parse_args()

    # Validate root directory
    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: Root directory does not exist: {root}")
        sys.exit(1)

    # Find all SVO files
    svo_files = find_all_svo_files(root)
    if not svo_files:
        print(f"No SVO files found under: {root}")
        print("Expected structure: <demo_folder>/recordings/SVO/*.svo2")
        return

    print("=" * 60)
    print(f"SVO to MP4 Conversion")
    print("=" * 60)
    print(f"Root directory: {root}")
    print(f"Found {len(svo_files)} SVO files")
    print(f"FPS: {args.fps}")
    print(f"Quality: {args.quality}")
    print("=" * 60)
    print()

    # Convert each SVO file
    success_count = 0
    failed_files = []

    for idx, svo_path in enumerate(svo_files, 1):
        print(f"[{idx}/{len(svo_files)}] Processing: {svo_path.relative_to(root)}")

        try:
            # Determine output paths
            recordings_dir = svo_path.parent.parent  # .../recordings
            mp4_dir = recordings_dir / "MP4"
            mp4_dir.mkdir(parents=True, exist_ok=True)

            camera_id = svo_path.stem  # filename without extension
            mp4_output_path = mp4_dir / f"{camera_id}.mp4"
            timestamps_output_path = mp4_dir / f"{camera_id}_timestamps.json"

            # Convert
            success = convert_svo_to_mp4(
                svo_path,
                mp4_output_path,
                timestamps_output_path,
                fps=args.fps,
                quality=args.quality,
                skip_if_exists=not args.overwrite
            )

            if success:
                success_count += 1
            else:
                failed_files.append(str(svo_path.relative_to(root)))

        except Exception as e:
            print(f"  ERROR: Unexpected error: {e}")
            failed_files.append(str(svo_path.relative_to(root)))

        print()

    # Summary
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total files:     {len(svo_files)}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
