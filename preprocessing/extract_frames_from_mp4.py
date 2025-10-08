#!/usr/bin/env python3
"""
Extract frames from MP4 videos in demo recordings.

This script processes demo folders containing MP4 videos and extracts them into
individual frames organized by camera name, matching the expected structure for
process_collected_demos.py.

Expected input structure:
  demo_folder/
    recordings/
      MP4/
        <camera_id_1>.mp4
        <camera_id_2>.mp4
        <camera_id_3>.mp4

Output structure:
  demo_folder/
    recordings/
      frames/
        hand_camera/
          000.jpg
          001.jpg
          ...
        varied_camera_1/
          000.jpg
          001.jpg
          ...
        varied_camera_2/
          000.jpg
          001.jpg
          ...
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


# Map camera IDs to camera names expected by process_collected_demos.py
# You may need to adjust this mapping based on your camera IDs
CAMERA_ID_TO_NAME = {
    "15512737": "hand_camera",      # wrist/hand camera
    "25455306": "varied_camera_1",  # top camera
    "27085680": "varied_camera_2",  # right/side camera
}


def extract_frames_from_video(video_path: Path, output_dir: Path) -> int:
    """
    Extract frames from a video file using ffmpeg.

    Args:
        video_path: Path to the input MP4 file
        output_dir: Directory to save extracted frames

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to extract frames as JPG with zero-padded filenames
    # -q:v 2 sets high quality for JPEG (1-31, lower is better)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:v", "2",
        str(output_dir / "%03d.jpg")
    ]

    logger.info(f"Extracting frames from {video_path.name} to {output_dir}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Count extracted frames
        frame_count = len(list(output_dir.glob("*.jpg")))
        logger.info(f"  Extracted {frame_count} frames")
        return frame_count

    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting frames: {e.stderr}")
        raise


def process_demo_folder(demo_folder: Path, camera_id_map: dict = None) -> None:
    """
    Process a single demo folder by extracting frames from all MP4 videos.

    Args:
        demo_folder: Path to demo folder containing recordings/MP4/
        camera_id_map: Optional custom mapping from camera IDs to camera names
    """
    if camera_id_map is None:
        camera_id_map = CAMERA_ID_TO_NAME

    mp4_dir = demo_folder / "recordings" / "MP4"
    frames_base_dir = demo_folder / "recordings" / "frames"

    if not mp4_dir.exists():
        logger.warning(f"No MP4 directory found in {demo_folder}")
        return

    # Find all MP4 files
    mp4_files = list(mp4_dir.glob("*.mp4"))

    if not mp4_files:
        logger.warning(f"No MP4 files found in {mp4_dir}")
        return

    logger.info(f"Processing {demo_folder.name}")
    logger.info(f"  Found {len(mp4_files)} MP4 files")

    for mp4_file in mp4_files:
        # Extract camera ID from filename (without extension)
        camera_id = mp4_file.stem

        # Map to camera name
        camera_name = camera_id_map.get(camera_id)

        if camera_name is None:
            logger.warning(f"  Unknown camera ID: {camera_id}, skipping")
            continue

        # Create output directory for this camera
        output_dir = frames_base_dir / camera_name

        # Extract frames
        frame_count = extract_frames_from_video(mp4_file, output_dir)

    logger.info(f"  Completed {demo_folder.name}")


def process_group_folder(group_folder: Path, camera_id_map: dict = None) -> None:
    """
    Process all demo folders within a group folder.

    Args:
        group_folder: Path to group folder containing multiple demo folders
        camera_id_map: Optional custom mapping from camera IDs to camera names
    """
    demo_folders = [f for f in group_folder.iterdir() if f.is_dir()]

    logger.info(f"Found {len(demo_folders)} demo folders in {group_folder.name}")

    for demo_folder in demo_folders:
        try:
            process_demo_folder(demo_folder, camera_id_map)
        except Exception as e:
            logger.error(f"Error processing {demo_folder}: {e}")


def load_camera_mapping(mapping_file: Path) -> dict:
    """Load camera ID to name mapping from a JSON file."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from MP4 videos in demo recordings"
    )
    parser.add_argument(
        "--demo_folder",
        type=str,
        help="Path to a single demo folder"
    )
    parser.add_argument(
        "--group_folder",
        type=str,
        help="Path to a group folder containing multiple demo folders"
    )
    parser.add_argument(
        "--camera_map",
        type=str,
        help="Path to JSON file mapping camera IDs to names (optional)"
    )

    args = parser.parse_args()

    # Load custom camera mapping if provided
    camera_id_map = CAMERA_ID_TO_NAME
    if args.camera_map:
        camera_id_map = load_camera_mapping(Path(args.camera_map))
        logger.info(f"Loaded camera mapping: {camera_id_map}")

    if args.demo_folder:
        demo_path = Path(args.demo_folder)
        if not demo_path.exists():
            logger.error(f"Demo folder does not exist: {demo_path}")
            return
        process_demo_folder(demo_path, camera_id_map)

    elif args.group_folder:
        group_path = Path(args.group_folder)
        if not group_path.exists():
            logger.error(f"Group folder does not exist: {group_path}")
            return
        process_group_folder(group_path, camera_id_map)

    else:
        parser.print_help()
        logger.error("Please provide either --demo_folder or --group_folder")
        return

    logger.info("Done!")


if __name__ == "__main__":
    main()
