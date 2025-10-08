#!/usr/bin/env python3
"""
Reorganize frames from camera ID folders to camera name folders.

This script renames/moves frame directories from camera IDs (e.g., 15512737)
to camera names (e.g., hand_camera) to match the expected structure for
process_collected_demos.py.
"""

import argparse
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Map camera IDs to camera names
CAMERA_ID_TO_NAME = {
    "15512737": "hand_camera",
    "25455306": "varied_camera_1",
    "26368109": "varied_camera_1",  # Alternative camera ID
    "27085680": "varied_camera_2",
}


def reorganize_frames(demo_folder: Path, camera_map: dict = None) -> None:
    """Reorganize frames from camera ID folders to camera name folders."""
    if camera_map is None:
        camera_map = CAMERA_ID_TO_NAME

    frames_dir = demo_folder / "recordings" / "frames"

    if not frames_dir.exists():
        logger.warning(f"No frames directory in {demo_folder}")
        return

    logger.info(f"Processing {demo_folder.name}")

    for camera_id, camera_name in camera_map.items():
        id_dir = frames_dir / camera_id
        name_dir = frames_dir / camera_name

        if not id_dir.exists():
            continue

        # If name_dir exists, remove it first
        if name_dir.exists():
            logger.info(f"  Removing existing {camera_name}")
            shutil.rmtree(name_dir)

        # Move/rename the directory
        logger.info(f"  Moving {camera_id} -> {camera_name}")
        shutil.move(str(id_dir), str(name_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize frame directories from camera IDs to camera names"
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

    args = parser.parse_args()

    if args.demo_folder:
        demo_path = Path(args.demo_folder)
        if not demo_path.exists():
            logger.error(f"Demo folder does not exist: {demo_path}")
            return
        reorganize_frames(demo_path)

    elif args.group_folder:
        group_path = Path(args.group_folder)
        if not group_path.exists():
            logger.error(f"Group folder does not exist: {group_path}")
            return

        demo_folders = [f for f in group_path.iterdir() if f.is_dir()]
        logger.info(f"Found {len(demo_folders)} demo folders")

        for demo_folder in demo_folders:
            try:
                reorganize_frames(demo_folder)
            except Exception as e:
                logger.error(f"Error processing {demo_folder}: {e}")

    else:
        parser.print_help()
        logger.error("Please provide either --demo_folder or --group_folder")
        return

    logger.info("Done!")


if __name__ == "__main__":
    main()
