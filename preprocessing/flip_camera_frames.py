#!/usr/bin/env python3
"""
Flip frames vertically for specific camera views.

This script flips all frames in specified camera directories vertically.
Useful for correcting camera orientation issues.
"""

import argparse
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def flip_frames_in_directory(frames_dir: Path) -> int:
    """
    Flip all .jpg frames in a directory vertically.

    Args:
        frames_dir: Directory containing frames to flip

    Returns:
        Number of frames flipped
    """
    if not frames_dir.exists():
        logger.warning(f"Directory does not exist: {frames_dir}")
        return 0

    frame_files = sorted(frames_dir.glob("*.jpg"))

    if not frame_files:
        logger.warning(f"No .jpg files found in {frames_dir}")
        return 0

    logger.info(f"Flipping {len(frame_files)} frames in {frames_dir.name}")

    for frame_path in frame_files:
        try:
            # Open image
            img = Image.open(frame_path)

            # Flip vertically
            flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)

            # Save back to same location
            flipped_img.save(frame_path)

        except Exception as e:
            logger.error(f"Error flipping {frame_path.name}: {e}")
            continue

    logger.info(f"  ✓ Flipped {len(frame_files)} frames")
    return len(frame_files)


def process_demo_folder(demo_folder: Path, camera_names: list) -> None:
    """
    Flip frames for specified cameras in a demo folder.

    Args:
        demo_folder: Path to demo folder
        camera_names: List of camera names to flip (e.g., ['varied_camera_1', 'varied_camera_2'])
    """
    frames_base_dir = demo_folder / "recordings" / "frames"

    if not frames_base_dir.exists():
        logger.warning(f"No frames directory in {demo_folder.name}")
        return

    logger.info(f"Processing: {demo_folder.name}")

    total_flipped = 0
    for camera_name in camera_names:
        camera_dir = frames_base_dir / camera_name
        flipped_count = flip_frames_in_directory(camera_dir)
        total_flipped += flipped_count

    logger.info(f"  Total flipped: {total_flipped} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Flip camera frames vertically"
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
        "--cameras",
        nargs="+",
        default=["varied_camera_1", "varied_camera_2"],
        help="Camera names to flip (default: varied_camera_1 varied_camera_2)"
    )

    args = parser.parse_args()

    if args.demo_folder:
        demo_path = Path(args.demo_folder)
        if not demo_path.exists():
            logger.error(f"Demo folder does not exist: {demo_path}")
            return
        process_demo_folder(demo_path, args.cameras)

    elif args.group_folder:
        group_path = Path(args.group_folder)
        if not group_path.exists():
            logger.error(f"Group folder does not exist: {group_path}")
            return

        demo_folders = [f for f in group_path.iterdir() if f.is_dir()]
        logger.info(f"Found {len(demo_folders)} demo folders")
        logger.info(f"Cameras to flip: {', '.join(args.cameras)}")
        logger.info("")

        for demo_folder in demo_folders:
            try:
                process_demo_folder(demo_folder, args.cameras)
            except Exception as e:
                logger.error(f"Error processing {demo_folder.name}: {e}")

    else:
        parser.print_help()
        logger.error("Please provide either --demo_folder or --group_folder")
        return

    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()
