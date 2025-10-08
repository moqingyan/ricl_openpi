#!/usr/bin/env python3
"""
Script to horizontally mirror frames from varied_camera_1 and varied_camera_2.
Run this before process_collected_demos.py to flip the camera images.
"""

import os
import argparse
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def mirror_camera_frames(demo_dir):
    """Mirror frames for varied_camera_1 and varied_camera_2 in all demo folders."""

    # Handle both absolute and relative paths
    if os.path.isabs(demo_dir):
        abs_demo_dir = demo_dir
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_demo_dir = os.path.join(current_dir, demo_dir)

    logger.info(f'Processing directory: {abs_demo_dir}')

    # Get all demo folders
    demo_folders = [f for f in os.listdir(abs_demo_dir)
                   if os.path.isdir(os.path.join(abs_demo_dir, f))]

    logger.info(f'Found {len(demo_folders)} demo folders')

    # Process each demo folder
    for demo_folder in demo_folders:
        demo_path = os.path.join(abs_demo_dir, demo_folder)
        frames_base = os.path.join(demo_path, 'recordings', 'frames')

        if not os.path.exists(frames_base):
            logger.warning(f'No frames directory in {demo_path}, skipping')
            continue

        # Process varied_camera_1 and varied_camera_2
        for camera_name in ['varied_camera_1', 'varied_camera_2']:
            camera_dir = os.path.join(frames_base, camera_name)

            if not os.path.exists(camera_dir):
                logger.warning(f'Camera directory not found: {camera_dir}, skipping')
                continue

            logger.info(f'Mirroring frames in {camera_dir}')

            # Get all image files
            image_files = [f for f in os.listdir(camera_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Mirror each image
            for img_file in image_files:
                img_path = os.path.join(camera_dir, img_file)

                try:
                    # Open image, mirror it, and save back
                    img = Image.open(img_path)
                    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    mirrored_img.save(img_path)
                except Exception as e:
                    logger.error(f'Error processing {img_path}: {e}')

            logger.info(f'Mirrored {len(image_files)} frames in {camera_name}')

    logger.info('Done mirroring all camera frames!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mirror varied_camera_1 and varied_camera_2 frames horizontally')
    parser.add_argument('--dir', type=str, required=True,
                       help='Directory containing demo folders')

    args = parser.parse_args()
    mirror_camera_frames(args.dir)
