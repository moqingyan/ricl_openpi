#!/usr/bin/env python3
"""
Script to flip all images vertically in ricl_human_demo directory.
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


def flip_images_in_directory(root_dir, dry_run=False, image_extensions=None):
    """
    Flip all images vertically in the given directory recursively.
    
    Args:
        root_dir: Root directory to search for images
        dry_run: If True, only print what would be done without actually flipping
        image_extensions: List of image extensions to process (default: ['.jpg', '.jpeg', '.png'])
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Convert to lowercase for case-insensitive matching
    image_extensions = [ext.lower() for ext in image_extensions]
    
    # Find all image files
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist!")
        return
    
    print(f"Searching for images in {root_dir}...")
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(root_path.rglob(f"*{ext}"))
        all_image_files.extend(root_path.rglob(f"*{ext.upper()}"))
    
    # Filter to only include images in hand_camera directories
    image_files = [img for img in all_image_files if "hand_camera" in img.parts]
    
    print(f"Found {len(image_files)} images in hand_camera directories to flip")
    
    if dry_run:
        print("\n=== DRY RUN MODE - No images will be modified ===")
        print(f"Would flip {len(image_files)} images")
        if image_files:
            print("\nFirst 10 files that would be flipped:")
            for img_path in image_files[:10]:
                print(f"  {img_path}")
        return
    
    # Flip images
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(image_files, desc="Flipping images"):
        try:
            # Open image
            img = Image.open(img_path)
            
            # Flip vertically (top to bottom)
            flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Save back (overwrite original)
            flipped_img.save(img_path)
            
            success_count += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            error_count += 1
    
    print(f"\n=== Done ===")
    print(f"Successfully flipped: {success_count} images")
    if error_count > 0:
        print(f"Errors: {error_count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip all images vertically in a directory")
    parser.add_argument(
        "--dir",
        type=str,
        default="ricl_human_demo",
        help="Directory to process (default: ricl_human_demo)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually flipping images"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Image extensions to process (default: .jpg .jpeg .png)"
    )
    
    args = parser.parse_args()
    
    # Get the preprocessing directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, args.dir)
    
    print(f"Target directory: {target_dir}")
    
    if not args.dry_run:
        response = input(f"\nWARNING: This will flip ALL images in hand_camera directories under {target_dir} vertically.\n"
                        f"This operation will OVERWRITE the original images.\n"
                        f"Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Operation cancelled.")
            exit(0)
    
    flip_images_in_directory(target_dir, dry_run=args.dry_run, image_extensions=args.extensions)

