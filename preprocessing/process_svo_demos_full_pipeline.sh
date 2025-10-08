#!/bin/bash
# Full pipeline to process SVO demos:
# 1. Convert SVO to MP4
# 2. Extract frames from MP4 (with --clip_half for stereo videos)
# 3. Reorganize frames to camera names
# 4. Process demos with process_collected_demos.py

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <demo_root_folder> [--clip_half]"
    echo "Example: $0 /media/amishsethi/Elements/stack_rgb_ordered"
    echo "Example: $0 /media/amishsethi/Elements/stack_rgb_ordered --clip_half"
    exit 1
fi

DEMO_ROOT="$1"
CLIP_HALF=""

if [ "$#" -eq 2 ] && [ "$2" == "--clip_half" ]; then
    CLIP_HALF="--clip_half"
    echo "Using --clip_half to extract left half of stereo videos"
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_FRAMES_SCRIPT="/media/amishsethi/Elements/chris_data_aug_29_elements/extract_frames.py"
REORGANIZE_SCRIPT="$SCRIPT_DIR/reorganize_frames_by_camera_name.py"
CONVERT_SVO_SCRIPT="$SCRIPT_DIR/convert_svo_to_mp4.py"

echo "========================================="
echo "Full SVO Demo Processing Pipeline"
echo "========================================="
echo "Demo root: $DEMO_ROOT"
echo ""

# Step 1: Convert SVO to MP4
echo "Step 1: Converting SVO files to MP4..."
python3 "$CONVERT_SVO_SCRIPT" --root "$DEMO_ROOT"
echo ""

# Step 2: Extract frames from MP4
echo "Step 2: Extracting frames from MP4 files..."
cd "$DEMO_ROOT"
python3 "$EXTRACT_FRAMES_SCRIPT" --root . $CLIP_HALF
echo ""

# Step 3: Reorganize frames by camera name
echo "Step 3: Reorganizing frames to camera names..."
for demo_folder in "$DEMO_ROOT"/*/; do
    if [ -d "$demo_folder" ]; then
        echo "  Processing: $(basename "$demo_folder")"
        python3 "$REORGANIZE_SCRIPT" --demo_folder "$demo_folder"
    fi
done
echo ""

echo "========================================="
echo "Pipeline complete!"
echo "========================================="
echo ""
echo "Frames are now organized and ready for processing."
echo "To process demos with DINOv2 embeddings, run:"
echo "  cd /home/amishsethi/OUR_RICL/ricl_openpi"
echo "  python preprocessing/process_collected_demos.py --dir_of_dirs=\"$DEMO_ROOT\""
