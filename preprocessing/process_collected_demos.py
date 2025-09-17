import numpy as np
import json 
import h5py
import os
import argparse
from PIL import Image
from openpi.policies.utils import embed_with_batches, load_dinov2, EMBED_DIM
from openpi_client.image_tools import resize_with_pad
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def process(dir, prompts):
    # load model for embedding images
    dinov2 = load_dinov2()
    logger.info(f'loaded dinov2 for image embedding')

    # get current directory and append the dir argument to get demo_dir
    current_dir = os.path.dirname(os.path.abspath(__file__)) # get current directory
    demo_dir = f"{current_dir}/{dir}"
    logger.info(f'absolute path of the {demo_dir=}')

    # infer base_group and domain from the group folder name
    group_name = os.path.basename(demo_dir)
    if group_name.endswith('_robot'):
        domain = 'robot'
        base_group = group_name[:-len('_robot')]
    elif group_name.endswith('_human'):
        domain = 'human'
        base_group = group_name[:-len('_human')]
    else:
        domain = 'unknown'
        base_group = group_name

    # get all the folders (demos) in the demo_dir
    demo_folders = [f"{demo_dir}/{f}" for f in os.listdir(demo_dir) if os.path.isdir(f"{demo_dir}/{f}")]
    logger.info(f'number of demo folders: {len(demo_folders)}')

    # iterate over the demo_folders and read the trajectory.h5 files and the frames
    for demo_folder in demo_folders:
        if os.path.exists(f"{demo_folder}/processed_demo.npz"):
            logger.info(f'{demo_folder=} already processed')
            continue

        processed_demo = {}
        logger.info(f'processing {demo_folder=}')
        num_steps = None
        if domain != 'human':
            traj_h5 = h5py.File(f"{demo_folder}/trajectory.h5", 'r')
            skip_bools = traj_h5["observation"]["timestamp"]["skip_action"][:]
            keep_bools = ~skip_bools
            obs_gripper_pos = traj_h5["observation"]["robot_state"]["gripper_position"][:].reshape(-1, 1)[keep_bools]
            act_gripper_pos = traj_h5["action"]["gripper_position"][:].reshape(-1, 1)[keep_bools]
            obs_joint_pos = traj_h5["observation"]["robot_state"]["joint_positions"][keep_bools]
            act_joint_vel = traj_h5["action"]["joint_velocity"][keep_bools]
            processed_demo["state"] = np.concatenate([obs_joint_pos, obs_gripper_pos], axis=1)
            processed_demo["actions"] = np.concatenate([act_joint_vel, act_gripper_pos], axis=1)
            num_steps = processed_demo["state"].shape[0]
            assert processed_demo["state"].shape == processed_demo["actions"].shape == (num_steps, 8)

        for camera_name, key in zip(['hand_camera', 'varied_camera_1', 'varied_camera_2'], ['wrist_image', 'top_image', 'right_image']):
            frames_dir = f"{demo_folder}/recordings/frames/{camera_name}"
            logger.info(f'{frames_dir=}')
            frames = [f"{frames_dir}/{f}" for f in os.listdir(frames_dir)]
            if num_steps is None:
                num_steps = len(frames)
            assert len(frames) == num_steps, f'{len(frames)=} {num_steps=}'
            frames = [np.array(Image.open(frame)) for frame in frames]
            frames = np.stack(frames, axis=0)
            assert frames.shape == (num_steps, 720, 1280, 3) and frames.dtype == np.uint8, f'{frames.shape=} {frames.dtype=}'
            frames = resize_with_pad(frames, 224, 224)
            assert frames.shape == (num_steps, 224, 224, 3) and frames.dtype == np.uint8, f'{frames.shape=} {frames.dtype=}'
            processed_demo[key] = frames
            
            embeddings = embed_with_batches(frames, dinov2)
            assert embeddings.shape == (num_steps, EMBED_DIM), f'{embeddings.shape=}'
            processed_demo[f"{key}_embeddings"] = embeddings

        # randomly sample a prompt from the prompts
        prompt = np.random.choice(prompts)
        processed_demo["prompt"] = prompt

        # store metadata useful for cross-domain pairing
        processed_demo["domain"] = domain
        processed_demo["base_group"] = base_group

        # save the processed episode as a npz file
        np.savez(f"{demo_folder}/processed_demo.npz", **processed_demo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--dir_of_dirs", type=str, default=None)
    parser.add_argument("--prompts", nargs="+", type=str, default=None)
    args = parser.parse_args()

    assert args.dir is not None or args.dir_of_dirs is not None, "Either --dir or --dir_of_dirs must be provided"

    if args.dir is not None:
        assert args.prompts is not None, "If --dir is provided, --prompts must also be provided"
        process(args.dir, args.prompts)
    else:
        skipped_dirs = []
        for dir in os.listdir(args.dir_of_dirs):
            # Skip hidden/system files and directories like .git, .gitattributes, .dvc, etc.
            if dir.startswith('.'):
                logger.info(f'Skipping hidden file/directory {dir}')
                skipped_dirs.append(dir)
                continue
            # Only process directories, skip files
            if not os.path.isdir(f"{args.dir_of_dirs}/{dir}"):
                logger.info(f'Skipping file {dir} (not a directory)')
                skipped_dirs.append(dir)
                continue
            # derive prompt from base_group (strip date prefix and domain suffix)
            if dir.endswith('_robot'):
                base_group = dir[:-len('_robot')]
            elif dir.endswith('_human'):
                base_group = dir[:-len('_human')]
            else:
                base_group = dir
            temp_prompts = [" ".join(base_group.split("_")[1:])]
            logger.info(f'**About to start processing dir {args.dir_of_dirs}/{dir} with prompts {temp_prompts}**')
            process(f"{args.dir_of_dirs}/{dir}", temp_prompts)
        logger.info(f"Skipped directories: {skipped_dirs}")

    print(f'done!')






    
