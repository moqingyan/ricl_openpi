import os
from huggingface_hub import HfApi, create_repo, upload_folder
api = HfApi()

demo_type = "human"
next_frame = "fifth"
time_step = 6900

repo_id = f"zhengyc02/pi0_ricl_{demo_type}_demo_{next_frame}_frame_{time_step}"
folder = f"{os.path.expanduser('~')}/ricl_openpi/checkpoints/pi0_fast_droid_ricl/ricl_human_demo_flipped_wrist_back/{time_step}"
commit_message=f"Flipped varied camera 1 and 2. pi0_ricl with {demo_type} demo, using next {next_frame} frame as latent action, time step = {time_step}."

create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder,
    path_in_repo="",                
    commit_message=commit_message,
    ignore_patterns=["**/wandb/**"],    # optional exclusions
)
print("Uploaded to:", repo_id)
