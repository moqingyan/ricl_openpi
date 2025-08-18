import os
from huggingface_hub import HfApi, create_repo, upload_folder
api = HfApi()
repo_id = "zhengyc02/pi0_fast_droid_ricl_no_action_interpolation_3000"
folder = f"{os.path.expanduser('~')}/ricl_openpi/checkpoints/pi0_fast_droid_ricl/{{ricl_retrain}}/3000"
create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder,                 # note: braces are literal in your path; keep the quotes
    path_in_repo="",                    
    commit_message="pi0_fast_droid_ricl retrain with no action interpolation, time step = 3000.",
    ignore_patterns=["**/wandb/**"],    # optional exclusions
)
print("Uploaded to:", repo_id)
