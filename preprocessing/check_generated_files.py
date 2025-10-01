import os


directory = "./preprocessing/ricl_human_demo"

for task_folder in os.listdir(directory):
    if task_folder.startswith("."):
        continue
    if not os.path.isdir(f"{directory}/{task_folder}"):
        continue
    for ep_folder in os.listdir(f"{directory}/{task_folder}"):
        assert os.path.exists(f"{directory}/{task_folder}/{ep_folder}/processed_demo.npz"), f"{directory}/{task_folder}/{ep_folder}/processed_demo.npz does not exist"
        if task_folder.endswith("_robot"):
            assert os.path.exists(f"{directory}/{task_folder}/{ep_folder}/indices_and_distances.npz"), f"{directory}/{task_folder}/{ep_folder}/indices_and_distances.npz does not exist"

print("Done checking all files")