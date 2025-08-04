import numpy as np

file_path = f"./preprocessing/collected_demos_training/2025-03-14_move_apple_to_the_right/2025-03-14_20-04-47/indices_and_distances.npz"

indices_and_distances = np.load(file_path)

print(indices_and_distances.keys())

print(indices_and_distances["retrieved_indices"].shape)
print(indices_and_distances["query_indices"].shape)
print(indices_and_distances["distances"].shape)
