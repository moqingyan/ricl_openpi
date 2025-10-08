import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Create debug output directory
debug_dir = "/tmp/ricl_debug_images"
os.makedirs(debug_dir, exist_ok=True)

print(f"=== RICL Debug: Retrieved vs Query Images ===")
print(f"KNN k: {self._knn_k}")
print(f"Retrieved indices shape: {retrieved_indices.shape}")
print(f"Query embedding shape: {query_embedding.shape}")

# Save query image
query_img = obs["query_top_image"]

query_img_np = np.array(query_img)
query_img_pil = Image.fromarray((query_img_np * 255).astype(np.uint8))
query_path = f"{debug_dir}/query_top_image.jpg"
query_img_pil.save(query_path)
print(f"Saved query image: {query_path}")

# Save retrieved images (current timestep)
for ct in range(self._knn_k):
    ep_idx, step_idx = retrieved_indices[0, ct]
    print(f"Retrieved {ct}: episode {ep_idx}, step {step_idx}")
    
    # Current timestep image
    retrieved_img = more_obs[f"retrieved_{ct}_top_image"]

    retrieved_img_np = np.array(retrieved_img)
    retrieved_img_pil = Image.fromarray((retrieved_img_np * 255).astype(np.uint8))
    retrieved_path = f"{debug_dir}/retrieved_{ct}_top_image.jpg"
    retrieved_img_pil.save(retrieved_path)
    print(f"  Saved current: {retrieved_path}")
    
    # Next timestep image (if available)
    if f"retrieved_{ct}_next_top_image" in more_obs:
        next_img = more_obs[f"retrieved_{ct}_next_top_image"]
        next_img_np = np.array(next_img)
        next_img_pil = Image.fromarray((next_img_np * 255).astype(np.uint8))
        next_path = f"{debug_dir}/retrieved_{ct}_next_top_image.jpg"
        next_img_pil.save(next_path)
        print(f"  Saved next: {next_path}")

# Create side-by-side comparison
fig, axes = plt.subplots(2, self._knn_k + 1, figsize=(4 * (self._knn_k + 1), 8))
fig.suptitle("Query vs Retrieved Images")

# Top row: current timestep
axes[0, 0].imshow(query_img_np)
axes[0, 0].set_title("Query")
axes[0, 0].axis('off')

for ct in range(self._knn_k):
    ep_idx, step_idx = retrieved_indices[0, ct]
    retrieved_img = more_obs[f"retrieved_{ct}_top_image"]
    axes[0, ct + 1].imshow(np.array(retrieved_img))
    axes[0, ct + 1].set_title(f"Retrieved {ct}\n(ep{ep_idx}, step{step_idx})")
    axes[0, ct + 1].axis('off')

# Bottom row: next timestep (if available)
axes[1, 0].text(0.5, 0.5, "Query\n(no next)", ha='center', va='center', transform=axes[1, 0].transAxes)
axes[1, 0].axis('off')

for ct in range(self._knn_k):
    ep_idx, step_idx = retrieved_indices[0, ct]
    if f"retrieved_{ct}_next_top_image" in more_obs:
        next_img = more_obs[f"retrieved_{ct}_next_top_image"]
        axes[1, ct + 1].imshow(np.array(next_img))
        axes[1, ct + 1].set_title(f"Next {ct}\n(ep{ep_idx}, step{step_idx}+5)")
    else:
        axes[1, ct + 1].text(0.5, 0.5, "No next\nimage", ha='center', va='center', transform=axes[1, ct + 1].transAxes)
    axes[1, ct + 1].axis('off')

comparison_path = f"{debug_dir}/comparison.jpg"
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved comparison: {comparison_path}")
print(f"All debug images saved to: {debug_dir}")
print("=== End Debug ===")