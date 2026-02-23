import json
import matplotlib.pyplot as plt
import numpy as np
import os

# === Settings ===
input_path = "output/kitti_res50_90kIter/metrics.json"
output_dir = "output/kitti_res50_90kIter"
os.makedirs(output_dir, exist_ok=True)

# === Load and filter metrics ===
with open(input_path, "r") as f:
    data = [json.loads(line) for line in f if "iteration" in line]

filtered = [d for d in data if "total_loss" in d and d["iteration"] > 2]
if not filtered:
    print("No valid training data found.")
    exit()

iterations = [d["iteration"] for d in filtered]
total_loss = [d.get("total_loss", 0) for d in filtered]

# === Define valid prefixes
valid_prefixes = ("loss_bbox", "loss_ce", "loss_giou")

# === Extract all component loss keys
loss_keys = set()
for d in filtered:
    for key in d:
        if key.startswith(valid_prefixes) and key != "total_loss":
            loss_keys.add(key)

# === Build component series
components = {
    key: [d.get(key, float("nan")) for d in filtered] for key in sorted(loss_keys)
}

# === Plot 1: Total Loss ===
plt.figure(figsize=(10, 6))
plt.plot(iterations, total_loss, label="Total Loss", color="blue", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.title("Total Loss vs Iteration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_total.png"))
plt.close()

# === Plot 2: All Component Losses ===
plt.figure(figsize=(12, 8))
for key in components:
    plt.plot(iterations, components[key], label=key)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Components vs Iteration")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_components.png"))
plt.close()

# === Plot 3: Total Loss (Log Iteration) ===
log_iterations = np.log10(iterations)
plt.figure(figsize=(10, 6))
plt.plot(log_iterations, total_loss, label="Total Loss", color="blue", linewidth=2)
plt.xlabel("log10(Iteration)")
plt.ylabel("Total Loss")
plt.title("Total Loss vs log(Iteration)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_total_loss.png"))
plt.close()

# === Plot 4: Component Losses (Log Iteration) ===
plt.figure(figsize=(12, 8))
for key in components:
    plt.plot(log_iterations, components[key], label=key)
plt.xlabel("log10(Iteration)")
plt.ylabel("Loss")
plt.title("Loss Components vs log(Iteration)")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_loss_components.png"))
plt.close()

print("All plots saved in:", output_dir)
