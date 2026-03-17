import math
import matplotlib.pyplot as plt

# Learning rate config
lr_config = {
    "initial_lr": 0.000025,
    "final_lr": 0.0000025,
    "lr_decay_start": 100,
    "num_rounds": 150,
    "lr_schedule": "cosine",  # Options: "cosine", "exponential", "linear"
}

# Define learning rate scheduler function
def calculate_federated_lr(server_round, lr_config):
    schedule = lr_config.get("lr_schedule", "cosine")
    initial_lr = lr_config.get("initial_lr", 0.000025)
    final_lr = lr_config.get("final_lr", 0.0000025)
    decay_start = lr_config.get("lr_decay_start", 100)
    total_rounds = lr_config.get("num_rounds", 150)

    if server_round < decay_start:
        return initial_lr

    progress = (server_round - decay_start) / (total_rounds - decay_start)
    progress = min(1.0, progress)

    if schedule == "cosine":
        return final_lr + (initial_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == "exponential":
        decay_factor = (final_lr / initial_lr) ** progress
        return initial_lr * decay_factor
    else:  # linear
        return initial_lr - (initial_lr - final_lr) * progress

# Generate learning rates over rounds
rounds_lr = [calculate_federated_lr(r, lr_config) for r in range(lr_config["num_rounds"])]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(lr_config["num_rounds"]), rounds_lr, marker="o", label="Learning Rate")
plt.axvline(x=lr_config["lr_decay_start"], color="red", linestyle="--", label="Decay Start")
plt.title("Federated Learning Rate Decay (Cosine Schedule)")
plt.xlabel("Federated Round")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("fl_lr_schedule.png")
