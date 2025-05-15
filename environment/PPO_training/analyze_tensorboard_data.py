import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

logdir = "runs"
output_dir = "experiment_summary"
os.makedirs(output_dir, exist_ok=True)

# All metrics including components
metrics = [
    "Reward", "Policy Loss", "Value Loss", "Entropy",
    "RewardComponents/proximity",
    "RewardComponents/move_toward",
    "RewardComponents/hit",
    "RewardComponents/push_right",
    "RewardComponents/push_wrong",
    "RewardComponents/puck_speed",
    "RewardComponents/puck_direction",
    "RewardComponents/passive_penalty",
    "RewardComponents/behind_puck",
    "RewardComponents/overposition",
    "RewardComponents/goal_scored",
    "RewardComponents/goal_conceded",
]

all_data = {metric: [] for metric in metrics}

for run in os.listdir(logdir):
    run_path = os.path.join(logdir, run)
    if not os.path.isdir(run_path):
        continue
    try:
        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        for metric in metrics:
            if metric in tags:
                scalars = ea.Scalars(metric)
                df = pd.DataFrame({
                    "step": [s.step for s in scalars],
                    "value": [s.value for s in scalars],
                    "run": run,
                    "metric": metric
                })
                all_data[metric].append(df)
    except Exception as e:
        print(f"Error reading {run_path}: {e}")

# Save CSVs and generate PNG plots
for metric, dfs in all_data.items():
    if not dfs:
        continue
    df = pd.concat(dfs, ignore_index=True)
    safe_name = metric.replace("/", "_").lower()
    csv_path = os.path.join(output_dir, f"{safe_name}.csv")
    png_path = os.path.join(output_dir, f"{safe_name}.png")

    df.to_csv(csv_path, index=False)

    plt.figure()
    for run_id, group in df.groupby("run"):
        plt.plot(group["step"], group["value"], label=run_id)
    plt.title(f"{metric}")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

print("✅ Export complete. Check the 'experiment_summary/' folder.")
