#!/usr/bin/env python3

import os, yaml, matplotlib.pyplot as plt

x_labels = []
mse_vals = []

for file in sorted(os.listdir("logs")):
    if file.endswith(".yaml"):
        with open(os.path.join("logs", file)) as f:
            log = yaml.safe_load(f)
            pid = log["pid"]
            mse = log["metrics"]["mse"]

            label = (
                f"KpL={pid['linear']['kp']}, KiL={pid['linear']['ki']}, KdL={pid['linear']['kd']}\n"
                f"KpA={pid['angular']['kp']}, KiA={pid['angular']['ki']}, KdA={pid['angular']['kd']}"
            )

            x_labels.append(label)
            mse_vals.append(mse)

plt.figure(figsize=(12, 6))
plt.bar(x_labels, mse_vals, color='mediumseagreen')
plt.ylabel("Mean Squared Tracking Error")
plt.title("PID Performance Comparison (Linear & Angular)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(True)
plt.show()