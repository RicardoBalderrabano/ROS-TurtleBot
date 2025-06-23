#!/usr/bin/env python3

import yaml, os

def load_logs(folder):
    results = []
    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            with open(os.path.join(folder, file)) as f:
                results.append(yaml.safe_load(f))
    return sorted(results, key=lambda x: x["metrics"]["mse"])

logs = load_logs("logs")

print("🏁 PID Configs Ranked by MSE:")
for i, log in enumerate(logs):
    p = log['pid']
    mse = log['metrics']['mse']
    print(f"{i+1}. "
          f"KpL={p['linear']['kp']} KiL={p['linear']['ki']} KdL={p['linear']['kd']} | "
          f"KpA={p['angular']['kp']} KiA={p['angular']['ki']} KdA={p['angular']['kd']} "
          f"→ MSE={mse:.4f}")