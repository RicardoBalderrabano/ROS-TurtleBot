#!/usr/bin/env python3
import rosbag, yaml, sys, os
import numpy as np

def analyze(bag_path, kp_lin, ki_lin, kd_lin, kp_ang, ki_ang, kd_ang):
    errors = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=['/tracking_error']):
            errors.append(msg.data)

    if errors:
        mse = float(np.mean(np.square(errors)))
        log_data = {
            'pid': {
                'linear': {
                    'kp': kp_lin,
                    'ki': ki_lin,
                    'kd': kd_lin
                },
                'angular': {
                    'kp': kp_ang,
                    'ki': ki_ang,
                    'kd': kd_ang
                }
            },
            'metrics': {'mse': mse}
        }
        name = os.path.splitext(os.path.basename(bag_path))[0]
        log_path = os.path.join("logs", f"{name}.yaml")
        with open(log_path, "w") as f:
            yaml.dump(log_data, f)
        print(f"Logged: {log_path} (MSE={mse:.4f})")
    else:
        print("⚠ No tracking error data found in bag.")

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: analyze_and_log.py <bag_path> <kp_lin> <ki_lin> <kd_lin> <kp_ang> <ki_ang> <kd_ang>")
        sys.exit(1)

    analyze(
        sys.argv[1],
        float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
        float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    )