#!/usr/bin/env python3
import os
import subprocess
import time
import yaml
import sys
from signal import SIGTERM

# Configuration
PID_CONFIGS = [
    {
        "kp_linear": 0.5, "ki_linear": 0.01, "kd_linear": 0.1,
        "kp_angular": 1.0, "ki_angular": 0.05, "kd_angular": 0.2
    },
        {
        "kp_linear": 0.8, "ki_linear": 0.02, "kd_linear": 0.3,
        "kp_angular": 1.5, "ki_angular": 0.1, "kd_angular": 0.5
    },
        {
        "kp_linear": 1.0, "ki_linear": 0.5, "kd_linear": 0.5,
        "kp_angular": 2.0, "ki_angular": 0.1, "kd_angular": 0.5
    },
        {
        "kp_linear": 0.4, "ki_linear": 0.01, "kd_linear": 0.05,
        "kp_angular": 0.8, "ki_angular": 0.01, "kd_angular": 0.1
    },
    # ... other configs
]

TEST_DURATION = 100
BAG_DIR = os.path.abspath("bags")
CONFIG_DIR = os.path.abspath("config")
LOG_DIR = os.path.abspath("logs")
LAUNCH_FILE = "main.launch"

def check_ros_running():
    try:
        subprocess.run(["rostopic", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        print("❌ ROS is not running! Run 'roscore' first.")
        return False

def setup_directories():
    os.makedirs(BAG_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def run_test(pid_params):
    name = f"kpL{pid_params['kp_linear']}_kiL{pid_params['ki_linear']}_kdL{pid_params['kd_linear']}_kpA{pid_params['kp_angular']}_kiA{pid_params['ki_angular']}_kdA{pid_params['kd_angular']}"
    print(f"\n=== Testing: {name} ===")

    # 1. Create config
    config_path = os.path.join(CONFIG_DIR, "pid.yaml")
    with open(config_path, "w") as f:
        yaml.dump(pid_params, f)

    # 2. Start recording
    bag_file = os.path.join(BAG_DIR, f"{name}.bag")
    bag_proc = subprocess.Popen(
        ["rosbag", "record", "-O", bag_file, "/tracking_error", "/cmd_vel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)  # Wait for recorder to initialize

    # 3. Launch system
    launch_proc = subprocess.Popen(
        ["roslaunch", "balderrabano_rodriguez", LAUNCH_FILE, f"config:={config_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 4. Run test
    try:
        for remaining in range(TEST_DURATION, 0, -1):
            print(f"\rTime remaining: {remaining}s", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTest interrupted!")

    # 5. Cleanup
    launch_proc.terminate()
    bag_proc.terminate()
    time.sleep(3)

    # 6. Verify and analyze
    if os.path.exists(bag_file):
        analyze_results(bag_file, pid_params)
    else:
        print("❌ No bag file created!")

def analyze_results(bag_file, pid_params):
    script_path = os.path.join(os.path.dirname(__file__), "analyze_and_log.py")
    if os.path.exists(script_path):
        result = subprocess.run(
            [
                "python3", script_path,
                bag_file,
                str(pid_params['kp_linear']),
                str(pid_params['ki_linear']),
                str(pid_params['kd_linear']),
                str(pid_params['kp_angular']),
                str(pid_params['ki_angular']),
                str(pid_params['kd_angular'])
            ],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    else:
        print("Analysis script not found at", script_path)

if __name__ == "__main__":
    if not check_ros_running():
        sys.exit(1)
        
    setup_directories()
    print(f"Starting PID tests with {len(PID_CONFIGS)} configurations")
    
    for config in PID_CONFIGS:
        run_test(config)
    
    print("\nAll tests completed! Check bags/ and logs/ directories.")