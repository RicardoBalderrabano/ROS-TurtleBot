#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

class MetricsLogger:
    def __init__(self):
        # Trajectory Data
        self.actual_traj = []       # Actual robot path [(x, y, timestamp)]
        self.planned_segments = []  # List of planned path segments [segment1, segment2, ...]
        self.current_segment = []   # Active planned segment
        
        # Error Metrics
        self.segment_errors = []    # RMSE per segment
        self.max_deviations = []    # Max deviation per segment
        self.tracking_errors = []   # Time-series of tracking errors
        
        # ROS Interfaces
        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        rospy.Subscriber("/global_planner/path", Path, self.path_cb)
        rospy.Subscriber("/tracking_error", Float32, self.tracking_error_cb)
        
        self.error_pub = rospy.Publisher("/tracking_error_metrics", Float32, queue_size=10)
        
        rospy.on_shutdown(self.save_plot_and_metrics)

    def odom_cb(self, msg):
        """Log actual trajectory with timestamps."""
        pos = msg.pose.pose.position
        self.actual_traj.append((pos.x, pos.y, rospy.Time.now().to_sec()))

    def path_cb(self, msg):
        """Store new planned path segment."""
        segment = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.planned_segments.append(segment)
        self.current_segment = segment
        rospy.loginfo(f"New path segment received (length: {len(segment)})")

    def tracking_error_cb(self, msg):
        """Log tracking error values."""
        self.tracking_errors.append(msg.data)
        self.error_pub.publish(msg)  # Optional: republish for other nodes

    def compute_segment_metrics(self):
        """Calculate RMSE and max deviation for each segment."""
        if not self.actual_traj or not self.planned_segments:
            rospy.logwarn("Insufficient data for metrics computation.")
            return
        
        actual = np.array([(x, y) for x, y, _ in self.actual_traj])
        
        for i, segment in enumerate(self.planned_segments):
            if len(segment) < 2:
                continue
                
            # Align actual trajectory to segment length
            segment_actual = actual[:len(segment)]
            segment_planned = np.array(segment[:len(segment_actual)])
            
            # Compute errors
            errors = np.linalg.norm(segment_actual - segment_planned, axis=1)
            rmse = np.sqrt(np.mean(errors**2))
            max_dev = np.max(errors)
            
            self.segment_errors.append(rmse)
            self.max_deviations.append(max_dev)
            rospy.loginfo(
                f"Segment {i+1} Metrics: RMSE = {rmse:.3f}m, Max Dev = {max_dev:.3f}m"
            )

    def save_plot_and_metrics(self):
        """Generate plots and log metrics on shutdown."""
        if not self.actual_traj or not self.planned_segments:
            rospy.logwarn("No data to save.")
            return

        # Plot Trajectories
        plt.figure(figsize=(10, 6))
        actual = np.array([(x, y) for x, y, _ in self.actual_traj])
        
        # Plot all segments
        for i, segment in enumerate(self.planned_segments):
            seg = np.array(segment)
            plt.plot(seg[:, 0], seg[:, 1], '--', label=f'Planned Segment {i+1}')
        
        plt.plot(actual[:, 0], actual[:, 1], 'b-', linewidth=2, label='Actual Path')
        
        # Annotate with metrics
        for i, (err, dev) in enumerate(zip(self.segment_errors, self.max_deviations)):
            plt.figtext(
                0.15, 0.85 - i*0.05,
                f"Segment {i+1}: RMSE={err:.2f}m, Max Dev={dev:.2f}m",
                fontsize=9
            )
        
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Trajectory Tracking Performance")
        plt.legend()
        plt.grid(True)
        
        # Save to timestamped file
        timestamp = rospy.Time.now().to_sec()
        save_path = f"/home/ricardo/Pictures/trajectory_plot_{timestamp}.png"
        plt.savefig(save_path)
        rospy.loginfo(f"Saved plot to {save_path}")

        # Plot Tracking Error Over Time
        if self.tracking_errors:
            plt.figure()
            plt.plot(self.tracking_errors)
            plt.xlabel("Time Steps")
            plt.ylabel("Tracking Error (m)")
            plt.title("Tracking Error Over Time")
            error_plot_path = f"/home/ricardo/Pictures/tracking_error_{timestamp}.png"
            plt.savefig(error_plot_path)
            rospy.loginfo(f"Saved error plot to {error_plot_path}")

if __name__ == '__main__':
    rospy.init_node('metrics_logger_node')
    logger = MetricsLogger()
    rospy.spin()