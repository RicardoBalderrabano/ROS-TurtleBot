#!/usr/bin/env python3
import rospy
import actionlib
import numpy as np
from Balderrabano_Rodriguez.msg import ControllerFeedback, ControllerResult, ControllerAction
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion

class PIDController:
    def __init__(self):
        # Initialize action interface
        self._feedback = ControllerFeedback()
        self._result = ControllerResult()
        
        # PID Gains
        self.kp_linear = 0.4
        self.ki_linear = 0.01
        self.kd_linear = 0.05
        self.kp_angular = 0.8
        self.ki_angular = 0.01
        self.kd_angular = 0.1
        
        # Error accumulators
        self.integral_linear = 0
        self.integral_angular = 0
        self.prev_error_linear = 0
        self.prev_error_angular = 0
        
        # Robot constraints (TurtleBot3 Waffle)
        self.max_linear_vel = 0.26    # m/s
        self.max_angular_vel = 1.82   # rad/s
        self.wheelbase = 0.160        # Distance between wheels (meters)
        self.min_turn_radius = 0.1    # Minimum safe turning radius
        
        # State
        self.current_pose = np.zeros(3)  # x, y, theta
        self.target_point = np.zeros(2)  # x, y
        
        # ROS Setup
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self._as = actionlib.SimpleActionServer(
            "controller_as", 
            ControllerAction, 
            execute_cb=self.execute_cb, 
            auto_start=False
        )
        self._as.start()
        
        rospy.loginfo("PID Controller with curvature limiting ready")

    def odom_callback(self, msg):
        """Update current robot pose."""
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, self.current_pose[2] = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

    def apply_curvature_limits(self, linear_vel, angular_vel):
        """
        Enforces minimum turning radius constraints.
        Returns adjusted (v, ω) that respect kinematic constraints.
        """
        # Avoid division by zero
        if abs(linear_vel) < 0.001:
            return linear_vel, angular_vel
            
        # Calculate current curvature
        curvature = abs(angular_vel / linear_vel)
        max_curvature = 1.0 / self.min_turn_radius
        
        # If exceeding maximum curvature
        if curvature > max_curvature:
            # Scale down both velocities proportionally (smoother)
            scaling_factor = max_curvature / curvature
            linear_vel *= scaling_factor
            angular_vel *= scaling_factor
            
            rospy.logdebug(f"Curvature limited! Original: {curvature:.2f}, New: {max_curvature:.2f}")
        
        return linear_vel, angular_vel

    def calculate_pid(self):
        """Compute PID control outputs with curvature constraints."""
        # Position error
        dx = self.target_point[0] - self.current_pose[0]
        dy = self.target_point[1] - self.current_pose[1]
        distance_error = np.sqrt(dx**2 + dy**2)
        
        # Angle error (wrapped to [-π, π])
        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - self.current_pose[2]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # PID terms (linear velocity)
        self.integral_linear += distance_error
        derivative_linear = distance_error - self.prev_error_linear
        linear_vel = (
            self.kp_linear * distance_error +
            self.ki_linear * self.integral_linear +
            self.kd_linear * derivative_linear
        )
        
        # PID terms (angular velocity)
        self.integral_angular += angle_error
        derivative_angular = angle_error - self.prev_error_angular
        angular_vel = (
            self.kp_angular * angle_error +
            self.ki_angular * self.integral_angular +
            self.kd_angular * derivative_angular
        )
        
        # Update previous errors
        self.prev_error_linear = distance_error
        self.prev_error_angular = angle_error
        
        # Apply curvature constraints first
        linear_vel, angular_vel = self.apply_curvature_limits(linear_vel, angular_vel)
        
        # Then apply absolute velocity limits
        linear_vel = np.clip(linear_vel, -self.max_linear_vel, self.max_linear_vel)
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        return linear_vel, angular_vel

    def execute_cb(self, goal):
        """Action server execution with curvature-aware control."""
        self.target_point = np.array([goal.target_point.x, goal.target_point.y])
        rate = rospy.Rate(10)  # 10Hz
        success = True
        
        # Reset PID accumulators for new goal
        self.integral_linear = 0
        self.integral_angular = 0
        self.prev_error_linear = 0
        self.prev_error_angular = 0
        
        rospy.loginfo(f"New goal received: {self.target_point}")

        while not rospy.is_shutdown():
            if self._as.is_preempt_requested():
                rospy.loginfo("Goal preempted")
                self._as.set_preempted()
                success = False
                break
            
            # Compute control with curvature limits
            linear_vel, angular_vel = self.calculate_pid()
            
            # Publish command
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self.cmd_pub.publish(cmd)
            
            # Feedback
            distance = np.linalg.norm(self.target_point - self.current_pose[:2])
            self._feedback.fb = Float64(distance)
            self._as.publish_feedback(self._feedback)
            
            # Completion check
            if distance < 0.05:  # 5cm threshold
                rospy.loginfo("Goal reached!")
                break
                
            rate.sleep()
        
        # Stop robot
        self.cmd_pub.publish(Twist())
        if success:
            self._result.res = Float64(distance)
            self._as.set_succeeded(self._result)

if __name__ == '__main__':
    rospy.init_node('pid_controller')
    try:
        controller = PIDController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass