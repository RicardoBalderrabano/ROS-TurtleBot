#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import PoseStamped
import actionlib
from balderrabano_rodriguez.msg import ControllerGoal, ControllerAction

class LocalPlanner(object):

    def __init__(self, action_server_name, increment, resolution, min_distance):
        self.action_server_name = action_server_name
        self.client = actionlib.SimpleActionClient(action_server_name, ControllerAction)
        
        # State
        self.current_p = []               # Current position [x, y]
        self.current_path = []            # List of waypoints from global planner
        self.current_waypoint_idx = None  # Current waypoint index in path

        # Tuning parameters
        self.R = increment * resolution
        self.min_distance = min_distance

        # ROS interfaces
        rospy.Subscriber("global_planner/path", Path, self.path_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.path_pub = rospy.Publisher("local_planner/path", Path, queue_size=2)
        self.goal_reached_pub = rospy.Publisher("/goal_reached", Bool, queue_size=1)

    def path_callback(self, msg):
        """Callback when a new global path is published"""
        self.current_path = []
        for pose_stamped in msg.poses:
            p = [
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ]
            self.current_path.append(p)
        self.current_waypoint_idx = None

    def publish_path(self, x):
        """Publish a path to visualize the local trajectory"""
        if len(x) < 1:
            return

        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()

        path_msg = Path()
        path_msg.header = header
        for point in x:
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = point[2]
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

    def send_goal(self, point):
        """Send a waypoint goal to the controller"""
        goal = ControllerGoal()
        goal.target_point.x = point[0]
        goal.target_point.y = point[1]
        self.client.send_goal(goal)

    def select_waypoint(self):
        """Main logic to decide the next waypoint and handle goal reached"""
        if len(self.current_path) == 0:
            return

        path = np.array(self.current_path)

        # If just received a new path
        if self.current_waypoint_idx is None:
            self.current_waypoint_idx = 0
            self.send_goal(path[0])
            return

        # Check if robot is close to the final goal
        idx_final = path.shape[0] - 1
        final_point = self.current_path[idx_final]

        if np.linalg.norm(np.array(final_point[0:2]) - np.array(self.current_p)) < self.min_distance:
            rospy.loginfo("Reached final global goal!")
            self.goal_reached_pub.publish(Bool(data=True))
            self.current_path = []
            self.current_waypoint_idx = None
            return

        # Select next local waypoint if within range
        idx_next = min(self.current_waypoint_idx + 1, path.shape[0] - 1)
        next_point = self.current_path[idx_next]

        if np.linalg.norm(np.array(next_point[0:2]) - np.array(self.current_p)) <= self.R * 3:
            points = [self.current_path[self.current_waypoint_idx], next_point]
            self.current_waypoint_idx = idx_next
            self.send_goal(next_point)
            self.publish_path(points)

    def odom_callback(self, msg):
        """Update robot position from odometry"""
        self.current_p = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]

def main():
    rospy.init_node("local_planner_node", anonymous=True)

    # Parameters
    Ts = rospy.get_param('~Ts_robot')
    rate = rospy.Rate(1 / Ts)
    increment = rospy.get_param('~increment')
    min_distance = rospy.get_param('~min_distance')
    resolution = rospy.get_param('~resolution')
    action_server_name = "controller_as"

    # Wait for odometry to initialize
    rospy.wait_for_message("odom", Odometry)

    # Init planner
    node = LocalPlanner(action_server_name, increment, resolution, min_distance)

    rospy.loginfo('Waiting for Controller: ' + action_server_name)
    node.client.wait_for_server()
    rospy.loginfo('Controller connected: ' + action_server_name)

    # Main loop
    while not rospy.is_shutdown():
        node.select_waypoint()
        rate.sleep()

if __name__ == '__main__':
    main()