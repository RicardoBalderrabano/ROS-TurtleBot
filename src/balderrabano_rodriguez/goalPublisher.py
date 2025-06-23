#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

class GoalPublisher:
    def __init__(self):
        rospy.init_node('goal_publisher')

        # Waypoints [x, y]
        self.waypoints = [
            [-1.5, -2.9],          # p1
            [-3, 6],      # p2
            [5.25, 5.5]       # p3
        ]

        self.goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=2)
        rospy.Subscriber('/goal_reached', Bool, self.reached_callback)

        self.goal_times = []
        self.current_goal_idx = -1
        self.start_time = None
        self.waiting = False

    def reached_callback(self, msg):
        if msg.data and self.waiting:
            elapsed = rospy.Time.now() - self.start_time
            self.goal_times.append(elapsed.to_sec())
            rospy.loginfo(f"Goal {self.current_goal_idx+1} reached in {elapsed.to_sec():.2f}s")
            self.waiting = False

            if self.current_goal_idx + 1 < len(self.waypoints):
                self.next_goal()
            else:
                rospy.loginfo("All goals completed!")
                self.print_summary()

    def publish_goal(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        self.goal_pub.publish(goal)
        rospy.loginfo(f"Published goal: ({x}, {y})")

    def next_goal(self):
        self.current_goal_idx += 1

        if self.current_goal_idx < len(self.waypoints):
            x, y = self.waypoints[self.current_goal_idx]
            self.start_time = rospy.Time.now()
            self.waiting = True
            self.publish_goal(x, y)
        else:
            rospy.loginfo("All goals completed!")
            self.print_summary()

    def print_summary(self):
        rospy.loginfo("\n=== PERFORMANCE SUMMARY ===")
        for i, t in enumerate(self.goal_times):
            rospy.loginfo(f"p{i+1}: {t:.2f}s")
        total = sum(self.goal_times)
        rospy.loginfo(f"TOTAL: {total:.2f}s")

    def run(self):
        rospy.sleep(5)  # Let other nodes initialize
        self.next_goal()

if __name__ == '__main__':
    gp = GoalPublisher()
    gp.run()
    rospy.spin()