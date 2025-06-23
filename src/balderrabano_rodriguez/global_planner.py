#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from project_venturino.graph import Graph
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped
from scipy.ndimage import binary_dilation
import traceback


class GlobalPlanner(object):

    def __init__(self, inflate_radius, increment, min_distance):
        self.inflate_radius = inflate_radius
        self.min_distance = min_distance
        self.increment = increment

        self.graph_pub = rospy.Publisher("global_planner/nodes", PointCloud2, queue_size=2)
        self.a_start_pub = {
            "close": rospy.Publisher("global_planner/close_nodes", PointCloud2, queue_size=2),
            "open": rospy.Publisher("global_planner/open_nodes", PointCloud2, queue_size=2)
        }

        self.costmap_pub = rospy.Publisher("global_costmap", OccupancyGrid, queue_size=1)
        self.path_pub = rospy.Publisher("/global_planner/path", Path, queue_size=2)
        self.tracking_error_pub = rospy.Publisher("/tracking_error", Float32, queue_size=10)  # ← NEW

        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/goal", PoseStamped, self.goal_callback)

        self.width = 0
        self.height = 0
        self.resolution = 0
        self.origin = None
        self.costmap = None
        self.g = Graph()
        self.map_received = False

        self.p0 = []
        self.yaw = []
        self.goal = []
        self.current_waypoint = None  # ← NEW

    def goal_callback(self, msg):
        rospy.loginfo('New target goal received!')
        self.goal = [msg.pose.position.x, msg.pose.position.y]
        self.compute_path()

    def odom_callback(self, msg):
        self.p0 = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        if self.current_waypoint is not None:
            self.compute_tracking_error(self.current_waypoint)

    def compute_tracking_error(self, desired_point):
        dx = desired_point[0] - self.p0[0]
        dy = desired_point[1] - self.p0[1]
        error = np.sqrt(dx ** 2 + dy ** 2)
        self.tracking_error_pub.publish(Float32(error))
        return error

    def receive_map(self):
        msg = rospy.wait_for_message("map", OccupancyGrid)
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin

        inflate_radius_cell = int(self.inflate_radius / self.resolution)
        self.costmap = np.array(np.reshape(msg.data, (msg.info.height, msg.info.width)) / 100, dtype=bool)
        self.costmap = binary_dilation(self.costmap, iterations=inflate_radius_cell)
        self.map_received = True
        self.publish_global_costmap()
        self.costmap = np.flip(np.flip(self.costmap, axis=0).T, axis=0)

    def publish_global_costmap(self):
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = rospy.Time.now()
        occupancy_grid.header.frame_id = "map"
        occupancy_grid.info.width = self.width
        occupancy_grid.info.height = self.height
        occupancy_grid.info.resolution = self.resolution
        occupancy_grid.info.origin = self.origin
        occupancy_grid.data = self.costmap.flatten()
        self.costmap_pub.publish(occupancy_grid)

    def publish_vertices(self, v_list=None, topic=""):
        if self.width == 0:
            return

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        header = Header(frame_id="map", stamp=rospy.Time.now())

        points = [self.find_point_vertex(k) for k in (v_list if v_list else self.g.get_all_nodes())]
        points = np.array(points, dtype=np.float32)
        points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)

        pc2 = PointCloud2()
        pc2.header = header
        pc2.fields = fields
        pc2.height = 1
        pc2.width = points.shape[0]
        pc2.is_bigendian = False
        pc2.point_step = 12
        pc2.row_step = 12 * points.shape[0]
        pc2.is_dense = False
        pc2.data = points.tobytes()

        (self.graph_pub if v_list is None else self.a_start_pub[topic]).publish(pc2)

    def publish_path(self, path_cells):
        if len(path_cells) < 1:
            self.current_waypoint = None
            return

        header = Header(frame_id="map", stamp=rospy.Time.now())
        path_msg = Path(header=header)

        for cell in path_cells:
            pose = PoseStamped()
            p = self.find_point_vertex(cell)
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

        # Set and evaluate tracking error to the first waypoint
        self.current_waypoint = self.find_point_vertex(path_cells[0])
        if len(self.p0) > 0:
            self.compute_tracking_error(self.current_waypoint)

    def check_safe_space(self, idx):
        i_min = int(max(0, (idx[0] - 1) * self.increment))
        i_max = int(min(self.width - 1, (idx[0] + 1) * self.increment - 1))
        j_min = int(max(0, (idx[1] - 1) * self.increment))
        j_max = int(min(self.width - 1, (idx[1] + 1) * self.increment - 1))
        space_matrix = self.costmap[i_min:i_max, j_min:j_max]
        return np.sum(space_matrix) < 1

    def build_graph(self):
        if not self.map_received:
            rospy.logdebug('Empty map: impossible to build the graph!')
            return

        rospy.loginfo('Constructing initial grid graph...')
        R_bar = self.increment
        self.g.grid = np.zeros((int(self.width / R_bar) + 1, int(self.height / R_bar) + 1), dtype=bool)

        for i in range(self.g.grid.shape[0]):
            for j in range(self.g.grid.shape[1]):
                self.g.grid[i, j] = self.check_safe_space((i, j))

        rospy.loginfo('Graph constructed!')
        self.publish_vertices()

    def compute_path(self):
        if len(self.p0) < 1 or len(self.goal) < 1:
            return

        v0 = self.find_vertex(self.p0)
        vf = self.find_vertex(self.goal)
        pf = self.find_point_vertex(vf)

        self.goal[0], self.goal[1] = pf

        try:
            shortest_path, open_list, closed_list = self.g.a_star(v0, vf)
            self.publish_path(shortest_path)
            self.publish_vertices([v[1] for v in open_list], "open")
            self.publish_vertices(closed_list, "close")
            rospy.loginfo('New global path computed!')
        except KeyError:
            rospy.logwarn(traceback.format_exc())
            rospy.logerr("The new global path can't be computed!")

        if np.linalg.norm(np.array(self.p0) - np.array(self.goal)) <= self.min_distance:
            self.goal = []

    def find_vertex(self, p):
        cx = -(p[0] + self.origin.position.x) / (self.resolution * self.increment)
        cy = -(p[1] + self.origin.position.y) / (self.resolution * self.increment)
        return round(cx), round(cy)

    def find_point_vertex(self, v):
        p0 = -self.resolution * self.increment * v[0] - self.origin.position.x
        p1 = -self.resolution * self.increment * v[1] - self.origin.position.y
        return p0, p1

    @staticmethod
    def quaternion_to_yaw(q):
        return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def main():
    rospy.init_node('global_planner_node', anonymous=True, log_level=rospy.DEBUG)
    inflate_radius = rospy.get_param('~inflate_radius')
    increment = rospy.get_param('~increment')
    min_distance = rospy.get_param('~min_distance')

    node = GlobalPlanner(inflate_radius, increment, min_distance)
    node.receive_map()
    node.build_graph()

    rospy.spin()


if __name__ == '__main__':
    main()