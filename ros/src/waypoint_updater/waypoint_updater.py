#!/usr/bin/env python
import math
import numpy as np
import rospy
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from geometry_msgs.msg import PoseStamped

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100
MAX_DECEL = 3.0 

class WaypointUpdater(object):
    def __init__(self):        
        rospy.init_node('waypoint_updater')        

        # TODO: Add other member variables you need below
        # Other related parameters are initilised
        self.base_lane = None
        self.pose = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        # Setting the node to publish "final waypoints" message
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)    

        # Setting the messages to be subscribed to for reading variables
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)     
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)       
        
        self.loop()

    # Code for defining the loop behaviour
    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    # Code for getting the closest waypoint within the stored waypoints      
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        # Each waypoint in the waypoint_tree is stored as [position, index]
        # The query() function will return the closest waypoint to [x, y], and
        #  the "1" value specifies to return only one item. We are then taking
        #  only the index ([1])
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        # Check if the closest waypoint is ahead of, or behind the vehicle
        # We are looking for the waypoint in front of the vehicle here
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        # Alternatively, you can take the orientation of the vehicle, and the 
        #  orientation of a vector from the previous waypoint to the current
        #  waypoint and compare them to determine if they are facing in the 
        #  same direction.
                
        if val > 0:
            # Waypoint is behind the vehicle, so increment index forward by one
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    # Code for publishing the waypoint information as lane to the ROS node 
    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    # Code for generating a lane from the waypoints   
    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx > farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane
    
    # Code to define the behavior when the stopline is within reach
    def decelerate_waypoints(self, waypoints, closest_idx):
        t_way = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 5, 0) 

        for i, wp in enumerate(waypoints):            
            p = Waypoint()
            p.pose = wp.pose
            braking_vel = 0.0         
            dist = self.distance(waypoints, i, stop_idx)
            
            if dist > 0.0:
                time_to_complete_stop = math.sqrt(dist * 2.0 / MAX_DECEL)
                braking_vel = dist / time_to_complete_stop * 1.5
                
                if braking_vel < 0.1:
                    braking_vel = 0.0

            p.twist.twist.linear.x = min(braking_vel, wp.twist.twist.linear.x)
            t_way.append(p)
            
        return t_way
            
        
    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

        
    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_lane = waypoints        
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

            
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

        
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

        
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')