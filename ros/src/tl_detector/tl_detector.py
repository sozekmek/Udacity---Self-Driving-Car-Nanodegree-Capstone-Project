#!/usr/bin/env python
import rospy
import tf
import cv2
import yaml
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
DIFF_THRESHOLD = 100

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # Defining the parameters of the self object with initial values
        self.pose = None
        self.waypoints = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        # Setting up the config file with its related path
        config_path = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_path)

        # Setting message to be published
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        
        # Defining remaining attributes of the self object
        self.bridge = CvBridge()
        self.is_simulation = not self.config["is_site"]
        self.light_classifier = TLClassifier(self.is_simulation)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.image_count = -1
        self.has_image = False
        self.image_count_thres = 4

        # Defining messages to be subscribed to
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = None

        # We may want to use image_raw here to prevent loss of data when changing color schemes
        if self.is_simulation:
            sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        else:
            sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
            #sub6 = rospy.Subscriber('/image_raw', Image, self.image_cb)

        rospy.spin()

        
    def pose_cb(self, msg):
        self.pose = msg

        
    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        self.waypoint_tree = KDTree(waypoints_2d)

        
    def traffic_cb(self, msg):
        self.lights = msg.lights

        
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        
        self.image_count += 1        
        light_wp = None        

        #print("imageID:{0}".format(self.image_count))
        if self.image_count % self.image_count_thres == 0:
            self.has_image = True        
            self.camera_image = msg
            light_wp, state = self.process_traffic_lights()       
            
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                # Only store traffic light waypoints if the light is red (otherwise, drive through)
                # Possibly update this code to account for yellow or stale green lights
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                
            self.state_count += 1
        else:
            state = self.state
    

    def get_closest_waypoint(self, pose_x, pose_y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        # Obtaining the closest waypoint ID from the waypoint tree
        closest_idx = self.waypoint_tree.query([pose_x, pose_y], 1)[1]
        
        return closest_idx
    
    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        print('running the Clf')
        return self.light_classifier.get_classification(cv_image)

    
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
         
        closest_light = None
        line_wp_idx = None
        light_precense = False
                
        stop_line_positions = self.config['stop_line_positions']
        
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            
            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Code for finding the index of the stop line waypoint index
                line = stop_line_positions[i]
                # Storing the closest waypoint index under a temporary variable
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Finding the difference between the current closest waypoint and the possible closer waypoint
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
        
            if diff > DIFF_THRESHOLD:
                self.image_count_thres = 10
            else:
                self.image_count_thres = 4
                # Setting traffic light precense to true
                light_precense = True                
            
            print('Distance to the closes traffic light is ', diff, self.image_count_thres)
            
        # Code for defining behavior when there is a traffic light
        if closest_light and light_precense:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')