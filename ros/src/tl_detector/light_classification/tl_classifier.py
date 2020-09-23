from styx_msgs.msg import TrafficLight

import cv2
import numpy as np
import tensorflow as tf
import datetime


import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class TLClassifier(object):
    def __init__(self, is_simulation):
        '''
        self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge()
        '''

        # Code for differentiating model usage at simulation and at workspace
        if is_simulation:
            self.MODEL_NAME = 'light_classification/simulation'
        else:
            self.MODEL_NAME = 'light_classification/trial'

        # Setting the path to the interference graph for both cases
        self.PATH_TO_FROZEN_GRAPH = self.MODEL_NAME + '/inf_graph.pb'

        # Code for initilising a Tensorflow model into the classifier
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                ser_graph = fid.read()
                od_graph_def.ParseFromString(ser_graph)
                tf.import_graph_def(od_graph_def, name='')
             
            # Setting the attributes of the self object
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')                        
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        # Code for starting the Tensorflow session        
        self.session = tf.Session(graph=self.detection_graph)
        self.threshold = 0.5
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        with self.detection_graph.as_default():
            im_expand = np.expand_dims(image, axis=0)
            class_init_t = datetime.datetime.now()

            (boxes, scores, classes, num_detections) = self.session.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: im_expand})

            # Code for keeping track of the time whenever needed fo debugging
            class_end_t = datetime.datetime.now()
            delta_time = class_end_t - class_init_t            

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)        

        # Code for printing out the light class and relating it to the TrafficLight object
        if scores[0] > self.threshold:
            if classes[0] == 1:
                print('Detected traffic light color is GREEN')
                return TrafficLight.GREEN
            elif classes[0] == 3:
                print('Detected traffic light color is YELLOW')
                return TrafficLight.YELLOW
            elif classes[0] == 2:
                print('Detected traffic light color is RED')
                return TrafficLight.RED
            
        # Code for returning unknown state in case classification fails
        return TrafficLight.UNKNOWN