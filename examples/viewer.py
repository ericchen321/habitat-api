#!/usr/bin/env python
#note need to run viewer with python2!!!
PKG = 'numpy_tutorial'
import roslib; roslib.load_manifest(PKG)

import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#import os
#import sys
#sys.path=[b for b in sys.path if "2.7" not in b]
#sys.path.insert(0,os.getcwd())

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError


pub = rospy.Publisher('ros_img_rgb', Image,queue_size=10)
def callback(data):
    print (rospy.get_name(), "I heard %s"%str(data.data))
    img = (np.reshape(data.data,(256, 256, 3))).astype(np.uint8)

    image_message = CvBridge().cv2_to_imgmsg(img, encoding="rgb8")
    pub.publish(image_message)

def listener():
    rospy.init_node('listener_node')
    rospy.Subscriber("rgb", numpy_msg(Floats), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()