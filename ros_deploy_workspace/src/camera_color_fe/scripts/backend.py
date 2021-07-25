#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from camera_color_fe.msg import colorir

import numpy as np
import time
import sys
import cv2
from postprocess import process

pub = None


def callback(msg):
    start = time.time()
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', type(msg))

    def Image2Array(msg):
        data = np.fromstring(msg.data, dtype=np.uint8)
        data = data.reshape(msg.height, msg.width, -1)
        if data.shape[-1] == 1:
            data = data.reshape(data.shape[:2])
        return data

    raw = Image2Array(msg.raw)
    block = Image2Array(msg.block)
    point = Image2Array(msg.point)

    para, show = process(raw.astype(np.float32) / 255, block.astype(np.float32) / 255, point.astype(np.float32) / 255)

    show_msg = Image()
    show_msg.data = (show.clip(0, 1) * 255).astype(np.uint8).tostring()
    show_msg.width = show.shape[1]
    show_msg.height = show.shape[0]
    show_msg.step = show.shape[1] * 3
    show_msg.encoding = "bgr8"
    pub.publish(show_msg)

    print("return in %.3f s" % (time.time() - start))


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    global pub
    pub = rospy.Publisher('chatter', Image, queue_size=1)

    rospy.Subscriber('/retailer', colorir, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    print("sys.version =", sys.version)
    listener()
