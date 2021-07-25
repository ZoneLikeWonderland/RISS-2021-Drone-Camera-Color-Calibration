#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from camera_color_fe.msg import colorir

import numpy as np
import time
import sys
import cv2
from postprocess import pickout

pub = None
show = np.zeros((1000, 640*1+640+200, 3), np.float32)
coeff = np.array([1, 1, 1])


def callback(msg):
    start = time.time()
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', type(msg))
    #cv_image = bggr8tobgr(np.fromstring(msg.data, dtype=np.uint8), msg.width, msg.height)
    # print(cv_image.shape)

    def Image2Array(msg):
        data = np.fromstring(msg.data, dtype=np.uint8)
        data = data.reshape(msg.height, msg.width, -1)
        if data.shape[-1] == 1:
            data = data.reshape(data.shape[:2])
        return data

    raw = Image2Array(msg.raw)
    block = Image2Array(msg.block)
    point = Image2Array(msg.point)
    # print(raw.shape)
    # print(block.shape)
    # print(point.shape)

    global show
    show *= 0.9
    raw_show = cv2.resize(raw, (640, 480))
    show[:raw_show.shape[0], :raw_show.shape[1]] = raw_show
    show[:block.shape[0], raw_show.shape[1]:raw_show.shape[1] +
         block.shape[1]] = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
    show[block.shape[0]:block.shape[0]+point.shape[0], raw_show.shape[1]         :raw_show.shape[1]+block.shape[1]] = cv2.cvtColor(point, cv2.COLOR_GRAY2BGR)

    show_list = {}
    ret = pickout(raw.astype(np.float32)/255, block.astype(np.float32) /
                  255, point.astype(np.float32)/255, show_list)

    global coeff
    if ret is not None:
        new_coeff, intensity = ret
        coeff = new_coeff

    balanced_show = raw_show * coeff

    show[raw_show.shape[0]:raw_show.shape[0] *
         2, :raw_show.shape[1]] = balanced_show
    #img /= mul[..., None]

    for i in show_list:
        if len(show_list[i].shape) == 2:
            show_list[i] = cv2.cvtColor(show_list[i], cv2.COLOR_GRAY2BGR)
        # print i,show_list[i].shape

    try:
        show[block.shape[0]:block.shape[0]+point.shape[0], raw_show.shape[1]:raw_show.shape[1]+block.shape[1]] = show_list["c_show"]
        show[:200, raw_show.shape[1]+block.shape[1]:raw_show.shape[1] +
             block.shape[1]+200] = show_list["warp"]*255
        show[200:200*2, raw_show.shape[1]+block.shape[1]:raw_show.shape[1]+block.shape[1]+200] = show_list["canny"]
        show[200*2:200*3, raw_show.shape[1]+block.shape[1]:raw_show.shape[1] +
             block.shape[1]+200] = show_list["canny_labels"]*255
        show[200*3:200*4, raw_show.shape[1]+block.shape[1]:raw_show.shape[1]+block.shape[1]+200] = show_list["matrix"]*255
        show[200*4:200*5, raw_show.shape[1]+block.shape[1]:raw_show.shape[1] +
             block.shape[1]+200] = show_list["warp_calibrated"]*255
    except KeyError:
        pass

    cv2.waitKey(1)

    show_msg = Image()
    show_msg.data = show.clip(0, 255).astype(np.uint8).tostring()
    show_msg.width = show.shape[1]
    show_msg.height = show.shape[0]
    show_msg.step = show.shape[1]*3
    show_msg.encoding = "bgr8"
    pub.publish(show_msg)

    print("return in %.3f s" % (time.time()-start))


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
