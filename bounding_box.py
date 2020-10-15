#!/usr/bin/env python3
import os
import rospy
import sys
from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd
#%matplotlib inline
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
import random
import os.path as osp

class SSD_Net(object):
    def __init__(self):
        #self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.box_pred, queue_size=1)
        self.label_idx = {'missle': 0, 'backpack': 1, 'blueline': 2, 'drill': 3, 'can': 4}
        self.label_idx = sorted(self.label_idx.items(), key=lambda kv: kv[1])
        self.coords = Int64MultiArray()
        self.label = ['background']
        for i in self.label_idx:
            self.label.append(i[0])
        self.net = build_ssd('test', 300, 6)
        self.net.load_weights('/home/austin/SSD-SubT/weights/ncsisT/ncsisT.pth')
        self.pub_img = rospy.Publisher('/box_pred/img', Image, queue_size=1)
        self.pub_box = rospy.Publisher('/box_pred/box_coords', Int64MultiArray, queue_size=1)
        self.test_file = '/home/austin/DataSet/ncsist_dataset/ncsist/ImageSets/Main/test.txt'
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.box_pred, queue_size=1)
    def box_pred(self, data):
        image = bridge.imgmsg_to_cv2(data, desired_encoding = "bgr8")
        #rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        rgb_image = image
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = self.net(xx)
        #currentAxis = plt.gca()
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        detections = y.data
        objs = []
        for i in range(detections.size(1)): # detections.size(1) --> class size
             for j in range(5): # each class choose top 5 predictions
                 if detections[0, i, j, 0].cpu().numpy() > 0.5:
                     score = detections[0, i, j, 0]
                     pt = (detections[0, i, j,1:]*scale).cpu().numpy()
                     objs.append([pt[0], pt[1], pt[2]-pt[0]+1, pt[3]-pt[1]+1, i, score])
             color_BBX = (255, 255, 0)
             color_TEXT = (255, 0, 0)
             colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
             for i, obj in enumerate(objs):
                 #cv2.rectangle(rgb_image, (int(obj[0]), int(obj[1])),\
                 #            (int(obj[0] + obj[2]), int(obj[1] + obj[3])), color_BBX, 3)
                 #cv2.putText(rgb_image, label[obj[4]], (int(obj[0]), int(obj[1])), 0, 1, color_TEXT,2)
                 coords = (obj[0], obj[1]), obj[2], obj[3]
                 display_txt = '%s: %.2f'%(self.label[obj[4]], obj[5])
                 cv2.rectangle(rgb_image, (int(obj[0]), int(obj[1])), (int(obj[0] + obj[2]), int(obj[1] + obj[3])), (255, 255, 0), 4)
                 rospy.loginfo(display_txt)
                 self.coords.data = [int(obj[0]), int(obj[1]), int(obj[0] + obj[2]), int(obj[1] + obj[3])]
                 #currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=colors[i], linewidth=2))
                 #currentAxis.text(obj[0], obj[1], display_txt, bbox={'facecolor':colors[i], 'alpha':0.5})
        #rospy.loginfo(display_txt)
        self.pub_img.publish(bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8"))
        self.pub_box.publish(self.coords)

if __name__ == "__main__":
    rospy.init_node("bounding_box_pred", anonymous=False)
    ssd_net = SSD_Net()
    rospy.spin()



