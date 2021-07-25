import time
from fit import *
import json
import torch
import glob
import cv2
import numpy as np
import sys
sys.path.append(r"ros_deploy_workspace\src\camera_color_fe\scripts")
if True:
    from postprocess import *

WIDTH = 640
HEIGHT = 480


data = []

mul = None
x_base, y_base = None, None


if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    net = torch.load(r"C:\Users\14682\Documents\CODE\RISS\src\deploy\best_test_error.pth")
    net.to(device)
    net.eval()

    # cap = cv2.VideoCapture(0)
    # while True:

    l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_indoor_01.bag\_xic_stereo_left_image_raw\*.jpg"  # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_outdoor_01.bag\_xic_stereo_left_image_raw\*.jpg"  # done
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210607_indoor_spider_board_01.bag\_xic_stereo_left_image_raw\*.jpg" #ok

    xs = []
    ys = []
    coeff = np.array([1, 1, 1])
    last_fit_time = time.time()
    last_fit_n = 0
    for i, path in enumerate(glob.glob(l)):

        img = cv2.imread(path).astype(np.float32) / 255

        img_resize = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA)

        with torch.no_grad():
            pred = net(torch.tensor(img_resize).to(device).permute(2, 0, 1).unsqueeze(0)).squeeze().cpu().numpy()

        para, show = process(img, pred[0], pred[1])

        cv2.imshow("show_all", show)
        cv2.waitKey(1)

    json.dump(data, open("data.json", "w"))
    plt.ioff()
    plt.show()
