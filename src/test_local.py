import time
from fit import *
import json
import torch
import glob
import cv2
import numpy as np
import sys
import os
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
    device = "cpu"
    net = torch.load(r"C:\Users\14682\Documents\CODE\RISS\src\deploy\best_test_error.pth")
    net.to(device)
    net.eval()

    # cap = cv2.VideoCapture(0)
    # while True:

    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_indoor_01.bag\_xic_stereo_left_image_raw\*.jpg"  # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_indoor_02.bag\_xic_stereo_left_image_raw\*.jpg"  # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_outdoor_01.bag\_xic_stereo_left_image_raw\*.jpg"  # done
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_outdoor_02.bag\_xic_stereo_left_image_raw\*.jpg" # not good TODO: small and blur?
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210607_indoor_spider_board_01.bag\_xic_stereo_left_image_raw\*.jpg"  # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210607_indoor_spider_board_02.bag\_xic_stereo_left_image_raw\*.jpg" # FAIL TODO: blur?

    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\CB_01.bag\_xic_stereo_left_image_raw\*.jpg" # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\CB_01_Rotation.bag\_xic_stereo_left_image_raw\*.jpg" # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\CB_01_WhiteBase.bag\_xic_stereo_left_image_raw\*.jpg" # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\CB_02.bag\_xic_stereo_left_image_raw\*.jpg" # haixing
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\CB_02_Rotation.bag\_xic_stereo_left_image_raw\*.jpg" # haixing
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\CB_03.bag\_xic_stereo_left_image_raw\*.jpg" # haixing

    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\shadowCB_01.bag\_xic_stereo_left_image_raw\*.jpg" # good
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\shadowCB_02.bag\_xic_stereo_left_image_raw\*.jpg" # ok
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\shadowCB_03.bag\_xic_stereo_left_image_raw\*.jpg" # fine
    l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\shadowCB_04.bag\_xic_stereo_left_image_raw\*.jpg"  # TODO: decide which one to use

    # l = r"C:\Users\14682\Downloads\RISS-2021-Journal-Template\figures\etc\notsee.jpg"
    # l = r"C:\Users\14682\Downloads\image_left_03.png"

    result_dir = os.path.join("result", l.replace("\\", "_").replace("*", "_").replace(":", "_"))
    os.makedirs(result_dir, exist_ok=True)

    xs = []
    ys = []
    coeff = np.array([1, 1, 1])
    last_fit_time = time.time()
    last_fit_n = 0
    for i, path in enumerate(glob.glob(l)):

        img = cv2.imread(path).astype(np.float32) / 255

        img_resize = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA)

        with torch.no_grad():
            start = time.time()
            pred = net(torch.tensor(img_resize).to(device).permute(2, 0, 1).unsqueeze(0)).squeeze().cpu().numpy()
            print("forward time", time.time() - start)

        para, show = process(img, pred[0], pred[1], settings={
            "ENABLE_EXPIRED_HINT": False,
            "ENABLE_PLT": False,
            "COEFF_LENGTH": 99999
        })

        cv2.imshow("show_all", show)
        # cv2.imwrite(os.path.join(result_dir, f"{i:010d}.png"), show * 255)
        cv2.waitKey(1)
        print(i, path, para)
        # cv2.waitKey()

        # if cv2.waitKey(1) == ord("s") or (show[240, -640:-320, 0] == show[240, -640:-320, 2]).all():
        # cv2.imwrite("show.png", show * 255)
