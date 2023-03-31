 # -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_main():
    uint16_img = cv2.imread('/home/lixing/Documents/Thuman/thuman_fb_v2/human_fb_train/DEPTH/0300/0010_0_00.png', -1)  
    uint16_img -= uint16_img.min()
    uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
    uint16_img *= 255
    new_uint16_img = uint16_img.astype(np.uint8)
    im_color = cv2.applyColorMap(new_uint16_img, cv2.COLORMAP_HOT)
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/02_depth_hot.png', im_color)

if __name__ == "__main__":
    debug_main()