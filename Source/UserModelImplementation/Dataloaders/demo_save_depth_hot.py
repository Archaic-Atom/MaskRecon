 # -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
def debug_main():
    uint16_img = cv2.imread('/home/lixing/Programs/BodyReconstruction_stereo/ResultImg/000000_10.png', -1) 
    print(uint16_img.min(), uint16_img.max())
    uint16_img -= uint16_img.min()
    uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
    print(uint16_img)
    uint16_img *= 255
    new_uint16_img = uint16_img.astype(np.uint8)
    im_color = cv2.applyColorMap(new_uint16_img, cv2.COLORMAP_HOT)
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/02_depth_hot.png', im_color)
'''
def debug_main():
    uint16_img = cv2.imread('/home/lixing/Programs/BodyReconstruction_stereo/ResultImg/000000_10.png', -1) 
    mask = uint16_img >0
    print(uint16_img[mask].min(), uint16_img.max())
    print(uint16_img[300,300])
    uint16_img[mask] -= 20000
    print(uint16_img[300,300])
    uint16_img = uint16_img / (30000 - 20000)
    print(uint16_img[300,300])
    uint16_img[~mask] = 0
    uint16_img *= 255
    new_uint16_img = uint16_img.astype(np.uint8)
    im_color = cv2.applyColorMap(new_uint16_img, cv2.COLORMAP_HOT)
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/02_depth_hot.png', im_color)
if __name__ == "__main__":
    debug_main()