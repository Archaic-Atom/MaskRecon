 # -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_main():
    uint16_img = cv2.imread('/home/lixing/Documents/BodyReconstruction_test/RenderPeople/DEPTH/rp_isabelle_posed_013/0000_0_00.png', -1)  
    print((uint16_img.max()))
    uint16_img = uint16_img/uint16_img.max()*255
    #uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
    print((uint16_img.max()))
    #uint16_img *= 255
    #new_uint16_img = uint16_img.astype(np.uint8)
    #im_color = cv2.applyColorMap(new_uint16_img, cv2.COLORMAP_HOT)
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/02_depth_hot.png', uint16_img)

if __name__ == "__main__":
    debug_main()