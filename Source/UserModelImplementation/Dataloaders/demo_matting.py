# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_main():
    color_mask = np.array(cv2.imread('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/color_mask.png'))
    mask_front = np.array(cv2.imread('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/0000_0_00_mask.png'))
    color_mask = np.where(mask_front>0,color_mask,255)
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/color_mask_mat.png', color_mask)
    
    color_label = np.array(cv2.imread('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/color_label.png'))
    mask_back = np.array(cv2.imread('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/0180_0_00_mask.png'))
    color_label = np.where(mask_back>0,color_label,255)
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/color_label_mat.png', color_label)

if __name__ == "__main__":
    debug_main()