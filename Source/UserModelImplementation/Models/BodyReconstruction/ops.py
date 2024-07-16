import torch
from torch import nn
import numpy as np
import cv2
import time
from PIL import Image
import torchvision.transforms as transforms



# set gloabal tensor to init X,Y only for the first time
flag_XY = True
X = torch.zeros([0]).cuda()
Y = torch.zeros([0]).cuda()

def depth_to_normal(depth):
    #depth: [B,1,H,W]
    depth = depth.squeeze(1)
    depth = depth*1000.0
    b, h, w = depth.shape
    normal = torch.zeros(b,3,h-2,w-2)
    for i in range(b):
        depth_img = depth[i,:,:]
        dx=(-(depth_img[2:h,1:w-1]-depth_img[0:h-2,1:w-1])*0.5)
        dy=(-(depth_img[1:h-1,2:w]-depth_img[1:h-1,0:w-2])*0.5)
        dz=torch.ones((h-2,w-2)).cuda()
        dl =torch.sqrt(dx * dx + dy * dy + dz * dz)
        dx = torch.div(dx, dl) * 0.5 + 0.5
        dy = torch.div(dy, dl) * 0.5 + 0.5
        dz = torch.div(dz, dl) * 0.5 + 0.5
        normal_img = torch.stack([dx, dy, dz], dim=0)
        normal[i,...] = normal_img
    paddings = (1, 1, 1, 1, 0, 0, 0, 0)
    normal = torch.nn.functional.pad(normal, paddings, 'constant')  # (B,H,W,3)
    return normal



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.normal_(0.0, 0.02)


