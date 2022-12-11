# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import JackFramework as jf


import os
import re
import argparse
from PIL import Image
import numpy as np
import pandas as pd

COLOR_DIVDING = 255
DEPTH_DIVIDING = 1000
ACC_EPSILON = 1e-9


def read_label_list(list_path: str, label_names: str):
    input_dataframe = pd.read_csv(list_path)
    gt_dsp_path = input_dataframe[label_names].values
    return gt_dsp_path


def d_1(res: torch.tensor, gt: torch.tensor, start_threshold: int = 2,
        threshold_num: int = 4, relted_error: float = 0.05,
        invaild_value: int = 0) -> torch.tensor:
    mask = (gt != invaild_value) 
    mask.detach_()
    acc_res = []
    with torch.no_grad():
        total_num = mask.int().sum()
        error = torch.abs(res[mask] - gt[mask])
        print(error.shape)
        related_threshold = gt[mask] * relted_error
        for i in range(threshold_num):
            threshold = start_threshold + i
            # acc = (error > threshold) & (error > related_threshold)
            acc = (error > threshold)
            acc_num = acc.int().sum()
            error_rate = acc_num / (total_num + ACC_EPSILON)
            acc_res.append(error_rate)
        mae = error.sum() / (total_num + ACC_EPSILON)
    return acc_res, mae

def epe(res: torch.tensor, gt: torch.tensor, 
        invaild_value: int = 0) -> torch.tensor:
    mask = gt > invaild_value
    mask.detach_()
    with torch.no_grad():
        total_num = mask.int().sum()
        error = torch.abs(res[mask] - gt[mask])
        mae = error.sum() / (total_num + ACC_EPSILON)
    return mae

def mse(res: torch.tensor, gt: torch.tensor, 
        invaild_value: int = 0) -> torch.tensor:
    mask = gt > invaild_value
    mask.detach_()
    with torch.no_grad():
        total_num = mask.int().sum()
        error = torch.abs(res[mask] - gt[mask])
        mae = error.sum() / (total_num + ACC_EPSILON)
    return mae

def read_pfm(filename: str) -> tuple:
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


class Evalution(nn.Module):
    """docstring for Evalution"""

    def __init__(self, start_threshold: int = 2,
                 threshold_num: int = 4, relted_error: float = 0.05,
                 invaild_value: int = 0):
        super().__init__()
        self._start_threshold = start_threshold
        self._threshold_num = threshold_num
        self._relted_error = relted_error
        self._invaild_value = invaild_value

    def forward(self, res, gt, error_type):
        if error_type == 'd_1':
            return d_1(res, gt, self._start_threshold,
                   self._threshold_num, self._relted_error,
                   self._invaild_value)
        if error_type == 'mse':
            return mse(res, gt, 
                   self._invaild_value)

        


def get_data(img_path: str, gt_path: str, img_type: str) -> np.array:
    if img_type == 'color':
        img = np.array(Image.open(img_path), dtype=np.float32) / float(COLOR_DIVDING)
        img_gt = np.array(Image.open(gt_path), dtype=np.float32) / float(COLOR_DIVDING)
    if img_type == 'depth':
        img = np.array(Image.open(img_path), dtype=np.float32) / float(DEPTH_DIVIDING)
        img_gt = np.array(Image.open(gt_path), dtype=np.float32) / float(DEPTH_DIVIDING)

    return img, img_gt


def data2cuda(img: np.array, img_gt: np.array) -> torch.tensor:
    img = torch.from_numpy(img).float()
    img_gt = torch.from_numpy(img_gt.copy()).float()

    img = Variable(img, requires_grad=False)
    img_gt = Variable(img_gt, requires_grad=False)
    return img, img_gt


def print_total(total: np.array, err_total: int,
                total_img_num: int, threshold_num: int) -> str:
    offset = 1
    str_data = 'depth_total '
    for j in range(threshold_num):
        d1_str = '%dpx: %f ' % (j + offset, total[j] / total_img_num)
        str_data = str_data + d1_str
    str_data = str_data + 'mae: %f' % (err_total / total_img_num)
    print(str_data)
    return str_data

def color_print_total(err_total: int,
                total_img_num: int, str_data: str) -> str:
    str_data = str_data + 'mae: %f' % (err_total / total_img_num) 
    print(str_data)
    return str_data


def cal_total(id_num: int, total: np.array, err_total: int,
              acc_res: torch.tensor, mae: torch.tensor,
              threshold_num: int) -> None:
    str_data = str(id_num) + ' '
    for i in range(threshold_num):
        d1_res = acc_res[i].cpu()
        d1_res = d1_res.detach().numpy()
        total[i] = total[i] + d1_res
        str_data = str_data + str(d1_res) + ' '

    mae_res = mae.cpu()
    mae_res = mae_res.detach().numpy()
    err_total = err_total + mae_res

    str_data = str_data + str(mae_res)
    print(str_data)

    return total, err_total

def color_cal_total(id_num: int, err_total: int,
               mae: torch.tensor) -> None:
    str_data = str(id_num) + ' '

    mae_res = mae.cpu()
    mae_res = mae_res.detach().numpy()
    err_total = err_total + mae_res

    str_data = str_data + str(mae_res)
    print(str_data)

    return err_total


def parser_args() -> object:
    parser = argparse.ArgumentParser(
        description="The Evalution process")
    parser.add_argument('--img_path_format', type=str,
                        default='./ResultImg/%04d_color_pre.png',
                        help='img_path_format')
    parser.add_argument('--depth_path_format', type=str,
                        default='./ResultImg/%04d_depth_pre.png',
                        help='depth_path_format')
    parser.add_argument('--gt_list_path', type=str,
                        default='./Datasets/renderpeople_testing_list.csv',
                        help='gt list path')
    parser.add_argument('--epoch', type=int,
                        default=0, help='epoch num')
    parser.add_argument('--output_path', type=str,
                        default='./Result/test_output.txt',
                        help='output file')
    parser.add_argument('--error_mode', type=str,
                        default=epe,
                        help='error type')
    parser.add_argument('--caption', type=str,
                        default="BodyReconstruction_fb_mask",
                        help='caption of model')
    args = parser.parse_args()
    return args


def evalution(epoch: int, img_path_format: str, depth_path_format: str, gt_list_path: str,  
                output_path: str, caption: str) -> None:
    gt_color_path = read_label_list(gt_list_path, 'color_gt')
    gt_depth_path = read_label_list(gt_list_path, 'depth_gt')
    total_img_num = len(gt_color_path)

    start_threshold = 1
    threshold_num = 5

    # Variable
    depth_total = np.zeros(threshold_num)
    depth_err_total = 0
    color_err_total = 0

    # push model to CUDA
    eval_model = Evalution(start_threshold=start_threshold,
                           threshold_num=threshold_num)
  
    
    eval_model = torch.nn.DataParallel(eval_model).cuda()

    for i in range(total_img_num):
        color_img_path = img_path_format % (i)
        color_gt_path = gt_color_path[i]
        depth_img_path = depth_path_format % (i)
        depth_gt_path = gt_depth_path[i]

        color, color_gt = get_data(color_img_path, color_gt_path, img_type='color')
        depth, depth_gt = get_data(depth_img_path, depth_gt_path, img_type='depth')

        color, color_gt = data2cuda(color, color_gt)
        depth, depth_gt = data2cuda(depth, depth_gt)

        color_mae = eval_model(color, color_gt, error_type='mse')
        depth_mae = eval_model(depth, depth_gt, error_type='mse')
        color_err_total = color_cal_total(i,color_err_total, color_mae)
        depth_err_total = color_cal_total(i, depth_err_total, depth_mae)
        
        
        
        
    color_str = color_print_total(color_err_total, total_img_num, "color_error_total ")
    data_str = color_print_total(depth_err_total, total_img_num, "depth_error_total ")
    fd_file = jf.FileHandler.open_file(output_path, True)
    jf.FileHandler.write_file(fd_file, caption)
    jf.FileHandler.write_file(fd_file, f'epoch: {epoch}, {color_str}')
    jf.FileHandler.write_file(fd_file, f'epoch: {epoch}, {data_str}')


def main():
    args = parser_args()
    evalution(args.epoch, args.img_path_format, args.depth_path_format, args.gt_list_path, 
              args.output_path, args.caption)
    # setting
    # img_path_format = args.img_path_format
    # gt_list_path = './Datasets/scene_flow_testing_list.csv'
    # gt_list_path = './Datasets/kitti2012_training_list.csv'
    # gt_list_path = args.gt_list_path


if __name__ == '__main__':
    main()
