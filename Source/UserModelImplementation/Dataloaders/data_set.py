# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

import pandas as pd
import JackFramework as jf
import cv2
from PIL import Image
import scipy.misc



class BodyReconstructionDataset(Dataset):
    _DEPTH_UNIT = 1000.0
    _DEPTH_DIVIDING = 255.0

    def __init__(self, args: object, list_path: str,
                 is_training: bool = False) -> None:
        self.__args = args
        self.__is_training = is_training
        self.__list_path = list_path

        input_dataframe = pd.read_csv(list_path)

        self.__color_img_path = input_dataframe["color_img"].values
        self.__depth_img_path = input_dataframe["depth_img"].values
        self.__uv_img_path = input_dataframe["uv_img"].values
        self.__color_gt_path = input_dataframe["color_gt"].values
        self.__depth_gt_path = input_dataframe["depth_gt"].values

        if is_training:
            self.__get_path = self._get_training_path
            self.__data_steam = list(zip(self.__color_img_path,
                                         self.__depth_img_path,
                                         self.__uv_img_path,
                                         self.__color_gt_path,
                                         self.__depth_gt_path))
        else:
            self.__get_path = self._get_testing_path
            self.__data_steam = list(zip(self.__color_img_path,
                                         self.__depth_img_path,
                                         self.__uv_img_path))

    def __getitem__(self, idx: int):
        color_img_path, depth_img_path, uv_img_path, color_gt_path, depth_gt_path = self.__get_path(idx)
        return self._get_data(color_img_path, depth_img_path, uv_img_path, color_gt_path, depth_gt_path)

    def _get_training_path(self, idx: int) -> list:
        return self.__color_img_path[idx], self.__depth_img_path[idx], self.__uv_img_path[idx],\
            self.__color_gt_path[idx], self.__depth_gt_path[idx]

    def _get_testing_path(self, idx: int) -> list:
        return self.__color_img_path[idx], self.__depth_img_path[idx], self.__uv_img_path[idx],\
            self.__color_gt_path[idx], self.__depth_gt_path[idx]

    def _get_data(self, color_img_path, depth_img_path, uv_img_path, color_gt_path, depth_gt_path):
        if self.__is_training:
            return self._read_training_data(color_img_path, depth_img_path, uv_img_path, 
                                            color_gt_path, depth_gt_path)
        return self._read_testing_data(color_img_path, depth_img_path, uv_img_path, 
                                            color_gt_path, depth_gt_path)

    def _read_training_data(self, color_img_path: str,
                            depth_img_path: str,
                            uv_img_path: str,
                            color_gt_path: str,
                            depth_gt_path: str) -> object:
        args = self.__args

        width = args.imgWidth
        hight = args.imgHeight

        color_img = jf.ImgIO.read_img(color_img_path)
        depth_img = self._read_png_depth(depth_img_path)
        uv_img = np.array(scipy.misc.imread(uv_img_path), np.float32)
        color_gt = jf.ImgIO.read_img(color_gt_path)
        depth_gt = self._read_png_depth(depth_gt_path)

        depth_gt = -depth_gt + 1.0
        color_gt = np.flip(color_gt, 1).copy()
        depth_gt = np.flip(depth_gt, 1).copy()

        color_img, depth_img, uv_img, color_gt, depth_gt = jf.DataAugmentation.random_crop(
            [color_img, depth_img, uv_img, color_gt, depth_gt],
            color_img.shape[1], color_img.shape[0], width, hight)

        #color_img = jf.DataAugmentation.standardize(color_img)
        #color_gt = jf.DataAugmentation.standardize(color_gt)

        color_img = color_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        color_gt = color_gt / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        uv_img = uv_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)

        #color_img = jf.DataAugmentation.normalize(color_img)
        #color_gt = jf.DataAugmentation.normalize(color_gt)

        color_img = color_img.transpose(2, 0, 1)
        depth_img = depth_img.transpose(2, 0, 1)
        uv_img = uv_img.transpose(2, 0, 1)
        color_gt = color_gt.transpose(2, 0, 1)
        depth_gt = depth_gt.transpose(2, 0, 1)
        return color_img, depth_img, uv_img, color_gt, depth_gt

    def _read_testing_data(self, color_img_path: str,
                           depth_img_path: str,
                           uv_img_path: str,
                           color_gt_path: str,
                           depth_gt_path: str) -> object:
        args = self.__args

        color_img = jf.ImgIO.read_img(color_img_path)
        depth_img = self._read_png_depth(depth_img_path)
        uv_img = np.array(scipy.misc.imread(uv_img_path), np.float32)

        #color_img = jf.DataAugmentation.standardize(color_img)
        color_img = color_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        uv_img = uv_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)

        color_img = color_img.transpose(2, 0, 1)
        depth_img = depth_img.transpose(2, 0, 1)
        uv_img = uv_img.transpose(2, 0, 1)

        return color_img, depth_img, uv_img
    
    def __len__(self):
        return len(self.__data_steam)

    @staticmethod
    def _read_png_depth(path: str) -> torch.tensor:
        gt_depth = jf.ImgIO.read_img(path)
        gt_depth = np.ascontiguousarray(
            gt_depth, dtype=np.float32) / float(BodyReconstructionDataset._DEPTH_UNIT)
        return gt_depth

def debug_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="The deep learning framework (based on pytorch)")
    parser.add_argument('--imgWidth', type=int,
                        default=512,
                        help='croped width')
    parser.add_argument('--imgHeight', type=int,
                        default=256,
                        help='croped height')
    parser.add_argument('--dataset', type=str,
                        default='kitti2015',
                        help='dataset')
    args = parser.parse_args()
    training_sampler = None
    data_set = BodyReconstructionDataset(args, './Datasets/thuman_training_list.csv', True)
    training_dataloader = torch.utils.data.DataLoader(
        data_set,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=training_sampler
    )

    for iteration, batch_data in enumerate(training_dataloader):
        print(iteration)
        print(batch_data[0].size())
        print(batch_data[1].size())
        print(batch_data[2].size())
        print(batch_data[3].size())
        print(batch_data[4].size())
        print('___________')
        print(batch_data[0][:,:,100,220])
        print(batch_data[1][:,:,100,220])
        print(batch_data[2][:,:,100,220])
        print(batch_data[3][:,:,100,220])
        print(batch_data[4][:,:,100,220])
    


if __name__ == '__main__':
    debug_main()
