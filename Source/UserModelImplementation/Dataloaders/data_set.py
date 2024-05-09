# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

import pandas as pd
import JackFramework as jf
import imageio
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import random
from PIL.ImageFilter import GaussianBlur
import cv2
try:
    from .mask_augmentation import MaskAug
except ImportError:
    from mask_augmentation import MaskAug


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

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                   hue=0.1)
        ])

        self.mask_aug = MaskAug(args.imgHeight, args.imgWidth, block_size=2, ratio=0.15)

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

        tw = args.imgWidth
        th = args.imgHeight

        

        color_img = Image.open(color_img_path)
        depth_img = Image.open(depth_img_path)
        uv_img = Image.open(uv_img_path)
        color_gt = Image.open(color_gt_path)
        depth_gt = Image.open(depth_gt_path)
        
        w, h = color_img.size

        #print("color_img", color_img)
        #print("depth_img", depth_img)
      
        # pad images
        pad_size = int(0.1 * tw)
        color_img = ImageOps.expand(color_img, pad_size, fill=0)
        depth_img = ImageOps.expand(depth_img, pad_size, fill=0)
        uv_img = ImageOps.expand(uv_img, pad_size, fill=0)
        color_gt = ImageOps.expand(color_gt, pad_size, fill=0)
        depth_gt = ImageOps.expand(depth_gt, pad_size, fill=0)

        #print("color_img_pad", color_img)
        #print("depth_img_pad", depth_img)

        
        # random flip
        if np.random.rand() > 0.5:
            color_img = transforms.RandomHorizontalFlip(p=1.0)(color_img)
            depth_img = transforms.RandomHorizontalFlip(p=1.0)(depth_img)
            uv_img = transforms.RandomHorizontalFlip(p=1.0)(uv_img)
            color_gt = transforms.RandomHorizontalFlip(p=1.0)(color_gt)
            depth_gt = transforms.RandomHorizontalFlip(p=1.0)(depth_gt)

        # random scale
        rand_scale = random.uniform(0.9, 1.1)
        w = int(rand_scale * w)
        h = int(rand_scale * h)
        color_img = color_img.resize((w, h), Image.BILINEAR)
        depth_img = depth_img.resize((w, h), Image.NEAREST)
        uv_img = uv_img.resize((w, h), Image.NEAREST)
        color_gt = color_gt.resize((w, h),  Image.BILINEAR)
        depth_gt = depth_gt.resize((w, h), Image.NEAREST)
        

        # augmentation
        color_img = self.aug_trans(color_img)

        # random blur
        aug_blur = 0.00002
        blur = GaussianBlur(np.random.uniform(0, aug_blur))
        color_img = color_img.filter(blur)

        


        color_img = np.array(color_img, np.float32)
        depth_img = np.array(depth_img, np.float32)
        uv_img = np.array(uv_img, np.float32)
        color_gt = np.array(color_gt, np.float32)
        depth_gt = np.array(depth_gt, np.float32)
        print("depth shape:", depth_img.shape)
        depth_img = np.expand_dims(depth_img, axis=2)
        depth_gt = np.expand_dims(depth_gt, axis=2)
        print("depth shape final:", depth_img.shape)

        


        # standardize image to [0-1]
        color_img = color_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        uv_img = uv_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        color_gt = color_gt / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        depth_img =np.ascontiguousarray(
            depth_img, dtype=np.float32) / float(BodyReconstructionDataset._DEPTH_UNIT)
        depth_gt =np.ascontiguousarray(
            depth_gt, dtype=np.float32) / float(BodyReconstructionDataset._DEPTH_UNIT)

        # transform back camera pars to same with front camera       
        depth_gt = -depth_gt + 1.0
        color_img = np.flip(color_img, 1).copy()
        uv_img = np.flip(uv_img, 1).copy()
        depth_img = np.flip(depth_img, 1).copy()

        # random crop
        color_img, depth_img, uv_img, color_gt, depth_gt = jf.DataAugmentation.random_crop(
            [color_img, depth_img, uv_img, color_gt, depth_gt],
            color_img.shape[1], color_img.shape[0], tw, th)

        if args.mask:
            color_img_mask, mask = self.mask_aug(color_img)
            #color_gt[:, :, 0] = color_gt[:, :, 0] * mask
            #color_gt[:, :, 1] = color_gt[:, :, 1] * mask
            #color_gt[:, :, 2] = color_gt[:, :, 2] * mask
        else:
            color_img_mask = color_img
        
        
        color_img = color_img.transpose(2, 0, 1)
        depth_img = depth_img.transpose(2, 0, 1)
        uv_img = uv_img.transpose(2, 0, 1)
        color_gt = color_gt.transpose(2, 0, 1)
        depth_gt = depth_gt.transpose(2, 0, 1)
        color_img_mask = color_img_mask.transpose(2, 0, 1)

        depth_img[np.isinf(depth_img)] = 0
        depth_gt[np.isinf(depth_gt)] = 0

        if args.mask:
            #return color_img_mask, depth_img.astype('float32') * mask.astype('float32'), \
            #    uv_img, color_gt, depth_gt.astype('float32') * mask.astype('float32'), color_img, depth_img
            return color_img_mask, depth_img.astype('float32') * mask.astype('float32'), \
                uv_img, color_gt, depth_gt, color_img, depth_img
        else:
            return color_img, depth_img, uv_img, color_gt, depth_gt, color_img, depth_img




    def _crop_tensor(self, tensor, h, w):
        _, H, W = tensor.shape
        i1 = np.int(H/2 - h/2)
        i2 = np.int(H/2 + h/2)
        i3 = np.int(W/2 - w/2)
        i4 = np.int(W/2 + w/2)
        return tensor[:, i1:i2, i3:i4]

    def _read_testing_data(self, color_img_path: str,
                           depth_img_path: str,
                           uv_img_path: str,
                           color_gt_path: str,
                           depth_gt_path: str) -> object:
        args = self.__args
        crop_w = args.imgWidth
        crop_h = args.imgHeight

        #color_img = jf.ImgIO.read_img(color_img_path)
        color_img = np.array(Image.open(color_img_path).convert('RGB'), np.float32)
        depth_img = self._read_png_depth(depth_img_path)
        uv_img = np.array(imageio.imread(uv_img_path), np.float32)
        
        if depth_img.shape[2] == 4:
            depth_img = depth_img[:,:,0]
            depth_img = np.expand_dims(depth_img, axis=2)
        color_img = np.flip(color_img, 1).copy()
        uv_img = np.flip(uv_img, 1).copy()
        depth_img = np.flip(depth_img, 1).copy()

       
        #uv_img=cv2.resize(uv_img, (512, 512), interpolation=cv2.INTER_AREA)
        #color_img=cv2.resize(color_img, (512, 512), interpolation=cv2.INTER_AREA)
        #depth_img=cv2.resize(depth_img, (512, 512), interpolation=cv2.INTER_AREA)
        #depth_img = np.expand_dims(depth_img, axis=2)
        
        # normalgan dataset
        #depth_img=depth_img/float(4)

        # buff data
        #depth_img = depth_img / float(60)

        # real data
        #color_img = np.rot90(color_img, 1)
        
        #zed data
        #depth_img = depth_img / float(3.0) 

        #tang data
        #depth_img = depth_img / float(4.0) 

        #xu teacher
        #depth_img = depth_img / float(1.5) 

        # stereopifu data
        #depth_img = depth_img / float(60)

        #depth_img = np.where(depth_img>3000, 0, depth_img)
        #depth_temp = np.concatenate((depth_img, depth_img, depth_img), axis=2)
        #color_img = np.where(depth_temp==0, 0, color_img)

        mask = color_img >0.01
        mask = mask[:,:,0]
        mask = np.expand_dims(mask, axis=2)
        depth_img = depth_img * mask


        #cv2.imwrite('./depth.png', depth_img.astype(np.uint16))


        
        #print(color_img.shape, depth_img.shape, uv_img.shape)

        #color_img = jf.DataAugmentation.standardize(color_img)
        color_img = color_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        uv_img = uv_img / float(BodyReconstructionDataset._DEPTH_DIVIDING)
        
        color_img = color_img.transpose(2, 0, 1)
        depth_img = depth_img.transpose(2, 0, 1)
        uv_img = uv_img.transpose(2, 0, 1)

        _, h, w = color_img.shape
        #print("testing images", color_img.shape, depth_img.shape, uv_img.shape)
        if (h!=crop_h) or (w!=crop_h):
            color_img = self._crop_tensor(color_img, crop_h, crop_w)
            depth_img = self._crop_tensor(depth_img, crop_h, crop_w)
            uv_img = self._crop_tensor(uv_img, crop_h, crop_w)
        

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
