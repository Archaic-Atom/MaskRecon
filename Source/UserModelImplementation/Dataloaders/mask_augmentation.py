# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2


class MaskAug(object):
    """docstring for ClassName"""
    COLOR_GRAY = 127

    def __init__(self, img_height: int, img_width: int,
                 block_size: int = 2, ratio: float = 0.3) -> None:
        super().__init__()
        assert((img_height % block_size == 0) and (img_width % block_size == 0))
        self._img_height = img_height
        self._img_width = img_width
        self._block_size = block_size
        self._block_width_num = int(img_width / block_size)
        self._block_height_num = int(img_height / block_size)
        self._block_num = int(img_height * img_width / block_size / block_size)
        self._block_num_list = list(range(0, self._block_num))
        self._sample_num = int(self._block_num * ratio)

    def _generate_mask(self) -> np.array:
        random_sample_list = random.sample(self._block_num_list, self._sample_num)
        mask = np.ones([self._img_height, self._img_width], dtype = float)
        for sample_id in random_sample_list:
            height_id = sample_id // self._block_width_num
            width_id = sample_id % self._block_width_num
            cy = height_id * self._block_size
            cx = width_id * self._block_size
            mask[cy:cy + self._block_size, cx:cx + self._block_size] = 0
        return mask

    def __call__(self, img) -> np.array:
        mask = self._generate_mask()
        img[:, :, 0] = img[:, :, 0] * mask
        img[:, :, 1] = img[:, :, 1] * mask
        img[:, :, 2] = img[:, :, 2] * mask
        return img, mask


def debug_main():
    mask_aug = MaskAug(512, 512, block_size=4, ratio=0.3)
    left_img = np.array(cv2.imread('/home/lixing/Documents/BodyReconstruction_test/RenderPeople/RENDER/rp_adanna_posed_001/0000_0_00.jpg'))

    left_img_mask, mask = mask_aug(left_img)
    left_img_mask[:, :, 0] = left_img[:, :, 0] * mask + (1 - mask) * MaskAug.COLOR_GRAY
    left_img_mask[:, :, 1] = left_img[:, :, 1] * mask + (1 - mask) * MaskAug.COLOR_GRAY
    left_img_mask[:, :, 2] = left_img[:, :, 2] * mask + (1 - mask) * MaskAug.COLOR_GRAY
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/color_mask_2_0.3.png', left_img_mask)
    depth_img = np.array(cv2.imread('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/depth_front.png'))
    depth_img_mask = depth_img
    depth_img_mask[:, :, 0] = depth_img[:, :, 0] * mask + (1 - mask) * MaskAug.COLOR_GRAY
    depth_img_mask[:, :, 1] = depth_img[:, :, 1] * mask + (1 - mask) * MaskAug.COLOR_GRAY
    depth_img_mask[:, :, 2] = depth_img[:, :, 2] * mask + (1 - mask) * MaskAug.COLOR_GRAY
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/depth_mask.png', depth_img_mask)
    original_x, original_y, height, width = 150, 300, 200, 200
    gt = np.array(cv2.imread('/home/lixing/Documents/BodyReconstruction_test/RenderPeople/RENDER/rp_adanna_posed_001/0000_0_00.jpg'))
    gt[:, :, 0] = gt[:, :, 0] * (1 - mask) + (mask) * MaskAug.COLOR_GRAY
    gt[:, :, 1] = gt[:, :, 1] * (1 - mask) + (mask) * MaskAug.COLOR_GRAY
    gt[:, :, 2] = gt[:, :, 2] * (1 - mask) + (mask) * MaskAug.COLOR_GRAY
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/color_label.png', gt)
    depth_gt = np.array(cv2.imread('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/depth_front.png'))
    depth_gt[:, :, 0] = depth_gt[:, :, 0] * (1 - mask) + (mask) * MaskAug.COLOR_GRAY
    depth_gt[:, :, 1] = depth_gt[:, :, 1] * (1 - mask) + (mask) * MaskAug.COLOR_GRAY
    depth_gt[:, :, 2] = depth_gt[:, :, 2] * (1 - mask) + (mask) * MaskAug.COLOR_GRAY
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/depth_label.png', depth_gt)

    left_img = np.array(cv2.imread('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/0000_0_00.jpg'))
    crop_img = np.zeros([height, width, 3], dtype = int)
    crop_img[:, :, 0] = left_img[original_y:original_y + height, original_x:original_x + width, 0] \
        * mask[original_y:original_y + height, original_x:original_x + width] \
        + (1 - mask[original_y:original_y + height, original_x:original_x + width]) * MaskAug.COLOR_GRAY
    crop_img[:, :, 1] = left_img[original_y:original_y + height, original_x:original_x + width, 1] \
        * mask[original_y:original_y + height, original_x:original_x + width]\
        + (1 - mask[original_y:original_y + height, original_x:original_x + width]) * MaskAug.COLOR_GRAY
    crop_img[:, :, 2] = left_img[original_y:original_y + height, original_x:original_x + width, 2] \
        * mask[original_y:original_y + height, original_x:original_x + width] \
        + (1 - mask[original_y:original_y + height, original_x:original_x + width]) * MaskAug.COLOR_GRAY
    cv2.imwrite('/home/lixing/Programs/BodyReconstruction_Mask_2/Example/crop_color_2_0.2.png', crop_img)
if __name__ == "__main__":
    debug_main()
