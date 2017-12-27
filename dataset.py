import os
from os import listdir
from os.path import isfile, join
import cv2
import random
import numpy as np

from typing import *
from torch.utils.data import Dataset
from numpy import ndarray as NPArray

MODE_TRAIN = 'train'
MODE_VAL = 'val'
MODE_TEST = 'test'
DICT_CLASSES = {'7': 0, '8': 1, '11': 2, '12':  3, '13': 4, '17': 5, '19': 6, '20': 7, '21': 8, '22': 9, '23': 10,
                '24': 11, '25': 12, '26': 13, '27': 14, '28': 15, '31': 16, '32': 17, '33': 18}
OLD_CLASSES = ['7', '8', '11', '12', '13', '17', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '31', '32',
               '33']


class CityScapesDataset(Dataset):

    def __init__(self, img_path: str, gt_path: str, mode: str):
        """
        :param dataset_dir_path: path of the images
        :param gt_path: path of the labels
        :param mode: 'train', 'test', 'val'
        """
        self.img_path = img_path
        self.gt_path = gt_path
        self.mode = mode
        self.dataset_size = len(os.listdir(os.path.join(self.img_path, mode)))

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, i: int) -> Tuple[NPArray, NPArray]:

        images = [img for img in listdir(join(self.img_path, self.mode)) if
                  isfile(join(self.img_path, self.mode, img))]
        gts = [gt for gt in listdir(join(self.gt_path, self.mode)) if
                  isfile(join(self.gt_path, self.mode, gt))]

        try:
            image_original = cv2.imread(join(self.img_path, self.mode, images[i]))
            image = image_original.transpose((2, 0, 1))
            image = image.astype(np.float32) / 127.5 - 1
            gt = cv2.imread(join(self.gt_path, self.mode, gts[i])).astype(np.float32)
            gt = gt.transpose((2, 0, 1))[0, :, :]
            gt = class_conversion(gt, gt.shape[0], gt.shape[1])  # classes numbers must be changed and have value [0, 18]

            return image, gt, image_original

        except:
            return self.__getitem__(random.randint(0, self.dataset_size))


def class_conversion(gt, width, height):
    for cls in OLD_CLASSES:
        gt = gt.reshape(width*height)
        gt[gt == int(cls)] = DICT_CLASSES[str(cls)]
        gt = gt.reshape((width, height))
    return gt


# def one_hot_encoding(gt):
#     gt = gt.transpose((2, 0, 1))[0, :, :]
#     gt_one_hot = np.zeros(shape=(19, gt.shape[0], gt.shape[1]))
#     for i in range(gt_one_hot.shape[0]):
#         class_indices = np.squeeze(gt == i)
#         gt_one_hot[i, class_indices] = 1.0
#     return gt_one_hot