import os
from typing import *
from datetime import datetime
from torch import FloatTensor
from torch.optim import Optimizer
from torch.autograd import Variable
from torch.nn import Module as Model
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.modules.loss import _Loss as Loss
from torch import nn
import torch
import cv2
import torch.backends.cudnn as cudnn
from avg_meter import AVGMeter
from tensorboard import SummaryWriter
from PIL import Image


CITYSCAPE_PALLETE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

class Tester(object):
    def __init__(self, img_path: str, output_dir_path: str, model: Model):

        self.model = model
        self.img_path = img_path
        self.output_dir_path = output_dir_path
        self.model.eval()

    def save_image(self, image, pred, output_dir, palettes, step):
        """
        Saves a given (B x C x H x W) into an image file.
        If given a mini-batch tensor, will save the tensor as a grid of images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        _, pred = torch.max(pred, 1)
        pred = pred.cpu().data.numpy()
        pred = np.array(palettes[pred.squeeze()], dtype=np.float32)
        image = np.array(image, dtype=np.float32)
        blend = cv2.addWeighted(image, 0.6, pred, 0.4, 0)
        cv2.imwrite(output_dir + '/seg_{}'.format(str(step)) + '.png', blend)

    def run(self):
        index = 0
        for img in os.listdir(self.img_path):
            print('Processing image {0} / {1} \n'.format(index, len(os.listdir(self.img_path))))
            image_original = cv2.imread(self.img_path + '/' + img)
            img = np.expand_dims(image_original.transpose((2, 0, 1)), 0) / 127.5 - 1
            img = Variable(torch.from_numpy(img)).cuda()
            img = img.float()
            pred = self.model(img)[0]
            self.save_image(image_original, pred, self.output_dir_path, CITYSCAPE_PALLETE, index)
            index += 1


