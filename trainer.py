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


class Trainer(object):
    def __init__(self, exp_path: str, model: Model, loss_fun: Loss, optimizer: Optimizer,
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, log_freq: int, lr: float):

        self.exp_path = exp_path
        self.model = model
        self.loss_fun = loss_fun.cuda()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log_freq = log_freq
        self.counter = 0
        self.lr_initial = lr
        self.global_step = 0
        self.lr_current = lr
        self.losses = AVGMeter(log_freq=log_freq)
        self.writer = SummaryWriter(log_dir=self.exp_path+'/'+datetime.now().strftime("%Y-%m-%d@%H_%M"))

        if os.path.exists(exp_path + '/last.checkpoint'):
            print('[loading checkpoint \'{}\']'.format(self.exp_path + '/last.checkpoint'))
            self.start_epoch, self.start_step, self.best_train_loss = self.load_checkpoint(
                model=self.model, optimizer=self.optimizer, path=self.exp_path + '/last.checkpoint')
        else:
            self.start_epoch, self.start_step, self.best_train_loss = 0, 0, None

    def run(self, epochs: int):
        cudnn.benchmark = True
        for epoch in range(self.start_epoch, epochs):
            self.adjust_learning_rate(epoch)
            self.train(epoch)

    def train(self, epoch: int):
        self.model.train()

        for step, (image, gt, _) in enumerate(self.train_loader):
            step = step + self.start_step
            self.global_step += step
            image, gt = image.cuda(), gt.cuda(async=True)

            self._fit(image, gt)

            progress = step / len(self.train_loader)
            progress_bar = ('█' * int(30 * progress)) + ('┈' * (30 - int(30 * progress)))
            print('\r[{}] Epoch {} | Step {}: ◖{}◗ │ {:4.2f}% │ Loss: {:.4f}'.format(
                datetime.now().strftime("%Y-%m-%d@%H:%M"), epoch, step,
                progress_bar, 100 * progress,
                              1000 * self.losses.avgl
            ), end='')

            if step % self.log_freq == 0:
                self.log(epoch, step)
            print(int(progress))
            if int(progress) == 1 and epoch % 5 == 0:
                self.log(epoch, step)
                self.validate(epoch, eval_score=accuracy)
                self.start_step = 0
                break

    def log(self, epoch: int, step: int):
        print('\n Checkpoint!')

        # save best model
        current_loss = self.losses.avgl
        if self.best_train_loss is None or self.best_train_loss > current_loss:
            self.best_train_loss = current_loss
            self.save_checkpoint(
                epoch=epoch,
                step=step + 1,
                path=self.exp_path + '/best.checkpoint'
            )

        # save last model
        self.save_checkpoint(
            epoch=epoch,
            step=step + 1,
            path=self.exp_path + '/last.checkpoint'
        )

    def _fit(self, image: FloatTensor, gt):
        gt = gt.long()
        if type(self.loss_fun) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            gt = gt.float()

        x = Variable(image)
        gt = Variable(gt)
        self.optimizer.zero_grad()

        pred = self.model(x)[0]

        loss = self.loss_fun(pred, gt)
        self.writer.add_scalar('Training_Loss', 1000*loss.data[0], self.global_step)  # Logging the loss at each step for tensorboard
        self.losses.append(loss.data[0])
        loss.backward()

        self.optimizer.step(closure=None)

    def load_checkpoint(self, model: nn.Module, optimizer: Optimizer,
                        path: str) -> Tuple[int, int, float, float] or bool:
        try:
            checkpoint = torch.load(path)
        except:
            return False

        model.load_state_dict(checkpoint['weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        best_loss = checkpoint['best_loss']
        return epoch, step, best_loss

    def save_checkpoint(self, epoch: int, step: int, path: str = 'checkpoint/training.checkpoint') -> bool:
        checkpoint = {
            'weights': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'best_loss': self.best_train_loss,
        }
        try:
            torch.save(checkpoint, path)
            return True
        except:
            return False

    def validate(self, epoch, eval_score):
        print('Epoch '+str(epoch)+' : VALIDATION \n')
        losses = AVGMeter(log_freq=self.log_freq)
        accuracies = AVGMeter(log_freq=self.log_freq)

        # switch to evaluate mode
        self.model.eval()

        for i, (image, gt, image_original) in enumerate(self.val_loader):
            gt = gt.long()
            if type(self.loss_fun) in [torch.nn.modules.loss.L1Loss,
                                   torch.nn.modules.loss.MSELoss]:
                gt = gt.float()

            image, gt = image.cuda(), gt.cuda(async=True)

            image = Variable(image, volatile=True)
            gt = Variable(gt, volatile=True)

            # compute output
            pred = self.model(image)[0]
            loss = 1000*self.loss_fun(pred, gt)
            #self.writer.add_scalar('Validation_Loss', 1000*loss.data[0], self.global_step)

            # save 10 exemplary segmentations and show on tensorboard
            if i % 50 == 0:
                seg = save_image(image_original, pred, self.exp_path+'/Val_images/Epoch_{}'.format(str(epoch)), CITYSCAPE_PALLETE, i)
                #self.writer.add_image('Validation , Epoch {0}, Step {1}'.format(epoch, i), torch.from_numpy(seg), epoch + i)

            # measure accuracy and record loss
            losses.append(loss.data[0])
            if eval_score is not None:
                accuracies.append(eval_score(pred, gt))
                #self.writer.add_scalar('Validation_Accuracy', eval_score(pred, gt), self.global_step)

            print('Test: [{0}/{1}]\t'
                  'Loss {loss.last_val:.4f} ({loss.avg:.4f})\t'
                  'Score {score.last_val:.3f} ({score.avg:.3f})'.format(
                i, len(self.val_loader), loss=losses,
                score=accuracies), flush=True)

        print(' * Score {top1.avg:.3f}'.format(top1=accuracies))
        return accuracies.avg

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr_current = self.lr_initial * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_current


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data[0]

def save_image(image, pred, output_dir, palettes, step):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _, pred = torch.max(pred, 1)
    pred = pred.cpu().data.numpy()
    pred = np.array(palettes[pred.squeeze()], dtype=np.float32)
    image = np.array(image, dtype=np.float32)[0, :, :, :]

    blend = cv2.addWeighted(image, 0.6, pred , 0.4, 0)
    cv2.imwrite(output_dir + '/seg_{}'.format(str(step)) + '.png', blend)

    return blend