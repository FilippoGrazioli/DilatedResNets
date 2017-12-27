import os
import click
from click import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

from trainer import Trainer
import dataset
import drn

@click.command()
# @click.option('--exp_name', type=str, prompt='Enter --exp_name')
# @click.option('--img_path', type=Path(exists=True), prompt='Enter --img_path')
# @click.option('--gt_path', type=Path(exists=True), prompt='Enter --gt_path')
@click.option('--exp_name', type=str, default='Debug')
@click.option('--img_path', type=Path(exists=True), default='./cityscapes_fine_light/images')
@click.option('--gt_path', type=Path(exists=True), default='./cityscapes_fine_light/gt')
@click.option('--pretrained', type=bool, default=False)
@click.option('--epochs', type=int, default=100)
@click.option('--log_freq', type=int, default=100)
@click.option('--batch_size', type=int, default=1)
@click.option('--lr', type=float, default=0.01)
@click.option('--momentum', type=float, default=0.9)
@click.option('--weight_decay', type=float, default=1e-4)
@click.option('--classes', default=19, type=int)
@click.option('--pretrained_model', type=str, default=None)
@click.option('--arch', type=str, default='drn_c_26')
@click.option('--log_dir_path', type=Path(exists=False), default='log')
def train(exp_name: str, img_path: str, epochs: int, log_freq: int, pretrained_model: str,
          batch_size: int, lr: float, momentum: float, weight_decay: float,
          log_dir_path: str, gt_path: str, arch: str, classes: int, pretrained: bool):

    print('\n🚀 Starting Experiment \'{}\' 🚀\n'.format(exp_name))

    # Create experiment log directory if it does not exists
    exp_path = log_dir_path + '/' + exp_name
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Load dataset
    ds_train = dataset.CityScapesDataset(img_path, gt_path, 'train')
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    ds_test = dataset.CityScapesDataset(img_path, gt_path, 'test')
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    ds_val = dataset.CityScapesDataset(img_path, gt_path, 'val')
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    model = drn.DRNSeg(arch, classes, pretrained_model=pretrained_model, pretrained=pretrained).cuda()
    loss_fun = nn.NLLLoss2d(ignore_index=255).cuda()
    optimizer = optim.SGD(model.optim_parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    trainer = Trainer(
        exp_path=exp_path,
        model=model,
        loss_fun=loss_fun,
        optimizer=optimizer,
        train_loader=train_loader,
        log_freq=log_freq,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=lr
    )
    trainer.run(epochs)



if __name__ == '__main__':
    train()
