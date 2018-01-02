import os
import click
from click import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

from tester import Tester
import dataset
import drn

@click.command()
@click.option('--arch', type=str, default='drn_c_26')
@click.option('--classes', default=19, type=int)
@click.option('--pretrained', type=bool, default=False)
@click.option('--pretrained_model', type=Path(exists=False), default='./log/Debug/best.checkpoint')
@click.option('--img_path', type=Path(exists=True), default='./cityscapes_fine_light/images/test')
@click.option('--output_dir_path', type=Path(exists=False), default='./output')
def test(arch: str, classes: int, pretrained: bool, img_path: str, output_dir_path: str, pretrained_model: Path):

    print('\n🚀 Starting Testing 🚀\n')

    model = drn.DRNSeg(arch, classes, pretrained_model=None, pretrained=pretrained).cuda()
    checkpoints = torch.load(pretrained_model)
    model.load_state_dict(checkpoints['weights'])

    tester = Tester(img_path=img_path,
                    output_dir_path=output_dir_path,
                    model=model)

    tester.run()

if __name__ == '__main__':
    test()
