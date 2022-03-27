import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from unet.unet import UNet
import cv2
import argparse
import numpy as np
from datetime import datetime
from models import networks


parser = argparse.ArgumentParser('train UNET')
parser.add_argument('--dataset', type=str, default='DAGM_Class1', help='dataset_seg path')
parser.add_argument('--imgsize', type=int, default=512, help='image size')
parser.add_argument('--channel', type=int, default=1, help='image channel')
parser.add_argument('--epoch', type=int, default=50, help='train epoch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_step', type=int, default=20, help='lr decay epoch')
parser.add_argument('--gpu', type=int, default=0, help='lr decay epoch')

args = parser.parse_args()


def extract_flaws(bin_img):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=4)
    filtered_img = np.zeros_like(bin_img)
    for i in range(retval):
        if i == 0:
            continue
        if stats[i][4] > 100:
            filtered_img[labels == i] = 1
    return filtered_img


def train_unet():
    lr = args.lr
    lr_step = args.lr_step
    epoch = args.epoch
    img_size = (args.imgsize, args.imgsize)
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    ae = UNet(n_channels=args.channel, n_classes=1).to(device)
    criterion = nn.MSELoss()

    optimizer = SGD(ae.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
    step_lr = StepLR(optimizer, lr_step)

    dataset = args.dataset
    data_path = './datasets/' + dataset + '_seg/train'
    stamp = 'UNET_' + dataset + '_imgsz' + str(img_size[0]) + '_e' + str(epoch) + '_lr' + str(lr)
    time_stamp = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(log_dir='aelog/' + stamp + '_' + time_stamp)

    glbstep = 0
    for e in range(epoch):
        img_path = os.path.join(data_path, 'img')
        mask_path = os.path.join(data_path, 'mask')
        for index, img_name in enumerate(os.listdir(img_path)):
            print(e, index)
            if args.channel == 1:
                img = Image.open(os.path.join(img_path, img_name)).convert('L')
            else:
                img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
            mask = Image.open(os.path.join(mask_path, img_name)).convert('L')
            img = img.resize(img_size)
            mask = mask.resize(img_size)
            img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
            mask_tensor = F.to_tensor(mask).unsqueeze(0).to(device)
            mask_tensor = (mask_tensor - 0.5) * 2

            output = ae(img_tensor)
            loss = criterion(output, mask_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            glbstep += 1
            writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=glbstep)
            if glbstep % 2 == 0 and torch.max(mask_tensor) == 1.0:
                df = 'CHW'
                if args.channel == 1:
                    df = 'HW'
                writer.add_image('img/img', img_tensor.squeeze(), glbstep, dataformats=df)
                writer.add_image('img/output', output.squeeze() / 2 + 0.5, glbstep, dataformats='HW')
                writer.add_image('img/mask', mask_tensor.squeeze() / 2 + 0.5, glbstep, dataformats='HW')
        step_lr.step()
    torch.save(ae, stamp + time_stamp + '.pth')
    return ae


def train_unet256():
    lr = 0.001
    lr_step = 10
    epoch = 35
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ae = networks.define_G(3, 1, 64, 'unet_256', 'batch',
                                        True, 'normal', 0.02, [0])
    criterion = nn.MSELoss()

    # optimizer = SGD(ae.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
    optimizer = Adam(ae.parameters(), lr=lr, betas=(0.5, 0.999))
    step_lr = StepLR(optimizer, lr_step)

    data_path = './datasets/KSDD_seg/train'
    stamp = 'UNET_funetune_e' + str(epoch) + '_lr' + str(lr)
    time_stamp = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(log_dir='aelog/' + stamp + '_' + time_stamp)

    glbstep = 0
    for e in range(epoch):
        img_path = os.path.join(data_path, 'img')
        mask_path = os.path.join(data_path, 'mask')
        for index, img_name in enumerate(os.listdir(img_path)):
            print(e, index)
            img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
            mask = Image.open(os.path.join(mask_path, img_name)).convert('L')
            img = img.resize((256, 256))
            mask = mask.resize((256, 256))  # (768, 576)
            img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
            mask_tensor = F.to_tensor(mask).unsqueeze(0).to(device)
            mask_tensor = (mask_tensor - 0.5) * 2

            output = ae(img_tensor)
            loss = criterion(output, mask_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            glbstep += 1
            writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=glbstep)
            if glbstep % 2 == 0 and torch.max(mask_tensor) == 1.0:
                writer.add_image('img/img', img_tensor.squeeze(), glbstep)
                writer.add_image('img/output', output.squeeze() / 2 + 0.5, glbstep, dataformats='HW')
                writer.add_image('img/mask', mask_tensor.squeeze() / 2 + 0.5, glbstep, dataformats='HW')
        step_lr.step()
        if e % 10 == 0 and e != 0:
            torch.save(ae, stamp + time_stamp + '_' + str(e) + '.pth')
    torch.save(ae, stamp + time_stamp + '.pth')
    return ae




if __name__ == '__main__':
    train_unet()
    # tensorboard --logdir=aelog