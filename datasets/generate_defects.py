import os
import sys
import random

import cv2
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision.transforms import functional as F

sys.path.append(".")
sys.path.append("..")
from models import networks
from data.base_dataset import get_transform
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataroot = '../datasets/DAGM_Class4_no_defect'
    opt.checkpoints_dir = '../checkpoints'
    opt.model = 'generate_defect'
    opt.netG = 'vanilla'
    opt.dataset_mode = 'vanilla'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.name = 'DAGM_Class4_filted'  # model path
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            model.forward()
            defect = torch.tensor((model.output / 2 + 0.5) * 255, dtype=torch.int).squeeze(0)
            if opt.output_nc == 1:
                defect = defect.squeeze(0)
            defect = np.array(defect.cpu().numpy(), dtype=np.uint8)
            cv2.imshow('1', defect)
            cv2.waitKey(0)
            pass
