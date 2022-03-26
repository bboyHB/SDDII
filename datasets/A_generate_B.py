import os
import sys
import random
import torch
import argparse
from PIL import Image
from torchvision.transforms import functional as F

sys.path.append(".")
sys.path.append("..")
from models import networks
from data.base_dataset import get_transform
from options.train_options import TrainOptions


opt = TrainOptions().parse()
opt.no_flip = True
transform = get_transform(opt, grayscale=opt.input_nc == 1)

# mix_generate = True
G = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
G_dict = torch.load(opt.modelpath)  # '../checkpoints/RSDDs2_cycle/latest_net_G_A.pth'
G_dict = {'module.'+k: v for k, v in dict(G_dict).items()}
G.load_state_dict(G_dict)

with torch.no_grad():
    data_dir = opt.datadir
    output_dir = data_dir + '_A_and_B'
    train_A = os.path.join(output_dir, 'A', 'train')
    train_B = os.path.join(output_dir, 'B', 'train')
    test_A = os.path.join(output_dir, 'A', 'test')
    test_B = os.path.join(output_dir, 'B', 'test')
    os.makedirs(train_A)
    os.makedirs(train_B)
    os.makedirs(test_A)
    os.makedirs(test_B)
    for t in ['train', 'test']:
        if t == 'train':
            out_A = train_A
            out_B = train_B
            # num2generate = 1024
        else:
            out_A = test_A
            out_B = test_B
            # num2generate = 512
        A_root = os.path.join(data_dir, t+'A')
        A_names = os.listdir(A_root)
        num2generate = len(A_names)
        for i in range(num2generate):
            # if mix_generate and i % 100 == 0:
            #     temp_p = './checkpoints/RSDDs1_cycle/' + str((i * 5) % 200 + 5) + '_net_G_A.pth'
            #     G_dict = torch.load(temp_p)  # '../checkpoints/RSDDs2_cycle/latest_net_G_A.pth'
            #     G_dict = {'module.' + k: v for k, v in dict(G_dict).items()}
            #     G.load_state_dict(G_dict)

            # name = random.choice(A_names)
            name = A_names[i]
            A_path = os.path.join(A_root, name)
            A_img = Image.open(A_path).convert('RGB')
            A_img_tensor = transform(A_img).unsqueeze(0)
            A_defect = G(A_img_tensor)
            B_img = F.to_pil_image((A_defect / 2 + 0.5).cpu().squeeze())
            A_img = A_img.resize(B_img.size)
            A_img.save(os.path.join(out_A, name[:-4] + '_' + str(i) + name[-4:]))
            B_img.save(os.path.join(out_B, name[:-4] + '_' + str(i) + name[-4:]))