
import torch
from PIL import Image

from models import networks
from data.base_dataset import get_transform
from options.train_options import TrainOptions
from torchvision.transforms import functional as F

opt = TrainOptions().parse()
transform = get_transform(opt)

G = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0])
G_dict = torch.load('checkpoints/6/latest_net_G_A.pth')
G_dict = {'module.'+k: v for k, v in dict(G_dict).items()}
G.load_state_dict(G_dict)

R = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0])
R_dict = torch.load('checkpoints/6/latest_net_G_B.pth')
R_dict = {'module.'+k: v for k, v in dict(R_dict).items()}
R.load_state_dict(R_dict)

with torch.no_grad():
    A_path = 'C:/Users/DeepLearning/Desktop/pytorch-CycleGAN-and-pix2pix-master/datasets/DAGM/Class6/testA/0001.PNG'
    A_img = transform(Image.open(A_path).convert('RGB')).unsqueeze(0)
    A_defect = G(A_img)
    A_repair = R(A_defect)
    F.to_pil_image((A_img / 2 + 0.5).cpu().squeeze()).show()
    F.to_pil_image((A_defect / 2 + 0.5).cpu().squeeze()).show()
    F.to_pil_image((A_repair / 2 + 0.5).cpu().squeeze()).show()

# python train.py --dataroot ./datasets/KSDD --preprocess crop --crop_size 500
# python train.py --dataroot ./datasets/KSDD --preprocess none --num_threads 0

#