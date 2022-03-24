#代码中的noisef为信号等级，例如我需要0.7的噪声，传入参数我传入的是1-0.7
from PIL import Image
import numpy as np
import random

import torchvision.transforms as transforms

norm_mean = (0.5, 0.5, 0.5)
norm_std = (0.5, 0.5, 0.5)
class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

class Gaussian_noise(object):
    """增加高斯噪声
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    """

    def __init__(self, mean, sigma):

        self.mean = mean
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 将图片灰度标准化
        img_ = np.array(img).copy()
        img_ = img_ / 255.0
        # 产生高斯 noise
        noise = np.random.normal(self.mean, self.sigma, img_.shape)
        # 将噪声和图片叠加
        gaussian_out = img_ + noise
        # 将超过 1 的置 1，低于 0 的置 0
        gaussian_out = np.clip(gaussian_out, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        gaussian_out = np.uint8(gaussian_out*255)
        # 将噪声范围搞为 0-255
        # noise = np.uint8(noise*255)
        return Image.fromarray(gaussian_out).convert('RGB')

def image_transform(noisef):
    """对训练集和测试集的图片作预处理转换
        train_transform：加噪图
        _train_transform：原图（不加噪）
        test_transform：测试图（不加噪）
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 重设大小
        #transforms.RandomCrop(32,padding=4),
        AddPepperNoise(noisef, p=0.9),                 #加椒盐噪声

        #Gaussian_noise(0, noisef),  # 加高斯噪声

        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(norm_mean,norm_std),
    ])
    _train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        #transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean,norm_std),

    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        #transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean,norm_std),

    ])
    return train_transform, _train_transform, test_transform
