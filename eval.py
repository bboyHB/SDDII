import os
import cv2
import time
import torch
import random
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from unet.unet import UNet
from models import networks
from data.base_dataset import get_transform
from options.train_options import TrainOptions
from torchvision.transforms import functional as F
from datasets.process_data import cal_fids


def cut_image(img_pil, height, stride):
    ret = []
    now = 0
    for j in range((img_pil.height - height) // stride + 1):
        ret.append(img_pil.crop((0, now, img_pil.width, now + height)))
        now += stride
    else:
        if (img_pil.height - height) % stride != 0:
            ret.append(img_pil.crop((0, img_pil.height - height, img_pil.width, img_pil.height)))
    return ret


def gether_seg(segs, size, height, stride):
    now = 0
    final_seg = np.zeros((size[1], size[0]))
    for i in range(len(segs)):
        if i == len(segs) - 1:
            final_seg[size[1] - height:, :] += segs[i]
        else:
            final_seg[now:now + height, :] += segs[i]
        now += stride
    final_seg[final_seg > 0] = 255
    return final_seg


def seg_iou_single_img(seg, gt):
    intersection = np.logical_and(seg == 255, gt == 255)
    union = np.logical_or(seg == 255, gt == 255)
    num_intersection = np.sum(intersection)
    num_union = np.sum(union)
    if num_union == 0:
        return 0
    iou = num_intersection / num_union
    return iou


def num_intersection_union(seg, gt):
    intersection = np.logical_and(seg == 255, gt == 255)
    union = np.logical_or(seg == 255, gt == 255)
    num_intersection = np.sum(intersection)
    num_union = np.sum(union)
    return num_intersection, num_union


def pixel_precision_recall_f1(seg, gt):
    """
    :param seg: 2d array, int,
            estimated targets as returned by a classifier
    :param gt: 2d array, int,
            ground truth
    :return:
        precision, recall, f1: float
    """
    seg_flt, gt_flt = seg.flatten(), gt.flatten()
    seg_flt[seg_flt > 0] = 1
    gt_flt[gt_flt > 0] = 1
    f1 = f1_score(y_true=gt_flt, y_pred=seg_flt)
    precision = precision_score(y_true=gt_flt, y_pred=seg_flt)
    recall = recall_score(y_true=gt_flt, y_pred=seg_flt)
    return precision, recall, f1


def filt_small_pixel_block(final_seg):
    num_pixel_threshold = 50
    num_components, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        np.array(final_seg, dtype=np.uint8))
    keep = list((stats[:, 4] > num_pixel_threshold))
    for i in range(num_components):
        if not keep[i]:
            label_map[label_map == i] = 0
    label_map[label_map != 0] = 255
    return np.array(label_map, dtype=np.uint8)


def filt_big_pixel_block(final_seg):
    num_pixel_threshold = 3000
    num_components, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        np.array(final_seg, dtype=np.uint8))
    keep = list((stats[:, 4] < num_pixel_threshold))
    for i in range(num_components):
        if not keep[i]:
            label_map[label_map == i] = 0
    label_map[label_map != 0] = 255
    return np.array(label_map, dtype=np.uint8)

def eval_compare():
    dataset_name = 'RSDDs1'
    opt = TrainOptions().parse()
    opt.no_flip = True
    transform = get_transform(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, [0])
    R_dict = torch.load('checkpoints/' + dataset_name + '_cycle/latest_net_G_B.pth')
    R_dict = {'module.'+k: v for k, v in dict(R_dict).items()}
    R.load_state_dict(R_dict)
    R.to(device)

    R_p2p = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_256', 'batch',
                                            not opt.no_dropout, opt.init_type, opt.init_gain, [0])
    R_p2p_dict = torch.load('checkpoints/' + dataset_name + '_pix2pix/latest_net_G.pth')
    # R_p2p_dict = torch.load('checkpoints/RSDDs2_pix2pix/10_net_G.pth')
    R_p2p_dict = {'module.'+k: v for k, v in dict(R_p2p_dict).items()}
    R_p2p.load_state_dict(R_p2p_dict)
    R_p2p.to(device)

    eval_path = './datasets/' + dataset_name + '_origin/test'
    img_path = os.path.join(eval_path, 'img')
    mask_path = os.path.join(eval_path, 'mask')

    U = torch.load('UNET_RSDDs1_imgsz256_e50_lr0.012021-05-15_10-56-48.pth', device)


    with torch.no_grad():
        thresh_hold = 0.15
        real = []
        pred_cycle = []
        pred_p2p = []
        pred_unet = []
        ious_cycle = []
        ious_p2p = []
        ious_unet = []
        # intersection_cycle = 0
        # intersection_p2p = 0
        # intersection_unet = 0
        # union_cycle = 0
        # union_p2p = 0
        # union_unet = 0
        time_cycle = []
        time_p2p = []
        time_unet = []
        precision_cycle = []
        precision_p2p = []
        precision_unet = []
        recall_cycle = []
        recall_p2p = []
        recall_unet = []
        f1_cycle = []
        f1_p2p = []
        f1_unet = []

        for name in os.listdir(img_path):
            A_path = os.path.join(img_path, name)
            A_img = Image.open(A_path).convert('RGB')
            A_mask = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE)
            if np.max(A_mask) > 0:
                real.append(1)
            else:
                real.append(0)

            segs = []
            now = time.time()
            for chip in cut_image(A_img, A_img.width, A_img.width // 2):
                chip_tensor = transform(chip).unsqueeze(0).to(device)
                chip_repair = R(chip_tensor)
                diff = torch.sum(torch.abs(chip_repair - chip_tensor), dim=1) / 6.0
                # diff = torch.sum(torch.clamp(chip_repair - chip_tensor, min=0), dim=1) / 6.0
                diff = torch.nn.functional.interpolate(diff.unsqueeze(0), (A_img.width, A_img.width))
                diff[diff >= thresh_hold] = 1
                diff[diff < thresh_hold] = 0
                segs.append(diff.squeeze().cpu().numpy())
            final_seg_cycle = gether_seg(segs, A_img.size, A_img.width, A_img.width // 2)
            # iu_cycle = num_intersection_union(final_seg_cycle, A_mask)
            # intersection_cycle += iu_cycle[0]
            # union_cycle += iu_cycle[1]
            ious_cycle.append(seg_iou_single_img(final_seg_cycle, A_mask))
            p_r_f1_cycle = pixel_precision_recall_f1(final_seg_cycle, A_mask)
            precision_cycle.append(p_r_f1_cycle[0])
            recall_cycle.append(p_r_f1_cycle[1])
            f1_cycle.append(p_r_f1_cycle[2])
            time_cycle.append(time.time()-now)
            if np.max(final_seg_cycle) > 0:
                pred_cycle.append(1)
            else:
                pred_cycle.append(0)

            segs_p2p = []
            now = time.time()
            for chip in cut_image(A_img, A_img.width, A_img.width // 2):
                chip_tensor = transform(chip).unsqueeze(0).to(device)
                chip_repair = R_p2p(chip_tensor)
                if dataset_name == 'RSDDs1':
                    diff = torch.sum(torch.clamp(chip_repair - chip_tensor, min=0), dim=1) / 6.0
                elif dataset_name == 'RSDDs2':
                    diff = torch.sum(torch.abs(chip_repair - chip_tensor), dim=1) / 6.0
                else:
                    exit()
                diff = torch.nn.functional.interpolate(diff.unsqueeze(0), (A_img.width, A_img.width))
                # hist, _ = np.histogram(diff.squeeze().cpu().numpy())
                diff[diff >= thresh_hold] = 1
                diff[diff < thresh_hold] = 0
                segs_p2p.append(diff.squeeze().cpu().numpy())
            final_seg_p2p = gether_seg(segs_p2p, A_img.size, A_img.width, A_img.width // 2)
            if dataset_name == 'RSDDs1':
                final_seg_p2p = filt_small_pixel_block(final_seg_p2p)
            # final_seg_p2p = filt_big_pixel_block(final_seg_p2p)
            # iu_p2p = num_intersection_union(final_seg_p2p, A_mask)
            # intersection_p2p += iu_p2p[0]
            # union_p2p += iu_p2p[1]
            ious_p2p.append(seg_iou_single_img(final_seg_p2p, A_mask))
            p_r_f1_p2p = pixel_precision_recall_f1(final_seg_p2p, A_mask)
            precision_p2p.append(p_r_f1_p2p[0])
            recall_p2p.append(p_r_f1_p2p[1])
            f1_p2p.append(p_r_f1_p2p[2])
            time_p2p.append(time.time() - now)
            if np.max(final_seg_p2p) > 0:
                pred_p2p.append(1)
            else:
                pred_p2p.append(0)

            segs_unet = []
            now = time.time()
            for chip in cut_image(A_img, A_img.width, A_img.width // 2):
                chip_tensor = transform(chip).unsqueeze(0).to(device)
                chip_repair = U(chip_tensor)
                diff = chip_repair / 2 + 0.5
                diff = torch.nn.functional.interpolate(diff, (A_img.width, A_img.width))
                diff[diff >= 0.5] = 1
                diff[diff < 0.5] = 0
                segs_unet.append(diff.squeeze().cpu().numpy())
            final_seg_unet = gether_seg(segs_unet, A_img.size, A_img.width, A_img.width // 2)
            # final_seg_unet = filt_small_pixel_block(final_seg_unet)
            # iu_unet = num_intersection_union(final_seg_unet, A_mask)
            # intersection_unet += iu_unet[0]
            # union_unet += iu_unet[1]
            ious_unet.append(seg_iou_single_img(final_seg_unet, A_mask))
            p_r_f1_unet = pixel_precision_recall_f1(final_seg_unet, A_mask)
            precision_unet.append(p_r_f1_unet[0])
            recall_unet.append(p_r_f1_unet[1])
            f1_unet.append(p_r_f1_unet[2])
            time_unet.append(time.time() - now)
            if np.max(final_seg_unet) > 0:
                pred_unet.append(1)
            else:
                pred_unet.append(0)

            img_origin_np = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE)
            height_per = img_origin_np.shape[0]
            width_per = img_origin_np.shape[1]
            interval = 5
            compare_img = np.ones((height_per, width_per*5 + interval*4)) * 255
            now_x, now_y = 0, 0
            imgs2paste = [img_origin_np, final_seg_unet, final_seg_cycle, final_seg_p2p, A_mask]
            for img2paste in imgs2paste:
                compare_img[now_y: now_y+height_per, now_x: now_x+width_per] = img2paste
                now_x += interval + width_per
            cv2.imwrite(dataset_name+'_eval_output/'+name[:-3]+'png', compare_img)
            # 显示结果
            # cv2.imshow('1', cv2.imread(A_path))
            # cv2.imshow('2', A_mask)
            # cv2.imshow('3', final_seg_unet)
            # cv2.imshow('4', final_seg_cycle)
            # cv2.imshow('5', final_seg_p2p)
            # cv2.waitKey(0)

        # print('total pixel_IOU UNET:', intersection_unet / union_unet)
        # print('total pixel_IOU CycleGAN:', intersection_cycle / union_cycle)
        # print('total pixel_IOU CycleGAN+Pix2Pix:', intersection_p2p / union_p2p)
        print('average IOU UNET:', sum(ious_unet) / len(ious_unet))
        print('average IOU CycleGAN:', sum(ious_cycle) / len(ious_cycle))
        print('average IOU CycleGAN+Pix2Pix:', sum(ious_p2p) / len(ious_p2p))
        print('average precision Score UNET:', sum(precision_unet) / len(precision_unet))
        print('average precision Score CycleGAN:', sum(precision_cycle) / len(precision_cycle))
        print('average precision Score CycleGAN+Pix2Pix:', sum(precision_p2p) / len(precision_p2p))
        print('average recall Score UNET:', sum(recall_unet) / len(recall_unet))
        print('average recall Score CycleGAN:', sum(recall_cycle) / len(recall_cycle))
        print('average recall Score CycleGAN+Pix2Pix:', sum(recall_p2p) / len(recall_p2p))
        print('average F1 Score UNET:', sum(f1_unet) / len(f1_unet))
        print('average F1 Score CycleGAN:', sum(f1_cycle) / len(f1_cycle))
        print('average F1 Score CycleGAN+Pix2Pix:', sum(f1_p2p) / len(f1_p2p))
        print('average time UNET:', sum(time_unet) / len(time_unet))
        print('average time CycleGAN:', sum(time_cycle) / len(time_cycle))
        print('average time CycleGAN+Pix2Pix:', sum(time_p2p) / len(time_p2p))

        print('img-level ACC UNET: ', accuracy_score(real, pred_unet))
        print('img-level ACC CycleGAN: ', accuracy_score(real, pred_cycle))
        print('img-level ACC CycleGAN+Pix2Pix: ', accuracy_score(real, pred_p2p))
        print('img-level F1 Score UNET: ', f1_score(real, pred_unet))
        print('img-level F1 Score CycleGAN: ', f1_score(real, pred_cycle))
        print('img-level F1 Score CycleGAN+Pix2Pix: ', f1_score(real, pred_p2p))
        print('img-level confusion matrix UNET:')
        print(confusion_matrix(real, pred_unet))
        print('img-level confusion matrix CycleGAN:')
        print(confusion_matrix(real, pred_cycle))
        print('img-level confusion matrix CycleGAN+Pix2Pix:')
        print(confusion_matrix(real, pred_p2p))
        pass


def eval_when_train_p2p(opt, R_p2p):
    temp_no_flip = opt.no_flip
    opt.no_flip = True
    transform = get_transform(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # R_p2p = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_256', 'batch',
    #                           not opt.no_dropout, opt.init_type, opt.init_gain, [0])
    # R_p2p_dict = torch.load('checkpoints/RSDDs1_pix2pix/latest_net_G.pth')
    # R_p2p_dict = {'module.' + k: v for k, v in dict(R_p2p_dict).items()}
    # R_p2p.load_state_dict(R_p2p_dict)
    # R_p2p.to(device)

    eval_path = './datasets/RSDDs1_origin/test'
    img_path = os.path.join(eval_path, 'img')
    mask_path = os.path.join(eval_path, 'mask')

    thresh_hold = 0.15
    real = []
    pred_p2p = []
    ious_p2p = []
    time_p2p = []
    f1_p2p = []
    with torch.no_grad():
        for name in os.listdir(img_path):
            A_path = os.path.join(img_path, name)
            A_img = Image.open(A_path).convert('RGB')
            A_mask = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE)
            if np.max(A_mask) > 0:
                real.append(1)
            else:
                real.append(0)

            segs_p2p = []
            now = time.time()
            for chip in cut_image(A_img, A_img.width, A_img.width // 2):
                chip_tensor = transform(chip).unsqueeze(0).to(device)
                chip_repair = R_p2p(chip_tensor)
                # diff = torch.sum(torch.abs(chip_repair - chip_tensor), dim=1) / 6.0
                diff = torch.sum(torch.clamp(chip_repair - chip_tensor, min=0), dim=1) / 6.0
                diff = torch.nn.functional.interpolate(diff.unsqueeze(0), (A_img.width, A_img.width))
                # hist, _ = np.histogram(diff.squeeze().cpu().numpy())
                diff[diff >= thresh_hold] = 1
                diff[diff < thresh_hold] = 0
                segs_p2p.append(diff.squeeze().cpu().numpy())
            final_seg_p2p = gether_seg(segs_p2p, A_img.size, A_img.width, A_img.width // 2)
            final_seg_p2p = filt_small_pixel_block(final_seg_p2p)
            ious_p2p.append(seg_iou_single_img(final_seg_p2p, A_mask))
            f1_p2p.append(pixel_precision_recall_f1(final_seg_p2p, A_mask)[2])
            time_p2p.append(time.time() - now)
            if np.max(final_seg_p2p) > 0:
                pred_p2p.append(1)
            else:
                pred_p2p.append(0)

        IOU = sum(ious_p2p) / len(ious_p2p)
        F1 = sum(f1_p2p) / len(f1_p2p)
        print('average IOU CycleGAN+Pix2Pix:', IOU)
        print('average F1 Score CycleGAN+Pix2Pix:', F1)
        # print('average time CycleGAN+Pix2Pix:', sum(time_p2p) / len(time_p2p))
        #
        # print('img-level ACC CycleGAN+Pix2Pix: ', accuracy_score(real, pred_p2p))
        # print('img-level F1 Score CycleGAN+Pix2Pix: ', f1_score(real, pred_p2p))
        # print('img-level confusion matrix CycleGAN+Pix2Pix:')
        # print(confusion_matrix(real, pred_p2p))
    opt.no_flip = temp_no_flip
    return IOU, F1

def eval_fid_when_train_cyclegan(opt, R_cyc):
    temp_no_flip = opt.no_flip
    temp_four_rotate = opt.four_rotate
    opt.no_flip = True
    opt.four_rotate = False
    transform = get_transform(opt, grayscale=opt.input_nc == 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_to_generate = 2048
    real_A_path = os.path.join(opt.dataroot, "trainA")
    real_B_path = os.path.join(opt.dataroot, "trainB")
    fake_B_path = os.path.join(opt.checkpoints_dir, opt.name, "B")
    if not os.path.exists(fake_B_path):
        os.makedirs(fake_B_path)

    with torch.no_grad():
        real_As = os.listdir(real_A_path)
        for i in range(num_to_generate):
            name = random.choice(real_As)
            A_path = os.path.join(real_A_path, name)
            if opt.input_nc == 1:
                A_img = Image.open(A_path).convert('L')
            else:
                A_img = Image.open(A_path).convert('RGB')

            A_img_tensor = transform(A_img).unsqueeze(0).to(device)
            fake_B_tensor = R_cyc(A_img_tensor)
            fake_B = F.to_pil_image((fake_B_tensor / 2 + 0.5).cpu().squeeze())
            fake_B.save(os.path.join(fake_B_path, f'fake_B_{i}.png'))
        fids = cal_fids(fake_B_path, real_A_path)
        print('FID:', fids)
    opt.no_flip = temp_no_flip
    opt.four_rotate = temp_four_rotate
    return fids

if __name__ == '__main__':
    eval_compare()


# python train.py --dataroot ./datasets/KSDD --preprocess crop --crop_size 500
# python train.py --dataroot ./datasets/KSDD --preprocess none --num_threads 0

'''
average IOU UNET: 0.4572931652391946
average IOU CycleGAN: 0.2786865383168425
average IOU CycleGAN+Pix2Pix: 0.1919577978374283
average F1 Score UNET: 0.6886167753628988
average F1 Score CycleGAN: 0.5090605476619966
average F1 Score CycleGAN+Pix2Pix: 0.3524709868306195
'''

