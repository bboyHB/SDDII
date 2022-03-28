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
from datasets.process_data import cal_fids, extract_diff, extract_biggest_connected_component


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
    opt = TrainOptions().parse()
    dataset_name = opt.eval_dataset_name
    output_dir = dataset_name+'_eval_output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if dataset_name not in [f'DAGM_Class{i}' for i in range(1, 11)] + ['RSDDs1', 'RSDDs2']:
        return
    opt.no_flip = True
    transform = get_transform(opt, grayscale=opt.input_nc == 1)

    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    R = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    R_dict = torch.load('checkpoints/' + dataset_name + f'_cycle{opt.suffix}/{opt.modelpath}_net_G_B.pth')
    R_dict = {'module.'+k: v for k, v in dict(R_dict).items()}
    R.load_state_dict(R_dict)
    R.to(device)

    R_p2p = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_256', 'batch',
                              not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    R_p2p_dict = torch.load('checkpoints/' + dataset_name + f'_pix2pix/{opt.epoch}_net_G.pth')
    # R_p2p_dict = torch.load('checkpoints/RSDDs2_pix2pix/10_net_G.pth')
    R_p2p_dict = {'module.'+k: v for k, v in dict(R_p2p_dict).items()}
    R_p2p.load_state_dict(R_p2p_dict)
    R_p2p.to(device)

    if dataset_name[:4] == 'RSDD':
        eval_path = './datasets/' + dataset_name + '_origin/test'
        thresh_hold = opt.threshold / 255  # 0.15
        thresh_hold2 = opt.threshold2 / 255
        thresh_hold3 = opt.threshold3 / 255
    else:
        eval_path = './datasets/' + dataset_name + '_seg/test'
        thresh_hold = opt.threshold
        thresh_hold2 = opt.threshold2
        thresh_hold3 = opt.threshold3
    img_path = os.path.join(eval_path, 'img')
    mask_path = os.path.join(eval_path, 'mask')

    U = torch.load(f'UNET_{dataset_name}_imgsz{opt.load_size}_e{opt.epoch_count}_lr{opt.lr}.pth', device)

    with torch.no_grad():
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
            if opt.input_nc == 1:
                A_img = Image.open(A_path).convert('L')
            else:
                A_img = Image.open(A_path).convert('RGB')
            A_mask = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE)
            if np.max(A_mask) > 0:
                real.append(1)
            else:
                real.append(0)

            segs = []
            now = time.time()
            if dataset_name[:4] == 'RSDD':
                for chip in cut_image(A_img, A_img.width, A_img.width // 2):
                    chip_tensor = transform(chip).unsqueeze(0).to(device)
                    chip_repair = R(chip_tensor)
                    diff = torch.sum(torch.abs(chip_repair - chip_tensor), dim=1) / 6.0
                    # diff = torch.sum(torch.clamp(chip_repair - chip_tensor, min=0), dim=1) / 6.0
                    diff = torch.nn.functional.interpolate(diff.unsqueeze(0), (A_img.width, A_img.width))
                    diff[diff >= thresh_hold2] = 1
                    diff[diff < thresh_hold2] = 0
                    segs.append(diff.squeeze().cpu().numpy())
                final_seg_cycle = gether_seg(segs, A_img.size, A_img.width, A_img.width // 2)
            else:
                A_img_tensor = transform(A_img).unsqueeze(0).to(device)
                A_img_repair = R(A_img_tensor)
                diff = torch.abs((A_img_repair / 2 + 0.5) * 255 - (A_img_tensor / 2 + 0.5) * 255)
                diff[diff > thresh_hold2] = 255
                diff[diff <= thresh_hold2] = 0
                final_seg_cycle = diff.squeeze().cpu().numpy().astype(np.uint8)
                final_seg_cycle = thresh_combine_open_close(final_seg_cycle)
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
            if dataset_name[:4] == 'RSDD':
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
            else:
                A_img_tensor = transform(A_img).unsqueeze(0).to(device)
                A_img_repair = R_p2p(A_img_tensor)
                if opt.seg_plan_b:
                    A_img_tensor_numpy = A_img_tensor.squeeze().cpu().numpy()
                    A_img_repair_numpy = A_img_repair.squeeze().cpu().numpy()
                    final_seg_p2p = extract_diff(A_img_tensor_numpy, A_img_repair_numpy, thresh_hold,
                                                 (opt.first_kernel, opt.first_kernel),
                                                 (opt.second_kernel, opt.second_kernel),
                                                 (opt.third_kernel, opt.third_kernel), True)
                else:
                    diff = torch.abs((A_img_repair / 2 + 0.5) * 255 - (A_img_tensor / 2 + 0.5) * 255)
                    diff[diff > thresh_hold] = 255
                    diff[diff <= thresh_hold] = 0
                    final_seg_p2p = diff.squeeze().cpu().numpy().astype(np.uint8)
                    final_seg_p2p = thresh_combine_open_close(final_seg_p2p, opt.third_kernel, opt.f, opt.onlymax)
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
            if dataset_name[:4] == 'RSDD':
                for chip in cut_image(A_img, A_img.width, A_img.width // 2):
                    chip_tensor = transform(chip).unsqueeze(0).to(device)
                    chip_repair = U(chip_tensor)
                    diff = chip_repair / 2 + 0.5
                    diff = torch.nn.functional.interpolate(diff, (A_img.width, A_img.width))
                    diff[diff >= thresh_hold3] = 1
                    diff[diff < thresh_hold3] = 0
                    segs_unet.append(diff.squeeze().cpu().numpy())
                final_seg_unet = gether_seg(segs_unet, A_img.size, A_img.width, A_img.width // 2)
            else:
                A_img_tensor = transform(A_img).unsqueeze(0).to(device)
                A_img_repair = U(A_img_tensor)
                diff = torch.abs((A_img_repair / 2 + 0.5) * 255 - (A_img_tensor / 2 + 0.5) * 255)
                diff[diff > thresh_hold3] = 255
                diff[diff <= thresh_hold3] = 0
                final_seg_unet = diff.squeeze().cpu().numpy().astype(np.uint8)

                if dataset_name[-1] in ('4', '7', '0'):
                    if dataset_name[-1] in ('7'):
                        final_seg_unet = thresh_combine_open_close(final_seg_unet)
                    inverse = np.array(final_seg_unet, dtype=np.uint8)
                    final_seg_unet[inverse == 0] = 255
                    final_seg_unet[inverse == 255] = 0
                    if dataset_name[-1] in ('7'):
                        final_seg_unet = extract_biggest_connected_component(final_seg_unet)
                    if dataset_name[-1] in ('0'):
                        final_seg_unet = extract_biggest_connected_component(final_seg_unet)
                        final_seg_unet = cv2.morphologyEx(final_seg_unet, cv2.MORPH_CLOSE, np.ones((opt.third_kernel, opt.third_kernel), np.uint8))
                elif dataset_name[-1] == '6':
                    final_seg_unet = thresh_combine_open_close(final_seg_unet, 9, f=True, onlymax=True)
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
            names_to_save = ('origin', 'unet', 'cycle', 'p2p', 'mask')
            imgs_to_save = (img_origin_np, final_seg_unet, final_seg_cycle, final_seg_p2p, A_mask)
            for i, img in enumerate(imgs_to_save):
                pre_suf = name.split('.')
                name2save = f'{pre_suf[0]}_{names_to_save[i]}.png'
                cv2.imwrite(os.path.join(output_dir, name2save), img)
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


def save_one_total_eval_img(img_origin_np, final_seg_unet, final_seg_cycle, final_seg_p2p, A_mask, output_dir, name, interval = 5):
    height_per = img_origin_np.shape[0]
    width_per = img_origin_np.shape[1]
    compare_img = np.ones((height_per, width_per * 5 + interval * 4)) * 255
    now_x, now_y = 0, 0
    imgs2paste = [img_origin_np, final_seg_unet, final_seg_cycle, final_seg_p2p, A_mask]
    for img2paste in imgs2paste:
        compare_img[now_y: now_y + height_per, now_x: now_x + width_per] = img2paste
        now_x += interval + width_per
    cv2.imwrite(os.path.join(output_dir, name), compare_img)

def eval_when_train_p2p(opt, R_p2p):
    temp_no_flip = opt.no_flip
    opt.no_flip = True
    transform = get_transform(opt, grayscale=opt.input_nc == 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.name[:4] == 'RSDD':
        eval_path = './datasets/' + opt.name[:6] + '_origin/test'
    elif opt.name[:4] == 'DAGM':
        eval_path = './datasets/' + opt.name[:11] + '_seg/Test'
    else:
        eval_path = 'not_exist'
    img_path = os.path.join(eval_path, 'img')
    mask_path = os.path.join(eval_path, 'mask')

    real = []
    pred_p2p = []
    ious_p2p = []
    pre_p2p = []
    rec_p2p = []
    time_p2p = []
    f1_p2p = []
    with torch.no_grad():
        for name in os.listdir(img_path):
            A_path = os.path.join(img_path, name)
            A_img = Image.open(A_path).convert('RGB')
            A_mask = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE)
            # if np.max(A_mask) > 0:
            #     real.append(1)
            # else:
            #     real.append(0)

            segs_p2p = []
            now = time.time()
            if opt.name[:4] == 'RSDD':
                thresh_hold = 0.15
                for chip in cut_image(A_img, A_img.width, A_img.width // 2):
                    chip_tensor = transform(chip).unsqueeze(0).to(device)
                    chip_repair = R_p2p(chip_tensor)
                    # diff = torch.sum(torch.abs(chip_repair - chip_tensor), dim=1) / 6.0
                    # ***************clamp估计不行，因为修复前后，有可能像素更大也可能更小
                    # 另外，除以6是因为通道数为3，原像素范围为[-1,1]，逐像素相减然后压缩后，区间范围变为[-3,3]，除以6进行归一化
                    diff = torch.sum(torch.clamp(chip_repair - chip_tensor, min=0), dim=1) / 6.0
                    # 这边是由于当时训练输入size为512，和实际图片size的160或55不一样，所以要重新resize，用于与二值图计算IOU
                    diff = torch.nn.functional.interpolate(diff.unsqueeze(0), (A_img.width, A_img.width))
                    # hist, _ = np.histogram(diff.squeeze().cpu().numpy())
                    diff[diff >= thresh_hold] = 1
                    diff[diff < thresh_hold] = 0
                    segs_p2p.append(diff.squeeze().cpu().numpy())
                final_seg_p2p = gether_seg(segs_p2p, A_img.size, A_img.width, A_img.width // 2)
                final_seg_p2p = filt_small_pixel_block(final_seg_p2p)
                ious_p2p.append(seg_iou_single_img(final_seg_p2p, A_mask))
                f1_p2p.append(pixel_precision_recall_f1(final_seg_p2p, A_mask)[2])
            elif opt.name[:4] == 'DAGM':
                cls = opt.name.split('_')[1][5:]
                # only_max = True
                # if cls in ('1', '6', '10'):
                #     if cls in ('6', '10'):
                #         thresh_hold = 0
                #         only_max = False
                #     kernel = (2, 2)
                # else:
                #     kernel = (3, 3)
                # A_img_tensor = transform(A_img).unsqueeze(0).to(device)
                # A_img_repair = R_p2p(A_img_tensor)
                # A_img_tensor_numpy = A_img_tensor.squeeze().cpu().numpy()
                # A_img_repair_numpy = A_img_repair.squeeze().cpu().numpy()
                # final_seg_p2p = extract_diff(A_img_tensor_numpy, A_img_repair_numpy, thresh_hold, kernel, only_max)
                # ious_p2p.append(seg_iou_single_img(final_seg_p2p, A_mask))
                # f1_p2p.append(pixel_precision_recall_f1(final_seg_p2p, A_mask)[2])
                thresh_hold = opt.threshold
                A_img_tensor = transform(A_img).unsqueeze(0).to(device)
                A_img_repair = R_p2p(A_img_tensor)
                diff = torch.abs((A_img_repair / 2 + 0.5) * 255 - (A_img_tensor / 2 + 0.5) * 255)
                diff[diff > thresh_hold] = 255
                diff[diff <= thresh_hold] = 0
                final_seg_p2p = diff.squeeze().cpu().numpy().astype(np.uint8)
                final_seg_p2p = thresh_combine_open_close(final_seg_p2p)
                ious_p2p.append(seg_iou_single_img(final_seg_p2p, A_mask))
                precision, recall, f1 = pixel_precision_recall_f1(final_seg_p2p, A_mask)
                pre_p2p.append(precision)
                rec_p2p.append(recall)
                f1_p2p.append(f1)
            else:
                break
            # time_p2p.append(time.time() - now)
            # if np.max(final_seg_p2p) > 0:
            #     pred_p2p.append(1)
            # else:
            #     pred_p2p.append(0)

        IOU = sum(ious_p2p) / len(ious_p2p)
        Pre = sum(pre_p2p) / len(pre_p2p)
        Rec = sum(rec_p2p) / len(rec_p2p)
        F1 = sum(f1_p2p) / len(f1_p2p)
        print('average IOU CycleGAN+Pix2Pix:', IOU)
        print('average Pre CycleGAN+Pix2Pix:', Pre)
        print('average Rec CycleGAN+Pix2Pix:', Rec)
        print('average F1 Score CycleGAN+Pix2Pix:', F1)
        # print('average time CycleGAN+Pix2Pix:', sum(time_p2p) / len(time_p2p))
        #
        # print('img-level ACC CycleGAN+Pix2Pix: ', accuracy_score(real, pred_p2p))
        # print('img-level F1 Score CycleGAN+Pix2Pix: ', f1_score(real, pred_p2p))
        # print('img-level confusion matrix CycleGAN+Pix2Pix:')
        # print(confusion_matrix(real, pred_p2p))
    opt.no_flip = temp_no_flip
    return IOU, Pre, Rec, F1

def thresh_combine_open_close(th_img, k=5, f=False, onlymax=False):
    img_diff_open = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    img_diff_close = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_diff_close)
    filted = np.zeros_like(img_diff_close, dtype=np.uint8)
    remain_region = (img_diff_open / 255) * labels
    remain = set()
    for i in remain_region.flat:
        remain.add(i)
    remain.remove(0)
    for r in remain:
        filted[labels == r] = 255
    # **************
    if f:
        filted = cv2.morphologyEx(filted, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if onlymax:
        filted = extract_biggest_connected_component(filted)
    return filted

def eval_fid_when_train_cyclegan(opt, R_cyc):
    temp_no_flip = opt.no_flip
    temp_four_rotate = opt.four_rotate
    opt.no_flip = True
    opt.four_rotate = False
    transform = get_transform(opt, grayscale=opt.input_nc == 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_to_generate = 128
    real_A_path = os.path.join(opt.dataroot, "trainA")
    real_B_train_path = os.path.join(opt.dataroot, "trainB")
    real_B_test_path = os.path.join(opt.dataroot, "testB")
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
        fids_1 = cal_fids(fake_B_path, real_B_test_path)
        # fids_2 = cal_fids(fake_B_path, real_B_train_path)
        print('FID:', fids_1)
        # print('FID:', fids_2)
    opt.no_flip = temp_no_flip
    opt.four_rotate = temp_four_rotate
    return fids_1  # , fids_2

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

