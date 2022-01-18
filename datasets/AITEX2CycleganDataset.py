import os
import shutil
import random
import cv2
import numpy as np


def split_cut_AITEX(width=256, stride=128, train_ratio=0.6):
    orin_data_path = 'C:/Users/DeepLearning/Desktop/AITEX'
    output_path = 'AITEX'
    trainA_path = os.path.join(output_path, 'trainA')
    trainB_path = os.path.join(output_path, 'trainB')
    testA_path = os.path.join(output_path, 'testA')
    testB_path = os.path.join(output_path, 'testB')

    defect_data_path = os.path.join(orin_data_path, 'Defect_images')
    mask_data_path = os.path.join(orin_data_path, 'Mask_images')

    def combine_mask_1_and_2(m1, m2):
        mask_path_1 = os.path.join(mask_data_path, m1)
        mask_path_2 = os.path.join(mask_data_path, m2)
        mask1 = cv2.imread(mask_path_1)
        mask2 = cv2.imread(mask_path_2)
        if mask1 is None or mask2 is None:
            return
        mask_combine = mask1 + mask2
        cv2.imwrite(os.path.join(mask_data_path, m1[:-5] + '.png'), mask_combine)
        os.remove(mask_path_1)
        os.remove(mask_path_2)
    # 原数据集中有一张图对应两个mask，将其合并
    combine_mask_1_and_2('0097_030_03_mask1.png', '0097_030_03_mask2.png')
    combine_mask_1_and_2('0044_019_04_mask1.png', '0044_019_04_mask2.png')

    items = os.listdir(defect_data_path)
    os.makedirs(trainA_path)
    os.makedirs(trainB_path)
    os.makedirs(testA_path)
    os.makedirs(testB_path)

    random.shuffle(items)
    train_num = int(len(items) * train_ratio)

    for i, item in enumerate(items):
        if i < train_num:
            outA = trainA_path
            outB = trainB_path
        else:
            outA = testA_path
            outB = testB_path

        png = cv2.imread(os.path.join(defect_data_path, item))
        mask = cv2.imread(os.path.join(mask_data_path, item[:-4] + '_mask.png'))
        if mask is None:
            continue
        png, mask = no_blank(png, mask)
        now = 0
        for j in range((mask.shape[1] - width) // stride + 1):
            png_cut = png[:, now: now + width, :]
            mask_cut = mask[:, now: now + width, :]
            if np.max(mask_cut) == 0:
                cv2.imwrite(os.path.join(outA, item[:-4] + '_' + str(j) + '.png'), png_cut)
            else:
                cv2.imwrite(os.path.join(outB, item[:-4] + '_' + str(j) + '.png'), png_cut)
            now += stride
        else:
            if (mask.shape[1] - width) % stride != 0:
                mask_cut = mask[:, mask.shape[1] - width:, :]
                png_cut = png[:, mask.shape[1] - width:, :]
                if np.max(mask_cut) == 0:
                    cv2.imwrite(os.path.join(outA, item[:-4] + '_' + str(j+1) + '.png'), png_cut)
                else:
                    cv2.imwrite(os.path.join(outB, item[:-4] + '_' + str(j+1) + '.png'), png_cut)


def no_blank(img_cv2, mask_cv2):
    norm = img_cv2.shape[0] * img_cv2.shape[2] * 255
    img_xy_sum = np.sum(img_cv2, axis=-1)
    img_x_sum = np.sum(img_xy_sum, axis=0) / norm
    start = 0
    while img_x_sum[start] > 0.9:
        start += 1
    return img_cv2[:, start:, :], mask_cv2[:, start:, :]

split_cut_AITEX()