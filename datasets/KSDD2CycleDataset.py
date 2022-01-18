import os
import shutil
import random
import cv2
import numpy as np


def split_no_cut():
    KSDD_path = 'C:/Users/DeepLearning/Desktop/KolektorSDD'
    output_path = 'KSDD'
    trainA_path = os.path.join(output_path, 'trainA')
    trainB_path = os.path.join(output_path, 'trainB')
    testA_path = os.path.join(output_path, 'testA')
    testB_path = os.path.join(output_path, 'testB')
    items = os.listdir(KSDD_path)
    os.makedirs(trainA_path)
    os.makedirs(trainB_path)
    os.makedirs(testA_path)
    os.makedirs(testB_path)

    random.shuffle(items)

    for i, item in enumerate(items):
        if i < 25:
            outA = trainA_path
            outB = trainB_path
        else:
            outA = testA_path
            outB = testB_path
        for name in [x for x in os.listdir(os.path.join(KSDD_path, item)) if x.endswith('bmp')]:
            bmp = cv2.imread(os.path.join(KSDD_path, item, name))
            if np.max(bmp) == 0:
                shutil.copy(os.path.join(KSDD_path, item, name.split('_')[0] + '.jpg'),
                            os.path.join(outA, item + '_' + name.split('_')[0] + '.jpg'))
            elif np.max(bmp) == 255:
                shutil.copy(os.path.join(KSDD_path, item, name.split('_')[0] + '.jpg'),
                            os.path.join(outB, item + '_' + name.split('_')[0] + '.jpg'))
            else:
                print(os.path.join(KSDD_path, item, name), np.max(bmp))


def split_cut_KSDD(orin_data_path, height=500, stride=250, train_ratio=0.6):
    # orin_data_path = 'C:/Users/DeepLearning/Desktop/KolektorSDD'
    output_path = 'KSDD'

    cycle_path = output_path + '_cycle'
    trainA_path = os.path.join(cycle_path, 'trainA')
    trainB_path = os.path.join(cycle_path, 'trainB')
    testA_path = os.path.join(cycle_path, 'testA')
    testB_path = os.path.join(cycle_path, 'testB')

    orin_path = output_path + '_origin'
    orin_train_img_path = os.path.join(orin_path, 'train', 'img')
    orin_train_mask_path = os.path.join(orin_path, 'train', 'mask')
    orin_test_img_path = os.path.join(orin_path, 'test', 'img')
    orin_test_mask_path = os.path.join(orin_path, 'test', 'mask')

    items = os.listdir(orin_data_path)
    os.makedirs(trainA_path)
    os.makedirs(trainB_path)
    os.makedirs(testA_path)
    os.makedirs(testB_path)
    os.makedirs(orin_train_img_path)
    os.makedirs(orin_train_mask_path)
    os.makedirs(orin_test_img_path)
    os.makedirs(orin_test_mask_path)

    random.shuffle(items)
    train_num = int(len(items)*train_ratio)

    for i, item in enumerate(items):
        if i < train_num:
            outA = trainA_path
            outB = trainB_path
            out_orin_img = orin_train_img_path
            out_orin_mask = orin_train_mask_path
        else:
            outA = testA_path
            outB = testB_path
            out_orin_img = orin_test_img_path
            out_orin_mask = orin_test_mask_path
        for name in [x for x in os.listdir(os.path.join(orin_data_path, item)) if x.endswith('bmp')]:
            bmp = cv2.imread(os.path.join(orin_data_path, item, name))
            jpg = cv2.imread(os.path.join(orin_data_path, item, name.split('_')[0] + '.jpg'))
            cv2.imwrite(os.path.join(out_orin_mask, item + '_' + name.split('_')[0] + '.jpg'), bmp)
            cv2.imwrite(os.path.join(out_orin_img, item + '_' + name.split('_')[0] + '.jpg'), jpg)
            now = 0
            for j in range((bmp.shape[0]-height)//stride+1):
                bmp_cut = bmp[now: now+height, :, :]
                jpg_cut = jpg[now: now+height, :, :]
                if np.max(bmp_cut) == 0:
                    cv2.imwrite(os.path.join(outA, item + '_' + name.split('_')[0] + '_' + str(j) + '.jpg'), jpg_cut)
                else:
                    cv2.imwrite(os.path.join(outB, item + '_' + name.split('_')[0] + '_' + str(j) + '.jpg'), jpg_cut)
                now += stride
            else:
                if (bmp.shape[0]-height) % stride != 0:
                    bmp_cut = bmp[bmp.shape[0]-height:, :, :]
                    jpg_cut = jpg[bmp.shape[0]-height:, :, :]
                    if np.max(bmp_cut) == 0:
                        cv2.imwrite(os.path.join(outA, item + '_' + name.split('_')[0] + '_' + str(j+1) + '.jpg'), jpg_cut)
                    else:
                        cv2.imwrite(os.path.join(outB, item + '_' + name.split('_')[0] + '_' + str(j+1) + '.jpg'), jpg_cut)


def split_supervised_seg_data(output_path, orin_path):
    orin_train_path = os.path.join(orin_path, 'train')
    orin_train_img_path = os.path.join(orin_train_path, 'img')
    orin_train_mask_path = os.path.join(orin_train_path, 'mask')
    orin_test_path = os.path.join(orin_path, 'test')
    orin_test_img_path = os.path.join(orin_test_path, 'img')
    orin_test_mask_path = os.path.join(orin_test_path, 'mask')
    for t in ['train', 'test']:
        out_t_img = os.path.join(output_path, t, 'img')
        out_t_mask = os.path.join(output_path, t, 'mask')
        os.makedirs(out_t_img)
        os.makedirs(out_t_mask)
        orin_t_img_path = eval('orin_' + t + '_img_path')
        orin_t_mask_path = eval('orin_' + t + '_mask_path')
        names = os.listdir(orin_t_img_path)
        for name in names:
            img = cv2.imread(os.path.join(orin_t_img_path, name))
            mask = cv2.imread(os.path.join(orin_t_mask_path, name))
            height = mask.shape[0]
            width = mask.shape[1]
            h2cut = width
            stride = h2cut // 2
            now = 0
            for j in range((height - h2cut) // stride + 1):
                img_cut = img[now: now + h2cut, :, :]
                mask_cut = mask[now: now + h2cut, :, :]
                cv2.imwrite(os.path.join(out_t_img, name[:-4] + '_' + str(j) + '.jpg'), img_cut)
                cv2.imwrite(os.path.join(out_t_mask, name[:-4] + '_' + str(j) + '.jpg'), mask_cut)
                now += stride
            else:
                if (height - h2cut) % stride != 0:
                    img_cut = img[height - h2cut:, :, :]
                    mask_cut = mask[height - h2cut:, :, :]
                    cv2.imwrite(os.path.join(out_t_img, name[:-4] + '_' + str(j) + '.jpg'), img_cut)
                    cv2.imwrite(os.path.join(out_t_mask, name[:-4] + '_' + str(j) + '.jpg'), mask_cut)


if __name__ == '__main__':
    split_cut_KSDD('C:/Users/DeepLearning/Desktop/KolektorSDD')
    split_supervised_seg_data('KSDD_seg', 'KSDD_origin')