import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_biggest_connected_component(img):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    index = np.argmax(stats[1:, 4]) + 1
    filted_img = img * (labels == index)
    return filted_img

def defect_num_location_aspect_area_distribution():
    root1 = "RSDDs1_seg/train/mask"
    root2 = "RSDDs2_seg/train/mask"
    for r in (root1, root2):
        img_names = os.listdir(r)
        side_length = 160 if r == root1 else 55
        num_defects = [0] * 30
        centroids_ratios = []
        aspect_ratios = []
        areas = []
        area_width_height_product_ratios = []
        defect_hist_sum = np.zeros((256, 1), dtype=np.float32)
        defect_free_hist_sum = np.zeros((256, 1), dtype=np.float32)
        for img_name in img_names:
            bin_img_path = os.path.join(r, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            # if np.max((img > 0) * (img < 255)):
            #     cv2.imshow('0', img)
            #     cv2.waitKey(0)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            num_defects[num_labels - 1] += 1
            # 无瑕疵跳过
            if num_labels == 1:
                defect_free_img_path = bin_img_path.replace("mask", "img")
                defect_free_img = cv2.imread(defect_free_img_path, cv2.IMREAD_GRAYSCALE)
                defect_free_hist = cv2.calcHist([defect_free_img], [0], None, [256], [0, 256])
                defect_free_hist_sum += defect_free_hist
                continue
            defect_img_path = bin_img_path.replace("mask", "img")
            defect_img = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
            defect_hist = cv2.calcHist([defect_img], [0], bin_img, [256], [0, 256])
            defect_hist_sum += defect_hist
            centroids_ratio = centroids[1:] / side_length
            centroids_ratios.extend(centroids_ratio)
            width_height_ratio = stats[1:, 2] / stats[1:, 3]
            aspect_ratios.extend(width_height_ratio)
            areas.extend(stats[1:, 4])
            width_height_product = stats[1:, 2] * stats[1:, 3]
            area_width_height_product_ratio = stats[1:, 4] / width_height_product
            area_width_height_product_ratios.extend(area_width_height_product_ratio)
        plt.plot(defect_free_hist_sum)
        plt.show()
        plt.plot(defect_hist_sum)
        plt.show()
        print(num_defects)
        num_defects_show = num_defects[1:max_index_none_zero(num_defects)+1]
        x_show = [str(x) for x in range(1, len(num_defects_show)+1)]
        plt.bar(x_show, num_defects_show)
        for a, b in zip(x_show, num_defects_show):
            plt.text(a, b, b, ha='center', va='bottom')
        plt.show()
        centroids_ratios = [list(x) for x in centroids_ratios]
        print(centroids_ratios)
        plt.scatter([x[0] for x in centroids_ratios], [x[1] for x in centroids_ratios])
        plt.show()
        print(aspect_ratios)
        plt.plot(range(len(aspect_ratios)), sorted(aspect_ratios))
        plt.show()
        print(areas)
        plt.plot(range(len(areas)), sorted(areas))
        plt.show()
        print(area_width_height_product_ratios)
        plt.plot(range(len(area_width_height_product_ratios)), sorted(area_width_height_product_ratios))
        plt.show()
        pass


def max_index_none_zero(array):
    for i in range(len(array)-1, -1, -1):
        if array[i] != 0:
            return i
    else:
        return -1

if __name__ == '__main__':
    defect_num_location_aspect_area_distribution()
