import os

import cv2
import numpy as np
import matplotlib.font_manager
from matplotlib import pyplot as plt

# 中文
# matplotlib.font_manager._rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 默认字体大小是10
plt.rcParams.update({'font.size': 14})
legend_line_width = 8
# 解决刻度符号乱码https://www.pythonheidong.com/blog/article/763890/93d7e6ca059b94c214cf/
# site-package/matplotlib/mathtext.py 826行 fontname = 'it'

def extract_biggest_connected_component(img):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    index = np.argmax(stats[1:, 4]) + 1
    filted_img = img * (labels == index)
    return filted_img

def defect_num_location_aspect_area_distribution_RSDDs():
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
            # 展示瑕疵最多的情况
            # if num_labels == 5:
            #     cv2.imshow('1', defect_img)
            #     cv2.waitKey(0)
            #     cv2.imshow('1', bin_img)
            #     cv2.waitKey(0)
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
        plt.savefig(f'RSDDs{r[5]}_defect_free_hist_sum.pdf', format='pdf', dpi=600)
        plt.show()
        plt.plot(defect_hist_sum)
        plt.savefig(f'RSDDs{r[5]}_defect_hist_sum.pdf', format='pdf', dpi=600)
        plt.show()
        print(num_defects)
        num_defects_show = num_defects[1:max_index_none_zero(num_defects)+1]
        x_show = [str(x) for x in range(1, len(num_defects_show)+1)]
        plt.bar(x_show, num_defects_show)
        for a, b in zip(x_show, num_defects_show):
            plt.text(a, b, b, ha='center', va='bottom')
        plt.savefig(f'RSDDs{r[5]}_num_defects.pdf', format='pdf', dpi=600)
        plt.show()
        centroids_ratios = [list(x) for x in centroids_ratios]
        print(centroids_ratios)
        plt.scatter([x[0] for x in centroids_ratios], [x[1] for x in centroids_ratios])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(f'RSDDs{r[5]}_centroids_ratios.pdf', format='pdf', dpi=600)
        plt.show()
        print(aspect_ratios)
        plt.plot(range(len(aspect_ratios)), sorted(aspect_ratios))
        plt.savefig(f'RSDDs{r[5]}_aspect_ratios.pdf', format='pdf', dpi=600)
        plt.show()
        print(areas)
        plt.plot(range(len(areas)), sorted(areas))
        plt.savefig(f'RSDDs{r[5]}_areas.pdf', format='pdf', dpi=600)
        plt.show()
        print(area_width_height_product_ratios)
        plt.plot(range(len(area_width_height_product_ratios)), sorted(area_width_height_product_ratios))
        plt.savefig(f'RSDDs{r[5]}_area_width_height_product_ratios.pdf', format='pdf', dpi=600)
        plt.show()
        pass

def centroids_ratios_distribution_RSDDs():
    root1 = "RSDDs1_seg/train/mask"
    root2 = "RSDDs2_seg/train/mask"
    for r in (root1, root2):
        img_names = os.listdir(r)
        side_length = 160 if r == root1 else 55
        centroids_ratios = []
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
            # 无瑕疵跳过
            if num_labels == 1:
                continue
            centroids_ratio = centroids[1:] / side_length
            centroids_ratios.extend(centroids_ratio)
        centroids_ratios = [list(x) for x in centroids_ratios]
        print(centroids_ratios)
        plt.scatter([x[0] for x in centroids_ratios], [x[1] for x in centroids_ratios])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'RSDDs{r[5]}_centroids_ratios.pdf', format='pdf', dpi=600)
        plt.show()
        pass

def areas_distribution_RSDDs():
    root1 = "RSDDs1_seg/train/mask"
    root2 = "RSDDs2_seg/train/mask"
    names = ('I型RSDDs数据集', 'II型RSDDs数据集')

    for i, r in enumerate((root1, root2)):
        img_names = os.listdir(r)
        side_length = 160 if r == root1 else 55
        areas = []
        for img_name in img_names:
            bin_img_path = os.path.join(r, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            # 无瑕疵跳过
            if num_labels == 1:
                continue
            areas.extend(stats[1:, 4] / (side_length * side_length))
        areas = sorted(areas)
        plt.plot(areas, label=names[i])
        plt.yscale('log')
        plt.tight_layout()
    leg = plt.legend(loc=4)
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.savefig(f'RSDDs_defect_areas.pdf', format='pdf', dpi=600)
    plt.show()

def width_height_product_distribution_RSDDs():
    root1 = "RSDDs1_seg/train/mask"
    root2 = "RSDDs2_seg/train/mask"
    names = ('I型RSDDs数据集', 'II型RSDDs数据集')

    for i, r in enumerate((root1, root2)):
        img_names = os.listdir(r)
        side_length = 160 if r == root1 else 55
        areas = []
        for img_name in img_names:
            bin_img_path = os.path.join(r, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            # 无瑕疵跳过
            if num_labels == 1:
                continue
            width_height_product = stats[1:, 2] * stats[1:, 3]
            areas.extend(width_height_product / (side_length * side_length))
        areas = sorted(areas)
        plt.plot(areas, label=names[i])
        plt.yscale('log')
        plt.tight_layout()
    leg = plt.legend(loc=4)
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.savefig(f'RSDDs_defect_width_height_product.pdf', format='pdf', dpi=600)
    plt.show()

def aspect_ratios_distribution_RSDDs():
    root1 = "RSDDs1_seg/train/mask"
    root2 = "RSDDs2_seg/train/mask"
    names = ('I型RSDDs数据集', 'II型RSDDs数据集')

    for i, r in enumerate((root1, root2)):
        img_names = os.listdir(r)
        aspect_ratios = []
        for img_name in img_names:
            bin_img_path = os.path.join(r, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            # 无瑕疵跳过
            if num_labels == 1:
                continue
            width_height_ratio = stats[1:, 2] / stats[1:, 3]
            aspect_ratios.extend(width_height_ratio)
        aspect_ratios = sorted(aspect_ratios)
        plt.plot(aspect_ratios, label=names[i])
        plt.yscale('log')
        plt.tight_layout()
    leg = plt.legend(loc=4)
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.savefig(f'RSDDs_defect_aspect_ratios.pdf', format='pdf', dpi=600)
    plt.show()

def aspect_ratios_distribution_DAGM():
    for i in range(1, 11):
        root = f"DAGM_Class{i}_seg/train/mask"
        img_names = os.listdir(root)
        aspect_ratios = []
        for img_name in img_names:
            bin_img_path = os.path.join(root, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            defect_area = np.sum(bin_img) / 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            width_height_ratio = stats[1:, 2] / stats[1:, 3]
            aspect_ratios.extend(width_height_ratio)
        aspect_ratios = sorted(aspect_ratios)
        plt.plot(aspect_ratios, label=f'类别{i}')
        plt.yscale('log')
        plt.tight_layout()
    leg = plt.legend()
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.savefig(f'DAGM_defect_aspect_ratios.pdf', format='pdf', dpi=600)
    plt.show()

def areas_distribution_DAGM():
    for i in range(1, 11):
        root = f"DAGM_Class{i}_seg/train/mask"
        img_names = os.listdir(root)
        areas = []
        for img_name in img_names:
            bin_img_path = os.path.join(root, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            areas.extend(stats[1:, 4] / (512 * 512))
        areas = sorted(areas)
        plt.plot(areas, label=f'类别{i}')
        plt.yscale('log')
        plt.tight_layout()
    leg = plt.legend()
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.savefig(f'DAGM_defect_areas.pdf', format='pdf', dpi=600)
    plt.show()

def width_height_product_distribution_DAGM():
    for i in range(1, 11):
        root = f"DAGM_Class{i}_seg/train/mask"
        img_names = os.listdir(root)
        areas = []
        for img_name in img_names:
            bin_img_path = os.path.join(root, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            width_height_product = stats[1:, 2] * stats[1:, 3]
            areas.extend(width_height_product / (512 * 512))
        areas = sorted(areas)
        plt.plot(areas, label=f'类别{i}')
        plt.yscale('log')
        plt.tight_layout()
    leg = plt.legend()
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.savefig(f'DAGM_defect_width_height_product.pdf', format='pdf', dpi=600)
    plt.show()

def color_distribution_DAGM():
    DAGM_path = 'E:\DAGM'
    for i in range(1, 11):
        root = f"DAGM_Class{i}_seg/train/mask"
        img_names = os.listdir(root)
        defect_hist_sum = np.zeros((256, 1), dtype=np.float32)
        for img_name in img_names:
            bin_img_path = os.path.join(root, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            defect_area = np.sum(bin_img) / 255
            defect_img_path = bin_img_path.replace("mask", "img")
            defect_img = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
            defect_hist = cv2.calcHist([defect_img], [0], bin_img, [256], [0, 256])
            defect_hist_sum += defect_hist / defect_area
        defect_hist_sum /= len(img_names)
        print(np.sum(defect_hist_sum))
        plt.plot(defect_hist_sum, label=f'类别{i}')
    plt.ylim([0, 0.05])
    leg = plt.legend()
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.tight_layout()
    plt.savefig(f'DAGM_defect_hist_sum.pdf', format='pdf', dpi=600)
    plt.show()

    for i in range(1, 11):
        root2 = os.path.join(DAGM_path, f'Class{i}', 'Train')
        defect_free_hist_sum = np.zeros((256, 1), dtype=np.float32)
        pic_names = {name for name in os.listdir(root2) if name.endswith('PNG')}
        defect_names = {name[:4] + name[-4:] for name in os.listdir(os.path.join(root2, 'Label')) if name.endswith('PNG')}
        normal_names = pic_names - defect_names
        for normal_name in normal_names:
            defect_free_img_path = os.path.join(root2, normal_name)
            defect_free_img = cv2.imread(defect_free_img_path, cv2.IMREAD_GRAYSCALE)
            defect_free_hist = cv2.calcHist([defect_free_img], [0], None, [256], [0, 256])
            defect_free_hist_sum += defect_free_hist / (512 * 512)
        defect_free_hist_sum /= len(normal_names)
        print(np.sum(defect_free_hist_sum))
        plt.plot(defect_free_hist_sum, label=f'类别{i}')
    leg = plt.legend()
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.tight_layout()
    plt.savefig(f'DAGM_defect_free_hist_sum.pdf', format='pdf', dpi=600)
    plt.show()
    pass

def color_distribution_RSDDs():
    root1 = "RSDDs1_seg/train/mask"
    root2 = "RSDDs2_seg/train/mask"
    names = ('I型RSDDs数据集', 'II型RSDDs数据集')
    for i, r in enumerate((root1, root2)):
        img_names = os.listdir(r)
        defect_hist_sum = np.zeros((256, 1), dtype=np.float32)
        count_defect = 0
        for img_name in img_names:
            bin_img_path = os.path.join(r, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            defect_area = np.sum(bin_img) / 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            # 无瑕疵跳过
            if num_labels == 1:
                continue
            defect_img_path = bin_img_path.replace("mask", "img")
            defect_img = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
            defect_hist = cv2.calcHist([defect_img], [0], bin_img, [256], [0, 256])
            defect_hist_sum += defect_hist / defect_area
            count_defect += 1
        defect_hist_sum /= count_defect
        print(np.sum(defect_hist_sum))
        plt.plot(defect_hist_sum, label=names[i])
    leg = plt.legend()
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.tight_layout()
    plt.savefig(f'RSDDs_defect_hist_sum.pdf', format='pdf', dpi=600)
    plt.show()

    for i, r in enumerate((root1, root2)):
        img_names = os.listdir(r)
        side_length = 160 if r == root1 else 55
        defect_free_hist_sum = np.zeros((256, 1), dtype=np.float32)
        count_defect_free = 0
        for img_name in img_names:
            bin_img_path = os.path.join(r, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            # 无瑕疵跳过
            if num_labels == 1:
                defect_free_img_path = bin_img_path.replace("mask", "img")
                defect_free_img = cv2.imread(defect_free_img_path, cv2.IMREAD_GRAYSCALE)
                defect_free_hist = cv2.calcHist([defect_free_img], [0], None, [256], [0, 256])
                defect_free_hist_sum += defect_free_hist / (side_length**2)
                count_defect_free += 1
                continue
        defect_free_hist_sum /= count_defect_free
        print(np.sum(defect_free_hist_sum))
        plt.plot(defect_free_hist_sum, label=names[i])
    leg = plt.legend()
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=legend_line_width)
    plt.tight_layout()
    plt.savefig(f'RSDDs_defect_free_hist_sum.pdf', format='pdf', dpi=600)
    plt.show()

def defect_num_location_aspect_area_distribution_DAGM(i, no_tick=False):
    DAGM_path = 'E:\DAGM'
    root = f"DAGM_Class{i}_seg/train/mask"
    img_names = os.listdir(root)
    side_length = 512
    num_defects = [0] * 10
    centroids_ratios = []
    aspect_ratios = []
    areas = []
    area_width_height_product_ratios = []
    defect_hist_sum = np.zeros((256, 1), dtype=np.float32)
    defect_free_hist_sum = np.zeros((256, 1), dtype=np.float32)
    for img_name in img_names:
        bin_img_path = os.path.join(root, img_name)
        bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
        # 阈值二值化，去除一些小颗粒
        bin_img[bin_img < 200] = 0
        bin_img[bin_img >= 200] = 255
        # if np.max((img > 0) * (img < 255)):
        #     cv2.imshow('0', img)
        #     cv2.waitKey(0)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
        num_defects[num_labels - 1] += 1
        # DAGM不会无瑕疵跳过
        if num_labels == 1:
            defect_free_img_path = bin_img_path.replace("mask", "img")
            defect_free_img = cv2.imread(defect_free_img_path, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('1', defect_free_img)
            cv2.waitKey(0)
            continue

        defect_img_path = bin_img_path.replace("mask", "img")
        defect_img = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
        # if num_labels == 5:
        #     cv2.imshow('1', defect_img)
        #     cv2.waitKey(0)
        #     cv2.imshow('1', bin_img)
        #     cv2.waitKey(0)
        defect_hist = cv2.calcHist([defect_img], [0], bin_img, [256], [0, 256])
        defect_hist_sum += defect_hist / (512*512)
        # 归一化
        centroids_ratio = centroids[1:] / side_length
        centroids_ratios.extend(centroids_ratio)
        width_height_ratio = stats[1:, 2] / stats[1:, 3]
        aspect_ratios.extend(width_height_ratio)
        # 归一化
        areas.extend(stats[1:, 4] / (512*512))
        width_height_product = stats[1:, 2] * stats[1:, 3]
        area_width_height_product_ratio = stats[1:, 4] / width_height_product
        area_width_height_product_ratios.extend(area_width_height_product_ratio)
    root2 = os.path.join(DAGM_path, f'Class{i}', 'Train')
    pic_names = {name for name in os.listdir(root2) if name.endswith('PNG')}
    defect_names = {name[:4] + name[-4:] for name in os.listdir(os.path.join(root2, 'Label')) if name.endswith('PNG')}
    normal_names = pic_names - defect_names
    for normal_name in normal_names:
        defect_free_img_path = os.path.join(root2, normal_name)
        defect_free_img = cv2.imread(defect_free_img_path, cv2.IMREAD_GRAYSCALE)
        defect_free_hist = cv2.calcHist([defect_free_img], [0], None, [256], [0, 256])
        defect_free_hist_sum += defect_free_hist / (512*512)
    defect_free_hist_sum /= len(normal_names)
    plt.plot(defect_free_hist_sum)
    if no_tick:
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'DAGM_Class{i}_defect_free_hist_sum.pdf', format='pdf', dpi=600)
    plt.show()
    defect_hist_sum /= len(img_names)
    plt.plot(defect_hist_sum)
    if no_tick:
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'DAGM_Class{i}_defect_hist_sum.pdf', format='pdf', dpi=600)
    plt.show()
    # 瑕疵数统计，目前不需要了
    # print(num_defects)
    # num_defects_show = num_defects[1:max_index_none_zero(num_defects)+1]
    # x_show = [str(x) for x in range(1, len(num_defects_show)+1)]
    # plt.bar(x_show, num_defects_show)
    # for a, b in zip(x_show, num_defects_show):
    #     plt.text(a, b, b, ha='center', va='bottom')
    # plt.show()
    centroids_ratios = [list(x) for x in centroids_ratios]
    print(centroids_ratios)
    plt.scatter([x[0] for x in centroids_ratios], [x[1] for x in centroids_ratios])
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f'DAGM_Class{i}_centroids_ratios.pdf', format='pdf', dpi=600)
    plt.show()
    print(aspect_ratios)
    plt.plot(range(len(aspect_ratios)), sorted(aspect_ratios))
    if no_tick:
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'DAGM_Class{i}_aspect_ratios.pdf', format='pdf', dpi=600)
    plt.show()
    print(areas)
    plt.plot(range(len(areas)), sorted(areas))
    if no_tick:
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'DAGM_Class{i}_areas.pdf', format='pdf', dpi=600)
    plt.show()
    print(area_width_height_product_ratios)
    plt.plot(range(len(area_width_height_product_ratios)), sorted(area_width_height_product_ratios))
    if no_tick:
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'DAGM_Class{i}_area_width_height_product_ratios.pdf', format='pdf', dpi=600)
    plt.show()

def centroids_ratios_distribution_DAGM():
    for i in range(1, 11):
        root = f"DAGM_Class{i}_seg/train/mask"
        img_names = os.listdir(root)
        side_length = 512
        centroids_ratios = []
        for img_name in img_names:
            bin_img_path = os.path.join(root, img_name)
            bin_img = cv2.imread(bin_img_path, cv2.IMREAD_GRAYSCALE)
            # 阈值二值化，去除一些小颗粒
            bin_img[bin_img < 200] = 0
            bin_img[bin_img >= 200] = 255
            # if np.max((img > 0) * (img < 255)):
            #     cv2.imshow('0', img)
            #     cv2.waitKey(0)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            # DAGM不会无瑕疵跳过
            if num_labels == 1:
                continue
            # 归一化
            centroids_ratio = centroids[1:] / side_length
            centroids_ratios.extend(centroids_ratio)
        centroids_ratios = [list(x) for x in centroids_ratios]
        print(centroids_ratios)
        plt.scatter([x[0] for x in centroids_ratios], [x[1] for x in centroids_ratios])
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'DAGM_Class{i}_centroids_ratios.pdf', format='pdf', dpi=600)
        plt.show()

def max_index_none_zero(array):
    for i in range(len(array)-1, -1, -1):
        if array[i] != 0:
            return i
    else:
        return -1

if __name__ == '__main__':
    # color_distribution_DAGM()
    # color_distribution_RSDDs()
    # centroids_ratios_distribution_DAGM()
    # centroids_ratios_distribution_RSDDs()
    # aspect_ratios_distribution_DAGM()
    # aspect_ratios_distribution_RSDDs()
    # areas_distribution_DAGM()
    # areas_distribution_RSDDs()
    width_height_product_distribution_DAGM()
    width_height_product_distribution_RSDDs()


# 自定义xy轴的拉伸比例 https://matplotlib.org/stable/gallery/scales/custom_scale.html