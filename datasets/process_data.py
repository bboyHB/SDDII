import argparse
import base64
import json
import os
import glob
import cv2
import shutil
import numpy as np

from labelme.logger import logger
from labelme import utils
from skimage.feature import local_binary_pattern, hog


def extract_mask_from_label(num, same_dir=True):
    root = os.path.join(labeled_DAGM_path, f'Class{num}_relabel')
    content = ('Train', 'Test')
    for t in content:
        out_dir = f'DAGM_Class{num}_mask/{t}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        t_path = os.path.join(root, t)
        # file_names = os.listdir(t_path)
        # png_file_names = [x for x in file_names if x.lower().endswith('png')]
        # json_file_names = [x for x in file_names if x.lower().endswith('json')]
        # assert len(png_file_names) == len(json_file_names), 'png, json的文件数不一致'
        label_name_to_value = {"_background_": 0}
        json_files = glob.glob(os.path.join(t_path, "*.json"))
        for json_file in json_files:
            data = json.load(open(json_file))
            imageData = data.get("imageData")

            if not imageData:
                # if json files are at the same dir with PNGs, same_dir=True
                if same_dir:
                    imagePath = json_file.replace("json", "png")
                else:
                    imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            img = utils.img_b64_to_arr(imageData)

            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl, _ = utils.shapes_to_label(
                img.shape, data["shapes"], label_name_to_value
            )

            # utils.lblsave(os.path.join(out_dir, os.path.basename(json_file).replace("json", "png")), lbl)
            img_white = lbl * 255
            cv2.imwrite(os.path.join(out_dir, os.path.basename(json_file).replace("json", "png")), img_white)

            logger.info("Saved mask of {} to: {}".format(os.path.basename(json_file), out_dir))

        # label_names = [None] * (max(label_name_to_value.values()) + 1)
        # for name, value in label_name_to_value.items():
        #     label_names[value] = name
        # with open(os.path.join(out_dir, "label_names.txt"), "w") as f:
        #     for lbl_name in label_names:
        #         f.write(lbl_name + "\n")

        logger.info("Saved label info to: {}".format(out_dir))

def extract_mask_from_label_and_output_seg_dir(i, same_dir=True):
    root = os.path.join(labeled_DAGM_path, f'Class{i}_relabel')
    content = ('Train', 'Test')
    for t in content:
        out_dir = f'DAGM_Class{i}_seg/{t.lower()}'
        out_img_dir = os.path.join(out_dir, 'img')
        out_mask_dir = os.path.join(out_dir, 'mask')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        if not os.path.exists(out_mask_dir):
            os.makedirs(out_mask_dir)
        t_path = os.path.join(root, t)
        # file_names = os.listdir(t_path)
        # png_file_names = [x for x in file_names if x.lower().endswith('png')]
        # json_file_names = [x for x in file_names if x.lower().endswith('json')]
        # assert len(png_file_names) == len(json_file_names), 'png, json的文件数不一致'
        label_name_to_value = {"_background_": 0}
        json_files = glob.glob(os.path.join(t_path, "*.json"))
        for json_file in json_files:
            data = json.load(open(json_file))
            imageData = data.get("imageData")

            if not imageData:
                # if json files are at the same dir with PNGs, same_dir=True
                if same_dir:
                    imagePath = json_file.replace("json", "PNG")
                else:
                    imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            img = utils.img_b64_to_arr(imageData)

            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl, _ = utils.shapes_to_label(
                img.shape, data["shapes"], label_name_to_value
            )

            img_bin = lbl * 255
            cv2.imwrite(os.path.join(out_mask_dir, os.path.basename(json_file).replace("json", "png")), img_bin)
            cv2.imwrite(os.path.join(out_img_dir, os.path.basename(json_file).replace("json", "png")), img)
            logger.info("Saved mask of {} to: {}".format(os.path.basename(json_file), out_dir))

        # label_names = [None] * (max(label_name_to_value.values()) + 1)
        # for name, value in label_name_to_value.items():
        #     label_names[value] = name
        # with open(os.path.join(out_dir, "label_names.txt"), "w") as f:
        #     for lbl_name in label_names:
        #         f.write(lbl_name + "\n")

        logger.info("Saved label info to: {}".format(out_dir))

def extract_defects_for_generate_defect_gan():
    root = os.path.join(labeled_DAGM_path, 'Class4_relabel')
    content = ('Train', 'Test')
    for t in content:
        out_dir = f'DAGM_Class4_filted/{t.lower()}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        t_path = os.path.join(root, t)
        label_name_to_value = {"_background_": 0}
        json_files = glob.glob(os.path.join(t_path, "*.json"))
        for json_file in json_files:
            data = json.load(open(json_file))
            imageData = data.get("imageData")

            if not imageData:
                imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            img = utils.img_b64_to_arr(imageData)

            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl, _ = utils.shapes_to_label(
                img.shape, data["shapes"], label_name_to_value
            )

            img_filted = img * lbl

            cv2.imwrite(os.path.join(out_dir, os.path.basename(json_file).replace("json", "png")), img_filted)

            logger.info("Saved mask of {} to: {}".format(os.path.basename(json_file), out_dir))

        logger.info("Saved label info to: {}".format(out_dir))


def save_defective_samples():
    root = os.path.join(labeled_DAGM_path, 'Class4')
    content = ('Train', 'Test')
    new_path = 'DAGM_Class4_no_defect'
    for c in content:
        new_sub_dir = os.path.join(new_path, c.lower())
        if not os.path.exists(new_sub_dir):
            os.makedirs(new_sub_dir)
        defectives = [x[:4]+x[-4:] for x in os.listdir(os.path.join(root, c, 'Label')) if x.endswith('PNG')]
        names = [x for x in os.listdir(os.path.join(root, c)) if x.endswith('PNG')]
        non_defectives = [x for x in names if x not in defectives]
        for name in non_defectives:
            shutil.copyfile(os.path.join(root, c, name), os.path.join(new_sub_dir, name))

def extract_biggest_connected_component(img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    if num_labels == 1:
        return img
    index = np.argmax(stats[1:, 4]) + 1
    filted_img = img * (labels == index)
    return filted_img

def cal_fids(path1, path2):
    import pytorch_fid.fid_score as fid
    import torch
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    num_avail_cpus = len(os.sched_getaffinity(0))  # os.cpu_count()
    num_workers = min(num_avail_cpus, 8)
    batch_size = 4
    output = {}
    paths = (path1, path2)
    for dims in (64, 192):  # , 768, 2048
        fid_value = fid.calculate_fid_given_paths(paths,
                                                  batch_size,
                                                  device,
                                                  dims,
                                                  num_workers)
        output[str(dims)] = fid_value
    return output

def extract_diff(img1, img2, thresh_hold, first_kernel=(2, 2), second_kernel=(9, 9), third_kernel=(5, 5), only_max=True):
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数

    des1_lbp = local_binary_pattern(img1, n_points, radius)
    _, des1_hog = hog(img1, visualize=True)
    des2_lbp = local_binary_pattern(img2, n_points, radius)
    _, des2_hog = hog(img2, visualize=True)
    des_lbp_diff = np.abs(des1_lbp - des2_lbp)

    lbp_diff_open = cv2.morphologyEx(des_lbp_diff, cv2.MORPH_OPEN, np.ones(first_kernel, np.uint8))
    lbp_diff_close = cv2.morphologyEx(lbp_diff_open, cv2.MORPH_CLOSE, np.ones(second_kernel, np.uint8))
    lbp_diff_close_open = cv2.morphologyEx(lbp_diff_close, cv2.MORPH_OPEN, np.ones(third_kernel, np.uint8))

    uint8_lbp_diff_close_open = np.array(lbp_diff_close_open, dtype=np.uint8)
    ret1, th_img1 = cv2.threshold(uint8_lbp_diff_close_open, thresh_hold, 255, cv2.THRESH_BINARY)
    if only_max:
        th_img1 = extract_biggest_connected_component(th_img1)
    return th_img1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", default='E:\DAGM')
    args = parser.parse_args()
    labeled_DAGM_path = args.path
    # extract_defects_for_generate_defect_gan()
    # save_defective_samples()
    # extract_mask_from_label(3)
    for i in range(1, 11):
        extract_mask_from_label_and_output_seg_dir(i)
