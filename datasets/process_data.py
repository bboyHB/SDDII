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


def extract_mask_from_label():
    root = os.path.join(labeled_DAGM_path, 'Class4_relabel')
    content = ('Train', 'Test')
    for t in content:
        out_dir = f'DAGM_Class4_mask/{t}'
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

            utils.lblsave(os.path.join(out_dir, os.path.basename(json_file).replace("json", "png")), lbl)

            logger.info("Saved mask of {} to: {}".format(os.path.basename(json_file), out_dir))

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        with open(os.path.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in label_names:
                f.write(lbl_name + "\n")

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
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    index = np.argmax(stats[1:, 4]) + 1
    filted_img = img * (labels == index)
    return filted_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", default='E:\DAGM')
    args = parser.parse_args()
    labeled_DAGM_path = args.path
    extract_defects_for_generate_defect_gan()
    save_defective_samples()
