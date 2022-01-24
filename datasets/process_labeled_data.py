import argparse
import base64
import json
import os
import glob
import cv2

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

def extract_class4_for_generate_defect_gan():
    root = os.path.join(labeled_DAGM_path, 'Class4_relabel')
    content = ('Train', 'Test')
    for t in content:
        out_dir = f'DAGM_Class4_filted/{t}'
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", default='E:\DAGM')
    args = parser.parse_args()
    labeled_DAGM_path = args.path
    extract_class4_for_generate_defect_gan()