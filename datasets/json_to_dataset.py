import argparse
import base64
import json
import os
import os.path as osp
import glob

from labelme.logger import logger
from labelme import utils


def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    json_dir = args.json_dir
    if args.out is None:
        out_dir = osp.join(osp.dirname(json_dir), "mask")
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    label_name_to_value = {"_background_": 0}
    json_files = glob.glob(osp.join(json_dir, "*.json"))

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

        utils.lblsave(osp.join(out_dir, osp.basename(json_file).replace("json", "png")), lbl)

        logger.info("Saved mask of {} to: {}".format(osp.basename(json_file), out_dir))

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        for lbl_name in label_names:
            f.write(lbl_name + "\n")

    logger.info("Saved label info to: {}".format(out_dir))


if __name__ == "__main__":
    main()
