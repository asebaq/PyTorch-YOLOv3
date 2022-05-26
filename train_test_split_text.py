# USAGE
# python train_test_split_text.py --img_dir vitality_one --img_ext png


import glob
import os
import numpy as np
import sys
import shutil
import argparse


def train_test_split_text(img_dir, img_ext):
    origin_dir = os.path.abspath(os.path.join("data", img_dir, "all_images", "train"))
    # Copy files
    for src in glob.iglob(os.path.join(origin_dir, "*." + img_ext)):
        # filename = src.split("\\")[-1]
        filename = src.split("/")[-1]
        dist = os.path.abspath(os.path.join("data", img_dir, "images", filename))
        shutil.copyfile(src, dist)

    origin_dir = os.path.abspath(os.path.join("data", img_dir, "all_images", "val"))
    # Copy files
    for src in glob.iglob(os.path.join(origin_dir, "*." + img_ext)):
        # filename = src.split("\\")[-1]
        filename = src.split("/")[-1]
        dist = os.path.abspath(os.path.join("data", img_dir, "samples", filename))
        shutil.copyfile(src, dist)

    origin_dir = os.path.abspath(os.path.join("data", img_dir, "all_images", "train", "yolo"))
    # Copy files
    for src in glob.iglob(os.path.join(origin_dir, "*.txt")):
        # filename = src.split("\\")[-1]
        filename = src.split("/")[-1]
        dist = os.path.abspath(os.path.join("data", img_dir, "labels", filename))
        shutil.copyfile(src, dist)

    current_dir = os.path.abspath(os.path.join("data", img_dir, "images"))
    split_pct = 1
    file_train = open(os.path.abspath(os.path.join("data", img_dir, "train.txt")), "w")
    file_val = open(os.path.abspath(os.path.join("data", img_dir, "valid.txt")), "w")
    counter = 1
    index_test = round(100 / split_pct)

    for fullpath in glob.iglob(os.path.join(current_dir, "*." + img_ext)):
        title, ext = os.path.splitext(os.path.basename(fullpath))
        if counter == index_test:
            counter = 1
            # file_val.write(current_dir + "\\" + title + "." + img_ext + "\n")
            file_val.write(current_dir + "/" + title + "." + img_ext + "\n")
        else:
            # file_train.write(current_dir + "\\" + title + "." + img_ext + "\n")
            file_train.write(current_dir + "/" + title + "." + img_ext + "\n")
            counter = counter + 1

    file_train.close()
    file_val.close()


def main():
    # Initiate argument parser
    ap = argparse.ArgumentParser(description="Sample dataset split")
    ap.add_argument("-i", "--img_dir",
                    help="Path to the folder where the images are stored",
                    default=os.getcwd(),
                    type=str)
    ap.add_argument("-e", "--img_ext",
                    help="Image extension.",
                    default="png",
                    type=str)

    args = vars(ap.parse_args())
    train_test_split_text(args["img_dir"], args["img_ext"])


if __name__ == '__main__':
    main()
