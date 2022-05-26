# USAGE
# python remove_alpha.py --img_dir data/morphology/samples --ext png

import glob
import os
import sys
import argparse
import cv2
import numpy as np


def rgb_only(img_path):
    img = cv2.imread(img_path)
    print(img.shape)
    im = img[:,:,:3]
    cv2.imwrite(img_path, im)


def correct_alphas(path, ext):
    for img_path in glob.glob(path + '/*.' + ext):
        img = cv2.imread(img_path)
        print(img.shape[2])
        if img.shape[2] > 3:
            rgb_only(img_path)
            print(img_path)
    print("Successfully removed alpha channels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correct xml files."
    )
    parser.add_argument("--img_dir", help="Directory path to images.", type=str)
    parser.add_argument("--ext", help="Images extension.", type=str)
    args = parser.parse_args()
    correct_alphas(args.img_dir, args.ext)
