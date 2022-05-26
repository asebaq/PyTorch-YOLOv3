# USAGE
# python train_test_split.py --img_dir './morph_anno/' --split_pct 20 --img_ext jpg

import glob
import os
import sys
import argparse
from tqdm import tqdm
from shutil import copyfile
import numpy as np


def train_test_split(img_dir, split_pct, img_ext):
    train_dir = os.path.join(img_dir, 'train')
    val_dir = os.path.join(img_dir, 'val')
    test_dir = os.path.join(img_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = sorted(glob.iglob(os.path.join(img_dir, '*.' + img_ext)))
    split_pct /= 2
    val_size = split_pct / 100
    num_data = len(images)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    split = int(np.floor(val_size * num_data))
    train_idx, val_idx, test_idx = indices[split * 2:], indices[:split], indices[split:split * 2]

    print(len(train_idx), len(val_idx), len(test_idx))
    for idx in tqdm(train_idx):
        img_fullpath = images[idx]
        img_title, img_ext = os.path.splitext(os.path.basename(img_fullpath))
        os.replace(img_fullpath, train_dir + '/' + img_title + img_ext)
        os.replace(img_fullpath.split('.')[0] + '.xml', train_dir + '/' + img_title + '.xml')

    for idx in tqdm(val_idx):
        img_fullpath = images[idx]
        img_title, img_ext = os.path.splitext(os.path.basename(img_fullpath))
        os.replace(img_fullpath, val_dir + '/' + img_title + img_ext)
        os.replace(img_fullpath.split('.')[0] + '.xml', val_dir + '/' + img_title + '.xml')

    for idx in tqdm(test_idx):
        img_fullpath = images[idx]
        img_title, img_ext = os.path.splitext(os.path.basename(img_fullpath))
        os.replace(img_fullpath, test_dir + '/' + img_title + img_ext)
        os.replace(img_fullpath.split('.')[0] + '.xml', test_dir + '/' + img_title + '.xml')


def main():
    # Initiate argument parser
    ap = argparse.ArgumentParser(description="Sample dataset split")
    ap.add_argument("-i", "--img_dir",
                    help="Path to the folder where the images are stored",
                    default=os.getcwd(),
                    type=str)

    ap.add_argument("-p", "--split_pct",
                    help="Train test split percentage.",
                    default=1,
                    type=int)

    ap.add_argument("-x", "--img_ext",
                    help="Images extensions.",
                    default='jpg',
                    type=str)
    args = vars(ap.parse_args())

    train_test_split(args['img_dir'], args['split_pct'], args['img_ext'])
    print('Successfully split images into train and test splits.')


if __name__ == '__main__':
    main()
