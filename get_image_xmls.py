# USAGE
# python get_image_xmls.py --img_dir data/waste/test --img_ext jpg

import glob
import os
import sys
import argparse


def get_images(img_dir, img_ext):
    files_dir = os.path.normpath(os.path.join(img_dir, '../files'))
    os.makedirs(files_dir, exist_ok=True)

    for fullpath in glob.glob(os.path.join(img_dir, '*.xml')):
        title, ext = os.path.splitext(os.path.basename(fullpath))
        img_path = fullpath[:-3] + img_ext
        if os.path.isfile(img_path):
            # os.replace(fullpath.replace('/', '\\'), files_dir + '\\' + title + ext)
            # os.replace(fullpath[:-3].replace('/', '\\') + img_ext, files_dir + '\\' + title + '.' + img_ext)
            os.replace(fullpath, files_dir + '/' + title + ext)
            os.replace(fullpath[:-3]+ img_ext, files_dir + '/' + title + '.' + img_ext)

        #     print(fullpath[:-3].replace('/', '\\') + img_ext)
        #     print(files_dir + '\\' + title + '.' + img_ext)
        #     print(fullpath.replace('/', '\\'))
        #     print(files_dir + '\\' + title + ext)
        # break


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
  
    get_images(args['img_dir'], args['img_ext'])
    print('Successfully got images with xmls.')


if __name__ == '__main__':
    main()
