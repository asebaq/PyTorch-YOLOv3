import os
from glob import glob
from tqdm import tqdm


def get_name(filename):
    return filename.split('/')[-1][:-4]


def main():
    data_path = './data/waste/all_images'
    images = glob(os.path.join(data_path, '*.jpg'))
    images.sort()
    images = [get_name(image) for image in images]
    print(len(images))

    xmls = glob(os.path.join(data_path, '*.xml'))
    xmls.sort()
    xmls = [get_name(xml) for xml in xmls]
    print(len(xmls))

    for image in tqdm(images):
        if image not in xmls:
            print(f'Image {image} does not have an XML file')

    for xml in tqdm(xmls):
        if xml not in images:
            print(f'XML {xml} does not have an image file')


if __name__ == '__main__':
    main()
