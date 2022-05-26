from glob import glob
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


def correct_xmls(path):
    for xml_file in tqdm(glob(os.path.join(path, '*.xml'))):
        title, ext = os.path.splitext(os.path.basename(xml_file))
        tree = ET.parse(xml_file)
        root = tree.getroot()
        root.find('folder').text = xml_file.split('/')[-2]
        root.find('filename').text = title + '.jpg'
        root.find('path').text = xml_file[:-3] + 'jpg'
        tree.write(xml_file)
    print('Successfully corrected xml files.')


if __name__ == '__main__':
    data_path = './data/waste/all_images'
    correct_xmls(data_path)
