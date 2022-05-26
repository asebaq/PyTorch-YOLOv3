from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def get_image(img_path):
    img_size = 416
    # Extract image as PyTorch tensor
    img = transforms.ToTensor()(Image.open(img_path))
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, img_size)
    return img

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf_thres = 0.8
    nms_thres = 0.4
    weights_path = 'vitality_checkpoints/yolov3_ckpt_99.pth'
    model_def = 'config/yolov3-vitality.cfg'
    img_size = 416
    class_path = 'data/vitality/classes.names'
    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_path = 'data/vitality/samples/1_22.png'
    img = cv2.imread(img_path)
    print("Performing object detection:")
    prev_time = time.time()
    
    # Configure input
    input_img = get_image(img_path)
    input_img = input_img.unsqueeze_(0)
    input_img = Variable(input_img.type(Tensor))
    img_detections = []
    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("Inference Time: {}".format(inference_time))

    img_detections.extend(detections)
    for detections in img_detections:
        print(detections.shape)
        # Rescale boxes to original image
        if detections is not None:
            print("img.shape:", img.shape)
            print(detections)

            detections = rescale_boxes(detections, img_size, img.shape[:2])
            print(detections)
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            # Bounding-box colors
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 0)]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                color = colors[int(cls_pred)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)

    cv2.imwrite('results/' + img_path.split('/')[-1], img)
                    