import os
import random

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import models

from resnet_yolo import resnet50
from yolo_loss import YoloLoss
from dataset import VocDetectorDataset
from eval_voc import evaluate
from predict import predict_image_cv
from config import VOC_CLASSES, COLORS
from kaggle_submission import output_submission_csv
import matplotlib.pyplot as plt
from object_detector import ObjectDetector

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_network_path = "best_detector.pth"

print('Loading saved network from {}'.format(load_network_path))
net = resnet50().to(device)
net.load_state_dict(torch.load(load_network_path))

# create objeect detector and set kalman filter settings
new_obj_time = 5
obj_life = 5
match_tolerance = 0.1
proc_variance = 0.3
proc_covariance = 0.01
measure_variance = 0.3
measure_covariance = 0.01
obj_detector = ObjectDetector(new_obj_time, obj_life, match_tolerance, proc_variance, proc_covariance, measure_variance, measure_covariance)

# set file ins and outs
input_file = 'video_files/nyc_long.mp4'
output_file = 'video_files/nyc_long_with_filterTEST.mp4'
# turn on or off kalman filter methods
post_processing = True



# load a video capture on the input file
video_input = cv2.VideoCapture(input_file)
# get total number of frames
total_frame_count = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))

# create vid output writer with same params as the input video
w, h, fps = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)), video_input.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_output = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

# set ret to True initially
ret = True
frames_labeled = 0
print("labeling video...")

# mark start time
start = time.time()

while ret:

    ret, image = video_input.read()

    # if there is no next image then break
    if not ret:
        break

    # print progrss
    if frames_labeled % 100 == 0:
        print("Finished {:.2f}%".format(100 * frames_labeled / total_frame_count))

    best_boxes = predict_image_cv(net, image)
    obj_detector.update_objects(best_boxes)

    if post_processing:
        best_boxes = obj_detector.get_best_boxes()

    for box in best_boxes:

        left_up = box[0]
        right_bottom = box[1]
        class_name = box[2]
        prob = box[3]

        color = COLORS[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    video_output.write(image)
    frames_labeled += 1

# mark end time
end = time.time()

print("Finished {:.2f}%".format(100))
print("Time elapsed: {:.2f}".format(end - start))

# close video captures
video_input.release()
video_output.release()