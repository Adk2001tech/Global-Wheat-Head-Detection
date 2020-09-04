# COLNE REPO. AND INSTALL ALL DEPENDENCY
#!git clone https://github.com/fizyr/keras-retinanet.git
#!pip install .
#!python setup.py build_ext --inplace

import os
import numpy as np

import matplotlib.pyplot as plt
import cv2

import time
from tqdm.notebook import tqdm


import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

path= 'resnet50_csv_07.h5'

model = models.load_model(path, backbone_name='resnet50')
model = models.convert_model(model)

print(model.summary())



#############   MAIN FUNCTION   #################

def perd_from_model(model, image, th=0.5, box_only=False):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # correct for image scale
    boxes /= scale
    
    if box_only:
        return scores, boxes

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)
    return draw

#######################################


video = cv2.VideoCapture('../../wheat2.mp4') 
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(video.get(3)) 
frame_height = int(video.get(4)) 

size = (frame_width, frame_height) 

# Below VideoWriter object will create 
# a frame of above defined The output 
# is stored in 'filename.avi' file. 
result = cv2.VideoWriter('../../wheat22.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 
	
for i in tqdm(range(length)):
    ret, frame = video.read() 
    if ret:
      frame= perd_from_model(model, frame, th=0.3, box_only=False)

      result.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) 
    else: 
      break

# When everything done, release 
# the video capture and video 
# write objects 
video.release() 
result.release() 
	

print("The video was successfully saved")
