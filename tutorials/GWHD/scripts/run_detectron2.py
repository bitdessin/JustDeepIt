import sys
import os
import glob
import numpy as np
from agrolens.models import FasterRCNN, RetinaNet, YOLO3
from agrolens.utils import ImageAnnotation


def train():
    images = 'train'
    annotation = 'train.json'
    class_label = 'class_label.txt'
    ws = 'outputs/d2'
    os.makedirs(ws, exist_ok=True)
    
    net = OD(class_label, model_arch='fasterrcnn',
             workspace=ws, backend='detectron2')
    net.train(annotation, images,
              batchsize=8, epoch=100, lr=0.0001,
              gpu=1, cpu=16)
    net.save(os.path.join(ws, 'gwhd2021.fasterrcnn.d2.pth'))


def inference():
    images = 'test'
    class_label = 'class_label.txt'
    ws = 'outputs/detectron2'
    
    net = OD(class_label, model_arch='fasterrcnn',
             model_weight=os.path.join(ws, 'gwhd2021.fasterrcnn.d2.pth'),
             workspace=ws, backend='detectron2')
    detect_outputs = net.inference(images, score_cutoff=0.7, batchsize=8, gpu=1, cpu=16)
    
    for detect_output in detect_outputs:
        detect_output.draw('bbox', os.path.join(ws, os.path.basename(detect_output.image_path)))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        inference()

