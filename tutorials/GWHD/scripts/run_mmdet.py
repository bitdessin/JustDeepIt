import sys
import os
import numpy as np
import torch
import glob
from justdeepit.models import OD


def train():
    images = 'train'
    annotation = 'train.json'
    class_label = 'class_label.txt'
    ws = 'outputs/mmdet'
    os.makedirs(ws, exist_ok=True)
    
    # download faster_rcnn_r101_fpn_mstrain_3x_coco_20210524_110822-4d4d2ca8.pth from
    # https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn
    init_weight = os.path.join(os.path.dirname(__file__),
                    'faster_rcnn_r101_fpn_mstrain_3x_coco_20210524_110822-4d4d2ca8.pth')
    
    net = OD(class_label, model_arch='fasterrcnn', model_weight=init_weight,
             workspace=ws, backend='mmdetection')
    net.train(annotation, images,
              batchsize=8, epoch=1000, lr=0.001,
              gpu=1, cpu=16)
    net.save(os.path.join(ws, 'gwhd2021.fasterrcnn.mmdet.pth'))


def inference():
    images = 'test'
    class_label = 'class_label.txt'
    ws = 'outputs/mmdet'
    
    net = OD(class_label, model_arch='fasterrcnn',
             model_weight=os.path.join(ws, 'gwhd2021.fasterrcnn.mmdet.pth'),
             workspace=ws, backend='mmdetection')
    detect_outputs = net.inference(images, score_cutoff=0.7, batchsize=8, gpu=1, cpu=16)
    
    for detect_output in detect_outputs:
        detect_output.draw('bbox', os.path.join(ws, os.path.basename(detect_output.image_path)))



if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        inference()

