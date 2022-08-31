import sys
import os
import joblib
import numpy as np
import torch
import glob
from justdeepit.models import IS
from justdeepit.utils import ImageAnnotation, ImageAnnotations


def train(dataset_dpath, model_backend):
    
    # traininng images
    train_images = os.path.join(dataset_dpath, 'train')
    train_images_annotation = os.path.join(dataset_dpath, 'train.json')
    class_label = os.path.join(dataset_dpath, 'class_label.txt')
    # temporary folder
    ws_dpath = os.path.join('outputs', model_backend) 
    os.makedirs(ws_dpath, exist_ok=True)
    
    net = IS(class_label, model_arch='maskrcnn', workspace=ws_dpath, backend=model_backend)
    net.train(train_images_annotation, train_images,
              batchsize=4, epoch=100, lr=0.0001, gpu=1, cpu=16)
    net.save(os.path.join(ws_dpath, 'sugarbeets.pth'))


def test(dataset_dpath, model_backend):
    
    # test images
    test_images = os.path.join(dataset_dpath, 'test')
    class_label = os.path.join(dataset_dpath, 'class_label.txt')
    # temporary folder
    ws_dpath = os.path.join('outputs', model_backend) 
    trained_weight = os.path.join(ws_dpath, 'sugarbeets.pth')
    
    net = IS(class_label, model_arch='maskrcnn',
             model_weight=trained_weight, workspace=ws_dpath, backend=model_backend)
    detect_outputs = net.inference(images, score_cutoff=0.5, batchsize=4, gpu=1, cpu=16)
    for detect_output in detect_outputs:
        output_img_fpath = os.path.join(ws_dpath, os.path.basename(detect_output.image_path))
        detect_output.draw('rgbmask', output_img_fpath,
                           class2rgb={'weeds': [255, 0, 0], 'sugarbeets': [0, 255, 0]})
     
    img_anns = ImageAnnotations(detect_outputs)
    img_anns.format('coco', os.path.join(ws_dpath, 'annotation.json'))




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_justdeepit.py train')
        print('       python scripts/run_justdeepit.py test')
        sys.exit(0)
    
    run_mode = sys.argv[1]
    model_backend = sys.argv[2] if len(sys.argv) >= 3 else 'detectron2'
    
    dataset_dpath = '.'
    
    if run_mode == 'train':
        train(dataset_dpath, model_backend)
    elif run_mode == 'test':
        test(dataset_dpath, model_backend)
    else:
        print('unsupported running mode, set `train` or `test`.')



