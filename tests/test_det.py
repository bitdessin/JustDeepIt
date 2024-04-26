import os
import platform
import glob
import logging
import unittest
import torch
from justdeepit.models import OD

logging.basicConfig(level=logging.WARNING)
logging.getLogger('detectron2.utils.events').setLevel(level=logging.WARNING)
logging.getLogger('fvcore.common.checkpoint').setLevel(level=logging.WARNING)
logging.getLogger('mmdet').setLevel(level=logging.WARNING)



class TestMMDet(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset = os.path.join('sample_datasets', 'det')
        self.train_images = os.path.join(self.dataset, 'images')
        self.train_ann_coco = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.train_ann_voc = os.path.join(self.dataset, 'annotations', 'PascalVOC')
        self.class_label = os.path.join(self.dataset, 'class_label.txt')
        self.query_images = os.path.join(self.dataset, 'images')

        self.ws = os.path.join('outputs', 'test_det')
        self.tmp = os.path.join(self.ws, 'tmp')
        os.makedirs(self.ws, exist_ok=True)  
        os.makedirs(self.tmp, exist_ok=True)  
    
    
    def __test_model(self, model_arch):
        train_images = {
            'images': self.train_images,
            'annotations': self.train_ann_coco,
            'annotation_format': 'coco'
        }

        # training
        m = OD(self.class_label, model_arch=model_arch, workspace=self.tmp)
        trained_weight = os.path.join(self.ws, f'{model_arch}.pth')
        m.train(train_images, batchsize=4, epoch=120, gpu=1, cpu=8)
        m.save(trained_weight)
        
        # detection
        m = OD(self.class_label, model_arch=model_arch, model_weight=trained_weight, workspace=self.tmp)
        outputs = m.inference(self.query_images, batchsize=4, gpu=1, cpu=8)
        for output in outputs:
            output.draw('bbox', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + f'.{model_arch}.png'),
                        label=True, score=True, alpha=0.3)
        outputs.format('coco', os.path.join(self.ws, f'result_coco.{model_arch}.son'))
    
    
    
    def test_fasterrcnn(self):
        self.__test_model('fasterrcnn')


    
    def test_retinanet(self):
        self.__test_model('retinanet')



    def test_yolo3(self):
        self.__test_model('yolov3')



#    # failed on mmdet-v3
 #   def test_ssd(self):
  #      self.__test_model('ssd')
    
    
    
    def test_fcos(self):
        self.__test_model('fcos')
    

    
    def test_voc(self):
        model_arch = 'fasterrcnn'
        train_images = {
            'images': self.train_images,
            'annotations': self.train_ann_voc,
            'format': 'voc'
        }

        # training
        m = OD(self.class_label, model_arch=model_arch, workspace=self.tmp)
        trained_weight = os.path.join(self.ws, f'{model_arch}.voc.pth')
        m.train(train_images, batchsize=4, epoch=120, gpu=1, cpu=8)
        m.save(trained_weight)
        
        # detection
        m = OD(self.class_label, model_arch=model_arch, model_weight=trained_weight, workspace=self.tmp)
        outputs = m.inference(self.query_images, batchsize=4, gpu=1, cpu=8)
        for output in outputs:
            output.draw('bbox', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + f'.{model_arch}.voc.png'),
                        label=True, score=True)
        outputs.format('coco', os.path.join(self.ws, f'result_coco.{model_arch}.voc.json'))



if __name__ == '__main__':
    unittest.main()

