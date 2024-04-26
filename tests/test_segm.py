import os
import platform
import glob
import logging
import unittest
import torch
from justdeepit.models import IS

logging.basicConfig(level=logging.WARNING)
logging.getLogger('detectron2.utils.events').setLevel(level=logging.WARNING)
logging.getLogger('fvcore.common.checkpoint').setLevel(level=logging.WARNING)
logging.getLogger('mmdet').setLevel(level=logging.WARNING)


class TestMMDet(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset = os.path.join('sample_datasets', 'segm')
        self.train_images = os.path.join(self.dataset, 'images')
        self.train_ann = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.class_label = os.path.join(self.dataset, 'class_label.txt')
        self.query_images = os.path.join(self.dataset, 'images')

        self.ws = os.path.join('outputs', 'test_segm')
        self.tmp = os.path.join(self.ws, 'tmp')
        os.makedirs(self.ws, exist_ok=True)  
        os.makedirs(self.tmp, exist_ok=True)

    
    def __test_model(self, model_arch):
        train_images = {
            'images': self.train_images,
            'annotations': self.train_ann,
            'annotation_format': 'coco'
        }

        # training
        m = IS(self.class_label, model_arch=model_arch, workspace=self.tmp)
        trained_weight = os.path.join(self.ws, f'{model_arch}.pth')
        m.train(train_images, batchsize=4, epoch=120, gpu=1, cpu=8)
        m.save(trained_weight)
        
        # inference
        m = IS(self.class_label, model_arch=model_arch, model_weight=trained_weight, workspace=self.tmp)
        outputs = m.inference(self.query_images, batchsize=4, gpu=1, cpu=8)
        for output in outputs:
            output.draw('bbox+contour', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + f'.{model_arch}.png'),
                        label=True, score=True, alpha=0.3)
        outputs.format('coco', os.path.join(self.ws, f'result_coco.{model_arch}.son'))
        

    def test_maskrcnn(self):
        self.__test_model('maskrcnn')


    def test_cascademaskrcnn(self):
        self.__test_model('cascademaskrcnn')



if __name__ == '__main__':
    unittest.main()

