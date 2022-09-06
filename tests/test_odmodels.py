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



class TestD2(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset = os.path.join('sample_datasets', 'od')
        self.ws = os.path.join('outputs', 'odmodel_detectron2')
        self.tmp = os.path.join(self.ws, 'tmp')
        
        self.train_images = os.path.join(self.dataset, 'images')
        self.query_images = os.path.join(self.dataset, 'images')
        self.coco = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.voc = os.path.join(self.dataset, 'annotations', 'PascalVOC')
        self.class_label = os.path.join(self.dataset, 'class_label.txt')
        
        self.batchsize = 4
        self.epoch = 110
        self.lr = 0.001
        self.gpu = 0
        self.cpu = 4
        if platform.system() == 'Darwin':
            self.cpu = 0
        if torch.cuda.is_available():
            self.gpu = 1
            self.cpu = 8
        
        os.makedirs(self.ws, exist_ok=True)  
        os.makedirs(self.tmp, exist_ok=True)  

    
    
    def __test_model(self, model):
        
        # training
        net = OD(self.class_label, model_arch=model, workspace=self.tmp, backend='detectron2')
        trained_weight = os.path.join(self.ws, model + '.pth')
        net.train(self.train_images, self.coco,
                  batchsize=self.batchsize, epoch=self.epoch, lr=self.lr,
                  gpu=self.gpu, cpu=self.cpu)
        net.save(trained_weight)
        
        
        # detection
        net = OD(self.class_label, model_arch=model, model_weight=trained_weight, workspace=self.tmp, backend='detectron2')
        outputs = net.inference(self.query_images, batchsize=self.batchsize, gpu=self.gpu, cpu=self.cpu)
        for output in outputs:
            output.draw('bbox', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + '.' + model + '.png'),
                        label=True, score=True)
        outputs.format('coco', os.path.join(self.ws, 'result_coco.' + model + '.json'))
        
        
    
    def test_fasterrcnn(self):
        self.__test_model('fasterrcnn')
   
    
    
    def test_retinanet(self):
        self.__test_model('retinanet')



    def test_voc(self):

        # training
        model = 'fasterrcnn'
        net = OD(self.class_label, model_arch=model, workspace=self.tmp, backend='detectron2')
        trained_weight = os.path.join(self.ws, model + '.voc.pth')
        net.train(self.train_images, self.voc, 'voc',
                  batchsize=self.batchsize, epoch=self.epoch, lr=self.lr,
                  gpu=self.gpu, cpu=self.cpu)
        net.save(trained_weight)

        # detection
        net = OD(self.class_label, model_arch=model, model_weight=trained_weight, workspace=self.tmp, backend='detectron2')
        outputs = net.inference(self.query_images, batchsize=self.batchsize, gpu=self.gpu, cpu=self.cpu)
        for output in outputs:
            output.draw('bbox', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + '.' + model + '.voc.png'),
                        label=True, score=True)
        outputs.format('coco', os.path.join(self.ws, 'result_coco.' + model + '.voc.json'))




class TestMMDet(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset = os.path.join('sample_datasets', 'od')
        self.ws = os.path.join('outputs', 'odmodel_mmdet')
        self.tmp = os.path.join(self.ws, 'tmp')
        
        self.train_images = os.path.join(self.dataset, 'images')
        self.query_images = os.path.join(self.dataset, 'images')
        self.coco = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.voc = os.path.join(self.dataset, 'annotations', 'PascalVOC')
        self.class_label = os.path.join(self.dataset, 'class_label.txt')
        
        self.batchsize = 4
        self.epoch = 510
        self.lr = 0.001
        self.cutoff = 0.7
        self.gpu = 0
        self.cpu = 2
        if platform.system() == 'Darwin':
            self.cpu = 0
        if torch.cuda.is_available():
            self.gpu = 1
            self.cpu = 8
        
        os.makedirs(self.ws, exist_ok=True)  
        os.makedirs(self.tmp, exist_ok=True)  
    
    
    
    def __test_model(self, model, gpu):
        
        if not torch.cuda.is_available():
            print('>>> MMDetection does not support CPU training, skipped test. >>>')
            return True
        
        # training
        net = None
        net = OD(self.class_label, model_arch=model, workspace=self.tmp, backend='mmdet')
        trained_weight = os.path.join(self.ws, model + '.pth')
        net.train(self.train_images, self.coco,
                  batchsize=self.batchsize, epoch=self.epoch, lr=self.lr, score_cutoff=self.cutoff,
                  cpu=self.cpu, gpu=gpu)
        net.save(trained_weight)
        
        
        # detection
        net = OD(self.class_label, model_arch=model, model_weight=trained_weight, workspace=self.tmp, backend='mmdet')
        outputs = net.inference(self.query_images, batchsize=self.batchsize, gpu=self.gpu, cpu=self.cpu)
        for output in outputs:
            output.draw('bbox', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + '.' + model + '.png'),
                        label=True, score=True)
        outputs.format('coco', os.path.join(self.ws, 'result_coco.' + model + '.json'))
    
    
    
    
    def test_fasterrcnn(self):
        self.__test_model('fasterrcnn', self.gpu)


    
    def test_cascadercnn(self):
        self.__test_model('cascadercnn', self.gpu)


    
    def test_dynamicrcnn(self):
        self.__test_model('dynamicrcnn', self.gpu)


    
    def test_retinanet(self):
        self.__test_model('retinanet', self.gpu)



    def test_yolo3(self):
        self.__test_model('yolov3', self.gpu)



    def test_ssd(self):
        self.__test_model('ssd', self.gpu)
    
    
    
    def test_fcos(self):
        self.__test_model('fcos', self.gpu)
    
    
    def test_voc(self):
        
        if not torch.cuda.is_available():
            print('>>> MMDetection does not support CPU training, skipped test. >>>')
            return True

        # training
        model = 'fasterrcnn'
        net = OD(self.class_label, model_arch=model, workspace=self.tmp, backend='mmdet')
        trained_weight = os.path.join(self.ws, model + '.voc.pth')
        net.train(self.train_images, self.voc, 'voc',
                  batchsize=self.batchsize, epoch=self.epoch, lr=self.lr,
                  cpu=self.cpu, gpu=self.gpu)
        net.save(trained_weight)
        
        # detection
        net = OD(self.class_label, model_arch=model, model_weight=trained_weight, workspace=self.tmp, backend='mmdet')
        outputs = net.inference(self.query_images, batchsize=self.batchsize, gpu=self.gpu, cpu=self.cpu)
        for output in outputs:
            output.draw('bbox', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + '.' + model + '.voc.png'),
                        label=True, score=True)
        outputs.format('coco', os.path.join(self.ws, 'result_coco.' + model + '.voc.json'))



if __name__ == '__main__':
    unittest.main()

