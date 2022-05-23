import os
import shutil
import platform
import glob
import threading
import unittest
import torch
import justdeepit


class TestODAPP(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = os.path.join('sample_datasets', 'od')
        self.ws = os.path.join('outputs', 'odapp')
        os.makedirs(self.ws, exist_ok=True)
        
        self.batchsize = 4
        self.epoch = 100
        self.cutoff = 0.7
        self.lr = 0.0001
        self.gpu = 0
        self.cpu = 4
        if platform.system() == 'Darwin':
            self.cpu = 0
        if torch.cuda.is_available():
            self.gpu = 1
            self.cpu = 8


    
    def __test_app(self, model_arch, weight_fpath, images_dpath, annotations_fpath, annotation_format, output_dpath, backend):
        
        class_label_fpath = os.path.join(self.dataset, 'class_label.txt')
        
        app = justdeepit.webapp.OD(self.ws)
        app.init_workspace()
        app.save_initial_model(class_label_fpath, model_arch, None, None, backend)
        
        # training
        app.sort_train_images(class_label_fpath, images_dpath, annotations_fpath, annotation_format)
        app.train_model(class_label_fpath, model_arch, None, weight_fpath, 
                        self.batchsize, self.epoch, self.lr, self.cutoff, self.cpu, self.gpu, backend)
        
        # detection
        for fpath in glob.glob(os.path.join(images_dpath, '**')):
            shutil.copy(fpath, os.path.join(self.ws, 'query_dataset'))
        app.sort_query_images(os.path.join(self.ws, 'query_dataset'))
        app.detect_objects(class_label_fpath, model_arch, None, weight_fpath,
                           self.cutoff, self.batchsize, self.cpu, self.gpu, backend)
        
        
        # clean up
        if os.path.exists(output_dpath):
            shutil.rmtree(output_dpath)
        shutil.copytree(self.ws, output_dpath)
        os.remove(os.path.join(weight_fpath))
        if os.path.exists(os.path.join(os.path.splitext(weight_fpath)[0] + '.yaml')):
            os.remove(os.path.join(os.path.splitext(weight_fpath)[0] + '.yaml'))
        if os.path.exists(os.path.join(os.path.splitext(weight_fpath)[0] + '.py')):
            os.remove(os.path.join(os.path.splitext(weight_fpath)[0] + '.py'))
    


    def test_app_fasterrcnn_d2(self):
        output_dpath = self.ws + '_d2_fasterrcnn'
        weight_fpath = os.path.join(self.ws, 'fasterrcnn.pth')
        images_dpath = os.path.join(self.dataset, 'images')
        ann_fpath = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.__test_app('fasterrcnn', weight_fpath, images_dpath, ann_fpath, 'COCO', output_dpath, 'detectron2')
    
    
    
    def test_app_fasterrcnn_mmdet(self):
        output_dpath = self.ws + '_mmdet_fasterrcnn'
        weight_fpath = os.path.join(self.ws, 'fasterrcnn.pth')
        images_dpath = os.path.join(self.dataset, 'images')
        ann_fpath = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.__test_app('fasterrcnn', weight_fpath, images_dpath, ann_fpath, 'COCO', output_dpath, 'mmdetection')
        
   
    
    def test_app_retinanet_mmdet(self):
        output_dpath = self.ws + '_mmdet_retinanet'
        weight_fpath = os.path.join(self.ws, 'retinanet.pth')
        images_dpath = os.path.join(self.dataset, 'images')
        ann_fpath = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.__test_app('retinanet', weight_fpath, images_dpath, ann_fpath, 'COCO', output_dpath, 'mmdetection')
        
        

    def test_app_yolo3_mmdet(self):
        output_dpath = self.ws + '_mmdet_yolo3'
        weight_fpath = os.path.join(self.ws, 'yolo3.pth')
        images_dpath = os.path.join(self.dataset, 'images')
        ann_fpath = os.path.join(self.dataset, 'annotations', 'COCO', 'instances_default.json')
        self.__test_app('yolo3', weight_fpath, images_dpath, ann_fpath, 'COCO', output_dpath, 'mmdetection')



if __name__ == '__main__':
    unittest.main()

