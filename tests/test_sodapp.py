import os
import platform
import shutil
import glob
import threading
import unittest
import torch
import justdeepit



class TestSODAPPMask(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset = os.path.join('sample_datasets', 'sod')
        self.ws = os.path.join('outputs', 'sodappmask')
        os.makedirs(self.ws, exist_ok=True)    
        os.makedirs(os.path.join(self.ws, 'train_dataset'), exist_ok=True)    
        os.makedirs(os.path.join(self.ws, 'query_dataset'), exist_ok=True)    
        
        self.image_suffix = '_image.jpg'
        self.mask_suffix = '_mask.png'
        
        self.batchsize = 4
        self.epoch = 110
        self.lr = 0.001
        self.gpu = 0
        self.cpu = 4
        if torch.cuda.is_available():
            self.gpu = 1
            self.cpu = 8
        
        
    
    def __test_app(self, weight, strategy, is_series):
        inference_strategy = 'resize' if strategy == 'resize' else 'slide'
        
        app = justdeepit.webapp.SOD(self.ws)
        app.init_workspace()
        
        for fpath in glob.glob(os.path.join(self.dataset, '*.jpg')):
            shutil.copy(fpath, os.path.join(self.ws, 'train_dataset'))
            shutil.copy(fpath, os.path.join(self.ws, 'query_dataset'))
        for fpath in glob.glob(os.path.join(self.dataset, '*.png')):
            shutil.copy(fpath, os.path.join(self.ws, 'train_dataset'))
        
        # training
        app.sort_train_images(os.path.join(self.ws, 'train_dataset'), None, 'mask',
                              self.image_suffix, self.mask_suffix)
        app.train_model('u2net', weight, self.batchsize, self.epoch, self.lr, self.cpu, self.gpu, strategy, 340)
        
        # detection
        app.sort_query_images(os.path.join(self.ws, 'query_dataset'))
        app.detect_objects('u2net', weight, self.batchsize, inference_strategy,
                           0.8, 0, 0, 340, self.cpu, self.gpu)
        
        # summarization
        app.summarize_objects(self.cpu, False, 0, 0)
        
    
    
    def test_app_1(self):
        weight = os.path.join(self.ws, 'sodapp.t1.pth')
        self.__test_app(weight, 'resize', False)
    
    def test_app_2(self):
        weight = os.path.join(self.ws, 'sodapp.t2.pth')
        self.__test_app(weight, 'randomcrop', True)
    


if __name__ == '__main__':
    unittest.main()


